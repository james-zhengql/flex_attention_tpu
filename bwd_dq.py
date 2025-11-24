import functools
from typing import Callable

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax import random
import jax.numpy as jnp
import numpy as np
import time
import statistics as stats


dimension_numbers = (((1,), (1,)), ((), ()))
MIN_BLOCK_SIZE = 128

def mha_reference(
    q,
    k,
    v,
    ab: jax.Array | None = None,
    *,
    sm_scale: float = 1.0,
    save_residuals: bool = False,
):
  logits = jnp.einsum("bhqc,bhkc->bhqk", q, k)
  if ab is not None:
    logits += ab
  if sm_scale != 1.0:
    logits *= sm_scale

  # --- causal masking (disabled for now but can enable later)
  mask = None
  # if causal:
  #   _, _, q_seq_len, _ = q.shape
  #   _, _, kv_seq_len, _ = k.shape
  #   mask_shape = (q_seq_len, kv_seq_len)
  #   row_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 0)
  #   col_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 1)
  #   causal_mask = (col_ids <= row_ids)[None, None, :, :]
  #   mask = causal_mask if mask is None else jnp.logical_and(mask, causal_mask)

  logits = logits if mask is None else logits + jnp.where(mask, 0.0, -1e9)

  m = logits.max(axis=-1)
  unnormalized = jnp.exp(logits - m[..., None])
  l = unnormalized.sum(axis=-1)
  weights = unnormalized / l[..., None]
  out = jnp.einsum("bhqk,bhkc->bhqc", weights, v)
  if save_residuals:
    return out, l, m
  return out

def flop_count_attention(b, h, q, k, d):
    """
    Rough FLOP count for one forward pass of scaled dot-product attention:
      QK^T  : 2 * b * h * q * k * d        (matrix multiplication)
      Softmax: ~ b * h * q * k             (small, we ignore it)
      (softmax @ V): 2 * b * h * q * k * d (another matmul)
    Total ≈ 4 * b * h * q * k * d FLOPs
    """
    return 4.0 * b * h * q * k * d

def mha_bwd_reference(
    q,
    k,
    v,
    o,
    do,
    l,
    m,
    ab: jax.Array | None = None,
    *,
    causal: bool = False,
    mask_value: float = None,
    sm_scale: float = 1.0,
    save_residuals: bool = True,
):

  # logits = jnp.einsum("bhqc,bhkc->bhqk", q, k)
  # if ab is not None:
  #   logits += ab
  # if sm_scale != 1.0:
  #   logits *= sm_scale

  # # # # no causal masking
  # # # mask = None
  # # # logits = logits if mask is None else logits + jnp.where(mask, 0.0, -1e9)

  # # m = logits.max(axis=-1)
  # unnormalized = jnp.exp(logits - m[..., None])
  # # l = unnormalized.sum(axis=-1)
  # # weights = unnormalized / l[..., None]
  # # o = jnp.einsum("bhkq,bhkc->bhqc", weights, v)

  # weights = unnormalized / l[..., None]

  # # dv = P^T * do
  # dv = jnp.einsum("bhqk,bhqc->bhkc", weights, do)

  # # dp = do * V^T
  # dp = jnp.einsum("bhqc,bhkc->bhqk", do, v)

  # # software backward
  # sum_d = jnp.sum(do*o, axis=-1)
  # # ds = weights * (dp - sum_d)
  # ds = weights * (dp - sum_d[..., None])

  # # dq = ds * k
  # dq = jnp.einsum("bhqk,bhkc->bhqc", ds, k)

  # # dk = ds^T * q
  # dk = jnp.einsum("bhqk,bhqc->bhkc", ds, q)

  # return dq, dk, dv

  logits = jnp.einsum(
      "bhqc,bhkc->bhqk",
      q.astype(jnp.float32),
      k.astype(jnp.float32),
  )

  if sm_scale != 1.0:
    logits *= sm_scale

  unnormalized = jnp.exp(logits - m[..., None])
  p = unnormalized / l[..., None]
  dv = jnp.einsum("bhpt,bhpd->bhtd", p, do.astype(jnp.float32)).astype(v.dtype)

  dp = jnp.einsum(
      "bhpd,bhtd->bhpt", do.astype(jnp.float32), v.astype(jnp.float32)
  )

  di = jnp.sum(o.astype(jnp.float32) * do.astype(jnp.float32), axis=-1)[
      ..., None
  ]  # [batch_size, num_heads, q_seq_len]

  ds = (dp - di) * p
  ds = ds * sm_scale

  dk = jnp.einsum("bhsd,bhst->bhtd", q.astype(jnp.float32), ds).astype(k.dtype)
  dq = jnp.einsum("bhst,bhtd->bhsd", ds, k.astype(jnp.float32)).astype(q.dtype)

  # dab is just ds
  dab = ds if ab is not None else None

  return dq, dk, dv

def flash_attention_bwd_dq(
    q,
    k,
    v,
    ab,
    segment_ids,
    l,
    m,
    o,
    do,
    d,
    *,
    block_b,
    block_q_major: int | None,
    block_k_major: int | None,
    block_k: int | None,
    sm_scale: float,
    causal: bool,
    mask_value: float,
    debug: bool,
):
    m = jnp.broadcast_to(m[..., None], (*m.shape, MIN_BLOCK_SIZE))
    l = jnp.broadcast_to(l[..., None], (*l.shape, MIN_BLOCK_SIZE))
    # Preprocess contraction for bwd pass
    d = jnp.broadcast_to(d[..., None], (*d.shape, block_k_major))
    batch_size, head_num, q_seq_len, head_dim = q.shape
    _, _, kv_seq_len, _ = k.shape

    # Grid specification
    grid = (
        pl.cdiv(batch_size, block_b),
        head_num,
        pl.cdiv(q_seq_len, block_q_major),
        pl.cdiv(kv_seq_len, block_k_major),
    )


    # input qo
    def qo_index_map(batch_index, head_index, q_seq_index, _):
        return (batch_index, head_index, q_seq_index, 0)

    qo_spec = pl.BlockSpec((block_b, 1, block_q_major, head_dim), qo_index_map)
    do_spec = qo_spec
    assert qo_spec.block_shape is not None
    assert q.ndim == len(qo_spec.block_shape)
    assert o.ndim == len(qo_spec.block_shape)
    assert do.ndim == len(qo_spec.block_shape)

    # input kv
    def kv_index_map(batch_index, head_index, q_seq_index, kv_seq_index):
        next_kv_index = kv_seq_index
        return (batch_index, head_index, next_kv_index, 0)

    kv_spec = pl.BlockSpec((block_b, 1, block_k_major, head_dim), kv_index_map)
    assert kv_spec.block_shape is not None
    assert k.ndim == len(kv_spec.block_shape)
    assert v.ndim == len(kv_spec.block_shape)

    # input lm
    def lm_index_map(batch_index, head_index, q_seq_index, _):
        return (batch_index, head_index, q_seq_index, 0)

    lm_spec = pl.BlockSpec((block_b, 1, block_q_major, MIN_BLOCK_SIZE), lm_index_map)
    assert lm_spec.block_shape is not None
    assert l.ndim == len(lm_spec.block_shape)
    assert m.ndim == len(lm_spec.block_shape)

    def ab_index_map(batch_index, head_index, q_seq_index, kv_seq_index):
        return (batch_index, head_index, q_seq_index, kv_seq_index)

    dab_spec = (
      pl.BlockSpec((block_b, 1, block_q_major, block_k_major), ab_index_map)
      if ab is not None
      else None
    )

    def d_index_map(batch_index, head_index, q_seq_index, _):
        return (batch_index, head_index, q_seq_index, 0)
    # d_spec = pl.BlockSpec((batch_size,head_num,kv_seq_len,MIN_BLOCK_SIZE),d_index_map)
    d_spec = pl.BlockSpec((batch_size, 1, block_q_major, MIN_BLOCK_SIZE), d_index_map)
    assert d_spec.block_shape is not None

    # Allocate scratch buffers
    
    dq_scratch = pltpu.VMEM((block_b, 1, block_q_major, head_dim), jnp.float32)
    scratch_shapes = [dq_scratch]


    dq_spec = pl.BlockSpec((block_b, 1, block_q_major, head_dim), qo_index_map)

    # in_spec specify
    in_specs = [
      qo_spec,      # q
      kv_spec,      # k
      kv_spec,      # v
      dab_spec,     # bias
      lm_spec,      # l
      lm_spec,      # m
      do_spec,      # do
      d_spec,       # d
    ]

    # out_spec specify
    out_specs = [
      dq_spec,
    ]

    out_shapes = [
      jax.ShapeDtypeStruct(q.shape, q.dtype),
    ]


    kernel = functools.partial(
        _flash_attention_bwd_dq,
        causal = causal,
        sm_scale = sm_scale,
        block_q = block_q_major,
        block_k = block_k_major,
        q_seq_len = q_seq_len,
        kv_seq_len = kv_seq_len
    )

    dq, *aux = pl.pallas_call(
      kernel,
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=0,
          grid=grid,
          in_specs=in_specs,
          out_specs=out_specs,
          scratch_shapes=scratch_shapes,
      ),
      out_shape=out_shapes,
      debug=debug,
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "parallel", "parallel", "arbitrary")
      ),
    )(q, k, v, ab, l, m, do, d)

    return dq

def _flash_attention_bwd_dq(q_tile_ref, *args, **kwargs):

    block_b = q_tile_ref.shape[0]
    # Create the real kernel from the factory
    kernel = flash_attention_bwd_dq_kernel

    for batch_idx in range(block_b):
        kernel(
            (batch_idx, 0),
            q_tile_ref,
            *args,
            **kwargs,
        )

def flash_attention_bwd_dq_kernel(
      batch_idx,
      q_tile_ref,
      k_tile_ref,
      v_tile_ref,
      ab_tile_ref,
      l_tile_ref,
      m_tile_ref,
      dO_tile_ref,
      di_tile_ref,
      dq_tile_ref,
      dq_scratch_ref,
      *,
      causal,
      sm_scale,
      block_q,
      block_k,
      q_seq_len,
      kv_seq_len,
  ):

    kv_tile_idx = pl.program_id(axis = 3)
    q_tile_idx = pl.program_id(axis = 2)
    _, _, block_k_major, _ = k_tile_ref.shape

    block_k_repeats, rem = divmod(block_k, MIN_BLOCK_SIZE)
    # block_q_repeats, rem = divmod(block_q, MIN_BLOCK_SIZE)

    @pl.when(kv_tile_idx == 0)
    def start_new_seq():
        dq_scratch_ref[batch_idx] = jnp.zeros(dq_scratch_ref.shape[2:], dq_scratch_ref.dtype)

    @pl.when(True)
    def body():
        @pl.loop(0, block_k_major, step=block_k, unroll=True)
        def _body(start_kv):
            k = k_tile_ref[(*batch_idx, pl.dslice(start_kv, block_k), slice(None))]
            v = v_tile_ref[(*batch_idx, pl.dslice(start_kv, block_k), slice(None))]

            q = q_tile_ref[batch_idx]

            l = l_tile_ref[batch_idx]
            # jax.print.debug(f"l shape {l.shape}")
            m = m_tile_ref[batch_idx]

            # l = l.reshape((1, 1, block_q, MIN_BLOCK_SIZE))

            dO = dO_tile_ref[batch_idx]
            D = di_tile_ref[batch_idx]

            dq_past = dq_scratch_ref[batch_idx]

            S = jax.lax.dot_general(q, k, dimension_numbers, preferred_element_type=jnp.float32)
            S = S * sm_scale
            
            unnormalized = jnp.exp(S - pltpu.repeat(m, block_k_repeats, axis=1))
            # jax.debug.print(jnp.linalg.norm(unnormalized))
            # pl.debug_print("logits norm: ", jnp.linalg.norm(unnormalized))
            P = unnormalized / pltpu.repeat(l, block_k_repeats, axis=1)
            # P = unnormalized

            dP = jax.lax.dot_general(dO,v,dimension_numbers,preferred_element_type=jnp.float32)
            dS = P * (dP  - pltpu.repeat(D, block_k_repeats, axis=1))
            dS = dS * sm_scale

            dq = dq_past + jax.lax.dot_general(dS,k,(((1,), (0,)), ((), ())),preferred_element_type=jnp.float32)

            dq_scratch_ref[batch_idx] = dq.astype(dq_scratch_ref.dtype)


    @pl.when(kv_tile_idx == (kv_seq_len // block_k) - 1)
    def store_res():
        dq_tile_ref[batch_idx] = dq_scratch_ref[batch_idx].astype(dq_tile_ref.dtype)



import math
# assume jax, jax.random, jnp, and your helper functions are already imported:
# from jax import random
# import jax.numpy as jnp
# from your_module import mha_reference, _flash_attention_impl, _flash_attention_impl_ref

def main():
    key = random.PRNGKey(0)
    batch = 1
    heads = 1
    q_len = 128
    kv_len = 25600
    head_dim = 128

    k1, k2, k3 = random.split(key, 3)
    q = random.normal(k1, (batch, heads, q_len, head_dim), dtype=jnp.float32)
    k = random.normal(k2, (batch, heads, kv_len, head_dim), dtype=jnp.float32)
    v = random.normal(k3, (batch, heads, kv_len, head_dim), dtype=jnp.float32)
    ab = None
    segment_ids = None

    block_b = 1
    block_q = 128
    block_q_major = 128
    block_k_major = 256
    block_k = 256

    causal = False
    # stable scalar softmax scale
    sm_scale = float(1.0/jnp.sqrt(head_dim).astype(jnp.float32))
    # sm_scale = float(1.0)
    print(sm_scale)
    debug = False
    save_residuals = True

    print("Running reference attention (for numeric check)...")
    # Call mha_reference with save_residuals=True for consistent output structure
    ref_output = mha_reference(q, k, v, sm_scale=sm_scale, save_residuals=save_residuals)
    # Unpack ref_output as it will return (out, l, m) when save_residuals=True
    o, l, m = ref_output


    print("Running Pallas TPU flash attention kernel...")
    # out = —
    # # Assign the reference outputs directly for comparison
    # out_ref_o = ref_o
    # out_ref_l = ref_l
    # out_ref_m = ref_m

    # # unpack outputs correctly whether residuals were saved or not
    # if save_residuals:
    #     o, l, m = out
    #     # out_ref_o, out_ref_l, out_ref_m are already set from ref_output
    # else:
    #     o = out
    #     # out_ref_o is already set from ref_output

    # correctness check (compare output tensors)
    # diff = jnp.linalg.norm(o - out_ref_o) / jnp.linalg.norm(ref_o)
    # print(f"Relative L2 error vs reference: {diff:.3e}")
    # print("Output shape:", o.shape)

    # print("l output shape: ", l.shape)
    # print("m output shape: ", m.shape)

    # # performance comparison (disabled)
    # # results = run_bench_suite(...)
    # # print("\nSummary:", results)

    key = jax.random.PRNGKey(0)
    do = jax.random.normal(key, o.shape)
    d = jnp.sum(o.astype(jnp.float32) * do.astype(jnp.float32), axis=-1)
    print(d.shape)

    print("Running reference attention backward (for numeric check)...")
    dq_ref, dk_ref, dv_ref = mha_bwd_reference(q, k, v, o, do, l, m, sm_scale=sm_scale)
    print(dq_ref.shape)

    print("Running Pallas TPU flash attention bwd kernel...")
    bwd_dq = flash_attention_bwd_dq(
        q=q,k=k,v=v,ab=ab,segment_ids=segment_ids,l=l,m=m,o=o,do=do,d=d,
        block_b=block_b,block_q_major=block_q,
        block_k_major=block_k_major,block_k=block_k,
        sm_scale=sm_scale,
        causal=causal,
        mask_value=None,
        debug=debug,
    )

    print(bwd_dq.shape)

    # correctness check
    diff_bwd = jnp.linalg.norm(bwd_dq)
    print(f"kernel: {diff_bwd:.3e}")
    diff_bwd = jnp.linalg.norm(dq_ref)
    print(f"reference: {diff_bwd:.3e}")
    diff_bwd = jnp.linalg.norm(bwd_dq - dq_ref) / jnp.linalg.norm(dq_ref)
    print(f"Relative L2 error vs reference: {diff_bwd:.3e}")

if __name__ == "__main__":
    main()