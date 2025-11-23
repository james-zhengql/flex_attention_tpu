#@title Imports
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
from jax import lax 


dimension_numbers = (((1,), (1,)), ((), ()))
MIN_BLOCK_SIZE = 128
TRANS_B_DIM_NUMBERS = (((0,), (0,)), ((), ()))

def _flash_attention_kernel(q_tile_ref, *args, **kwargs):
    """Connects _flash_attention_impl to the generated kernel."""
    block_b = q_tile_ref.shape[0]

    # Create the real kernel from the factory
    kernel = make_flash_attention_kernel()

    for batch_idx in range(block_b):
        kernel(
            (batch_idx, 0),
            q_tile_ref,
            *args,
            **kwargs,
        )


def _flash_attention_impl(
    q,
    k,
    v,
    ab,
    segment_ids,
    save_residuals,
    causal,
    sm_scale,
    block_b,
    block_q,
    block_k_major,
    block_k,
    debug,
):
  batch_size, num_heads, q_seq_len, head_dim = q.shape
  _, _, kv_seq_len, _ = k.shape

  # Grid specification
  grid = (
      pl.cdiv(batch_size, block_b),
      num_heads,
      pl.cdiv(q_seq_len, block_q),
      kv_seq_len // block_k_major,
  )

  def q_index_map(batch_index, head_index, q_seq_index, _):
    return (batch_index, head_index, q_seq_index, 0)

  def kv_index_map(batch_index, head_index, q_seq_index, kv_seq_index):
    next_kv_index = kv_seq_index
    return (batch_index, head_index, next_kv_index, 0)

  def ab_index_map(batch_index, head_index, q_seq_index, kv_seq_index):
    next_q_index = q_seq_index
    next_kv_index = kv_seq_index
    return (batch_index, head_index, next_q_index, next_kv_index)

  def o_index_map(batch_index, head_index, q_seq_index, _):
    return (batch_index, head_index, q_seq_index, 0)

  def lm_index_map(batch_index, head_index, q_seq_index, _):
    return (batch_index, head_index, q_seq_index, 0)

  kernel = functools.partial(
      _flash_attention_kernel,
      causal=causal,
      sm_scale=sm_scale,
      block_k=block_k,
      kv_seq_len=kv_seq_len,
      block_k_major=block_k_major
  )

  out_shape = jax.ShapeDtypeStruct(shape=q.shape, dtype=q.dtype)
  out_shape = [out_shape]
  out_specs = [pl.BlockSpec((block_b, 1, block_q, head_dim), o_index_map)]

  # Allocate scratch buffers
  if block_k != kv_seq_len:
    m_scratch = pltpu.VMEM((block_b, 1, block_q, MIN_BLOCK_SIZE), jnp.float32)
    l_scratch = pltpu.VMEM((block_b, 1, block_q, MIN_BLOCK_SIZE), jnp.float32)
    acc_scratch = pltpu.VMEM((block_b, 1, block_q, head_dim), jnp.float32)
    scratch_shapes = [m_scratch, l_scratch, acc_scratch]
  else:
    scratch_shapes = []

  # Output specs
  if save_residuals:
    out_specs = [
        *out_specs,
        pl.BlockSpec((block_b, 1, block_q, MIN_BLOCK_SIZE), lm_index_map),
        pl.BlockSpec((block_b, 1, block_q, MIN_BLOCK_SIZE), lm_index_map),
    ]
    l = jax.ShapeDtypeStruct(
        (batch_size, num_heads, q_seq_len, MIN_BLOCK_SIZE), dtype=jnp.float32
    )
    m = jax.ShapeDtypeStruct(
        (batch_size, num_heads, q_seq_len, MIN_BLOCK_SIZE), dtype=jnp.float32
    )
    out_shape = (*out_shape, l, m)
  else:
    out_specs = [*out_specs, None, None]
    out_shape = (*out_shape, None, None)

  ab_block_spec = (
      pl.BlockSpec((block_b, 1, block_q, block_k_major), ab_index_map)
      if ab is not None else None)

  in_specs = [
      pl.BlockSpec((block_b, 1, block_q, head_dim), q_index_map),
      pl.BlockSpec((block_b, 1, block_k_major, head_dim), kv_index_map),
      pl.BlockSpec((block_b, 1, block_k_major, head_dim), kv_index_map),
      ab_block_spec,
  ]

  o, *aux = pl.pallas_call(
      kernel,
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=0,
          grid=grid,
          in_specs=in_specs,
          out_specs=out_specs,
          scratch_shapes=scratch_shapes,
      ),
      out_shape=out_shape,
      debug=debug,
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=("parallel", "parallel", "parallel", "arbitrary")
      ),
  )(q, k, v, ab)

  if save_residuals:
    l, m = (v[..., 0] for v in aux[-2:])
    return (o, l, m)
  else:
    return o


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


def make_flash_attention_kernel(mask_fn=None):
  """Factory returning a kernel with an optional custom mask function."""
  def flash_attention_fwd_kernel(
      batch_idx,
      q_tile_ref,
      k_tile_ref,
      v_tile_ref,
      ab_tile_ref,
      O_tile_ref,
      m_tile_ref,
      l_tile_ref,
      O_scratch_ref,
      m_scratch_ref,
      l_scratch_ref,
      *,
      causal,
      sm_scale,
      block_k,
      kv_seq_len,
      block_k_major,
  ):
    # block_k_major = 
    head_dim = k_tile_ref.shape[-1]
    # head_dim = 128
    kv_seq_idx = pl.program_id(3)

    @pl.when(kv_seq_idx == 0)
    def start_new_seq():
      m_scratch_ref[batch_idx] = jnp.full(
          m_scratch_ref.shape[2:], -jnp.inf, jnp.float32)
      l_scratch_ref[batch_idx] = jnp.zeros(
          l_scratch_ref.shape[2:], jnp.float32)
      O_scratch_ref[batch_idx] = jnp.zeros(
          O_scratch_ref.shape[2:], jnp.float32)

    if mask_fn is None:
      should_run = True

    @pl.when(should_run)
    def body():
    #   @pl.loop(0, block_k_major, step=block_k, unroll=True)
    #   def _body(start_k):
      @pl.loop(0, block_k_major // block_k, unroll = True)
      def _body(i):
        start_k = i * block_k
        m_past = m_scratch_ref[batch_idx]
        l_past = l_scratch_ref[batch_idx]
        O_past = O_scratch_ref[batch_idx]
        k_ref = k_tile_ref[(*batch_idx, pl.dslice(start_k, block_k), slice(None))]
        q_ref = q_tile_ref[batch_idx]

        S = jax.lax.dot_general(q_ref, k_ref, dimension_numbers, preferred_element_type=jnp.float32)
        S *= sm_scale

        if ab_tile_ref is not None:
          ab = ab_tile_ref[
              (*batch_idx, pl.dslice(None), pl.dslice(start_k, block_k))
          ].astype(jnp.float32)
          S += ab

        m_cur = jnp.max(S, axis=1)[:, None]
        m_next = jnp.maximum(m_cur, m_past)
        block_k_repeats, rem = divmod(block_k, MIN_BLOCK_SIZE)
        if rem:
          raise NotImplementedError(f"{block_k=} should be a multiple of {MIN_BLOCK_SIZE}")

        P = jnp.exp(S - pltpu.repeat(m_next, block_k_repeats, 1))
        l_corr = jnp.exp(m_past - m_next) * l_past
        l_next = l_corr + jnp.sum(P, axis=1)[:, None]

        head_dim_repeats, rem = divmod(head_dim, MIN_BLOCK_SIZE)
        if rem:
          raise NotImplementedError(f"{head_dim=} should be a multiple of {MIN_BLOCK_SIZE} if larger")

        l_broadcast = lambda l: pltpu.repeat(l, head_dim_repeats, 1)
        l_scratch_ref[batch_idx] = l_next
        m_scratch_ref[batch_idx] = m_next

        l_next_inv_safe = jnp.where(l_next == 0.0, 1.0, 1.0 / l_next)
        v_ref = v_tile_ref[(*batch_idx, pl.dslice(start_k, block_k), slice(None))]
        o_curr = jax.lax.dot(P.astype(v_ref.dtype), v_ref, preferred_element_type=jnp.float32)
        O_scratch_ref[batch_idx] = O_past * l_broadcast(l_corr) + o_curr
        O_scratch_ref[batch_idx] *= l_broadcast(l_next_inv_safe)

    @pl.when(kv_seq_idx == (kv_seq_len // block_k_major) - 1)
    def store_res():
      O_tile_ref[batch_idx] = O_scratch_ref[batch_idx].astype(O_tile_ref.dtype)
      # Only store m/l if they were requested (i.e., not None)
      if (m_tile_ref is not None) and (l_tile_ref is not None):
        m_tile_ref[batch_idx] = m_scratch_ref[batch_idx].astype(m_tile_ref.dtype)
        l_tile_ref[batch_idx] = l_scratch_ref[batch_idx].astype(l_tile_ref.dtype)

  return flash_attention_fwd_kernel


def flash_attention_bwd_dkv(
    k,
    v,
    q,
    ab,
    segment_ids,
    l,
    m,
    di,
    do,
    *,
    block_b,
    block_q,
    block_q_major: int | None,
    block_k_major: int | None,
    block_k: int | None,
    sm_scale: float,
    causal: bool,
    # mask_value: float,
    debug: bool,
):
    batch_size, head_num, q_seq_len, head_dim = q.shape
    _, _, kv_seq_len, _ = k.shape

    m = jnp.broadcast_to(m[..., None], (*m.shape, MIN_BLOCK_SIZE))
    l = jnp.broadcast_to(l[..., None], (*l.shape, MIN_BLOCK_SIZE))
    di = jnp.broadcast_to(di[..., None], (*di.shape, MIN_BLOCK_SIZE))

    grid = (
      batch_size,
      head_num,
      kv_seq_len // block_k_major,
      q_seq_len // block_q_major,
    )



    def qo_index_map(batch_index, head_index, kv_seq_index, q_seq_index):
      return (batch_index, head_index, q_seq_index, 0)

    qo_spec = pl.BlockSpec((1, 1, block_q_major, head_dim), qo_index_map)
    assert qo_spec.block_shape is not None
    assert q.ndim == len(qo_spec.block_shape)

    di_spec = qo_spec
    do_spec = qo_spec
    assert do.ndim == len(qo_spec.block_shape)

    def kv_index_map(batch_index, head_index, kv_seq_index, _ ):
      return (batch_index, head_index, kv_seq_index, 0)

    kv_spec = pl.BlockSpec((1, 1, block_k_major, head_dim),  kv_index_map)
    assert kv_spec.block_shape is not None
    assert k.ndim == len(kv_spec.block_shape)
    assert v.ndim == len(kv_spec.block_shape)

    def lm_index_map(batch_index, head_index, kv_seq_index, q_seq_index):
      return (batch_index, head_index, q_seq_index, 0)

    lm_spec = pl.BlockSpec((1, 1, block_q_major, MIN_BLOCK_SIZE), lm_index_map)
    assert lm_spec.block_shape is not None
    assert l.ndim == len(lm_spec.block_shape)
    assert m.ndim == len(lm_spec.block_shape)

    def ab_index_map(batch_index, head_index, kv_seq_index, q_seq_index):
        return (batch_index, head_index, q_seq_index, kv_seq_index)

    dab_spec = (
      pl.BlockSpec((1, 1, block_q_major, block_k_major), ab_index_map)
      if ab is not None
      else None
    )
    
    def dkv_index_map(batch_index, head_index, kv_seq_index, _):
      return (batch_index, head_index, kv_seq_index, 0)
  
    dk_spec = kv_spec
    dv_spec = kv_spec

    # in_spec specify
    in_specs = [
      qo_spec,      # q
      kv_spec,      # k
      kv_spec,      # v
      dab_spec,     # bias
      lm_spec,      # l
      lm_spec,      # m
      di_spec,      # di
      qo_spec,      # do   
    ]

    out_shapes = [
      jax.ShapeDtypeStruct(k.shape, k.dtype),
      jax.ShapeDtypeStruct(v.shape, v.dtype),
      jax.ShapeDtypeStruct(ab.shape, ab.dtype) if ab is not None else None,
    ]
    
    out_specs = [
        dk_spec,
        dv_spec,
        dab_spec,
    ]

    # Allocate scratch buffers
    # if block_k != kv_seq_len:
    dk_scratch = pltpu.VMEM((block_k_major, head_dim), jnp.float32)
    dv_scratch = pltpu.VMEM((block_k_major, head_dim), jnp.float32)
    # scratch_shapes = [dq_scratch, dk_scratch, dv_scratch]
    scratch_shapes = [dk_scratch, dv_scratch]
    # else:
    #     scratch_shapes = []

    kernel = functools.partial(
        flash_attention_dkv_kernel,
        causal = causal,
        sm_scale = sm_scale,
        block_q = block_q,
        block_k = block_k,
        q_seq_len = q_seq_len,
        block_q_major = block_q_major,
        block_k_major=block_k_major,
        # block_b = block_b,
    )

    # dq = jnp.zeros(dq_spec, jnp.float32)

    dk_out, dv_out, *aux = pl.pallas_call(
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
  )(q, k, v, ab, l, m, di, do)

    return dk_out, dv_out
    

def flash_attention_dkv_kernel(
      q_tile_ref,
      k_tile_ref,
      v_tile_ref,
      ab_tile_ref,
      l_tile_ref,
      m_tile_ref,
      di_tile_ref,
      dO_tile_ref,
      dk_tile_ref,
      dv_tile_ref,
      dab_tile_ref,
      dk_scratch_ref, 
      dv_scratch_ref,
      *,
      causal,
      sm_scale,
      block_q,
      block_k,
      q_seq_len,
      block_q_major,
      block_k_major
  ):

    _, _, q_seq_length, _ = q_tile_ref.shape
    kv_tile_idx = pl.program_id(axis = 2)
    q_tile_idx = pl.program_id(axis = 3)
    
    block_k_repeats, rem = divmod(block_k, MIN_BLOCK_SIZE)


    @pl.when(q_tile_idx == 0)
    def start_new_kv_seq():
        # dq_scratch_ref[:, :] = jnp.zeros(dq_scratch_ref.shape, dq_scratch_ref.dtype)
        dk_scratch_ref[:, :] = jnp.zeros(dk_scratch_ref.shape, dk_scratch_ref.dtype)
        dv_scratch_ref[:, :] = jnp.zeros(dv_scratch_ref.shape, dv_scratch_ref.dtype)

    @pl.when(True)
    def body():
      @pl.loop(0, block_q_major // block_q, unroll=True)
      def _body(j):
          start_q = j * block_q
        # @pl.loop(0, q_seq_length, step=block_q, unroll=True)
        # def _body(start_q):
          @pl.loop(0, block_k_major // block_k, unroll=True)
          def _body(i):
            start_k = i * block_k
            q  = q_tile_ref[0, 0, pl.dslice(start_q, block_q), :].astype(jnp.float32)
            dO = dO_tile_ref[0, 0, pl.dslice(start_q, block_q), :].astype(jnp.float32)
            di  = di_tile_ref[0, 0, pl.dslice(start_q, block_q), :].astype(jnp.float32)
            l  = l_tile_ref[0, 0, pl.dslice(start_q, block_q), :].astype(jnp.float32)
            m  = m_tile_ref[0, 0, pl.dslice(start_q, block_q), :].astype(jnp.float32)
            k = k_tile_ref[0, 0, pl.dslice(start_k, block_k), :].astype(jnp.float32)
            v = v_tile_ref[0, 0, pl.dslice(start_k, block_k), :].astype(jnp.float32)

            
            dk_past = dk_scratch_ref[pl.ds(start_k, block_k), :]
            dv_past = dv_scratch_ref[pl.ds(start_k, block_k), :]
                        
            S = jax.lax.dot_general(q, k, dimension_numbers, preferred_element_type=jnp.float32)  # block_q * block_k

            S = S * sm_scale
            # unnormalized = jnp.exp(S - m_block[:, None])
            unnormalized = jnp.exp(S - pltpu.repeat(m, block_k // MIN_BLOCK_SIZE, axis=1))   # (block_q, block_k)
            P = unnormalized / pltpu.repeat(l, block_k // MIN_BLOCK_SIZE, axis=1)                 # (block_q, block_k)

            # dv = dv_past + jnp.einsum('rb, rd->bd', P, dO)
            dv = dv_past + jax.lax.dot_general(P,dO,TRANS_B_DIM_NUMBERS,preferred_element_type=jnp.float32)
            # dv = dv_past + jnp.einsum("qk,qd->kd", P, dO)

            dv_scratch_ref[pl.dslice(start_k, block_k), :] = dv.astype(dv_scratch_ref.dtype)

            dP = jax.lax.dot_general(dO,v,dimension_numbers,preferred_element_type=jnp.float32)
  

            dS = P * (dP - pltpu.repeat(di, block_k // MIN_BLOCK_SIZE, axis=1))

            if sm_scale != 1.0:
              dS = dS * sm_scale  

            dk_update = jax.lax.dot(dS.T.astype(dO.dtype), q, preferred_element_type=jnp.float32)

            dk = dk_past + dk_update

            dk_scratch_ref[pl.dslice(start_k, block_k), :] = dk.astype(dk_scratch_ref.dtype)

    
    @pl.when(q_tile_idx == q_seq_len // block_q_major - 1)
    def store_res():
      dv_tile_ref[0, 0, :, :] = dv_scratch_ref[...].astype(dv_tile_ref.dtype)
      dk_tile_ref[0, 0, :, :] = dk_scratch_ref[...].astype(dk_tile_ref.dtype)
    
    # return flash_attention_dkv_kernel


def mha_reference_bwd(
    q, 
    k,
    v, 
    ab,
    segment_ids: None,
    o,
    l,
    m,
    do,
    causal: bool = False,
    mask_value: float = None,
    *,
    sm_scale: float = 1.0, 
):

  logits = jnp.einsum(
      "bhqc,bhkc->bhqk",
      q.astype(jnp.float32),
      k.astype(jnp.float32),
  )

  if sm_scale != 1.0:
    logits *= sm_scale

  if ab is not None:
    logits += abf

  mask = None
  if segment_ids is not None:
    mask = segment_ids.q[:, :, None] == segment_ids.kv[:, None, :]
    mask = mask[:, None, :, :]

  if causal:
    _, _, q_seq_len, _ = q.shape
    _, _, kv_seq_len, _ = k.shape
    mask_shape = (q_seq_len, kv_seq_len)
    row_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 0)
    col_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 1)
    causal_mask = (col_ids <= row_ids)[None, None, :, :]
    mask = causal_mask if mask is None else jnp.logical_and(mask, causal_mask)

  logits = logits if mask is None else logits + jnp.where(mask, 0.0, mask_value)

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

  return dq, dk, dv, dab


def _mha_reference_bwd(
    q,
    k,
    v,
    ab,
    o,
    l,
    m,
    do,
    *,
    segment_ids, 
    causal: bool,
    mask_value: float,
    sm_scale: float,
    save_residuals: bool,
    
):
  # del save_residuals
  # q, k, v, ab, segment_ids, o, l, m = residuals
  dq, dk, dv, dab = mha_reference_bwd(
      q,
      k,
      v,
      ab,
      segment_ids,
      o,
      l,
      m,
      do,
      causal=causal,
      mask_value=mask_value,
      sm_scale=sm_scale,
  )
  return dq, dk, dv, dab, None

def flop_count_attention(b, h, q, k, d):
    """
    Rough FLOP count for one forward pass of scaled dot-product attention:
      QK^T  : 2 * b * h * q * k * d        (matrix multiplication)
      Softmax: ~ b * h * q * k             (small, we ignore it)
      (softmax @ V): 2 * b * h * q * k * d (another matmul)
    Total ≈ 4 * b * h * q * k * d FLOPs
    """
    return 4.0 * b * h * q * k * d

def benchmark(fn, args, iters=30, warmup=5, name="fn"):
    # 1. Warmup phase — triggers JIT compilation and stabilizes cache
    for _ in range(warmup):
        y = fn(*args)
        # .block_until_ready() ensures we wait until computation is finished
        if isinstance(y, (tuple, list)):
            jax.tree_util.tree_map(
                lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, y
            )
        else:
            y.block_until_ready()

    # 2. Timed runs
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        y = fn(*args)
        # Synchronize (very important for accurate timing)
        if isinstance(y, (tuple, list)):
            jax.tree_util.tree_map(
                lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, y
            )
        else:
            y.block_until_ready()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    # 3. Compute summary statistics
    mean_t = sum(times) / len(times)
    med_t = stats.median(times)
    p10, p90 = np.percentile(np.array(times), [10, 90])

    print(f"[{name}] mean={mean_t*1e3:.2f} ms  median={med_t*1e3:.2f} ms  "
          f"p10={p10*1e3:.2f} ms  p90={p90*1e3:.2f} ms")

    # Return average and median latency (seconds)
    return mean_t, med_t

def build_fns_for_bench(
    q, k, v, ab, segment_ids, o, l, m, di, do,
    *,
    sm_scale=1.0,
    save_residuals=False,
    causal=False,
    block_b=1,
    block_q=128,
    block_k_major=128,
    block_k=128,
    debug=False,
    block_q_major=128,
):
    # ---------- Reference backward ----------
    ref_fn = functools.partial(
        _mha_reference_bwd,
        segment_ids=segment_ids,
        causal=causal,
        mask_value=None,
        sm_scale=sm_scale,
        save_residuals=save_residuals,
    )
    ref_jit = jax.jit(ref_fn)

    # ---------- Flash (Pallas) backward ----------
    flash_partial = functools.partial(
        flash_attention_bwd_dkv,
        ab=ab,
        segment_ids=segment_ids,
        causal=causal,
        sm_scale=sm_scale,
        block_b=block_b,
        block_q=block_q,
        block_q_major=block_q_major,
        block_k_major=block_k_major,
        block_k=block_k,
        debug=debug,
    )

  
    def flash_bwd_fn(q, k, v, l, m, di, do):
        return flash_partial(q=q, k=k, v=v, l=l, m=m, di=di, do=do)

    flash_jit = jax.jit(flash_bwd_fn)

    return ref_jit, flash_jit
def run_bench_suite(
    q, k, v, ab, segment_ids, o, l, m, di, do,
    *,
    sm_scale,
    block_b,
    block_q,
    block_k_major,
    block_k,
    block_q_major,
    causal=False,
):
    # Unpack shapes
    b, h, q_len, d = q.shape
    _, _, k_len, _ = k.shape

    gflops = flop_count_attention(b, h, q_len, k_len, d) / 1e9

    ref_jit, flash_jit = build_fns_for_bench(
        q, k, v, ab, segment_ids, o, l, m, di, do,
        sm_scale=sm_scale,
        save_residuals=False,
        causal=causal,
        block_b=block_b,
        block_q=block_q,
        block_k_major=block_k_major,
        block_k=block_k,
        debug=False,
        block_q_major=block_q_major,
    )

    print(f"\n== Benchmark config: b={b}, h={h}, q={q_len}, k={k_len}, d={d}, causal={causal} ==")
    print(f"Estimated FLOPs per call: {gflops:.2f} GFLOPs")

    # ---------- Reference backward ----------
    t_mean_ref, t_med_ref = benchmark(
        ref_jit, (q, k, v, ab, o, l, m, do),
        name="mha_reference_bwd[jit]",
    )
    print(f"  → Throughput: {gflops / t_med_ref:.2f} GFLOP/s")

    # ---------- Flash (Pallas) backward ----------
    t_mean_flash, t_med_flash = benchmark(
        flash_jit, (q, k, v, l, m, di, do),
        name="pallas_flash_bwd_dkv[jit]",
    )
    print(f"  → Throughput: {gflops / t_med_flash:.2f} GFLOP/s")

    # ---------- Numeric correctness check ----------
    dq_ref, dk_ref, dv_ref, dab_ref, _ = ref_jit(q, k, v, ab, o, l, m, do)
    dk_out, dv_out = flash_jit(q, k, v, l, m, di, do)

    # block_until_ready
    dq_ref.block_until_ready()
    dk_ref.block_until_ready()
    dv_ref.block_until_ready()
    dk_out.block_until_ready()
    dv_out.block_until_ready()

    rel_dk = jnp.linalg.norm(dk_out - dk_ref) / jnp.linalg.norm(dk_ref)
    rel_dv = jnp.linalg.norm(dv_out - dv_ref) / jnp.linalg.norm(dv_ref)
    print(f"Numeric dk diff (Relative L2): {rel_dk:.3e}")
    print(f"Numeric dv diff (Relative L2): {rel_dv:.3e}")

    return {
        "ref_ms_med": t_med_ref * 1e3,
        "flash_ms_med": t_med_flash * 1e3,
        "ref_gflops": gflops / t_med_ref,
        "flash_gflops": gflops / t_med_flash,
        "rel_l2_dk": float(rel_dk),
        "rel_l2_dv": float(rel_dv),
    }

def main():
  key = random.PRNGKey(0)
  batch = 1
  heads = 1
  q_len = 128 * 64
  kv_len = 128 * 64
  head_dim = 128

  k1, k2, k3, k4 = random.split(key, 4)
  q = random.normal(k1, (batch, heads, q_len, head_dim), dtype=jnp.float32)
  k = random.normal(k2, (batch, heads, kv_len, head_dim), dtype=jnp.float32)
  v = random.normal(k3, (batch, heads, kv_len, head_dim), dtype=jnp.float32)
  do = random.normal(k4, (batch, heads, q_len, head_dim), dtype=jnp.float32)
  ab = None
  segment_ids = None

  block_b = 1
  block_q_major = 512
  block_q = 128
  block_k_major = 512
  block_k = 128

  causal = False
  sm_scale = float(1.0 / jnp.sqrt(head_dim).astype(jnp.float32))
  debug = False
  save_residuals = True


  print("Running Pallas TPU flash attention kernel...")
  o, l, m = _flash_attention_impl(
      q=q, k=k, v=v, ab=ab, segment_ids=segment_ids,
      save_residuals=save_residuals,
      causal=causal, sm_scale=sm_scale,
      block_b=block_b, block_q=block_q_major,
      block_k_major=block_k_major, block_k=block_k,
      debug=debug,
  )

  di = jnp.sum(o.astype(jnp.float32) * do.astype(jnp.float32), axis=-1)

#performence comparison
#Run the benchmark comparison
  results = run_bench_suite(
      k=k, v=v, q=q, l=l, m=m, di=di, do=do, o=o, ab=ab, segment_ids=segment_ids,
      sm_scale=sm_scale,
      block_b=block_b,
      block_q=block_q,
      block_k_major=block_k_major,
      block_q_major=block_q_major,
      block_k=block_k,
      causal=causal,
  )

  print("\nSummary:", results)

if __name__ == "__main__":
  main()