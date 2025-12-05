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
from util import _inline_jaxpr_score_backward, make_jax_score_fn

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
    block_q,
    block_q_major: int | None,
    block_k_major: int | None,
    block_k: int | None,
    sm_scale: float,
    causal: bool,
    mask_value: float,
    debug: bool,
    score_fn = None
):
    # broadcast l,m,d
    l = jnp.broadcast_to(l[..., None], (*l.shape, MIN_BLOCK_SIZE))
    m = jnp.broadcast_to(m[..., None], (*m.shape, MIN_BLOCK_SIZE))
    d = jnp.broadcast_to(d[..., None], (*d.shape, MIN_BLOCK_SIZE))

    # Preprocess contraction for bwd pass
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
    d_spec = pl.BlockSpec((batch_size, 1, block_q_major, MIN_BLOCK_SIZE), d_index_map)
    assert d_spec.block_shape is not None
    assert d.ndim == len(d_spec.block_shape)

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

    if score_fn is not None:
      score_jaxpr = jax.make_jaxpr(score_fn)(
          jnp.zeros((block_q, head_dim), q.dtype),
          jnp.zeros((block_k, head_dim), k.dtype),
      )
    else:
        score_jaxpr = None
    
    kernel = functools.partial(
        _flash_attention_bwd_dq,
        causal = causal,
        sm_scale = sm_scale,
        block_q = block_q,
        block_k = block_k,
        q_seq_len = q_seq_len,
        kv_seq_len = kv_seq_len,
        score_jaxpr= score_fn
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
      score_jaxpr
  ):

    kv_tile_idx = pl.program_id(axis = 3)
    q_tile_idx = pl.program_id(axis = 2)
    _, _, block_k_major, _ = k_tile_ref.shape

    block_k_repeats, rem = divmod(block_k, MIN_BLOCK_SIZE)

    @pl.when(kv_tile_idx == 0)
    def start_new_seq():
        dq_scratch_ref[batch_idx] = jnp.zeros(dq_scratch_ref.shape[2:], dq_scratch_ref.dtype)

    @pl.when(True)
    def body():
        @pl.loop(0, block_k_major, step=block_k, unroll=True)
        def _body(start_kv):

            q = q_tile_ref[batch_idx]
            k = k_tile_ref[(*batch_idx, pl.dslice(start_kv, block_k), slice(None))]
            v = v_tile_ref[(*batch_idx, pl.dslice(start_kv, block_k), slice(None))]

            l = l_tile_ref[batch_idx]
            m = m_tile_ref[batch_idx]
            dO = dO_tile_ref[batch_idx]
            D = di_tile_ref[batch_idx]

            dq_past = dq_scratch_ref[batch_idx]

            # S = QK^T

            # --- CHANGE 1: Backward Pass Score ---
            if score_jaxpr is not None:
                S, score_grad_fn = jax.vjp(score_jaxpr, q, k)
            else:
                S = jax.lax.dot_general(q, k, dimension_numbers, preferred_element_type=jnp.float32)
                score_grad_fn = None
            S = S * sm_scale
            # P(i,j) = exp(S(i,j)-L(i))
            unnormalized = jnp.exp(S - pltpu.repeat(m, block_k_repeats, axis=1))
            P = unnormalized / pltpu.repeat(l, block_k_repeats, axis=1)
            # dP(i,j) = dO(i)V(i)^T
            dP = jax.lax.dot_general(dO,v,dimension_numbers,preferred_element_type=jnp.float32)
            # dS(i,j) = P(i,j) * (dP(i,j)-D(i))
            dS = P * (dP  - pltpu.repeat(D, block_k_repeats, axis=1))
            dS = dS * sm_scale
            # dQ(i) = dQi + dS(i,j) * k(j)

            # --- CHANGE 2: Backward Pass using Saved Residuals ---
            if score_grad_fn is not None:
              # This uses the residuals saved from step 1. 
              # It's fused and efficient. 
              # jax.vjp returns a tuple (dQ, dK)
              dq_update, dk_update = score_grad_fn(dS)
            else:
              dq_update = jax.lax.dot_general(dS,k,(((1,), (0,)), ((), ())),preferred_element_type=jnp.float32)

            dq = dq_past + dq_update

            dq_scratch_ref[batch_idx] = dq.astype(dq_scratch_ref.dtype)


    @pl.when(kv_tile_idx == (kv_seq_len // block_k_major) - 1)
    def store_res():
        dq_tile_ref[batch_idx] = dq_scratch_ref[batch_idx].astype(dq_tile_ref.dtype)

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
    q, k, v,
    l, m, o, do, d,
    *,
    ab=None,
    sm_scale=1.0,
    save_residuals=False,
    causal=False,
    block_b=1,
    block_q=128,
    block_q_major=128,
    block_k_major=128,
    block_k=128,
    debug=False,
    score_fn = None
):
    batch_size, head_num, q_seq_len, head_dim = q.shape
    _, _, kv_seq_len, _ = k.shape

    # JIT-compile reference implementation
    ref_fn = functools.partial(mha_bwd_reference, ab=None, sm_scale=sm_scale, save_residuals=False)
    ref_jit = jax.jit(ref_fn)

    # JIT-compile your custom Pallas kernel implementation
    flash_partial = functools.partial(
        flash_attention_bwd_dq,
        ab=ab,
        causal = causal,
        sm_scale = sm_scale,
        block_b = block_b,
        block_q = block_q,
        block_q_major = block_q_major,
        block_k_major = block_k_major,
        block_k = block_k,
        segment_ids = None,
        mask_value = None,
        debug = False,
        score_fn = score_fn
    )
    def flash_bwd_fn(q, k, v, l, m, o, do, d):
        return flash_partial(q=q, k=k, v=v, l=l, m=m, o=o, do=do, d=d)

    flash_jit = jax.jit(flash_bwd_fn)

    return ref_jit, flash_jit

def run_bench_suite(q, k, v, l, m, o, do, d, *, sm_scale, block_b, block_q, block_q_major, block_k_major, block_k, causal=False, score_fn=None):
    # Unpack shapes
    b, h, q_len, h_d = q.shape
    _, _, k_len, _ = k.shape

    # Compute FLOPs (for throughput calculation)
    gflops = flop_count_attention(b, h, q_len, k_len, h_d) / 1e9

    # Create JIT-compiled versions of both implementations
    ref_jit, flash_jit = build_fns_for_bench(
        q, k, v,
        l, m, o, do, d,
        sm_scale=sm_scale,
        save_residuals=False,  # exclude residual buffers for fair comparison
        causal=causal,
        block_b=block_b,
        block_q=block_q,
        block_q_major=block_q_major,
        block_k_major=block_k_major,
        block_k=block_k,
        debug=False,
        score_fn=score_fn
    )

    print(f"\n== Benchmark config: b={b}, h={h}, q={q_len}, k={k_len}, d={d}, causal={causal} ==")
    print(f"Estimated FLOPs per call: {gflops:.2f} GFLOPs")

    # ---- Reference (naive) implementation ----
    t_mean_ref, t_med_ref = benchmark(ref_jit, (q, k, v, o, do, l, m), name="mha_reference[jit]")
    print(f"  → Throughput: {gflops / t_med_ref:.2f} GFLOP/s")

    # ---- Pallas FlashAttention kernel ----
    t_mean_flash, t_med_flash = benchmark(flash_jit, (q, k, v, l, m, o, do, d), name="pallas_flash[jit]")
    print(f"  → Throughput: {gflops / t_med_flash:.2f} GFLOP/s")

    # ---- Numeric correctness check ----
    dq_ref, dk_ref, dv_ref = ref_jit(q, k, v, o, do, l, m)
    dq_ref = dq_ref.block_until_ready()
    dq_flash = flash_jit(q, k, v, l, m, o, do, d)
    if isinstance(dq_flash, (tuple, list)):
        dq_flash = dq_flash[0]
    dq_flash = dq_flash.block_until_ready()

    # Relative L2 error measures numerical difference between both results
    rel_err = jnp.linalg.norm(dq_flash - dq_ref) / jnp.linalg.norm(dq_ref)
    print(f"Numeric diff (Relative L2): {rel_err:.3e}")

    # Return summarized metrics
    return {
        "ref_ms_med": t_med_ref * 1e3,
        "flash_ms_med": t_med_flash * 1e3,
        "ref_gflops": gflops / t_med_ref,
        "flash_gflops": gflops / t_med_flash,
        "rel_l2": float(rel_err),
    }

import math
# assume jax, jax.random, jnp, and your helper functions are already imported:
# from jax import random
# import jax.numpy as jnp
# from your_module import mha_reference, _flash_attention_impl, _flash_attention_impl_ref

def main():
    key = random.PRNGKey(0)
    batch = 1
    heads = 1
    q_len = 12800
    kv_len = 12800
    head_dim = 256

    k1, k2, k3 = random.split(key, 3)
    q = random.normal(k1, (batch, heads, q_len, head_dim), dtype=jnp.float32)
    k = random.normal(k2, (batch, heads, kv_len, head_dim), dtype=jnp.float32)
    v = random.normal(k3, (batch, heads, kv_len, head_dim), dtype=jnp.float32)
    ab = None
    segment_ids = None

    block_b = 1
    block_q = 128
    block_q_major = 512
    block_k_major = 512
    block_k = 128

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

    # Generate inputs (do, d) for backward pass
    key = jax.random.PRNGKey(0)
    do = jax.random.normal(key, o.shape)
    d = jnp.sum(o.astype(jnp.float32) * do.astype(jnp.float32), axis=-1)

    # Add Score
    #performence comparison
    #Run the benchmark comparison
    def my_score(q, k):
        return jnp.einsum("qd, kd -> qk", q, k)
    
    jax_score = make_jax_score_fn(my_score)

    print("Running reference attention backward (for numeric check)...")
    # Call mha_bwd_reference
    dq_ref, dk_ref, dv_ref = mha_bwd_reference(q, k, v, o, do, l, m, sm_scale=sm_scale)

    print("Running Pallas TPU flash attention bwd kernel...")
    # Call custom kernel
    bwd_dq = flash_attention_bwd_dq(
        q=q,k=k,v=v,ab=ab,segment_ids=segment_ids,l=l,m=m,o=o,do=do,d=d,
        block_b=block_b,
        block_q=block_q,block_q_major=block_q_major,
        block_k_major=block_k_major,block_k=block_k,
        sm_scale=sm_scale,
        causal=causal,
        mask_value=None,
        debug=debug,
        score_fn=jax_score
    )

    # correctness check
    diff_bwd = jnp.linalg.norm(bwd_dq)
    print(f"kernel: {diff_bwd:.3e}")
    diff_bwd = jnp.linalg.norm(dq_ref)
    print(f"reference: {diff_bwd:.3e}")
    diff_bwd = jnp.linalg.norm(bwd_dq - dq_ref) / jnp.linalg.norm(dq_ref)
    print(f"Relative L2 error vs reference: {diff_bwd:.3e}")

    # performance check
    results = run_bench_suite(
      k=k, v=v, q=q, l=l, m=m, o=o, d=d, do=do,
      sm_scale=sm_scale,
      block_b=block_b,
      block_q=block_q,
      block_q_major=block_q_major,
      block_k_major=block_k_major,
      block_k=block_k,
      causal=causal,
      score_fn=jax_score
    )
    print("\nSummary:", results)

if __name__ == "__main__":
    main()