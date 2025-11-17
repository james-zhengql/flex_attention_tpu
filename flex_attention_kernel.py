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


dimension_numbers = (((1,), (1,)), ((), ()))
MIN_BLOCK_SIZE = 128


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
  ):
    block_k_major = k_tile_ref.shape[2]
    head_dim = k_tile_ref.shape[3]
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
      @pl.loop(0, block_k_major, step=block_k, unroll=True)
      def _body(start_k):
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
    *,
    ab=None,
    sm_scale=1.0,
    save_residuals=False,
    causal=False,
    block_b=1,
    block_q=128,
    block_k_major=128,
    block_k=128,
    debug=False,
):
    # JIT-compile reference implementation
    ref_fn = functools.partial(mha_reference, ab=None, sm_scale=sm_scale, save_residuals=False)
    ref_jit = jax.jit(ref_fn)

    # JIT-compile your custom Pallas kernel implementation
    flash_partial = functools.partial(
        _flash_attention_impl,
        ab=ab,
        segment_ids=None,
        save_residuals=save_residuals,
        causal=causal,
        sm_scale=sm_scale,
        block_b=block_b,
        block_q=block_q,
        block_k_major=block_k_major,
        block_k=block_k,
        debug=debug,
    )
    flash_jit = jax.jit(flash_partial)

    return ref_jit, flash_jit

def run_bench_suite(q, k, v, *, sm_scale, block_b, block_q, block_k_major, block_k, causal=False):
    # Unpack shapes
    b, h, q_len, d = q.shape
    _, _, k_len, _ = k.shape

    # Compute FLOPs (for throughput calculation)
    gflops = flop_count_attention(b, h, q_len, k_len, d) / 1e9

    # Create JIT-compiled versions of both implementations
    ref_jit, flash_jit = build_fns_for_bench(
        q, k, v,
        sm_scale=sm_scale,
        save_residuals=False,  # exclude residual buffers for fair comparison
        causal=causal,
        block_b=block_b,
        block_q=block_q,
        block_k_major=block_k_major,
        block_k=block_k,
        debug=False,
    )

    print(f"\n== Benchmark config: b={b}, h={h}, q={q_len}, k={k_len}, d={d}, causal={causal} ==")
    print(f"Estimated FLOPs per call: {gflops:.2f} GFLOPs")

    # ---- Reference (naive) implementation ----
    t_mean_ref, t_med_ref = benchmark(ref_jit, (q, k, v), name="mha_reference[jit]")
    print(f"  → Throughput: {gflops / t_med_ref:.2f} GFLOP/s")

    # ---- Pallas FlashAttention kernel ----
    t_mean_flash, t_med_flash = benchmark(flash_jit, (q, k, v), name="pallas_flash[jit]")
    print(f"  → Throughput: {gflops / t_med_flash:.2f} GFLOP/s")

    # ---- Numeric correctness check ----
    o_ref = ref_jit(q, k, v).block_until_ready()
    o_flash = flash_jit(q, k, v)
    if isinstance(o_flash, (tuple, list)):
        o_flash = o_flash[0]
    o_flash = o_flash.block_until_ready()

    # Relative L2 error measures numerical difference between both results
    rel_err = jnp.linalg.norm(o_flash - o_ref) / jnp.linalg.norm(o_ref)
    print(f"Numeric diff (Relative L2): {rel_err:.3e}")

    # Return summarized metrics
    return {
        "ref_ms_med": t_med_ref * 1e3,
        "flash_ms_med": t_med_flash * 1e3,
        "ref_gflops": gflops / t_med_ref,
        "flash_gflops": gflops / t_med_flash,
        "rel_l2": float(rel_err),
    }

