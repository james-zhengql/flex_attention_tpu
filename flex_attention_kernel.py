#@title Imports
import functools
from typing import Callable

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax import random
import jax.numpy as jnp
import numpy as np

dimension_numbers = (((1,), (1,)), ((), ())) 
MIN_BLOCK_SIZE = 128



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
  _verify_block("block_q", "q_seq_len", block_q, q_seq_len, should_divide=False)
  _verify_block("block_k_major", "kv_seq_len", block_k_major, kv_seq_len)
  _verify_block("block_k", "kv_seq_len", block_k, kv_seq_len)
  _verify_block("block_b", "batch", block_b, batch_size, should_divide=False)

  # TODO(apaszke): Tile over heads as well.
  grid = (
      pl.cdiv(batch_size, block_b),
      num_heads,
      pl.cdiv(q_seq_len, block_q),
      kv_seq_len // block_k_major,
  )

  def q_index_map(batch_index, head_index, q_seq_index, _):
    return (batch_index, head_index, q_seq_index, 0)

  def kv_index_map(batch_index, head_index, q_seq_index, kv_seq_index):
    if causal:
      # If the kv block is skipped, prefetch the next valid kv block, i.e. the
      # 0th one to be used for the next block_q rows.
      next_kv_index = lax.select(
          below_or_on_diag(q_seq_index, block_q, kv_seq_index, block_k_major),
          kv_seq_index,
          0,
      )
    else:
      next_kv_index = kv_seq_index
    return (batch_index, head_index, next_kv_index, 0)

  def ab_index_map(batch_index, head_index, q_seq_index, kv_seq_index):
    if causal: 
      should_run = below_or_on_diag(
          q_seq_index, block_q, kv_seq_index, block_k_major
      )
      # If the ab block is skipped, prefetch the next valid ab block, i.e. the
      # 0th kv to be used for the next block_q rows.
      next_q_index = lax.select(
          should_run,
          q_seq_index,
          lax.select(
              q_seq_index == (q_seq_len // block_q) - 1, 0, q_seq_index + 1
          ),
      )
      next_kv_index = lax.select(should_run, kv_seq_index, 0)
    else:
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
      mask_value=DEFAULT_MASK_VALUE,
      sm_scale=sm_scale,
      block_k=block_k,
      kv_seq_len=kv_seq_len,
  )
  out_shape = jax.ShapeDtypeStruct(shape=q.shape, dtype=q.dtype)
  out_shape = [out_shape]
  out_specs = [pl.BlockSpec((block_b, 1, block_q, head_dim), o_index_map)]

  if block_k != kv_seq_len:
    m_scratch = pltpu.VMEM((block_b, 1, block_q, MIN_BLOCK_SIZE), jnp.float32)
    l_scratch = pltpu.VMEM((block_b, 1, block_q, MIN_BLOCK_SIZE), jnp.float32)
    acc_scratch = pltpu.VMEM((block_b, 1, block_q, head_dim), jnp.float32)
    scratch_shapes = [m_scratch, l_scratch, acc_scratch]
  else:
    scratch_shapes = []

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
          dimension_semantics=(
              "parallel",
              "parallel",
              "parallel",
              "arbitrary",
          )
      ),
      cost_estimate=_fwd_cost_estimate(
          q,
          k,
          v,
          ab,
          segment_ids,
          causal=causal,
          sm_scale=sm_scale,
          kernel_inputs_specs=(q, k, v, ab, q_segment_ids, kv_segment_ids),
          kernel_outputs_specs=out_shape,
      ),
  )(q, k, v, ab, q_segment_ids, kv_segment_ids)
  if save_residuals:
    l, m = (v[..., 0] for v in aux[-2:])
    return (o, l, m)
  else:
    return o

def make_flash_attention_kernel(mask_fn=None):
  """Factory returning a kernel with an optional custom mask function."""
  # stationary q, move k and v
  def flash_attention_fwd_kernel(
                                batch_idx,
                                # input
                                q_tile_ref,
                                k_tile_ref,
                                v_tile_ref,
                                ab_tile_ref,
                                # output
                                O_tile_ref,
                                m_tile_ref,
                                l_tile_ref, 
                                # scratch
                                O_scratch_ref,
                                m_scratch_ref,
                                l_scratch_ref,
                                # scalar input
                                *,
                                causal,
                                sm_scale,
                                block_k,
                                kv_seq_len
                                ):
    
    # get k,v
    block_k_major = k_tile_ref.shape[2]
    head_dim = k_tile_ref.shape[-1]
    # kv_idx last dim in grid
    kv_seq_idx = pl.program_id(-1)

    # at the start of kv_seq_idx
    @pl.when(kv_seq_idx == 0)
    def start_new_seq():
      m_scratch_ref[batch_idx] = jnp.full(
        m_scratch_ref.shape[2:], -jnp.inf, jnp.float32)
      l_scratch_ref[batch_idx] = jnp.zeros(
        l_scratch_ref.shape[2:], jnp.float32)
      O_scratch_ref[batch_idx] = jnp.zeros(
        O_scratch_ref.shape[2:], jnp.float32)


    # use mask to decide whether should run
    if mask_fn is None:
      should_run = True
    # else:
    #   q_idx = jnp.arange(q_ref.shape[0])
    #   k_idx = start_k + jnp.arange(block_k)
    #   # Evaluate mask function (must be JAX-traceable)
    #   mask = mask_fn(q_idx, k_idx, batch_idx[0], batch_idx[1])
    
    
    @pl.when(should_run)
    @pl.loop(0, block_k_major, step=block_k, unroll=True)
    def body(start_k):
      m_past = m_scratch_ref[batch_idx]
      l_past = l_scratch_ref[batch_idx]
      O_past = O_scratch_ref[batch_idx]
      k_ref = k_tile_ref[(*batch_idx,pl.dslice(start_k,block_k)),slice(None)]
      q_ref = q_tile_ref[batch_idx]
      # S = q*k
      S = jax.lax.dot_general(
        q_ref, k_ref, dimension_numbers, preferred_element_type=jnp.float32,
      )

      S = S * sm_scale

      if ab_tile_ref is not None:
          ab = ab_tile_ref[
              (*batch_idx, pl.dslice(None), pl.dslice(start_k, block_k))
          ].astype(jnp.float32)
          S += ab
      
      # --- custom mask injection ---
      if mask_fn is not None:
          # Construct q/k indices within this tile
          q_idx = jnp.arange(q_ref.shape[0])
          k_idx = start_k + jnp.arange(block_k)
          # Evaluate mask function (must be JAX-traceable)
          mask = mask_fn(q_idx, k_idx, batch_idx[0], batch_idx[1])
          S = S + jnp.where(mask, 0.0, -1e9)

      # m_cur [block_q,1]
      m_cur = jnp.max(S, axis=1)[:,None] 
      # m_past from vmem [block_q,128], m_next also [block_q,128]
      m_next = max(m_cur, m_past)
      block_k_repeats, rem = divmod(block_k, MIN_BLOCK_SIZE)
      if rem:
        raise NotImplementedError(
            f"{block_k=} should be a multiple of {MIN_BLOCK_SIZE}"
        )

      P = jnp.exp(S - pltpu.repeat(m_next, block_k_repeats, 1))

      l_corr = jnp.exp(m_past - m_cur) * l_past

      l_next = l_corr + jnp.sum(P, axis = 1)[:, None]


      head_dim_repeats, rem = divmod(head_dim, MIN_BLOCK_SIZE)
      l_broadcast = lambda l: pltpu.repeat(l, head_dim_repeats, 1)
      if rem:
        raise NotImplementedError(
            f"{head_dim=} should be a multiple of {MIN_BLOCK_SIZE} if larger"
        )

      l_scratch_ref[batch_idx] = l_next
      m_scratch_ref[batch_idx] = m_next

      l_next_inv_safe = jnp.where(l_next == 0.0, 1.0, 1.0 / l_next)
      v = v_tile_ref[(*batch_idx, pl.dslice(start_k, block_k), slice(None))]
      o_curr = jax.lax.dot(
          P.astype(v.dtype), v, preferred_element_type=jnp.float32
      )
      # l_corr
      O_scratch_ref[batch_idx] = O_past*l_broadcast(l_corr)+o_curr
      O_scratch_ref[batch_idx] *= l_broadcast(l_next_inv_safe)

    # when is the last kv tile of the seq
    @pl.when(kv_seq_idx == (kv_seq_len // block_k_major) - 1)
    def store_res():
      O_tile_ref[batch_idx] = O_scratch_ref[batch_idx].astype(O_tile_ref.dtype)
      m_tile_ref[batch_idx] = m_scratch_ref[batch_idx].astype(m_tile_ref.dtype)
      l_tile_ref[batch_idx] = l_scratch_ref[batch_idx].astype(l_tile_ref.dtype)
      


      


def flash_attention_fwd(
    q,
    k,
    v,
    sm_scale,
    block_b,
    block_q,
    block_k_major,
    block_k,
  ):
  batch_size, num_heads, q_seq_len, head_dim = q.shape

  grid = (pl.cdiv(batch_size,block_b),
          num_heads,
          pl.cdiv(q_seq_len,block_q),
          pl.cdiv(q_seq_len,block_k_major))

  def q_index_map(batch_index, head_index, q_seq_index, _):
    return (batch_index, head_index, q_seq_index, 0)

  def kv_index_map(batch_index, head_index, q_seq_index, kv_seq_index):
    return (batch_index, head_index, kv_seq_index, 0)

  in_spec = (pl.BlockSpec((block_b, 1, block_q, head_dim), q_index_map),
             pl.BlockSpec((block_b, 1, block_k_major, head_dim), kv_index_map),
             pl.BlockSpec((block_b, 1, block_k_major, head_dim), kv_index_map))






def matmul_kernel(x_ref, y_ref, z_ref, nsteps, activation):
  @pl.when(pl.program_id(2) == 0)
  def _():
    acc_ref[...] = jnp.zeros_like(acc_ref)

  z_ref[...] += x_ref[...] @ y_ref[...]
  @pl.when(pl.program_id(2) == nsteps - 1)
  def _():
    z_ref[...] = activation(acc_ref[...]).astype(z_ref.dtype)

def matmul(
    x: jax.Array,
    y: jax.Array,
    *,
    bm: int = 128,
    bk: int = 128,
    bn: int = 128,
):
  m, k = x.shape
  _, n = y.shape
  return pl.pallas_call(
      matmul_kernel,
      out_shape=jax.ShapeDtypeStruct((m, n), x.dtype),
      in_specs=[pl.BlockSpec((bm, bk), lambda i, j, k: (i, k)),
                pl.BlockSpec((bk, bn), lambda i, j, k: (k, j))],
      out_specs=pl.BlockSpec((bm, bn), lambda i, j, k: (i, j)),
      grid=(m // bm, n // bn, k // bk),
      compiler_params=dict(
          dimension_semantics=("parallel", "parallel", "arbitrary")),
  )(x, y)
import timeit

def benchmark(f, ntrials: int = 100):
  def run(*args, **kwargs):
    # Compile function first
    jax.block_until_ready(f(*args, **kwargs))
    # Time function
    result = timeit.timeit(lambda: jax.block_until_ready(f(*args, **kwargs)),
                           number=ntrials)
    time = result / ntrials
    # print(f"Time: {time}")
    return time
  return run

def analyze_matmul(m: int, k: int, n: int, dtype: np.dtype,
                   mm_func):
  x = jnp.ones((m, k), dtype=dtype)
  y = jnp.ones((k, n), dtype=dtype)
  time = benchmark(mm_func)(x, y)
  print(f"----- {m} x {k} x {n} -----")
  print("Matmul time: ", time)
  print()

print("================bm=128, bk=128, bn=128===================")
mm = functools.partial(matmul, bm=128, bk=128, bn=128)
analyze_matmul(1024, 1024, 1024, jnp.float32, mm)