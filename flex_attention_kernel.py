#@title Imports
import functools
from typing import Callable

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from jax.extend import core
from jax._src.util import safe_map

from constants import dimension_numbers, MIN_BLOCK_SIZE



def _flash_attention_kernel(q_tile_ref, *args, score_jaxpr=None, **kwargs):
    """Connects _flash_attention_impl to the generated kernel."""
    block_b = q_tile_ref.shape[0]

    kernel = make_flash_attention_kernel(
        # score_jaxpr=score_jaxpr,
    )

    for batch_idx in range(block_b):
        kernel(
            (batch_idx, 0),
            q_tile_ref,
            *args,
            **kwargs,
        )



def _flex_attention_impl(
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
    score_fn,
):
  batch_size, num_heads, q_seq_len, head_dim = q.shape
  _, _, kv_seq_len, _ = k.shape

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

  if score_fn is not None:
      score_jaxpr = jax.make_jaxpr(score_fn)(
          jnp.zeros((block_q, head_dim), q.dtype),
          jnp.zeros((block_k, head_dim), k.dtype),
      )
  else:
      score_jaxpr = None


  kernel = functools.partial(
      _flash_attention_kernel,
      causal=causal,
      sm_scale=sm_scale,
      block_k=block_k,
      kv_seq_len=kv_seq_len,
      score_jaxpr=score_jaxpr,
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
          dimension_semantics=("parallel", "parallel", "parallel", "arbitrary")
      ),
  )(q, k, v, ab)

  if save_residuals:
    l, m = (v[..., 0] for v in aux[-2:])
    return (o, l, m)
  else:
    return o
  

def make_flash_attention_kernel(mask_fn=None, score_jaxpr=None):
  """Factory returning a kernel with an optional custom mask function."""
  def flash_attention_fwd_kernel(
      batch_idx,
      q_tile_ref,
      k_tile_ref,
      v_tile_ref,
      ab_tile_ref,
      O_tile_ref,
      l_tile_ref,
      m_tile_ref,
      m_scratch_ref,
      l_scratch_ref,
      O_scratch_ref,
      *,
      causal,
      sm_scale,
      block_k,
      kv_seq_len,
  ):
    block_k_major = k_tile_ref.shape[2]
    head_dim = k_tile_ref.shape[3]
    kv_seq_idx = pl.program_id(3)
    q_seq_idx = pl.program_id(2)   

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

        def _inline_jaxpr_score(q, k, closed_jaxpr):
          jaxpr = closed_jaxpr.jaxpr
          literals = closed_jaxpr.literals
          env = {}

          def read(var):
              if isinstance(var, core.Literal):
                  return var.val
              return env[var]

          def write(var, val):
              if not hasattr(val, 'dtype'):
                  val = jnp.asarray(val)
              env[var] = val

          write(jaxpr.invars[0], q)
          write(jaxpr.invars[1], k)
          for var, val in zip(jaxpr.constvars, literals):
              write(var, val)

          for eqn in jaxpr.eqns:
              invals = [read(v) for v in eqn.invars]
              outvals = eqn.primitive.bind(*invals, **eqn.params)
              if not eqn.outvars:
                  continue
              if eqn.primitive.multiple_results:
                  for var, val in zip(eqn.outvars, outvals):
                      write(var, val)
              else:
                  write(eqn.outvars[0], outvals)

          return read(jaxpr.outvars[0])

        if score_jaxpr is None:
            S = jax.lax.dot_general(
                q_ref, k_ref, dimension_numbers,
                preferred_element_type=jnp.float32
            )
            S = S * sm_scale
        else:
            S = _inline_jaxpr_score(q_ref, k_ref, score_jaxpr)

        S = jax.lax.dot_general(
                q_ref, k_ref, dimension_numbers,
                preferred_element_type=jnp.float32
            )

        if ab_tile_ref is not None:
          ab = ab_tile_ref[
              (*batch_idx, pl.dslice(None), pl.dslice(start_k, block_k))
          ].astype(jnp.float32)
          S += ab

        if causal:
            q_start = q_seq_idx * q_ref.shape[0]  # block_q
            k_start = kv_seq_idx * block_k_major + start_k

            q_pos = (q_start + jnp.arange(q_ref.shape[0]))[:, None]
            k_pos = (k_start + jnp.arange(block_k))[None, :]

            causal_mask = k_pos > q_pos
            S = jnp.where(causal_mask, -1e9, S)

        m_cur = jnp.max(S, axis=1)[:, None]
        m_next = jnp.maximum(m_cur, m_past)
        block_k_repeats, rem = divmod(block_k, MIN_BLOCK_SIZE)
        if rem:
          raise NotImplementedError(
              f"{block_k=} should be a multiple of {MIN_BLOCK_SIZE}"
          )

        P = jnp.exp(S - pltpu.repeat(m_next, block_k_repeats, 1))
        l_corr = jnp.exp(m_past - m_next) * l_past
        l_next = l_corr + jnp.sum(P, axis=1)[:, None]

        head_dim_repeats, rem = divmod(head_dim, MIN_BLOCK_SIZE)
        if rem:
          raise NotImplementedError(
              f"{head_dim=} should be a multiple of {MIN_BLOCK_SIZE} if larger"
          )

        l_broadcast = lambda l: pltpu.repeat(l, head_dim_repeats, 1)
        l_scratch_ref[batch_idx] = l_next
        m_scratch_ref[batch_idx] = m_next

        l_next_inv_safe = jnp.where(l_next == 0.0, 1.0, 1.0 / l_next)
        v_ref = v_tile_ref[(*batch_idx, pl.dslice(start_k, block_k), slice(None))]
        o_curr = jax.lax.dot(
            P.astype(v_ref.dtype), v_ref,
            preferred_element_type=jnp.float32
        )
        O_scratch_ref[batch_idx] = O_past * l_broadcast(l_corr) + o_curr
        O_scratch_ref[batch_idx] *= l_broadcast(l_next_inv_safe)

    @pl.when(kv_seq_idx == (kv_seq_len // block_k_major) - 1)
    def store_res():
      O_tile_ref[batch_idx] = O_scratch_ref[batch_idx].astype(O_tile_ref.dtype)
      if (m_tile_ref is not None) and (l_tile_ref is not None):
        l_tile_ref[batch_idx] = l_scratch_ref[batch_idx].astype(l_tile_ref.dtype)
        m_tile_ref[batch_idx] = m_scratch_ref[batch_idx].astype(m_tile_ref.dtype)
  return flash_attention_fwd_kernel
