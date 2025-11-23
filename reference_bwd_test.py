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
TRANS_B_DIM_NUMBERS = (((1,), (1,)), ((), ()))


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
    q,
    k,
    v,
    ab,
    segment_ids,
    l,
    m,
    do,
    di,
    *,
    block_q_major: int | None,
    block_q: int | None,
    block_k_major: int | None,
    block_k: int | None,
    sm_scale: float,
    causal: bool = False,
    # mask_value: float = DEFAULT_MASK_VALUE,
    debug: bool = False,
):
  batch_size, num_heads, q_seq_len, head_dim = q.shape
  _, _, kv_seq_len, _ = k.shape
  # _verify_block("block_q_major_dkv", "q_seq_len", block_q_major, q_seq_len)
  # _verify_block("block_q_dkv", "q_seq_len", block_q, q_seq_len)
  # _verify_block("block_k_major_dkv", "kv_seq_len", block_k_major, kv_seq_len)
  # _verify_block("block_k_dkv", "kv_seq_len", block_k, kv_seq_len)

  # Broadcast out scalar values
  m = jnp.broadcast_to(m[..., None], (*m.shape, MIN_BLOCK_SIZE))
  l = jnp.broadcast_to(l[..., None], (*l.shape, MIN_BLOCK_SIZE))
  # Preprocess contraction for bwd pass
  di = jnp.broadcast_to(di[..., None], (*di.shape, MIN_BLOCK_SIZE))

  # kv index needs to be before q index since q index is the contractng
  # dimension.
  grid = (
      batch_size,
      num_heads,
      kv_seq_len // block_k_major,
      q_seq_len // block_q_major,
  )

  def qo_index_map(batch_index, head_index, kv_seq_index, q_seq_index):
    if causal:
      # If the q block is skipped, stay at the 0th q block.
      next_q_index = lax.select(
          below_or_on_diag(
              q_seq_index, block_q_major, kv_seq_index, block_k_major
          ),
          q_seq_index,
          0,
      )
    else:
      next_q_index = q_seq_index

    return (batch_index, head_index, next_q_index, 0)

  qo_spec = pl.BlockSpec((1, 1, block_q_major, head_dim), qo_index_map)
  assert qo_spec.block_shape is not None
  assert q.ndim == len(qo_spec.block_shape)
  do_spec = qo_spec
  assert do.ndim == len(qo_spec.block_shape)

  def kv_index_map(batch_index, head_index, kv_seq_index, _):
    return (batch_index, head_index, kv_seq_index, 0)

  kv_spec = pl.BlockSpec((1, 1, block_k_major, head_dim), kv_index_map)
  assert kv_spec.block_shape is not None
  assert k.ndim == len(kv_spec.block_shape)
  assert v.ndim == len(kv_spec.block_shape)

  def lm_index_map(batch_index, head_index, _, q_seq_index):
    return (batch_index, head_index, q_seq_index, 0)

  lm_spec = pl.BlockSpec((1, 1, block_q_major, MIN_BLOCK_SIZE), lm_index_map)
  assert lm_spec.block_shape is not None
  assert l.ndim == len(lm_spec.block_shape)
  assert m.ndim == len(lm_spec.block_shape)

  di_spec = pl.BlockSpec((1, 1, block_q_major, MIN_BLOCK_SIZE), qo_index_map)
  assert di_spec.block_shape is not None
  assert di.ndim == len(di_spec.block_shape)

  def ab_index_map(batch_index, head_index, kv_seq_index, q_seq_index):
    return (batch_index, head_index, q_seq_index, kv_seq_index)

  dab_spec = (
      pl.BlockSpec((1, 1, block_q_major, block_k_major), ab_index_map)
      if ab is not None
      else None
  )

  q_segment_ids_spec = kv_segment_ids_spec = None
  q_segment_ids = kv_segment_ids = None
  if segment_ids is not None:

    def q_segment_ids_index_map(
        batch_index, head_index, kv_seq_index, q_seq_index
    ):
      del head_index
      if causal:
        next_q_index = lax.select(
            below_or_on_diag(
                q_seq_index, block_q_major, kv_seq_index, block_k_major
            ),
            q_seq_index,
            0,
        )
      else:
        next_q_index = q_seq_index
      return (batch_index, next_q_index, 0)

    def kv_segment_ids_index_map(batch_index, head_index, kv_seq_index, _):
      del head_index
      return (batch_index, 0, kv_seq_index)

    q_segment_ids_spec = pl.BlockSpec(
        (1, block_q_major, NUM_LANES), q_segment_ids_index_map
    )
    kv_segment_ids_spec = pl.BlockSpec(
        (1, NUM_SUBLANES, block_k_major), kv_segment_ids_index_map
    )

    q_segment_ids = jax.lax.broadcast_in_dim(
        segment_ids.q,
        (batch_size, q_seq_len, NUM_LANES),
        (
            0,
            1,
        ),
    )
    kv_segment_ids = jax.lax.broadcast_in_dim(
        segment_ids.kv,
        (batch_size, NUM_SUBLANES, kv_seq_len),
        (
            0,
            2,
        ),
    )

  in_specs = [
      qo_spec,
      kv_spec,
      kv_spec,
      dab_spec,
      q_segment_ids_spec,
      kv_segment_ids_spec,
      lm_spec,
      lm_spec,
      do_spec,
      di_spec,
  ]

  out_shapes = [
      jax.ShapeDtypeStruct((batch_size, num_heads, kv_seq_len, head_dim),
                           k.dtype),
      jax.ShapeDtypeStruct((batch_size, num_heads, kv_seq_len, head_dim),
                           v.dtype),
  ]
  def dkv_index_map(batch_index, head_index, kv_seq_index, _):
    return (batch_index, head_index, kv_seq_index, 0)

  dkv_spec = pl.BlockSpec((1, 1, block_k_major, head_dim), dkv_index_map)
  out_specs = [dkv_spec, dkv_spec]
  scratch_shapes = [
      pltpu.VMEM((block_k_major, head_dim), jnp.float32),  # type: ignore
      pltpu.VMEM((block_k_major, head_dim), jnp.float32),  # type: ignore
  ]

  kernel = functools.partial(
      _flash_attention_dkv_kernel,
      block_q=block_q,  # type: ignore
      block_k=block_k,  # type: ignore
      sm_scale=sm_scale,
      causal=causal,
      # mask_value=mask_value,
      q_seq_len=q_seq_len,
  )
  name_scope = f"flash_mha_bwd_dkv_{block_q_major=}_{block_q=}_{block_k_major=}_{block_k=}"
  with jax.named_scope(name_scope):
    dk, dv = pl.pallas_call(
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
                dimension_semantics=(
                    "parallel",
                    "parallel",
                    "parallel",
                    "arbitrary",
                )
        ),
    )(q, k, v, ab, q_segment_ids, kv_segment_ids, l, m, do, di)
    assert dk.shape == k.shape
    assert dv.shape == v.shape
  return dk, dv


def _flash_attention_dkv_kernel(
    q_tile_ref,
    k_tile_ref,
    v_tile_ref,
    ab_tile_ref,
    q_segment_ids_tile_ref,
    kv_segment_ids_tile_ref,
    l_tile_ref,
    m_tile_ref,
    do_tile_ref,
    di_tile_ref,
    dk_tile_ref,
    dv_tile_ref,
    dk_scratch_ref,
    dv_scratch_ref,
    *,
    sm_scale: float,
    causal: bool,
    # mask_value: float,
    q_seq_len: int,
    block_q: int,
    block_k: int,

):
  _, _, block_q_major, _ = q_tile_ref.shape
  _, _, block_k_major, _ = k_tile_ref.shape

  q_seq_index = pl.program_id(axis=3)
  kv_seq_index = pl.program_id(axis=2)

  @pl.when(q_seq_index == 0)
  def start_new_sequence():
    dk_scratch_ref[:, :] = jnp.zeros(dk_scratch_ref.shape, dk_scratch_ref.dtype)
    dv_scratch_ref[:, :] = jnp.zeros(dv_scratch_ref.shape, dv_scratch_ref.dtype)

  def q_body(j, _):
    start_q = j * block_q
    def k_body(i, _):
      start_k = i * block_k
      k = k_tile_ref[0, 0, pl.ds(start_k, block_k), :]
      v = v_tile_ref[0, 0, pl.ds(start_k, block_k), :]
      q = q_tile_ref[0, 0, pl.ds(start_q, block_q), :]  # [block_q, head_dim]
      l = l_tile_ref[0, 0, pl.ds(start_q, block_q), :]  # [block_q, 128]
      m = m_tile_ref[0, 0, pl.ds(start_q, block_q), :]  # [block_q, 128]
      do = do_tile_ref[0, 0, pl.ds(start_q, block_q), :]  # [block_q, 128]
      di = di_tile_ref[0, 0, pl.ds(start_q, block_q), :].astype(
          jnp.float32
      )  # [block_q, 128]

      capped_logits = lax.dot_general(
          q, k, TRANS_B_DIM_NUMBERS, preferred_element_type=jnp.float32
      )  # [block_q_major, block_k]

      if ab_tile_ref is not None:
        ab = ab_tile_ref[
            0,
            0,
            pl.dslice(j * block_q, block_q),
            pl.dslice(i * block_k, block_k),
        ].astype(jnp.float32)
        capped_logits += ab

      if sm_scale != 1.0:
        capped_logits *= sm_scale

      mask = None
      if q_segment_ids_tile_ref is not None:
        repeats, rem = divmod(block_k, NUM_LANES)
        if rem:
          raise NotImplementedError(
          )
        q_segment_ids = q_segment_ids_tile_ref[
            0, pl.ds(start_q, block_q), :
        ]  # [block_q, NUM_LANES].
        q_segment_ids = pltpu.repeat(
            q_segment_ids, repeats, axis=1
        )  # [block_q, block_k].
        kv_segment_ids = kv_segment_ids_tile_ref[
            :, 0, pl.ds(start_k, block_k)
        ]  # [1, block_k].
        mask = jnp.equal(q_segment_ids, kv_segment_ids).astype(jnp.bool_)

      if causal:
        mask_shape = (block_q, block_k)
        row_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 0)
        row_ids += q_seq_index * block_q_major + start_q
        col_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 1)
        col_ids += kv_seq_index * block_k_major + start_k
        causal_mask = col_ids <= row_ids
        mask = (
            causal_mask if mask is None else jnp.logical_and(mask, causal_mask)
        )

      capped_logits = (
          capped_logits
          if mask is None
          else capped_logits + jnp.where(mask, 0.0, mask_value)
      )

      p = jnp.exp(
          capped_logits - pltpu.repeat(m, block_k // MIN_BLOCK_SIZE, axis=1)
      )
      p = p * pltpu.repeat(
          1 / l, block_k // MIN_BLOCK_SIZE, axis=1
      )  # [block_q_major, block_k_major]
      dv = lax.dot(p.T.astype(do.dtype), do, preferred_element_type=jnp.float32)
      dv_scratch_ref[pl.ds(start_k, block_k), :] += dv.astype(
          dv_scratch_ref.dtype
      )

      # di: [block_q, 128]
      # do: [block_q, head_dim]
      # v: [block_k_major, head_dim]
      dp = lax.dot_general(
          do, v, TRANS_B_DIM_NUMBERS, preferred_element_type=jnp.float32
      )
      ds = (dp - pltpu.repeat(di, block_k // MIN_BLOCK_SIZE, axis=1)) * p

      if sm_scale != 1.0:
        ds = ds * sm_scale

      # ds: [block_q_major, block_k_major]
      # q: [block_q_major, head_dim]
      dk = lax.dot(ds.T.astype(do.dtype), q, preferred_element_type=jnp.float32)
      dk_scratch_ref[pl.ds(start_k, block_k), :] += dk.astype(
          dk_scratch_ref.dtype
      )
    lax.fori_loop(0, block_k_major // block_k, k_body, None, unroll=True)

  if causal:
    should_run = below_or_on_diag(
        q_seq_index, block_q_major, kv_seq_index, block_k_major
    )
  else:
    should_run = True

  @pl.when(should_run)
  def run():
    lax.fori_loop(0, block_q_major // block_q, q_body, None, unroll=True)

  @pl.when(q_seq_index == q_seq_len // block_q_major - 1)
  def end_of_q_sequence():
    dv_tile_ref[0, 0, :, :] = dv_scratch_ref[...].astype(dv_tile_ref.dtype)
    dk_tile_ref[0, 0, :, :] = dk_scratch_ref[...].astype(dk_tile_ref.dtype)



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
  # if sm_scale != 1.0:
  #   raise NotImplementedError

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
    segment_ids, 
    o,
    l,
    m,
    causal: bool,
    mask_value: float,
    sm_scale: float,
    save_residuals: bool,
    do,
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


def main():
  key = random.PRNGKey(0)
  batch = 1
  heads = 1
  q_len = 128 * 128
  kv_len = 128 * 128
  head_dim = 128

  k1, k2, k3, k4 = random.split(key, 4)
  q = random.normal(k1, (batch, heads, q_len, head_dim), dtype=jnp.float32)
  k = random.normal(k2, (batch, heads, kv_len, head_dim), dtype=jnp.float32)
  v = random.normal(k3, (batch, heads, kv_len, head_dim), dtype=jnp.float32)
  do = random.normal(k4, (batch, heads, q_len, head_dim), dtype=jnp.float32)
  ab = None
  segment_ids = None

  block_b = 1
  block_q_major = 256
  block_q = 128
  block_k_major = 128
  block_k = 128

  causal = False
  sm_scale = float(1.0 / jnp.sqrt(head_dim).astype(jnp.float32))
  debug = False
  save_residuals = True

  print("Running reference attention (for numeric check)...")
  o_ref, l_ref, m_ref = mha_reference(q, k, v, save_residuals=save_residuals, sm_scale=sm_scale)


  print("Running Pallas TPU flash attention kernel...")
  o, l, m = _flash_attention_impl(
      q=q, k=k, v=v, ab=ab, segment_ids=segment_ids,
      save_residuals=save_residuals,
      causal=causal, sm_scale=sm_scale,
      block_b=block_b, block_q=block_q,
      block_k_major=block_k_major, block_k=block_k,
      debug=debug,
  )

  print("Running Reference MHA Backward kernel (for numeric check)")
  dq_ref, dk_ref, dv_ref, dab_ref, _ = _mha_reference_bwd(
      q=q, k=k, v=v, ab=ab,  o=o, do=do, m=m, l=l, segment_ids=segment_ids,
      causal=False, mask_value=None, sm_scale=sm_scale, 
      save_residuals=save_residuals, 
  )

  di = jnp.sum(o.astype(jnp.float32) * do.astype(jnp.float32), axis=-1)

  print("Running Pallas TPU flash attention kernel kv backward logic")
  dk_out, dv_out = flash_attention_bwd_dkv(
      q=q, k=k, v=v, ab=ab, di=di, do=do, l=l, m=m, segment_ids=segment_ids,
      causal=causal, sm_scale=sm_scale,
      block_q=block_q,
      block_q_major=block_q_major,
      block_k_major=block_k_major, block_k=block_k, 
      debug=debug,
  )

  diff_dk = jnp.linalg.norm(dk_out - dk_ref) / jnp.linalg.norm(dk_ref)
  print(f"Relative dk L2 error vs reference: {diff_dk:.3e}")

  diff_dv = jnp.linalg.norm(dv_out - dv_ref) / jnp.linalg.norm(dv_ref)
  print(f"Relative dv L2 error vs reference: {diff_dv:.3e}")


if __name__ == "__main__":
  main()
