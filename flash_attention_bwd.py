#@title Imports
import functools
from typing import Callable

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax import lax
from jax.extend import core
from jax._src.util import safe_map


import jax.numpy as jnp
import numpy as np

dimension_numbers = (((1,), (1,)), ((), ()))
MIN_BLOCK_SIZE = 128





def _flash_attention_dq_kernel(
    q_tile_ref,
    k_tile_ref,
    v_tile_ref,
    ab_tile_ref,
    l_tile_ref,
    m_tile_ref,
    do_tile_ref,
    di_tile_ref,
    dq_tile_ref,
    ds_tile_ref,
    dq_scratch_ref,
    *,
    sm_scale: float,
    causal: bool,
    kv_seq_len: int,
    block_k: int,
    score_jaxpr = None
):
  _, _, block_k_major, _ = k_tile_ref.shape
  _, _, block_q_major, _ = q_tile_ref.shape

  kv_seq_index = pl.program_id(axis=3)
  q_seq_index = pl.program_id(axis=2)

  @pl.when(kv_seq_index == 0)
  def start_new_sequence():
    dq_scratch_ref[:, :] = jnp.zeros(dq_scratch_ref.shape, dq_scratch_ref.dtype)

  def body(i, _):
    k_slice = pl.ds(i * block_k, block_k)
    q = q_tile_ref[0, 0, :, :]
    k = k_tile_ref[0, 0, k_slice, :]  # [block_k, head_dim]
    v = v_tile_ref[0, 0, k_slice, :]  # [block_k, head_dim]
    l = l_tile_ref[0, 0, :, :]  # [block_q_major, 128]
    m = m_tile_ref[0, 0, :, :]  # [block_q_major, 128]
    do = do_tile_ref[0, 0, :, :]  # [block_q_major, head_dim]
    di = di_tile_ref[0, 0, :].astype(jnp.float32)  # [block_q_major, 128]

    capped_logits = jax.lax.dot_general(
        q, k, dimension_numbers, preferred_element_type=jnp.float32
    )

    if ab_tile_ref is not None:
      ab = ab_tile_ref[0, 0, :, pl.dslice(i * block_k, block_k)].astype(
          jnp.float32
      )
      capped_logits += ab

    if sm_scale != 1.0:
      capped_logits *= sm_scale


    p = jnp.exp(
        capped_logits - pltpu.repeat(m, block_k // MIN_BLOCK_SIZE, axis=1)
    )
    p = p * pltpu.repeat(
        1 / l, block_k // MIN_BLOCK_SIZE, axis=1
    )  # [block_q_major, block_k]

    # di: [block_q_major, 128]
    # do: [block_q_major, head_dim]
    # v: [block_k_major, head_dim]
    dp = jax.lax.dot_general(
        do,
        v,
        dimension_numbers,
        preferred_element_type=jnp.float32,
    )

    ds = (dp - pltpu.repeat(di, block_k // MIN_BLOCK_SIZE, axis=1)) * p
    if sm_scale != 1.0:
        ds = ds * sm_scale
    ##############################
    # CUSTOM SCORE BACKWARD HOOK
    ##############################
    if score_jaxpr is not None:
        dq_local, dk_local = _inline_jaxpr_score_backward(q, k, score_jaxpr, ds)

        # accumulate per-k-block dk
        dk_tile_slice = pl.ds(i * block_k, block_k)
        dq_scratch_ref[:, :] += dq_local.astype(dq_scratch_ref.dtype)
        # save dk into dv_scratch or dk_scratch depending on your code
        # (in dq kernel we ONLY compute dq, so DK belongs in ds buffer)
        if ds_tile_ref is not None:
            ds_tile_ref[0, 0, :, dk_tile_slice] = dk_local.astype(ds_tile_ref.dtype)

    else:
        # default FlashAttention dq path
        dq_scratch_ref[:, :] += lax.dot(
            ds.astype(k.dtype),
            k,
            preferred_element_type=jnp.float32,
        ).astype(dq_scratch_ref.dtype)


#   if causal:
#     should_run = below_or_on_diag(
#         q_seq_index, block_q_major, kv_seq_index, block_k_major
#     )
#     should_not_run = lax.select(should_run, False, True)
#   else:
  should_run = True
  should_not_run = False  # type: ignore

  @pl.when(should_run)
  def run():
    lax.fori_loop(0, block_k_major // block_k, body, None, unroll=True)

  @pl.when(should_not_run)
  def zero_out_ds():
    if ds_tile_ref is not None:
      ds_tile_ref[...] = jnp.zeros_like(ds_tile_ref)

  @pl.when(kv_seq_index == kv_seq_len // block_k_major - 1)
  def end_of_kv_sequence():
    dq_tile_ref[0, 0, :, :] = dq_scratch_ref[...].astype(dq_tile_ref.dtype)
    dq_scratch_ref[...] = jnp.zeros_like(dq_scratch_ref)


def _flash_attention_bwd_dq(
    q,
    k,
    v,
    ab,
    l,
    m,
    do,
    di,
    *,
    block_q_major: int | None,
    block_k_major: int | None,
    block_k: int | None,
    sm_scale: float,
    causal: bool,
    debug: bool,
    score_fn = None
):
  batch_size, num_heads, q_seq_len, head_dim = q.shape
  _, _, kv_seq_len, _ = k.shape
#   _verify_block("block_q_dq", "q_seq_len", block_q_major, q_seq_len)
#   _verify_block("block_k_major_dq", "kv_seq_len", block_k_major, kv_seq_len)
#   _verify_block("block_k_dq", "block_k", block_k, kv_seq_len)

  # Broadcast out scalar values
  m = jnp.broadcast_to(m[..., None], (*m.shape, MIN_BLOCK_SIZE))
  l = jnp.broadcast_to(l[..., None], (*l.shape, MIN_BLOCK_SIZE))
  # Preprocess contraction for bwd pass
  di = jnp.broadcast_to(di[..., None], (*di.shape, block_k_major))

  grid = (
      batch_size,
      num_heads,
      q_seq_len // block_q_major,
      kv_seq_len // block_k_major,
  )

  def qo_index_map(batch_index, head_index, q_seq_index, _):
    return (batch_index, head_index, q_seq_index, 0)

  qo_spec = pl.BlockSpec((1, 1, block_q_major, head_dim), qo_index_map)
  do_spec = qo_spec

  def kv_index_map(batch_index, head_index, q_seq_index, kv_seq_index):
    # if causal:
    #   # If the kv block is skipped, prefetch the next valid kv block, i.e. the
    #   # 0th one to be used for the next block_q rows.
    #   next_kv_index = lax.select(
    #       below_or_on_diag(
    #           q_seq_index, block_q_major, kv_seq_index, block_k_major
    #       ),
    #       kv_seq_index,
    #       0,
    #   )
    # else:
    next_kv_index = kv_seq_index
    return (batch_index, head_index, next_kv_index, 0)

  kv_spec = pl.BlockSpec((1, 1, block_k_major, head_dim), kv_index_map)
  assert kv_spec.block_shape is not None
  assert k.ndim == len(kv_spec.block_shape)
  assert v.ndim == len(kv_spec.block_shape)

  def lm_index_map(batch_index, head_index, q_seq_index, _):
    return (batch_index, head_index, q_seq_index, 0)

  lm_spec = pl.BlockSpec((1, 1, block_q_major, MIN_BLOCK_SIZE), lm_index_map)
  assert lm_spec.block_shape is not None
  assert l.ndim == len(lm_spec.block_shape)
  assert m.ndim == len(lm_spec.block_shape)

  di_spec = pl.BlockSpec((1, 1, block_q_major, MIN_BLOCK_SIZE), qo_index_map)
  assert di_spec.block_shape is not None
  assert di.ndim == len(di_spec.block_shape)

  def ab_index_map(batch_index, head_index, q_seq_index, kv_seq_index):
    return (batch_index, head_index, q_seq_index, kv_seq_index)

  dab_spec = (
      pl.BlockSpec((1, 1, block_q_major, block_k_major), ab_index_map)
      if ab is not None
      else None
  )


  in_specs = [
      qo_spec,
      kv_spec,
      kv_spec,
      dab_spec,
      lm_spec,
      lm_spec,
      do_spec,
      di_spec,
  ]

  out_shapes = [
      jax.ShapeDtypeStruct(q.shape, q.dtype),
      jax.ShapeDtypeStruct(ab.shape, ab.dtype) if ab is not None else None,
  ]
  dq_spec = pl.BlockSpec((1, 1, block_q_major, head_dim), qo_index_map)
  out_specs = [
      dq_spec,
      dab_spec,
  ]
  scratch_shapes = [pltpu.VMEM((block_q_major, head_dim), jnp.float32)]  # type: ignore

  if score_fn is not None:
      score_jaxpr = jax.make_jaxpr(score_fn)(
          jnp.zeros((block_q_major, head_dim), q.dtype),
          jnp.zeros((block_k, head_dim), k.dtype),
      )
      # print(score_jaxpr.jaxpr)
  else:
      score_jaxpr = None

  kernel = functools.partial(
      _flash_attention_dq_kernel,
      sm_scale=sm_scale,
      causal=causal,
      block_k=block_k,  # type: ignore
      kv_seq_len=kv_seq_len,
      score_jaxpr= score_jaxpr
  )

  name_scope = f"flash_mha_bwd_dq_{block_q_major=}_{block_k_major=}_{block_k=}"
  with jax.named_scope(name_scope):
    dq, ds = pl.pallas_call(
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
    )(q, k, v, ab, l, m, do, di)

  # dab is just ds
  return dq, ds