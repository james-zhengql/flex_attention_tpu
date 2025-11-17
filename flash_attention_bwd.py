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

def _flash_attention_bwd(q_tile_ref, *args, **kwargs):

    block_b = q_tile_ref.shape[0]

    # Create the real kernel from the factory
    kernel = flash_attention_bwd_kernel

    for batch_idx in range(block_b):
        kernel(
            (batch_idx, 0),
            q_tile_ref,
            *args,
            **kwargs,
        )

def flash_attention_bwd(
    q,
    k,
    v,
    ab,
    segment_ids,
    l,
    o,
    do,
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
    batch_size, head_num, q_seq_len, head_dim = q.shape
    _, _, kv_seq_len, _ = k.shape
    
    # Grid specification
    grid = (
        pl.cdiv(batch_size, block_b),
        head_num,
        pl.cdiv(q_seq_len, block_q_major),
        pl.cdiv(kv_seq_len, block_k_major),
    )

    def qo_index_map(batch_index, head_index, q_seq_index, _):
        return (batch_index, head_index, q_seq_index, 0)

    qo_spec = pl.BlockSpec((1, 1, block_q_major, head_dim), qo_index_map)
    do_spec = qo_spec

    def kv_index_map(batch_index, head_index, q_seq_index, kv_seq_index):
        # if causal:
        # # If the kv block is skipped, prefetch the next valid kv block, i.e. the
        # # 0th one to be used for the next block_q rows.
        #     next_kv_index = lax.select(
        #     below_or_on_diag(
        #         q_seq_index, block_q_major, kv_seq_index, block_k_major
        #     ),
        #     kv_seq_index,
        #     0,
        #     )
        # else:
        next_kv_index = kv_seq_index
        return (batch_index, head_index, next_kv_index, 0)
        
    kv_spec = pl.BlockSpec((batch_size, head_num, kv_seq_len, head_dim), kv_index_map)
    assert kv_spec.block_shape is not None
    assert k.ndim == len(kv_spec.block_shape)
    assert v.ndim == len(kv_spec.block_shape)

    def lm_index_map(batch_index, head_index, q_seq_index, _):
        return (batch_index, head_index, q_seq_index, 0)

    lm_spec = pl.BlockSpec((1, 1, block_q_major, MIN_BLOCK_SIZE), lm_index_map)
    assert lm_spec.block_shape is not None
    assert l.ndim == len(lm_spec.block_shape)

    def ab_index_map(batch_index, head_index, q_seq_index, kv_seq_index):
        return (batch_index, head_index, q_seq_index, kv_seq_index)

    dab_spec = (
      pl.BlockSpec((1, 1, block_q_major, block_k_major), ab_index_map)
      if ab is not None
      else None
    )

    # Allocate scratch buffers
    if block_k != kv_seq_len:
        dk_scratch = pltpu.VMEM((block_b, 1, block_k_major, head_dim), jnp.float32)
        dv_scratch = pltpu.VMEM((block_b, 1, block_k_major, head_dim), jnp.float32)
        # scratch_shapes = [dq_scratch, dk_scratch, dv_scratch]
        scratch_shapes = [dk_scratch, dv_scratch]
    else:
        scratch_shapes = []
    
    # in_spec specify
    in_specs = [
      qo_spec,      # q
      kv_spec,      # k
      kv_spec,      # v
      dab_spec,     # bias
      lm_spec,      # l
      qo_spec,      # o
      do_spec,      # do     
    ]

    out_shapes = [
      jax.ShapeDtypeStruct(k.shape, k.dtype),
      jax.ShapeDtypeStruct(v.shape, v.dtype),
      jax.ShapeDtypeStruct(ab.shape, ab.dtype) if ab is not None else None,
    ]
    
    def dk_index_map(b, h, kv_tile):
        k_start = kv_tile * block_k
        k_end   = jnp.minimum(k_start + block_k, kv_seq_len)
        return (
            b,                              # batch 
            h,                              # head 
            jnp.arange(k_start, k_end),     # K 
            jnp.arange(head_dim),           # D 
    )
    dq_spec = pl.BlockSpec((batch_size, head_num, block_q_major, head_dim), qo_index_map)
    dk_spec = pl.BlockSpec((batch_size, head_num, block_k_major, head_dim), dk_index_map)
    dv_spec = pl.BlockSpec((batch_size, head_num, block_k_major, head_dim), dk_index_map)
    out_specs = [
        dq_spec,
        dk_spec,
        dv_spec,
        dab_spec,
    ]

    kernel = functools.partial(
        _flash_attention_bwd,
        causal = causal,
        sm_scale = sm_scale,
        block_q = block_q_major,
        q_seq_len = q_seq_len
    )

    dq, dq, dv, *aux = pl.pallas_call(
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
          dimension_semantics=("parallel", "parallel", "parallel", "arbitrary", "parallel", "parallel", "parallel")
      ),
  )(q, k, v, ab, l, o, do)


def flash_attention_bwd_kernel(
      batch_idx,
      q_tile_ref,
      k_tile_ref,
      v_tile_ref,
      ab_tile_ref,
      O_tile_ref,
      dO_tile_ref,
      dq_tile_ref,
      dk_tile_ref,
      dv_tile_ref,
    #   m_tile_ref,
      l_tile_ref,
      dk_scratch_ref, 
      dv_scratch_ref,
    #   O_scratch_ref,
    #   m_scratch_ref,
    #   l_scratch_ref,
      *,
      causal,
      sm_scale,
      block_q,
      q_seq_len,
  ):
    _, _, q_seq_length = q_tile_ref.shape
    
    kv_tile_idx = pl.program_id(axis = 3)
    q_tile_idx = pl.program_id(axis = 2)

    k = k_tile_ref[batch_idx]
    v = v_tile_ref[batch_idx]

    D = jnp.sum(dO_tile_ref * O_tile_ref, axis = -1)
    block_q_repeats, rem = divmod(block_q, MIN_BLOCK_SIZE)
    @pl.when(kv_tile_idx == 0)
    def start_new_kv_seq():
        # dq_scratch_ref[:, :] = jnp.zeros(dq_scratch_ref.shape, dq_scratch_ref.dtype)
        dk_scratch_ref[batch_idx] = jnp.zeros(dk_scratch_ref[2:].shape, dk_scratch_ref.dtype)
        dv_scratch_ref[batch_idx] = jnp.zeros(dv_scratch_ref[2:].shape, dv_scratch_ref.dtype)
    
    def body():
        @pl.loop(0, q_seq_length, step=block_q, unroll=True)
        def _body(start_q):
            q = q_tile_ref[(*batch_idx, pl.dslice(start_q, block_q), slice(None))]
            l = l_tile_ref[(*batch_idx, pl.dslice(start_q, block_q), slice(None))]
            # D = D_tile_ref[(*batch_idx, pl.dslice(start_q, block_q), slice(None))]
            # O = O_tile_ref[(*batch_idx, pl.dslice(start_q, block_q), slice(None))]
            dO = dO_tile_ref[(*batch_idx, pl.dslice(start_q, block_q), slice(None))]
            dq = dq_tile_ref[(*batch_idx, pl.dslice(start_q, block_q), slice(None))]

            # dq_past = dq_scratch_ref[batch_idx]
            dk_past = dk_scratch_ref[batch_idx]
            dv_past = dv_scratch_ref[batch_idx]

            S = jax.lax.dot_general(q, k, dimension_numbers, preferred_element_type=jnp.float32)
            P = jnp.exp(S - pltpu.repeat(l, block_q_repeats, 1))
            # dv = dv_past + jnp.einsum('rb, rd->bd', P, dO)
            dv = dv_past + jax.lax.dot_general(P,dO,dimension_numbers,preferred_element_type=jnp.float32)
            dv_scratch_ref[batch_idx] = dv

            # dP = jnp.einsum("qc,kc->qk", dO, v)
            dP = jax.lax.dot_general(dO,v,dimension_numbers,preferred_element_type=jnp.float32)
            dS = P * (dP - D)

            # dq = dq_past + jnp.einsum("bhqk,bhkc->bhqc", dS, k)
            # dq = dq + jnp.einsum("bhqk,bhkc->bhqc", dS, k)
            dq = dq + jax.lax.dot_general(dS,k,dimension_numbers,preferred_element_type=jnp.float32)
            
            # dk = dk_past + jnp.einsum("bhqk,bhqc->bhkc", dS, q)
            dk = dk_past + jax.lax.dot_general(dS,q,dimension_numbers,preferred_element_type=jnp.float32)
            dk_scratch_ref[batch_idx] = dk

            dq_tile_ref[start_q] = dq.astype(dq_tile_ref.dtype)

    @pl.when(q_tile_idx == (q_seq_len // block_q) - 1)
    def store_res():
      dk_tile_ref[batch_idx] = dk_scratch_ref[batch_idx].astype(dk_tile_ref.dtype)
      dv_tile_ref[batch_idx] = dv_scratch_ref[batch_idx].astype(dv_tile_ref.dtype)
            
    # return flash_attention_bwd_kernel


