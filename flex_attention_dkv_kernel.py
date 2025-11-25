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
from flash_attention_fwd_ref import _flash_attention_impl_ref
from flex_attention_kernel import _flex_attention_impl
from util import _inline_jaxpr_score_backward, make_jax_score_fn

dimension_numbers = (((1,), (1,)), ((), ()))
MIN_BLOCK_SIZE = 128
TRANS_B_DIM_NUMBERS = (((0,), (0,)), ((), ()))


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
    score_fn = None
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

    di_spec = pl.BlockSpec((1, 1, block_q_major, MIN_BLOCK_SIZE), qo_index_map)
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

    if score_fn is not None:
      score_jaxpr = jax.make_jaxpr(score_fn)(
          jnp.zeros((block_q, head_dim), q.dtype),
          jnp.zeros((block_k, head_dim), k.dtype),
      )
    else:
        score_jaxpr = None

    kernel = functools.partial(
        flash_attention_dkv_kernel,
        causal = causal,
        sm_scale = sm_scale,
        block_q = block_q,
        block_k = block_k,
        q_seq_len = q_seq_len,
        block_q_major = block_q_major,
        block_k_major=block_k_major,
        score_jaxpr= score_fn
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
      block_k_major,
      score_jaxpr
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
                        
            if score_jaxpr is not None:
                S, score_grad_fn = jax.vjp(score_jaxpr, q, k)
            else:
                S = jax.lax.dot_general(q, k, dimension_numbers, preferred_element_type=jnp.float32)
                score_grad_fn = None

            S = S * sm_scale
            # unnormalized = jnp.exp(S - m_block[:, None])
            unnormalized = jnp.exp(S - pltpu.repeat(m, block_k // MIN_BLOCK_SIZE, axis=1))   # (block_q, block_k)
            P = unnormalized / pltpu.repeat(l, block_k // MIN_BLOCK_SIZE, axis=1)                 # (block_q, block_k)

            # dv = dv_past + jnp.einsum('rb, rd->bd', P, dO)
            dv = dv_past + jax.lax.dot_general(P,dO,TRANS_B_DIM_NUMBERS,preferred_element_type=jnp.float32)
            # dv = dv_past + jnp.einsum("qk,qd->kd", P, dO)

            dv_scratch_ref[pl.dslice(start_k, block_k), :] = dv.astype(dv_scratch_ref.dtype)

            dP = jax.lax.dot_general(dO,v,dimension_numbers,preferred_element_type=jnp.float32)
  
            # print(f"di shape{di.shape}")
            dS = P * (dP - pltpu.repeat(di, block_k // MIN_BLOCK_SIZE, axis=1))

            if sm_scale != 1.0:
              dS = dS * sm_scale  

            # --- CHANGE 2: Backward Pass using Saved Residuals ---
            if score_grad_fn is not None:
              # This uses the residuals saved from step 1. 
              # It's fused and efficient. 
              # jax.vjp returns a tuple (dQ, dK)
              dq_update, dk_update = score_grad_fn(dS)
            else:
              dk_update = jax.lax.dot(dS.T.astype(dO.dtype), q, preferred_element_type=jnp.float32)
            
            print(f"dk shape{dk_update.shape}")
            dk = dk_past + dk_update

            dk_scratch_ref[pl.dslice(start_k, block_k), :] = dk.astype(dk_scratch_ref.dtype)

    
    @pl.when(q_tile_idx == q_seq_len // block_q_major - 1)
    def store_res():
      dv_tile_ref[0, 0, :, :] = dv_scratch_ref[...].astype(dv_tile_ref.dtype)
      dk_tile_ref[0, 0, :, :] = dk_scratch_ref[...].astype(dk_tile_ref.dtype)
    