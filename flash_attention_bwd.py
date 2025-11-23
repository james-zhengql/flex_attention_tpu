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

###################################################
# BACKEND IMPLEMENTATIONS FOR PRIMITIVES
###################################################

def _bwd_add(inputs, out, d_out, params):
    x, y = inputs
    return d_out, d_out

def _bwd_mul(inputs, out, d_out, params):
    x, y = inputs
    return d_out * y, d_out * x

def _bwd_sub(inputs, out, d_out, params):
    x, y = inputs
    return d_out, -d_out

def _bwd_div(inputs, out, d_out, params):
    x, y = inputs
    return d_out / y, -d_out * x / (y * y)

def _bwd_tanh(inputs, out, d_out, params):
    return (d_out * (1 - out * out),)

def _bwd_reduce_sum(inputs, out, d_out, params):
    (x,) = inputs
    return (jnp.broadcast_to(d_out, x.shape),)

def _bwd_dot_general(inputs, out, d_out, params):
    lhs, rhs = inputs
    contracting_dims_r = ((1,), (0,)) 
    contracting_dims_l = ((0,), (0,)) 
    batch_dims = ((), ()) # Assuming no batch dims based on your debug output
    dn_r = (contracting_dims_r, batch_dims)
    dn_l = (contracting_dims_l, batch_dims)
    
    return (
        lax.dot_general(d_out, rhs, dn_r, preferred_element_type=jnp.float32),
        lax.dot_general(lhs, d_out, dn_l, preferred_element_type=jnp.float32),
    )

def _bwd_broadcast_in_dim(inputs, out, d_out, params):
    (x,) = inputs
    bdims = params["broadcast_dimensions"]
    out_shape = out.shape

    reduce_axes = tuple(i for i in range(len(out_shape)) if i not in bdims)

    dx = jnp.sum(d_out, axis=reduce_axes)
    dx = jnp.reshape(dx, x.shape)

    return (dx,)

_PRIMITIVE_BWD_TABLE = {
    lax.add_p: _bwd_add,
    lax.mul_p: _bwd_mul,
    lax.sub_p: _bwd_sub,
    lax.div_p: _bwd_div,
    lax.tanh_p: _bwd_tanh,
    lax.reduce_sum_p: _bwd_reduce_sum,
    lax.dot_general_p: _bwd_dot_general,
    lax.broadcast_in_dim_p: _bwd_broadcast_in_dim,
}

#############################################################
# INLINE SCORE BACKWARD (FULLY FIXED)
#############################################################

def _inline_jaxpr_score_backward(q, k, closed_jaxpr, d_score):
    """
    Executes the Forward and Backward pass of a custom score function 
    inside a Pallas kernel loop.
    
    Args:
        q: Query tile [BlockQ, HeadDim]
        k: Key tile [BlockK, HeadDim]
        closed_jaxpr: The custom score ClosedJaxpr object
        d_score: The incoming gradient w.r.t the score [BlockQ, BlockK]
    """
    jaxpr = closed_jaxpr.jaxpr
    consts = closed_jaxpr.literals
    
    # --- 1. Environments ---
    env = {} 
    grad_env = {}

    def read(var):
        if type(var) is core.Literal: return var.val
        return env[var]

    def write(var, val):
        env[var] = val

    def read_grad(var):
        if type(var) is core.Literal: return 0.0
        return grad_env.get(var, 0.0)

    def accumulate_grad(var, val):
        if type(var) is core.Literal: return
        # Initialize if missing, otherwise add
        if var not in grad_env:
            grad_env[var] = val
        else:
            grad_env[var] = grad_env[var] + val

    # --- 2. Forward Pass (Recompute Primals) ---
    # Map Q and K to the input variables of the Jaxpr
    # Assuming score_fn(q, k) -> score
    write(jaxpr.invars[0], q)
    write(jaxpr.invars[1], k)
    safe_map(write, jaxpr.constvars, consts)

    for eqn in jaxpr.eqns:
        invals = safe_map(read, eqn.invars)
        # We rely on JAX/Pallas to inline these primitive calls
        outvals = eqn.primitive.bind(*invals, **eqn.params)
        if not eqn.outvars: continue
        if eqn.primitive.multiple_results:
             safe_map(write, eqn.outvars, outvals)
        else:
             write(eqn.outvars[0], outvals)

    # --- 3. Backward Pass (Compute Gradients) ---
    # Seed the gradient with d_score (ds)
    # Assuming single output score function
    accumulate_grad(jaxpr.outvars[0], d_score)

    for eqn in jaxpr.eqns[::-1]:
        # Primal Inputs/Outputs
        primals_in = safe_map(read, eqn.invars)
        if eqn.primitive.multiple_results:
            primals_out = safe_map(read, eqn.outvars)
            d_out = safe_map(read_grad, eqn.outvars)
        else:
            primals_out = read(eqn.outvars[0])
            d_out = read_grad(eqn.outvars[0])

        # Look up VJP
        if eqn.primitive not in _PRIMITIVE_BWD_TABLE:
             raise NotImplementedError(f"Missing VJP for {eqn.primitive}")
             
        d_inputs = _PRIMITIVE_BWD_TABLE[eqn.primitive](
            primals_in, primals_out, d_out, eqn.params
        )
        
        safe_map(accumulate_grad, eqn.invars, d_inputs)

    # Return dQ and dK
    return read_grad(jaxpr.invars[0]), read_grad(jaxpr.invars[1])



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