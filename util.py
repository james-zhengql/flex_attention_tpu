import ast
import inspect
import textwrap
import jax.numpy as jnp
import jax
from jax import lax
from jax.extend import core
from jax._src.util import safe_map
from jax.experimental.pallas import tpu as pltpu

from constants import MIN_BLOCK_SIZE

dimension_numbers = (((1,), (1,)), ((), ()))
MIN_BLOCK_SIZE = 128
TRANS_B_DIM_NUMBERS = (((0,), (0,)), ((), ()))



# ============================================================
# AST transformer: safe, fusion-friendly JAX lowering
# ============================================================

class ToJaxTransformer(ast.NodeTransformer):
    """Rewrite Python ops → JAX ops, and inline-lower dot/einsum to lax.dot_general,
       with NO runtime assertions (to preserve fusion)."""

    def __init__(self):
        super().__init__()

    # ---------------------------------------------------------
    # Rewrite binary operators into JAX ops
    # ---------------------------------------------------------
    def visit_BinOp(self, node):
        self.generic_visit(node)
        left, right = node.left, node.right

        # Map + - * /
        ops = {
            ast.Add:  "add",
            ast.Sub:  "subtract",
            ast.Mult: "multiply",
            ast.Div:  "divide",
        }
        op_type = type(node.op)

        if op_type in ops:
            return ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="jnp", ctx=ast.Load()),
                    attr=ops[op_type],
                    ctx=ast.Load()
                ),
                args=[left, right],
                keywords=[],
            )

        return node

    # ---------------------------------------------------------
    # Rewrite function calls: dot → lax.dot_general, einsum → dot_general
    # ---------------------------------------------------------
    def visit_Call(self, node):
        self.generic_visit(node)
        fn = node.func

        # Replace built-in sum → jnp.sum
        if isinstance(fn, ast.Name) and fn.id == "sum":
            return ast.Call(
                func=ast.Attribute(value=ast.Name("jnp", ctx=ast.Load()),
                                   attr="sum", ctx=ast.Load()),
                args=node.args,
                keywords=[]
            )

        # jnp.tanh → already fine
        if isinstance(fn, ast.Attribute) and fn.attr == "tanh":
            return node

        # -----------------------------------------------------
        # jnp.dot(q, k) → inline lax.dot_general(q, k)
        # -----------------------------------------------------
        if isinstance(fn, ast.Attribute) and fn.attr == "dot":
            q, k = node.args   # assume dot(q, k)
            return self._make_inline_dot_general(q, k)

        # -----------------------------------------------------
        # jnp.einsum("...d,...d->...", q, k)
        # -----------------------------------------------------
        if isinstance(fn, ast.Attribute) and fn.attr == "einsum":
            pattern_node = node.args[0]

            if not (isinstance(pattern_node, ast.Constant)
                    and isinstance(pattern_node.value, str)):
                raise AssertionError(
                    "einsum pattern must be a literal string like '...d,...d->...'"
                )

            pattern = pattern_node.value
            # In util.py
            pattern = pattern_node.value.replace(" ", "") # Remove spaces first

            # Allow specific known patterns that are semantically equivalent
            ALLOWED_PATTERNS = [
                "...d,...d->...",
                "qd,kd->qk",
            ]

            if pattern not in ALLOWED_PATTERNS:
                raise AssertionError(f"Unsupported einsum pattern '{pattern}'. "
                    "Only '...d,...d->...' is allowed for FlashAttention TPU.")

            q, k = node.args[1], node.args[2]
            return self._make_inline_dot_general(q, k)

        return node

    # ---------------------------------------------------------
    # Build inline lax.dot_general call (no wrappers!)
    # ---------------------------------------------------------
    def _make_inline_dot_general(self, q, k):

        contracting_axis = ast.Constant(value=1)  # last dim of rank-2 tensors

        dims = ast.Tuple(
            elts=[
                ast.Tuple(  # contracting dims
                    elts=[
                        ast.Tuple(elts=[contracting_axis], ctx=ast.Load()),
                        ast.Tuple(elts=[contracting_axis], ctx=ast.Load())
                    ],
                    ctx=ast.Load()
                ),
                ast.Tuple(  # no batch dims
                    elts=[
                        ast.Tuple(elts=[], ctx=ast.Load()),
                        ast.Tuple(elts=[], ctx=ast.Load())
                    ],
                    ctx=ast.Load()
                )
            ],
            ctx=ast.Load()
        )

        return ast.Call(
            func=ast.Attribute(
                value=ast.Name(id="lax", ctx=ast.Load()),
                attr="dot_general",
                ctx=ast.Load(),
            ),
            args=[q, k, dims],
            keywords=[]
        )


# ============================================================
# Final wrapper: validate user_fn, rewrite AST to pure JAX IR
# ============================================================

def make_jax_score_fn(user_fn):
    """
    Validate + rewrite user score function into a pure JAX function
    that TPU Pallas kernels can fully fuse.
    """

    # -----------------------------
    # 0. Extract user source code
    # -----------------------------
    src = textwrap.dedent(inspect.getsource(user_fn))
    tree = ast.parse(src)

    # -----------------------------
    # 1. Static validation (compile-time)
    #    -> NO runtime asserts allowed
    # -----------------------------
    # (You can add more rules here if needed)
    # Ensure function has exactly 2 args (q, k)
    sig = inspect.signature(user_fn)
    if len(sig.parameters) != 2:
        raise AssertionError("score_fn must have exactly two arguments: (q, k)")

    # -----------------------------
    # 2. AST rewrite → inline lax.dot_general
    # -----------------------------
    transformer = ToJaxTransformer()
    tree = transformer.visit(tree)
    ast.fix_missing_locations(tree)
    
    # Pretty-print transformed code
    print("==== Transformed JAX Code ====")
    print(ast.unparse(tree))
    print("================================")
    # -----------------------------
    # 3. Compile rewritten AST
    # -----------------------------
    namespace = {"jnp": jnp, "lax": lax}
    code = compile(tree, filename="<ast>", mode="exec")
    exec(code, namespace)

    # -----------------------------
    # 4. Return the JIT-compiled function
    # -----------------------------
    rewritten_fn = namespace[user_fn.__name__]
    return rewritten_fn



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
    
    # 1. d(Loss)/dQ = d_out @ rhs.T (or d_out @ K)
    d_Q = lax.dot_general(d_out, rhs, dn_r, preferred_element_type=jnp.float32)
    
    # 2. d(Loss)/d(K.T) = lhs.T @ d_out
    d_KT = lax.dot_general(lhs, d_out, dn_l, preferred_element_type=jnp.float32)
    
    # Add transpose to d(Loss)/d(K.T) to get d(Loss)/dK, as requested
    d_K = lax.transpose(d_KT, permutation=(1, 0))
    
    return (d_Q, d_K)

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


def _inline_fused_attn_backward(q, k, v, dO, l, m, di, closed_jaxpr, sm_scale):
    """
    Executes:
    1. Forward Custom Score (generates S and saves env)
    2. Softmax & Attention Gradients (calculates P, dV, dS)
    3. Backward Custom Score (uses env and dS to get dQ, dK)
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
        if var not in grad_env:
            grad_env[var] = val
        else:
            grad_env[var] = grad_env[var] + val

    # =========================================================
    # PART 1: Forward Pass (Compute S and Save Env)
    # =========================================================
    write(jaxpr.invars[0], q)
    write(jaxpr.invars[1], k)
    safe_map(write, jaxpr.constvars, consts)

    for eqn in jaxpr.eqns:
        invals = safe_map(read, eqn.invars)
        outvals = eqn.primitive.bind(*invals, **eqn.params)
        if not eqn.outvars: continue
        if eqn.primitive.multiple_results:
             safe_map(write, eqn.outvars, outvals)
        else:
             write(eqn.outvars[0], outvals)

    # Extract the Score S from the jaxpr output
    S = read(jaxpr.outvars[0])

    # =========================================================
    # PART 2: The "Middle" Math (Flash Attention Backward Logic)
    # =========================================================
    # 1. Apply Scale
    S_scaled = S * sm_scale

    # 2. Recompute P (Probabilities)
    # Note: Assuming block_k, MIN_BLOCK_SIZE are available in scope or passed in
    # Use the dimensions from the inputs to be safe
    block_k = k.shape[0]
    # Adjust repeat logic as per your environment (pltpu.repeat or jnp.broadcast)
    # unnormalized = jnp.exp(S_scaled - m)
    unnormalized = jnp.exp(S_scaled - pltpu.repeat(m, block_k // MIN_BLOCK_SIZE, axis=1))
    P = unnormalized / pltpu.repeat(l, block_k // MIN_BLOCK_SIZE, axis=1)

    # 3. Compute dV update (This chunk's contribution to dV)
    # dv_chunk = P.T @ dO
    dv_chunk = jax.lax.dot_general(P, dO, TRANS_B_DIM_NUMBERS, preferred_element_type=jnp.float32)

    # 4. Compute dP
    # dP = dO @ V.T
    dP = jax.lax.dot_general(dO, v, dimension_numbers, preferred_element_type=jnp.float32)

    # 5. Compute dS (Gradient of Score)
    # dS = P * (dP - di)
    dS = P * (dP - pltpu.repeat(di, block_k // MIN_BLOCK_SIZE, axis=1))
    
    if sm_scale != 1.0:
        dS = dS * sm_scale

    # =========================================================
    # PART 3: Backward Pass (Compute dQ, dK using dS)
    # =========================================================
    
    # Seed the gradient with the dS we just calculated
    accumulate_grad(jaxpr.outvars[0], dS)

    for eqn in jaxpr.eqns[::-1]:
        primals_in = safe_map(read, eqn.invars)
        if eqn.primitive.multiple_results:
            primals_out = safe_map(read, eqn.outvars)
            d_out = safe_map(read_grad, eqn.outvars)
        else:
            primals_out = read(eqn.outvars[0])
            d_out = read_grad(eqn.outvars[0])

        if eqn.primitive not in _PRIMITIVE_BWD_TABLE:
             raise NotImplementedError(f"Missing VJP for {eqn.primitive}")
             
        d_inputs = _PRIMITIVE_BWD_TABLE[eqn.primitive](
            primals_in, primals_out, d_out, eqn.params
        )
        
        safe_map(accumulate_grad, eqn.invars, d_inputs)

    # Return dQ, dK, and the dV update
    return read_grad(jaxpr.invars[0]), read_grad(jaxpr.invars[1]), dv_chunk