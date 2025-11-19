import ast
import inspect
import textwrap
import jax.numpy as jnp
import jax
from jax import lax


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
            if pattern != "...d,...d->...":
                raise AssertionError(
                    f"Unsupported einsum pattern '{pattern}'. "
                    "Only '...d,...d->...' is allowed for FlashAttention TPU."
                )

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
    return jax.jit(rewritten_fn)
