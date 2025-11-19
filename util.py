import ast
import inspect
import textwrap
import jax.numpy as jnp
import jax
from jax import lax

# ------------------------------
# AST transformer: Python → JAX IR
# ------------------------------

class ToJaxTransformer(ast.NodeTransformer):
    """Rewrite Python math into JAX ops and rewrite dot operations into lax.dot_general."""

    def __init__(self):
        super().__init__()

    # --- handle binary ops: +, -, *, /
    def visit_BinOp(self, node):
        self.generic_visit(node)

        left = node.left
        right = node.right

        if isinstance(node.op, ast.Add):
            return ast.Call(
                func=ast.Attribute(value=ast.Name(id="jnp", ctx=ast.Load()),
                                   attr="add", ctx=ast.Load()),
                args=[left, right],
                keywords=[]
            )
        if isinstance(node.op, ast.Sub):
            return ast.Call(
                func=ast.Attribute(value=ast.Name(id="jnp", ctx=ast.Load()),
                                   attr="subtract", ctx=ast.Load()),
                args=[left, right],
                keywords=[]
            )
        if isinstance(node.op, ast.Mult):
            return ast.Call(
                func=ast.Attribute(value=ast.Name(id="jnp", ctx=ast.Load()),
                                   attr="multiply", ctx=ast.Load()),
                args=[left, right],
                keywords=[]
            )
        if isinstance(node.op, ast.Div):
            return ast.Call(
                func=ast.Attribute(value=ast.Name(id="jnp", ctx=ast.Load()),
                                   attr="divide", ctx=ast.Load()),
                args=[left, right],
                keywords=[]
            )

        return node

    # --- detect calls: sum, tanh, dot, einsum
    def visit_Call(self, node):
        self.generic_visit(node)

        # fn name
        fn = node.func

        # Replace built-in sum → jnp.sum
        if isinstance(fn, ast.Name) and fn.id == "sum":
            return ast.Call(
                func=ast.Attribute(value=ast.Name(id="jnp", ctx=ast.Load()),
                                   attr="sum", ctx=ast.Load()),
                args=node.args,
                keywords=[]
            )

        # Replace jnp.tanh
        if isinstance(fn, ast.Attribute) and fn.attr == "tanh":
            return node  # already jnp.tanh

        # Replace jnp.dot → lax.dot_general
        if isinstance(fn, ast.Attribute) and fn.attr == "dot":
            q, k = node.args   # assume dot(q, k)
            return self._make_dot_general_call(q, k)

        # Replace jnp.einsum("...d,...d->...", q, k)
        if isinstance(fn, ast.Attribute) and fn.attr == "einsum":
            # assume einsum("...d,...d->...", q, k)
            q = node.args[1]
            k = node.args[2]
            return self._make_dot_general_call(q, k)

        return node

    # --- helper: build dot_general AST call
    def _make_dot_general_call(self, q, k):
        return ast.Call(
            func=ast.Attribute(
                value=ast.Name(id="lax", ctx=ast.Load()),
                attr="dot_general",
                ctx=ast.Load(),
            ),
            args=[
                q,
                k,
                ast.Tuple(
                    elts=[
                        ast.Tuple(
                            elts=[
                                ast.Tuple(elts=[ast.Constant(value=-1)], ctx=ast.Load()),
                                ast.Tuple(elts=[ast.Constant(value=-1)], ctx=ast.Load())
                            ],
                            ctx=ast.Load()
                        ),
                        ast.Tuple(elts=[ast.Tuple(elts=[]), ast.Tuple(elts=[])], ctx=ast.Load())
                    ],
                    ctx=ast.Load()
                )
            ],
            keywords=[]
        )


# -------------------------------------------
# Final wrapper: user function → JAX function
# -------------------------------------------

def make_jax_score_fn(user_fn):
    """
    Convert user-provided Python score function into a JAX-friendly
    function with all dot-like operations rewritten into lax.dot_general.
    """

    # get source of Python function
    src = textwrap.dedent(inspect.getsource(user_fn))
    tree = ast.parse(src)

    # transform
    transformer = ToJaxTransformer()
    tree = transformer.visit(tree)
    ast.fix_missing_locations(tree)

    # compile back into Python code
    code = compile(tree, filename="<ast>", mode="exec")
    namespace = {"jnp": jnp, "lax": lax}
    exec(code, namespace)

    # return rewritten JAX function
    rewritten_fn = namespace[user_fn.__name__]

    # optionally JIT it
    return jax.jit(rewritten_fn)
