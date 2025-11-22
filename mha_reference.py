import jax
import jax.numpy as jnp


def mha_reference(
    q,
    k,
    v,
    ab: jax.Array | None = None,
    *,
    sm_scale: float = 1.0,
    save_residuals: bool = False,
    score_fn=None,
):
    """
    Reference multi-head attention for correctness checking.
    Supports optional custom score_fn(q_vec).
    """

    batch, heads, q_len, dim = q.shape
    _, _, kv_len, _ = k.shape

    if score_fn is None:
        logits = jnp.einsum("bhqc,bhkc->bhqk", q, k) * sm_scale
        if ab is not None:
            logits += ab
    else:
        # score_fn wants (Q,C) and (K,C), not full BHQC tensors
        def apply_block(q_bh, k_bh):
            return score_fn(q_bh, k_bh)


        # vmap over batch and heads:
        logits = jax.vmap(
            jax.vmap(
                lambda q_i, k_i: score_fn(q_i, k_i),
                in_axes=(0,0),   # q_i: (Q,C), k_i: (K,C)
            ),
            in_axes=(0,0)
        )(q, k)


        if ab is not None:
            logits += ab




    # -------------------------
    # Causal mask (optional)
    # -------------------------
    mask = None
    logits = logits if mask is None else logits + jnp.where(mask, 0.0, -1e9)

    # -------------------------
    # Numerically stable softmax
    # -------------------------
    m = logits.max(axis=-1)
    unn = jnp.exp(logits - m[..., None])
    l = unn.sum(axis=-1)
    weights = unn / l[..., None]

    # -------------------------
    # Weighted sum
    # -------------------------
    print("logits shape:", logits.shape)
    print("weights shape:", weights.shape)
    print("v shape:", v.shape)

    out = jnp.einsum("bhqk,bhkc->bhqc", weights, v)

    return (out, l, m) if save_residuals else out


def mha_bwd_reference(
    q,
    k,
    v,
    o,
    do,
    l,
    m,
    ab: jax.Array | None = None,
    *,
    causal: bool = False,
    mask_value: float = None,
    sm_scale: float = 1.0,
    save_residuals: bool = True,
    score_fn = None
):

    if score_fn is None:
        logits = jnp.einsum("bhqc,bhkc->bhqk", q, k)
    else:
        logits = score_fn(q, k)

    if ab is not None:
        logits += ab
    if sm_scale != 1.0:
        logits *= sm_scale

    # # no causal masking
    # mask = None
    # logits = logits if mask is None else logits + jnp.where(mask, 0.0, -1e9)

    # m = logits.max(axis=-1)
    unnormalized = jnp.exp(logits - m[..., None])
    # l = unnormalized.sum(axis=-1)
    # weights = unnormalized / l[..., None]
    # o = jnp.einsum("bhkq,bhkc->bhqc", weights, v)

    p = unnormalized / l[..., None]

    # dv = P^T * do
    dv = jnp.einsum("bhqk,bhqc->bhkc", p, do)

    # dp = do * V^T
    dp = jnp.einsum("bhqc,bhkc->bhqk", do, v)

    # software backward
    sum_d = jnp.sum(do*o, axis=-1)
    ds = p * (dp - sum_d[..., None])

    if score_fn is None:
        dq = jnp.einsum("bhqk,bhkc->bhqc", ds, k)
        dk = jnp.einsum("bhqk,bhqc->bhkc", ds, q)
    else:
        _, vjp_fn = jax.vjp(score_fn, q, k)
        dq, dk = vjp_fn(ds)


    return dq, dk, dv
