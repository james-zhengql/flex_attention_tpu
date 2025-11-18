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
    score_ctx=None,
):
    """
    Reference multi-head attention for correctness checking.
    Supports optional custom score_fn(q_vec, k_vec, score_ctx).
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
            return score_fn(q_bh, k_bh, score_ctx)   # (Q,K)

        # vmap over batch and heads:
        logits = jax.vmap(          # batch
                    jax.vmap(       # head
                        apply_block,
                        in_axes=(0, 0), out_axes=0
                    ),
                    in_axes=(0, 0), out_axes=0
                )(q, k)              # result: (B,H,Q,K)

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
