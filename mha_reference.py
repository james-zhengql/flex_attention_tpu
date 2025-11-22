import jax
import jax.numpy as jnp
from util import make_jax_score_fn

# -------------------------------------------------------
# 2. The Reference Implementation
# -------------------------------------------------------
def mha_reference(
    q,
    k,
    v,
    ab: jax.Array | None = None,
    *,
    sm_scale: float = 1.0,
    save_residuals: bool = False,
    score_fn=None,
    causal: bool = False,   
):

    batch, heads, q_len, dim = q.shape

    if score_fn is None:
        # Default Dot-Product
        logits = jnp.einsum("bhqc,bhkc->bhqk", q, k) * sm_scale
        if ab is not None:
            logits += ab
    else:
        # 2. Apply the Custom Score
        #    The user function expects (Q_seq, Dim), (K_seq, Dim)
        #    We must broadcast (vmap) it over Batch and Heads.
        
        batched_score = jax.vmap(
            jax.vmap(score_fn, in_axes=(0, 0)), 
            in_axes=(0, 0)
        )

        logits = batched_score(q, k)

        # Add bias if provided
        if ab is not None:
            logits += ab

        # If user score doesn't include sm_scale, you could apply it here.
        if sm_scale != 1.0:
            # keep behavior same as your old version: assume user handles scale
            pass

    if causal:
        k_len = logits.shape[-1]
        q_idx = jnp.arange(q_len)[:, None]      # (Q,1)
        k_idx = jnp.arange(k_len)[None, :]     # (1,K)
        causal_mask = k_idx > q_idx            # (Q,K)
        logits = jnp.where(causal_mask, -1e9, logits)

    # -------------------------
    # Softmax & Output
    # -------------------------
    m = jnp.max(logits, axis=-1, keepdims=True)
    unn = jnp.exp(logits - m)
    l = jnp.sum(unn, axis=-1, keepdims=True)
    weights = unn / l

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
