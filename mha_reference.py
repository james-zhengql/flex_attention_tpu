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
