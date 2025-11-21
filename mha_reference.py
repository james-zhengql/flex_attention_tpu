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
        
        # Inner vmap: Maps over Heads (axis 1)
        # Outer vmap: Maps over Batch (axis 0)
        batched_score = jax.vmap(
            jax.vmap(score_fn, in_axes=(0, 0)), 
            in_axes=(0, 0)
        )

        
        logits = batched_score(q, k)

        # Add bias if provided (standard broadcasting handles the rest)
        if ab is not None:
            logits += ab
            
        # Note: If your custom score function DOES NOT handle scale, apply it here.
        if sm_scale != 1.0:
            # Heuristic: Check if user function likely applied it? 
            # For safety in reference, we often assume user handles it in custom fn,
            # but if you want to force it:
            # logits = logits * sm_scale
            pass

    # -------------------------
    # Softmax & Output
    # -------------------------
    m = jnp.max(logits, axis=-1, keepdims=True)
    unn = jnp.exp(logits - m)
    l = jnp.sum(unn, axis=-1, keepdims=True)
    weights = unn / l

    out = jnp.einsum("bhqk,bhkc->bhqc", weights, v)

    return (out, l, m) if save_residuals else out