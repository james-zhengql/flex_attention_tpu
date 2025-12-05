import jax.numpy as jnp

def make_alibi_score_fn(slope=0.5):
    """Subtracts linear penalty based on distance."""
    def alibi_score(q_tile, k_tile):
        score = jnp.dot(q_tile, k_tile.T)
        # Simple tile-local distance for demo
        m, n = q_tile.shape[0], k_tile.shape[0]
        dist = jnp.abs(jnp.arange(m)[:, None] - jnp.arange(n)[None, :])
        return score - (slope * dist)
    return alibi_score

def make_softcap_score_fn(cap=30.0):
    """Soft-caps logits using Tanh (Gemma 2 style)."""
    def softcap_score(q_tile, k_tile):
        logits = jnp.dot(q_tile, k_tile.T)
        return cap * jnp.tanh(logits / cap)
    return softcap_score