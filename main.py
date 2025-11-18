
from jax import random

import jax.numpy as jnp
import benchmark

def main():
    key = random.PRNGKey(0)
    batch = 1
    heads = 1
    q_len = 256
    kv_len = 25600
    head_dim = 128

    k1, k2, k3 = random.split(key, 3)
    q = random.normal(k1, (batch, heads, q_len, head_dim), dtype=jnp.float32)
    k = random.normal(k2, (batch, heads, kv_len, head_dim), dtype=jnp.float32)
    v = random.normal(k3, (batch, heads, kv_len, head_dim), dtype=jnp.float32)
    ab = None
    segment_ids = None

    block_b = 1
    block_q = 128
    block_k_major = 128
    block_k = 128

    causal = False
    sm_scale = float(1.0 / jnp.sqrt(head_dim).astype(jnp.float32))
    debug = False
    save_residuals = True

    # ------------------------------------------------------
    # ⭐️ USER-DEFINED SCORE FUNCTION + CONTEXT
    # ------------------------------------------------------
    # Example: dot-product with extra sinusoidal modulation
    def user_score_fn_block(q_block, k_block, ctx):
        """
        q_block: (Q, C)
        k_block: (K, C)
        return: scores (Q, K)
        """
        # Example fused nonlinear score
        diff_sum = (q_block[:,None,:] - k_block[None,:,:]).sum(-1)   # (Q,K)
        scores = q_block @ k_block.T                                 # (Q,K)
        return scores * ctx["scale"] + ctx["alpha"] * diff_sum


    # User context (can be ANY PyTree)
    score_ctx = {
        "scale": sm_scale,
        "alpha": 0.3,
    }

    results = benchmark.run_bench_suite(
        q, k, v,
        sm_scale=1.0,
        block_b=1,
        block_q=128,
        block_k_major=128,
        block_k=128,
        causal=False,
        score_fn=None,
        score_ctx=None
    )
    print("\nSummary:", results)


if __name__ == "__main__":
    main()