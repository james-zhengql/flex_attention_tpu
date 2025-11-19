import jax
from jax import random

import jax.numpy as jnp
import benchmark

from typing import NamedTuple
from constants import dimension_numbers
from util import make_jax_score_fn

class ScoreContext(NamedTuple):
    scale: jnp.ndarray
    alpha: jnp.ndarray

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


    # def user_score_fn_block(q, k):
    #     """
    #     q_block: (Q, C)
    #     k_block: (K, C)
    #     ctx: {"scale": float, "alpha": float}

    #     returns:
    #         scores: (Q, K)
    #     """

    #     s = jax.lax.dot_general(
    #         q, k, dimension_numbers, preferred_element_type=jnp.float32
    #     )  # [block_q, block_k]

    #     # custom kernel with pallas
    #     q_sum = jnp.sum(q, axis=-1)  # Shape: (Q_block,)
    #     k_sum = jnp.sum(k, axis=-1)  # Shape: (K_block,)
        
    #     # Broadcast subtraction to get (Q, K) shape
    #     diff_sum = q_sum[:, None] - k_sum[None, :]

    #     s += diff_sum
    #     return s

    def my_score(q, k):
        q_sum = jnp.sum(q, axis=-1)  # Shape: (Q_block,)
        k_sum = jnp.sum(k, axis=-1)  # Shape: (K_block,)
        
        # Broadcast subtraction to get (Q, K) shape
        diff_sum = q_sum[:, None] - k_sum[None, :]
        return jnp.dot(q, k) + 0.3 * jnp.tanh(diff_sum)


    jax_score = make_jax_score_fn(my_score)

    results = benchmark.run_bench_suite(
        q, k, v,
        sm_scale=1.0,
        block_b=1,
        block_q=128,
        block_k_major=128,
        block_k=128,
        causal=False,
        score_fn=jax_score,
        which=["flash","flash_ref"]
    )
    print("\nSummary:", results)


if __name__ == "__main__":
    main()