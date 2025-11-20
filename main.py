import jax
from jax import random

import jax.numpy as jnp
import benchmark

from typing import NamedTuple
from constants import dimension_numbers, MIN_BLOCK_SIZE
from util import make_jax_score_fn
import flash_attention_bwd

def generate_l_m(q, k, sm_scale=1.0):
    """
    q: (batch, heads, Q, D)
    k: (batch, heads, K, D)
    Produces:
       m: (batch, heads, Q, MIN_BLOCK_SIZE)
       l: (batch, heads, Q, MIN_BLOCK_SIZE)
    Matching the forward kernel broadcast convention.
    """
    batch, heads, Q, D = q.shape
    _, _, K, _ = k.shape

    # Compute logits
    S = jnp.einsum("bhqd, bhkd -> bhqk", q, k) * sm_scale  # (b,h,Q,K)

    # m: max over key dimension
    m_raw = jnp.max(S, axis=-1)  # (b,h,Q)

    # l: sum of exp(logits - m)
    l_raw = jnp.sum(jnp.exp(S - m_raw[..., None]), axis=-1)  # (b,h,Q)

    # Expand to MIN_BLOCK_SIZE in the last dim (like forward kernel)
    m = jnp.broadcast_to(m_raw[..., None], (batch, heads, Q, MIN_BLOCK_SIZE))
    l = jnp.broadcast_to(l_raw[..., None], (batch, heads, Q, MIN_BLOCK_SIZE))

    return l, m


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

    # results = benchmark.run_bench_suite(
    #     q, k, v,
    #     sm_scale=1.0,
    #     block_b=1,
    #     block_q=128,
    #     block_k_major=128,
    #     block_k=128,
    #     causal=False,
    #     score_fn=jax_score,
    #     which=["flash","flash_ref"]
    # )
    # print("\nSummary:", results)


    # Generate synthetic l, m
    l_test, m_test = generate_l_m(q, k)

    # Fake upstream gradient
    dout = jnp.ones((batch, heads, q_len, head_dim), dtype=jnp.float32)

    # Zero di
    di = jnp.zeros((batch, heads, q_len, MIN_BLOCK_SIZE), dtype=jnp.float32)

    # Run backward kernel
    dq, ds = flash_attention_bwd._flash_attention_bwd_dq(
        q, k, v,
        ab=None,
        segment_ids=None,
        l=l_test,
        m=m_test,
        do=dout,
        di=di,
        causal=False,
        sm_scale=1.0,
        block_q_major=128,
        block_k_major=128,
        block_k=128,
        mask_value=-1e9,
        debug=False,
    )

    print("dq:", dq.shape)
    print("ds:", ds.shape)


if __name__ == "__main__":
    main()