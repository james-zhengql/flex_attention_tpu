import jax
from jax import random

import jax.numpy as jnp
import benchmark

from typing import NamedTuple
from constants import dimension_numbers, MIN_BLOCK_SIZE
from util import make_jax_score_fn
import flash_attention_bwd

def generate_l_m(q, k, sm_scale=1.0):
    batch, heads, Q, D = q.shape
    _, _, K, _ = k.shape

    S = jnp.einsum("bhqd, bhkd -> bhqk", q, k) * sm_scale
    m_raw = jnp.max(S, axis=-1)
    l_raw = jnp.sum(jnp.exp(S - m_raw[..., None]), axis=-1)

    return l_raw, m_raw


def main():
    key = random.PRNGKey(0)
    batch = 1
    heads = 1
    q_len = 25600
    kv_len = 25600
    head_dim = 128

    k1, k2, k3 = random.split(key, 3)
    q = random.normal(k1, (batch, heads, q_len, head_dim), dtype=jnp.float32)
    k = random.normal(k2, (batch, heads, kv_len, head_dim), dtype=jnp.float32)
    v = random.normal(k3, (batch, heads, kv_len, head_dim), dtype=jnp.float32)

    def my_score(q, k):
        return jnp.einsum("qd, kd -> qk", q, k) + 0.3 * jnp.tanh(jnp.einsum("qd, kd -> qk", q, k))

    jax_score = make_jax_score_fn(my_score)

    results = benchmark.run_bench_suite(
        q, k, v,
        sm_scale=1.0,
        block_b=1,
        block_q=1024,
        block_k_major=1024,
        block_k=512,
        causal=True,   
        score_fn=jax_score,
        which=["ref", "flash_ref"]
    )
    print("\nSummary:", results)


if __name__ == "__main__":
    main()
