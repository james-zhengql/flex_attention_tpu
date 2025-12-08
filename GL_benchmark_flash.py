import time
import jax
import jax.numpy as jnp
from flash_attn_kernel import flash_attention


def bench_flash(b, h, q_len, k_len, d, causal=False, warmup=10, iters=30):

    key = jax.random.PRNGKey(0)
    q = jax.random.normal(key, (b, h, q_len, d))
    k = jax.random.normal(key+1, (b, h, k_len, d))
    v = jax.random.normal(key+2, (b, h, k_len, d))

    fn = lambda q, k, v: flash_attention(q, k, v, causal)

    for _ in range(warmup):
        fn(q, k, v).block_until_ready()

    times = []
    for _ in range(iters):
        start = time.time()
        y = fn(q, k, v).block_until_ready()
        end = time.time()
        times.append((end - start) * 1000)

    print(f"Batch={b}, Heads={h}, Seq={q_len}, d={d}, causal={causal}")
    print(f"Mean: {sum(times)/len(times):.3f} ms")
    print(f"Min:  {min(times):.3f} ms")
    print(f"Max:  {max(times):.3f} ms")


if __name__ == "__main__":
    print("Devices:", jax.devices())

    bench_flash(
        b=1, h=4,
        q_len=2048, k_len=2048,
        d=128,
        causal=True,
    )
