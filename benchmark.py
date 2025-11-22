import functools
import jax

import jax.numpy as jnp
import numpy as np
import statistics as stats

import flash_attention_fwd_ref
import flex_attention_kernel
from mha_reference import mha_reference

import time



def flop_count_attention(b, h, q, k, d):
    """
    Rough FLOP count for one forward pass of scaled dot-product attention:
      QK^T  : 2 * b * h * q * k * d        (matrix multiplication)
      Softmax: ~ b * h * q * k             (small, we ignore it)
      (softmax @ V): 2 * b * h * q * k * d (another matmul)
    Total ≈ 4 * b * h * q * k * d FLOPs
    """
    return 4.0 * b * h * q * k * d

def benchmark(fn, args, iters=30, warmup=5, name="fn"):
    # 1. Warmup phase — triggers JIT compilation and stabilizes cache
    for _ in range(warmup):
        y = fn(*args)
        if isinstance(y, (tuple, list)):
            jax.tree_util.tree_map(
                lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, y
            )
        else:
            y.block_until_ready()

    # 2. Timed runs
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        y = fn(*args)
        if isinstance(y, (tuple, list)):
            jax.tree_util.tree_map(
                lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, y
            )
        else:
            y.block_until_ready()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    # 3. Compute summary statistics
    mean_t = sum(times) / len(times)
    med_t = stats.median(times)
    p10, p90 = np.percentile(np.array(times), [10, 90])

    print(f"[{name}] mean={mean_t*1e3:.2f} ms  median={med_t*1e3:.2f} ms  "
          f"p10={p10*1e3:.2f} ms  p90={p90*1e3:.2f} ms")

    return mean_t, med_t

def build_fns_for_bench(
    q, k, v,
    *,
    ab=None,
    sm_scale=1.0,
    save_residuals=False,
    causal=False,
    block_b=1,
    block_q=128,
    block_k_major=128,
    block_k=128,
    debug=False,
    score_fn=None,
    which=("ref", "flash", "flash_ref"),
):
    """
    Build only the functions requested in `which`.
    which: tuple/list containing any of:
        "ref"        – jax naive reference
        "flash"      – your Pallas FlashAttention kernel
        "flash_ref"  – the reference FA kernel
    """
    out = {}

    if "ref" in which:
        ref_partial = functools.partial(
            mha_reference,
            ab=None,
            sm_scale=sm_scale,
            save_residuals=False,
            score_fn=score_fn,
            causal=causal,   
        )
        out["ref_jit"] = jax.jit(ref_partial, static_argnames=("score_fn",))

    if "flash" in which:
        flash_partial = functools.partial(
            flex_attention_kernel._flash_attention_impl,
            ab=ab,
            segment_ids=None,
            save_residuals=save_residuals,
            causal=causal,
            sm_scale=sm_scale,
            block_b=block_b,
            block_q=block_q,
            block_k_major=block_k_major,
            block_k=block_k,
            debug=debug,
            score_fn=score_fn,
        )
        out["flash_jit"] = jax.jit(flash_partial, static_argnames=("score_fn",))

    if "flash_ref" in which:
        flash_ref_partial = functools.partial(
            flash_attention_fwd_ref._flash_attention_impl_ref,
            ab=ab,
            segment_ids=None,
            save_residuals=save_residuals,
            causal=causal,
            sm_scale=sm_scale,
            block_b=block_b,
            block_q=block_q,
            block_k_major=block_k_major,
            block_k=block_k,
            debug=debug,
        )
        out["flash_ref_jit"] = jax.jit(flash_ref_partial)

    return out

def compute_diff(ref_out, test_out):
    """Compute numeric diff between ref outputs and test outputs.
    
    Works for:
      - single tensors
      - tuples/lists of tensors
    Returns:
      dict mapping field index → relative L2 error
    """
    diffs = {}

    if not isinstance(ref_out, (tuple, list)):
        ref_out = (ref_out,)
    if not isinstance(test_out, (tuple, list)):
        test_out = (test_out,)

    if len(ref_out) != len(test_out):
        raise ValueError(
            f"Output count mismatch: ref has {len(ref_out)}, test has {len(test_out)}"
        )

    for i, (r, t) in enumerate(zip(ref_out, test_out)):
        if r is None and t is None:
            diffs[i] = None
            continue
        if r is None or t is None:
            raise ValueError(f"Mismatch: ref[{i}] is {r}, test[{i}] is {t}")

        if r.shape != t.shape:
            raise ValueError(
                f"Shape mismatch at output {i}: ref {r.shape} vs test {t.shape}"
            )

        diff = jnp.linalg.norm(t - r) / (jnp.linalg.norm(r) + 1e-6)
        diffs[i] = float(diff)

    return diffs


def run_bench_suite(
    q, k, v,
    *,
    sm_scale,
    block_b,
    block_q,
    block_k_major,
    block_k,
    causal=False,
    score_fn=None,
    which=("ref", "flash", "flash_ref")
):
    """
    Run benchmarks only for selected implementations.
    """
    b, h, q_len, d = q.shape
    _, _, k_len, _ = k.shape

    gflops = flop_count_attention(b, h, q_len, k_len, d) / 1e9

    compiled = build_fns_for_bench(
        q, k, v,
        sm_scale=sm_scale,
        causal=causal,
        block_b=block_b,
        block_q=block_q,
        block_k_major=block_k_major,
        block_k=block_k,
        score_fn=score_fn,
        which=which,
    )

    print(f"\n== Benchmark config: b={b}, h={h}, q={q_len}, k={k_len}, d={d}, causal={causal} ==")
    print(f"Estimated FLOPs per call: {gflops:.2f} GFLOPs\n")

    results = {}

    if "ref_jit" in compiled:
        fn = compiled["ref_jit"]
        t_mean, t_med = benchmark(fn, (q, k, v), name="mha_reference[jit]")
        print(f"  → Throughput: {gflops/t_med:.2f} GFLOP/s")
        results["ref"] = (t_mean, t_med)

    if "flash_jit" in compiled:
        fn = compiled["flash_jit"]
        t_mean, t_med = benchmark(fn, (q, k, v), name="pallas_flash[jit]")
        print(f"  → Throughput: {gflops/t_med:.2f} GFLOP/s")
        results["flash"] = (t_mean, t_med)

    if "flash_ref_jit" in compiled:
        fn = compiled["flash_ref_jit"]
        t_mean, t_med = benchmark(fn, (q, k, v), name="pallas_flash_ref[jit]")
        print(f"  → Throughput: {gflops/t_med:.2f} GFLOP/s")
        results["flash_ref"] = (t_mean, t_med)

    if "ref_jit" in compiled and "flash_ref_jit" in compiled:
        ref_out = compiled["ref_jit"](q, k, v)
        flash_out = compiled["flash_ref_jit"](q, k, v)

        ref_list = ref_out if isinstance(ref_out, (tuple, list)) else (ref_out,)
        flash_list = flash_out if isinstance(flash_out, (tuple, list)) else (flash_out,)

        diffs = compute_diff(ref_list, flash_list)
        print("\nNumeric diff:")
        for i, e in diffs.items():
            print(f"  output[{i}] rel L2 error = {e}")

    return results
