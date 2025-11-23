import functools
import jax
import jax.numpy as jnp
import numpy as np
import statistics as stats
import time

import flash_attention_fwd_ref
import flex_attention_kernel
import flash_attention_bwd
from mha_reference import mha_reference, mha_bwd_reference


# ============================================================
# FLOP COUNT
# ============================================================

def flop_count_attention(b, h, q, k, d):
    return 4.0 * b * h * q * k * d


# ============================================================
# GENERIC BENCH FUNCTION
# ============================================================

def benchmark(fn, args, iters=30, warmup=5, name="fn"):
    # ---- Warmup ----
    for _ in range(warmup):
        out = fn(*args)
        jax.tree_util.tree_map(
            lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
            out
        )

    # ---- Timed runs ----
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        out = fn(*args)
        jax.tree_util.tree_map(
            lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
            out
        )
        t1 = time.perf_counter()
        times.append(t1 - t0)

    mean_t = sum(times) / len(times)
    med_t = stats.median(times)
    p10, p90 = np.percentile(times, [10, 90])

    print(f"[{name}] mean={mean_t*1e3:.2f} ms  median={med_t*1e3:.2f} ms  "
          f"p10={p10*1e3:.2f} ms  p90={p90*1e3:.2f} ms")

    return mean_t, med_t


# ============================================================
# DIFF TOOL
# ============================================================

def compute_diff(ref_out, test_out):
    if not isinstance(ref_out, (tuple, list)):
        ref_out = (ref_out,)
    if not isinstance(test_out, (tuple, list)):
        test_out = (test_out,)

    diffs = {}
    for i, (r, t) in enumerate(zip(ref_out, test_out)):
        if r.shape != t.shape:
            raise ValueError(f"Shape mismatch: ref {r.shape} vs test {t.shape}")

        diffs[i] = float(
            jnp.linalg.norm(t - r) / (jnp.linalg.norm(r) + 1e-6)
        )

    return diffs


# ============================================================
# FN BUILDER
# ============================================================

def build_fns_for_bench(
    q, k, v,
    *,
    ab=None,
    sm_scale=1.0,
    save_residuals=True,
    causal=False,
    block_b=1,
    block_q=128,
    block_k_major=128,
    block_k=128,
    debug=False,
    score_fn=None,
    which=("ref", "flash", "flash_ref"),
):
    out = {}

    # ========================================================
    # REFERENCE MHA
    # ========================================================
    if "ref" in which:
        # Forward
        ref_fwd = functools.partial(
            mha_reference,
            ab=None,
            sm_scale=sm_scale,
            save_residuals=True,
            score_fn=score_fn,
        )
        out["ref_fwd_jit"] = jax.jit(ref_fwd, static_argnames=("score_fn",))

        # Backward (needs o, l, m from fwd)
        ref_bwd = functools.partial(
            mha_bwd_reference,
            ab=None,
            sm_scale=sm_scale,
            save_residuals=True,
            score_fn=score_fn,
        )
        out["ref_bwd_jit"] = jax.jit(ref_bwd, static_argnames=("score_fn",))

        # Combined: fwd + bwd
        def ref_fwd_bwd(q, k, v):
            o, l, m = ref_fwd(q, k, v)
            do = jnp.ones_like(o)
            dq, dk, dv = ref_bwd(q, k, v, o, do, l, m)
            return dq, dk, dv

        out["ref_fwd_bwd_jit"] = jax.jit(ref_fwd_bwd)

    # ========================================================
    # FLASH ATTENTION PALLAS KERNEL
    # ========================================================
    if "flash" in which:
        flash_fwd = functools.partial(
            flex_attention_kernel._flash_attention_impl,
            ab=ab,
            segment_ids=None,
            save_residuals=True,
            causal=causal,
            sm_scale=sm_scale,
            block_b=block_b,
            block_q=block_q,
            block_k_major=block_k_major,
            block_k=block_k,
            debug=debug,
            score_fn=score_fn,
        )
        out["flash_fwd_jit"] = jax.jit(flash_fwd, static_argnames=("score_fn",))

        # Backward — unavailable for flash (you didn't provide it)
        # So skip flash_bwd

        def flash_fwd_bwd(q, k, v):
            o, l, m = flash_fwd(q, k, v)
            return o, l, m  # no backward yet

        out["flash_fwd_bwd_jit"] = jax.jit(flash_fwd_bwd)

    # ========================================================
    # FLASH ATTENTION REFERENCE (FWD + BWD)
    # ========================================================
    if "flash_ref" in which:
        flash_ref_fwd = functools.partial(
            flash_attention_fwd_ref._flash_attention_impl_ref,
            ab=ab,
            segment_ids=None,
            save_residuals=True,
            causal=causal,
            sm_scale=sm_scale,
            block_b=block_b,
            block_q=block_q,
            block_k_major=block_k_major,
            block_k=block_k,
            debug=debug,
        )
        out["flash_ref_fwd_jit"] = jax.jit(flash_ref_fwd)

        flash_ref_bwd = functools.partial(
            flash_attention_bwd._flash_attention_bwd_dq,
            causal=causal,
            sm_scale=sm_scale,
            block_q_major=block_q,
            block_k_major=block_k_major,
            block_k=block_k,
            debug=debug,
            score_fn=score_fn,
        )
        out["flash_ref_bwd_jit"] = jax.jit(flash_ref_bwd)

        # Combined: fwd + bwd
        def flash_ref_fwd_bwd(q, k, v):
            o, l, m = flash_ref_fwd(q, k, v)
            do = jnp.ones_like(o)
            di = jnp.ones_like(l)
            dq, ds = flash_ref_bwd(q, k, v, ab= None, l=l, m=m, do=do, di=di)
            return dq, ds

        out["flash_ref_fwd_bwd_jit"] = jax.jit(flash_ref_fwd_bwd)

    return out


# ============================================================
# MAIN BENCH SUITE
# ============================================================

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
    which=("ref", "flash", "flash_ref"),
):
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

    # ======================================================
    # FORWARD BENCHMARKS
    # ======================================================
    for name in ("ref_fwd_jit", "flash_fwd_jit", "flash_ref_fwd_jit"):
        if name in compiled:
            fn = compiled[name]
            t_mean, t_med = benchmark(fn, (q, k, v), name=name)
            print(f"  → FWD Throughput: {gflops/t_med:.2f} GFLOP/s\n")
            results[name] = (t_mean, t_med)

    # ======================================================
    # BACKWARD BENCHMARKS
    # ======================================================
    for name in ("ref_bwd_jit", "flash_bwd_jit", "flash_ref_bwd_jit"):
        if name in compiled:
            fn = compiled[name]
            # backward needs extra args
            if name == "ref_bwd_jit":
                o, l, m = compiled["ref_fwd_jit"](q, k, v)
                do = jnp.ones_like(o)
                args = (q, k, v, o, do, l, m)
            elif name == "flash_ref_bwd_jit":
                o, l, m = compiled["flash_ref_fwd_jit"](q, k, v)
                do = jnp.ones_like(o)
                di = jnp.ones_like(l)
                args = (q, k, v, None, l, m, do, di)
            else:
                continue

            t_mean, t_med = benchmark(fn, args, name=name)
            print(f"  → BWD Throughput: {gflops/t_med:.2f} GFLOP/s\n")
            results[name] = (t_mean, t_med)

    # ======================================================
    # COMBINED BENCHMARKS
    # ======================================================
    for name in ("ref_fwd_bwd_jit", "flash_fwd_bwd_jit", "flash_ref_fwd_bwd_jit"):
        if name in compiled:
            fn = compiled[name]
            t_mean, t_med = benchmark(fn, (q, k, v), name=name)
            print(f"  → FWD+BWD Throughput: {gflops/t_med:.2f} GFLOP/s\n")
            results[name] = (t_mean, t_med)

    return results

