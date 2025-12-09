import functools
import jax
import jax.numpy as jnp
import numpy as np
import statistics as stats
import time

# Assumes these imports exist in your environment
import flash_attention_fwd_ref
import flex_attention_kernel
import flash_attention_bwd
from mha_reference import mha_reference, mha_bwd_reference
from jax_exp import mha_reference_no_custom_vjp,_flash_attention_impl


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

    if not times:
        return 0.0, 0.0

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
            # Squeeze simplified dims if necessary or just warn
            if r.size == t.size:
                t = t.reshape(r.shape)
            else:
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
    mask_fn=None,           
    block_mask_fn=None,     
    window_size: int | None = None,     # Sliding Window
    segment_ids: jax.Array | None = None, # Document Masking
    s2_stride: int | None = None,         # S2 Attention
    alibi_slope: float | None = None,
    which=("ref", "flash", "flash_ref"),
):
    out = {}

    # ========================================================
    # REFERENCE MHA
    # ========================================================
    if "ref" in which:
        # We pass ALL the configuration parameters to the reference
        # so it matches the logic inside mask_fn exactly.
        ref_fwd = functools.partial(
            mha_reference,
            ab=None,
            sm_scale=sm_scale,
            save_residuals=True,
            score_fn=score_fn,
            causal=causal,
            window_size=window_size,
            segment_ids=segment_ids,  # <--- Passed to Ref
            s2_stride=s2_stride,       # <--- Passed to Ref
            alibi_slope = alibi_slope
        )
        # segment_ids is an array, not static, so we don't put it in static_argnames
        out["ref_fwd_jit"] = jax.jit(ref_fwd, static_argnames=("score_fn",))

    # ========================================================
    # FLASH ATTENTION PALLAS KERNEL
    # ========================================================
    if "flash" in which:
        print(f"Compiling Flash Kernel with: Mask={mask_fn is not None}, BlockMask={block_mask_fn is not None}")
        
        # The Flash Kernel relies on the closures (mask_fn/block_mask_fn) 
        # to handle the logic for window/docs/s2. We don't pass s2_stride directly 
        # unless the kernel expects it, but we assume it's baked into mask_fn.
        flash_fwd = functools.partial(
            flex_attention_kernel._flex_attention_impl,
            ab=ab,
            segment_ids=None, # Usually None for Pallas as it's handled in mask_fn
            save_residuals=True,
            causal=causal,
            sm_scale=sm_scale,
            block_b=block_b,
            block_q=block_q,
            block_k_major=block_k_major,
            block_k=block_k,
            debug=debug,
            score_fn=score_fn,
            mask_fn=mask_fn,             
            block_mask_fn=block_mask_fn
        )
        
        out["flash_fwd_jit"] = jax.jit(
            flash_fwd, 
            static_argnames=("score_fn", "mask_fn", "block_mask_fn")
        )

    # ========================================================
    # FLASH ATTENTION REFERENCE (Loop-based)
    # ========================================================
    if "flash_ref" in which:
        flash_ref_fwd = functools.partial(
            _flash_attention_impl,
            ab=ab,
            segment_ids=segment_ids,
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
    mask_fn=None,         
    block_mask_fn=None,   
    window_size: int | None = None,
    segment_ids: jax.Array | None = None,
    s2_stride: int | None = None,
    alibi_slope: float | None = None,
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
        mask_fn=mask_fn,             
        block_mask_fn=block_mask_fn, 
        window_size=window_size,     # <--- Passed
        segment_ids=segment_ids,     # <--- Passed
        s2_stride=s2_stride,         # <--- Passed
        alibi_slope=alibi_slope,
        which=which,
    )

    print(f"\n== Benchmark config: b={b}, h={h}, q={q_len}, k={k_len}, d={d}, causal={causal} ==")
    print(f"Estimated FLOPs per call: {gflops:.2f} GFLOPs")
    if block_mask_fn:
        print("Note: FLOP count does not account for skipped blocks (sparsity).")

    results = {}

    # ======================================================
    # FORWARD BENCHMARKS
    # ======================================================
    for name in ("ref_fwd_jit", "flash_fwd_jit", "flash_ref_fwd_jit"):
        if name in compiled:
            fn = compiled[name]
            t_mean, t_med = benchmark(fn, (q, k, v), iters=10, name=name)
            if t_med > 0:
                print(f"  → FWD Throughput: {gflops/t_med:.2f} GFLOP/s\n")
            results[name] = (t_mean, t_med)
    
    print("--- Numeric Accuracy (vs ref_fwd_jit) ---")
    
    ref_target = "ref_fwd_jit"
    ref_fwd_out = None

    if ref_target in compiled:
        try:
            ref_fwd_out = compiled[ref_target](q, k, v)
            jax.tree_util.tree_map(lambda x: x.block_until_ready(), ref_fwd_out)
        except Exception as e:
            print(f"Reference run failed: {e}")

    if ref_fwd_out:
        for name, fn in compiled.items():
            if name == ref_target: continue 
            # # Skip flash_ref if it doesn't support the new masks, to avoid noise
            # if name == "flash_ref_fwd_jit" and (window_size or segment_ids is not None):
            #      print(f"[{name}] Skipped accuracy check (implementation may not support new masks)")
            #      continue

            try:
                test_out = fn(q, k, v)
                jax.tree_util.tree_map(lambda x: x.block_until_ready(), test_out)

                diff_res = compute_diff(ref_fwd_out, test_out)
                print(f"[{name}] vs [{ref_target}]: {diff_res}")
            except Exception as e:
                print(f"[{name}] Validation failed: {e}")
    else:
        print("Skipping accuracy check (no reference output).")

    return results