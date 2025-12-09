import functools
import jax
import jax.numpy as jnp
import numpy as np
import statistics as stats
import time

# Assumes these imports exist in your environment
import flex_attention_dq_bwd  # Assumed module containing flash_attention_bwd_dq
from mha_reference import mha_reference, mha_bwd_reference
# Helper for masks, if needed
from jax_exp import _flash_attention_bwd_dq

# ============================================================
# FLOP COUNT
# ============================================================

def flop_count_attention_bwd(b, h, q, k, d):
    # Backward pass is roughly 2.5x the FLOPs of forward pass
    # (Re-computing attention + computing dQ + computing dK + computing dV)
    # Forward is 4 * b * h * q * k * d
    return 10.0 * b * h * q * k * d


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
    # Normalize to tuples
    if not isinstance(ref_out, (tuple, list)):
        ref_out = (ref_out,)
    if not isinstance(test_out, (tuple, list)):
        test_out = (test_out,)

    # If test_out has fewer elements (e.g., only dQ), compare only those
    n_check = min(len(ref_out), len(test_out))
    
    names = ["dQ", "dK", "dV"] # Assumed order
    diffs = {}
    
    for i in range(n_check):
        r = ref_out[i]
        t = test_out[i]
        
        if r.shape != t.shape:
             # Squeeze simplified dims if necessary or just warn
            if r.size == t.size:
                t = t.reshape(r.shape)
            else:
                raise ValueError(f"Shape mismatch at index {i}: ref {r.shape} vs test {t.shape}")

        # Relative L2 Error
        diff = float(jnp.linalg.norm(t - r) / (jnp.linalg.norm(r) + 1e-6))
        
        # Key name (e.g., "dQ" or "0")
        key = names[i] if i < len(names) else str(i)
        diffs[key] = diff

    return diffs


# ============================================================
# FN BUILDER
# ============================================================

def build_fns_for_bench_bwd(
    q, k, v, l, m, o, do, d,
    *,
    ab=None,
    sm_scale=1.0,
    causal=False,
    block_b=1,
    block_q=128,
    block_k_major=128,
    block_k=128,
    debug=False,
    score_fn=None,
    mask_fn=None,
    block_mask_fn=None,
    segment_ids=None,
    which=("ref", "flash")
):
    out = {}

    # ========================================================
    # REFERENCE MHA BACKWARD
    # ========================================================
    if "ref" in which:
        # Reference signature: mha_bwd_reference(q, k, v, o, do, l, m, ...)
        ref_bwd = functools.partial(
            mha_bwd_reference,
            ab=None,
            sm_scale=sm_scale,
            save_residuals=False, # Usually False for BWD pass itself
            causal=causal
        )
        # Note: Argument order for mha_bwd_reference in your imports seems to be:
        # q, k, v, o, do, l, m
        # We wrap it to match the standard bench signature (q,k,v,l,m,o,do,d)
        def ref_wrapper(q, k, v, l, m, o, do, d):
            return ref_bwd(q, k, v, o, do, l, m)
            
        out["ref_bwd_jit"] = jax.jit(ref_wrapper)

    # ========================================================
    # FLASH ATTENTION BACKWARD (Pallas)
    # ========================================================
    if "flash" in which:
        # Assuming usage of flash_attention_bwd_dq as per your snippet
        # If there is a separate dK/dV kernel, it should be aggregated here.
        
        flash_partial = functools.partial(
            flex_attention_dq_bwd.flash_attention_bwd_dq,
            ab=ab,
            sm_scale=sm_scale,
            block_b=block_b,
            block_q=block_q,
            block_k_major=block_k_major,
            block_k=block_k,
            segment_ids=segment_ids,
            debug=debug,
            score_fn=score_fn,
            mask_fn=mask_fn,
            block_mask_fn=block_mask_fn
        )
        
        # Wrapper to enforce consistent arg signature
        def flash_wrapper(q, k, v, l, m, o, do, d):
            return flash_partial(q=q, k=k, v=v, l=l, m=m, o=o, do=do, d=d)

        out["flash_bwd_jit"] = jax.jit(flash_wrapper)

    return out


# ============================================================
# MAIN BENCH SUITE
# ============================================================

def run_bench_suite_bwd(
    q, k, v, l, m, o, do,
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
    segment_ids=None,
    which=("ref", "flash"),
):
    b, h, q_len, dim = q.shape
    _, _, k_len, _ = k.shape

    # Pre-compute 'd' (delta) for kernels that require it
    # d = sum(do * o)
    d = jnp.sum(o.astype(jnp.float32) * do.astype(jnp.float32), axis=-1)

    # 10.0 factor for backward pass FLOPs
    gflops = flop_count_attention_bwd(b, h, q_len, k_len, dim) / 1e9
    
    compiled = build_fns_for_bench_bwd(
        q, k, v, l, m, o, do, d,
        sm_scale=sm_scale,
        causal=causal,
        block_b=block_b,
        block_q=block_q,
        block_k_major=block_k_major,
        block_k=block_k,
        score_fn=score_fn,
        mask_fn=mask_fn,
        block_mask_fn=block_mask_fn,
        segment_ids=segment_ids,
        which=which
    )

    print(f"\n== BWD Benchmark config: b={b}, h={h}, q={q_len}, k={k_len}, d={dim}, causal={causal} ==")
    print(f"Estimated FLOPs per call: {gflops:.2f} GFLOPs (Factor 10x)")
    
    results = {}

    # ======================================================
    # BACKWARD BENCHMARKS
    # ======================================================
    for name in ("ref_bwd_jit", "flash_bwd_jit"):
        if name in compiled:
            fn = compiled[name]
            t_mean, t_med = benchmark(fn, (q, k, v, l, m, o, do, d), iters=10, name=name)
            if t_med > 0:
                print(f"  → BWD Throughput: {gflops/t_med:.2f} GFLOP/s\n")
            results[name] = (t_mean, t_med)
    
    # ======================================================
    # ACCURACY CHECK
    # ======================================================
    print("--- Numeric Accuracy (vs ref_bwd_jit) ---")
    
    ref_target = "ref_bwd_jit"
    ref_out = None

    if ref_target in compiled:
        try:
            # Run reference
            ref_out = compiled[ref_target](q, k, v, l, m, o, do, d)
            jax.tree_util.tree_map(lambda x: x.block_until_ready(), ref_out)
        except Exception as e:
            print(f"Reference run failed: {e}")

    if ref_out:
        for name, fn in compiled.items():
            if name == ref_target: continue 

            try:
                test_out = fn(q, k, v, l, m, o, do, d)
                jax.tree_util.tree_map(lambda x: x.block_until_ready(), test_out)

                diff_res = compute_diff(ref_out, test_out)
                print(f"[{name}] vs [{ref_target}]: {diff_res}")
            except Exception as e:
                print(f"[{name}] Validation failed: {e}")
    else:
        print("Skipping accuracy check (no reference output).")

    return results


# ============================================================
# ENTRY POINT
# ============================================================

def main():
    # Setup Random Keys
    key = jax.random.PRNGKey(0)
    k1, k2, k3, k4 = jax.random.split(key, 4)

    # Config
    batch = 1
    heads = 1
    q_len = 8192
    kv_len = 8192
    head_dim = 128
    
    block_b = 1
    block_q = 128
    block_k_major = 128
    block_k = 128
    causal = True
    sm_scale = float(1.0/jnp.sqrt(head_dim))
    
    print(f"Initializing inputs... (Shape: {q_len}x{head_dim})")
    q = jax.random.normal(k1, (batch, heads, q_len, head_dim), dtype=jnp.float32)
    k = jax.random.normal(k2, (batch, heads, kv_len, head_dim), dtype=jnp.float32)
    v = jax.random.normal(k3, (batch, heads, kv_len, head_dim), dtype=jnp.float32)
    
    # We need valid l, m, o to run backward benchmarks
    print("Running forward pass to generate activations (l, m, o)...")
    
    # Use reference MHA for reliable l/m/o generation
    ref_fwd_out = mha_reference(
        q, k, v, 
        sm_scale=sm_scale, 
        save_residuals=True, # Critical for l, m
        causal=causal
    )
    o, l, m = ref_fwd_out
    
    # Generate random gradient for output (do)
    do = jax.random.normal(k4, o.shape, dtype=jnp.float32)

    # Optional: Setup custom score/masks if your environment supports them
    try:
        from jax_exp import masks, make_jax_score_fn
        mask_fn, block_mask_fn = masks.make_causal_mask_fns(block_q=block_q, block_k_major=block_k_major)
        def my_score(q, k): return jnp.einsum("qd, kd -> qk", q, k)
        jax_score = make_jax_score_fn(my_score)
    except ImportError:
        print("Note: Custom masks/score_fn imports failed. Using defaults.")
        mask_fn, block_mask_fn, jax_score = None, None, None

    # Run Benchmark
    run_bench_suite_bwd(
        q, k, v, l, m, o, do,
        sm_scale=sm_scale,
        block_b=block_b,
        block_q=block_q,
        block_k_major=block_k_major,
        block_k=block_k,
        causal=causal,
        score_fn=jax_score,
        mask_fn=mask_fn,
        block_mask_fn=block_mask_fn
    )

if __name__ == "__main__":
    main()