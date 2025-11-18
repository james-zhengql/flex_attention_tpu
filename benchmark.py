import functools
import jax

import jax.numpy as jnp
import numpy as np

import flash_attention_fwd_ref
import flex_attention_kernel
import mha_reference

import time
import stats




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
        # .block_until_ready() ensures we wait until computation is finished
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
        # Synchronize (very important for accurate timing)
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

    # Return average and median latency (seconds)
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
    score_ctx=None,      
):
    # JIT-compile reference implementation
    ref_fn = functools.partial(mha_reference, ab=None, sm_scale=sm_scale, save_residuals=False,score_fn=score_fn,
        score_ctx=score_ctx)
    ref_jit = jax.jit(ref_fn)

    # JIT-compile your custom Pallas kernel implementation
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
        score_ctx=score_ctx,
    )

    flash_jit = jax.jit(flash_partial)

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
        score_fn=score_fn,
        score_ctx=score_ctx,
    )


    flash_ref_jit = jax.jit(flash_ref_partial)

    return ref_jit, flash_jit, flash_ref_jit

def run_bench_suite(q, k, v, *, sm_scale, block_b, block_q, block_k_major, block_k, causal=False,score_fn=None,score_ctx=None):
    # Unpack shapes
    b, h, q_len, d = q.shape
    _, _, k_len, _ = k.shape

    # Compute FLOPs (for throughput calculation)
    gflops = flop_count_attention(b, h, q_len, k_len, d) / 1e9

    # Create JIT-compiled versions of both implementations
    ref_jit, flash_jit, flash_ref_jit = build_fns_for_bench(
        q, k, v,
        sm_scale=sm_scale,
        save_residuals=False,  # exclude residual buffers for fair comparison
        causal=causal,
        block_b=block_b,
        block_q=block_q,
        block_k_major=block_k_major,
        block_k=block_k,
        debug=False,
        score_fn=score_fn,
        score_ctx=score_ctx
    )

    print(f"\n== Benchmark config: b={b}, h={h}, q={q_len}, k={k_len}, d={d}, causal={causal} ==")
    print(f"Estimated FLOPs per call: {gflops:.2f} GFLOPs")

    # ---- Reference (naive) implementation ----
    t_mean_ref, t_med_ref = benchmark(ref_jit, (q, k, v), name="mha_reference[jit]")
    print(f"  → Throughput: {gflops / t_med_ref:.2f} GFLOP/s")

    # ---- Pallas FlashAttention kernel ----
    t_mean_flash, t_med_flash = benchmark(flash_jit, (q, k, v), name="pallas_flash[jit]")
    print(f"  → Throughput: {gflops / t_med_flash:.2f} GFLOP/s")

    # ---- Pallas FlashAttention kernel ----
    t_mean_flash_ref, t_med_flash_ref = benchmark(flash_ref_jit, (q, k, v), name="pallas_flash_ref[jit]")
    print(f"  → Throughput: {gflops / t_med_flash:.2f} GFLOP/s")

    # ---- Numeric correctness check ----
    o_ref = ref_jit(q, k, v).block_until_ready()
    o_flash = flash_jit(q, k, v)
    if isinstance(o_flash, (tuple, list)):
        o_flash = o_flash[0]
    o_flash = o_flash.block_until_ready()

    # Relative L2 error measures numerical difference between both results
    rel_err = jnp.linalg.norm(o_flash - o_ref) / jnp.linalg.norm(o_ref)
    print(f"Numeric diff (Relative L2): {rel_err:.3e}")

    # Return summarized metrics
    return {
        "ref_ms_med": t_med_ref * 1e3,
        "flash_ms_med": t_med_flash * 1e3,
        "flash_ref_ms_med":t_med_flash_ref*1e3,
        "ref_gflops": gflops / t_med_ref,
        "flash_gflops": gflops / t_med_flash,
        "flash_ref_gflops": gflops / t_med_flash_ref,
        "rel_l2": float(rel_err),
    }