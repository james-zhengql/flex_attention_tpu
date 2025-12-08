import os
import shutil
import jax
import scores
from jax import random
import jax.numpy as jnp
import benchmark 
import masks # Assumes this contains the factory functions we defined earlier
import numpy as np
import util

# ==============================================================================
# Helper to Generate Fake Document IDs
# ==============================================================================

def generate_doc_lengths(total_len, num_docs, seed=0):
    """
    Generates a list of random document lengths that sum exactly to total_len.
    Ensures no document has 0 length.
    """
    np.random.seed(seed)
    
    if num_docs <= 0:
        return [total_len]
    if num_docs > total_len:
        raise ValueError("Cannot have more documents than tokens!")

    # 1. Generate split points
    # We pick (num_docs - 1) cuts in the sequence.
    # range(1, total_len) ensures we don't cut at 0 or end, avoiding 0-length docs.
    splits = np.sort(np.random.choice(range(1, total_len), num_docs - 1, replace=False))
    
    # 2. Add start (0) and end (total_len)
    boundaries = np.concatenate(([0], splits, [total_len]))
    
    # 3. Calculate lengths (distance between cuts)
    lengths = np.diff(boundaries)
    
    return lengths.tolist()

# ==============================================================================
# Main Execution Loop
# ==============================================================================
def main():
    print("=== FlexAttention Masking Benchmark & Verification ===\n")
    
    # 1. Hardware / Data Config
    # -------------------------
    key = random.PRNGKey(0)
    batch, heads = 1, 1
    q_len, kv_len = 16384, 16384
    head_dim = 128
    
    # Block Sizes (Must match what fits in your VMEM)
    block_q = 1024
    block_k_major = 1024
    block_k = 512

    # 2. Generate Inputs (BF16 for TPU Speed)
    # -------------------------
    print(f"Generating inputs: B={batch}, H={heads}, L={q_len}, D={head_dim} (BF16)...")
    k1, k2, k3, k4 = random.split(key, 4)
    q = random.normal(k1, (batch, heads, q_len, head_dim), dtype=jnp.bfloat16)
    k = random.normal(k2, (batch, heads, kv_len, head_dim), dtype=jnp.bfloat16)
    v = random.normal(k3, (batch, heads, kv_len, head_dim), dtype=jnp.bfloat16)

    # 3. Define Test Cases
    # -------------------------
    # Each case is a dictionary defining how to build the mask and what to tell the reference.
    
    test_cases = []

    # --- Case A: Standard Causal ---
    test_cases.append({
        "name": "Causal Attention",
        "factory": masks.make_causal_mask_fns,
        "factory_args": (), # No extra args needed for causal factory
        "ref_args": {"causal": True}
    })

    # # --- Case B: Sliding Window ---
    # window_size = 1024
    # test_cases.append({
    #     "name": f"Sliding Window (W={window_size})",
    #     "factory": masks.make_sliding_window_mask_fns,
    #     "factory_args": (window_size,), # Pass window_size to factory
    #     "ref_args": {"causal": False, "window_size": window_size}
    # })
    
    # # --- Case C: Jagged Documents (Randomized) ---
    # # Generate 5 random document lengths that sum to 8192
    # doc_lengths = generate_doc_lengths(total_len=q_len, num_docs=5, seed=42)
    # print(f"Generated Doc Lengths: {doc_lengths}") 
    # # Example Output: [1200, 350, 4000, 2000, 642]

    # # 1. Create Reference IDs for validation
    # #    (Builds the array [0,0,.., 1,1,.., 2,2,..])
    # ref_ids_list = []
    # for i, length in enumerate(doc_lengths):
    #     ref_ids_list.append(jnp.full((length,), i, dtype=jnp.int32))
    
    # jagged_ids_ref = jnp.concatenate(ref_ids_list)
    # jagged_ids_ref = jnp.tile(jagged_ids_ref[None, :], (batch, 1))

    # test_cases.append({
    #     "name": f"Jagged Masking ({len(doc_lengths)} Docs)",
    #     "factory": masks.make_jagged_mask_fns,
    #     # Pass the list of lengths to the factory
    #     "factory_args": (doc_lengths,),  
    #     # Pass the ID array to the reference
    #     "ref_args": {"causal": True, "segment_ids": jagged_ids_ref}
    # })

    # # --- Case D: ALiBi (Score Function) ---
    # # Returns (None, None) for masks, passes score_fn to ref_args
    
    # alibi_fn = util.make_jax_score_fn(scores.make_alibi_score_fn(slope=0.5))
    # test_cases.append({
    #     "name": "ALiBi Attention",
    #     "factory": lambda *args: (None, None), # Dummy factory
    #     "factory_args": (),
    #     "ref_args": { "score_fn": alibi_fn,"alibi_slope":0.5}
    # })

    # # --- Case E: Tanh Soft-Capping (Score Function) ---
    # tanh_fn = util.make_jax_score_fn(scores.make_softcap_score_fn(cap=30.0))
    # test_cases.append({
    #     "name": "Tanh Soft-Capping",
    #     "factory": lambda *args: (None, None),
    #     "factory_args": (),
    #     "ref_args": { "score_fn": tanh_fn}
    # })

    # 4. Run Loop
    # -------------------------
    for case in test_cases:
        print(f"\n" + "="*60)
        print(f"RUNNING: {case['name']}")
        print("="*60)

        # A. Build the Masks
        # Unpack factory args (block sizes + specific args)
        factory_fn = case['factory']
        extra_args = case['factory_args']
        
        try:
            mask_fn, block_mask_fn = factory_fn(block_q, block_k_major, *extra_args)
        except AttributeError:
            print(f"Skipping {case['name']} (Factory function not found in masks.py)")
            continue

        # B. Prepare Reference Arguments
        # Start with defaults
        current_ref_args = {
            "sm_scale": 1.0,
            "block_b": 1,
            "block_q": block_q,
            "block_k_major": block_k_major,
            "block_k": block_k,
            "which": ["flash", "ref","flash_ref"], # Compare Flash vs Reference
            "mask_fn": mask_fn,
            "block_mask_fn": block_mask_fn,
        }
        # Update with case-specific args (causal=True, window_size=..., etc.)
        current_ref_args.update(case["ref_args"])

        # C. Run Benchmark
        results = benchmark.run_bench_suite(
            q, k, v,
            **current_ref_args
        )
        
        # D. Print Quick Status
        # Check diffs in the output (if printed by benchmark)
        # Note: benchmark.py prints the diffs automatically.

    print("\n=== All Tests Completed ===")

if __name__ == "__main__":
    main()


# import os
# import shutil
# import jax
# import jax.numpy as jnp
# from jax import random

# # Assuming these are imported from your modules
# import benchmark 
# from main import make_causal_mask_fns 

# def main():
#     # --- 1. SETUP ---
#     key = random.PRNGKey(0)
#     # Increased sequence length to ensuring heavy compute
#     batch, heads, q_len, dim = 1, 1, 25600, 128
    
#     # Generate Inputs
#     # CRITICAL: Use bfloat16 for TPU. float32 is slow; float16 crashes.
#     k1, k2, k3 = random.split(key, 3)
#     q = random.normal(k1, (batch, heads, q_len, dim), dtype=jnp.bfloat16)
#     k = random.normal(k2, (batch, heads, q_len, dim), dtype=jnp.bfloat16)
#     v = random.normal(k3, (batch, heads, q_len, dim), dtype=jnp.bfloat16)

#     # Get Functions
#     mask_fn, block_mask_fn = make_causal_mask_fns(1024, 1024)
#     bench_fns = benchmark.build_fns_for_bench(
#         q, k, v,
#         sm_scale=1.0, block_b=1, block_q=1024, block_k_major=1024, block_k=512,
#         causal=True, mask_fn=mask_fn, block_mask_fn=block_mask_fn,
#         which=["flash"]
#     )
#     flash_jit = bench_fns["flash_fwd_jit"]

#     # --- 2. WARMUP ---
#     print("Warming up (compiling)...")
#     warmup_out = flash_jit(q, k, v)
#     jax.tree_util.tree_map(lambda x: x.block_until_ready(), warmup_out)
#     print("Warmup done.")

#     # --- 3. PROFILE ---
#     trace_dir = "/tmp/tpu_profile"
    
#     if os.path.exists(trace_dir):
#         shutil.rmtree(trace_dir)

#     print(f"Starting trace... saving to {trace_dir}")
#     jax.profiler.start_trace(trace_dir)

#     # Run loop
#     for _ in range(100):
#         # FIX IS HERE: Call the function fresh every time!
#         out = flash_jit(q, k, v)
        
#         # Block on the NEW output, not the warmup output
#         jax.tree_util.tree_map(lambda x: x.block_until_ready(), out)

#     jax.profiler.stop_trace()
#     print("Trace finished!")
    
#     # Optional: Snapshot memory usage to verify you aren't leaking HBM
#     jax.profiler.save_device_memory_profile(f"{trace_dir}/memory.prof")

# if __name__ == "__main__":
#     main()

# import jax
# import jax.numpy as jnp
# from jax import random
# import pandas as pd
# import time

# # Import your existing modules
# import benchmark 
# from main import make_causal_mask_fns

# # --- TPU v5e Specs (Approximate) ---
# # Peak Compute (BF16 Matrix Mul): ~197 TFLOPS
# # Peak Memory Bandwidth: ~819 GB/s
# TPU_PEAK_TFLOPS = 197.0
# TPU_PEAK_BW = 819.0

# def get_theoretical_metrics(b, h, l, d, causal=True, dtype_bytes=2):
#     # 1. Calculate IO (Bytes)
#     #    We Read Q, K, V and Write O. Total 4 arrays of size (B, H, L, D)
#     total_elements = 4 * (b * h * l * d)
#     total_bytes = total_elements * dtype_bytes
    
#     # 2. Calculate FLOPs
#     #    Standard Attention is 4 * B * H * L * L * D
#     #    Causal masks half the matrix, so divide by 2
#     total_flops = 4 * b * h * (l * l) * d
#     if causal:
#         total_flops /= 2
        
#     return total_flops, total_bytes

# def main():
#     key = random.PRNGKey(0)
    
#     # Constants
#     BATCH = 1
#     HEADS = 16       # Use more heads to saturate the hardware
#     DIM = 128
    
#     # Block sizes (Keep constant)
#     BLOCK_Q = 1024
#     BLOCK_K = 1024
    
#     # The Sweep: Loop over Sequence Lengths
#     # Powers of 2: 1024, 2048, 4096, 8192, 16384, 32768
#     SEQ_LENS = [1024 * (2**i) for i in range(5)]
    
#     results_data = []

#     print(f"{'SeqLen':<10} | {'Time(ms)':<10} | {'TFLOP/s':<10} | {'Intensity':<10}")
#     print("-" * 55)

#     for L in SEQ_LENS:
#         # 1. Generate Inputs (New shape each time)
#         # CRITICAL: Use bfloat16 for TPU performance
#         k1, k2, k3, key = random.split(key, 4)
#         q = random.normal(k1, (BATCH, HEADS, L, DIM), dtype=jnp.bfloat16)
#         k = random.normal(k2, (BATCH, HEADS, L, DIM), dtype=jnp.bfloat16)
#         v = random.normal(k3, (BATCH, HEADS, L, DIM), dtype=jnp.bfloat16)

#         # 2. Setup Kernel
#         mask_fn, block_mask_fn = make_causal_mask_fns(BLOCK_Q, BLOCK_K)

#         # 3. Run Benchmark
#         #    IMPORTANT: Only run "flash". "ref" will hang/crash at L=16k.
#         bench_out = benchmark.run_bench_suite(
#             q, k, v,
#             sm_scale=1.0,
#             block_b=1,
#             block_q=BLOCK_Q,
#             block_k_major=BLOCK_K,
#             block_k=512,
#             causal=False,
#             mask_fn=mask_fn,
#             block_mask_fn=block_mask_fn,
#             which=["ref"]  # <--- Only benchmark your kernel
#         )
        
#         # 4. Extract Metrics
#         #    bench_out["flash_fwd_jit"] returns (mean_time, median_time)
#         #    We use median_time to be robust against jitter
#         _, time_sec = bench_out["ref_fwd_jit"]
        
#         flops, bytes_moved = get_theoretical_metrics(BATCH, HEADS, L, DIM, causal=True)
        
#         tflops_per_sec = (flops / 1e12) / time_sec
#         intensity = flops / bytes_moved  # FLOPs per Byte

#         print(f"{L:<10} | {time_sec*1000:<10.2f} | {tflops_per_sec:<10.2f} | {intensity:<10.2f}")
        
#         results_data.append({
#             "SeqLen": L,
#             "Time_Sec": time_sec,
#             "TFLOPs": tflops_per_sec,
#             "Intensity": intensity
#         })

#     # Save to CSV for plotting
#     df = pd.DataFrame(results_data)
#     df.to_csv("roofline_data.csv", index=False)
#     print("\nSaved sweep data to roofline_data.csv")

# if __name__ == "__main__":
#     main()