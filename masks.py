import jax.numpy as jnp

def make_causal_mask_fns(block_q, block_k_major):
    """
    Returns a tuple of (mask_fn, block_mask_fn) closures capturing the block sizes.
    """
    
    # 1. Element-wise mask: Passed to the inner kernel loop
    #    Returns True if the interaction is valid (Keep), False if masked (Ignore)
    def causal_mask_fn(q_idx, k_idx):
        return q_idx[:, None] >= k_idx[None, :]

    # 2. Block-wise mask: Used by Pallas to skip entire blocks (Block Sparsity)
    #    Returns True if the block pair needs to run, False if fully masked.
    def causal_block_mask_fn(q_block_idx, k_block_idx):
        # Calculate the absolute position of the first token in the K block
        k_start = k_block_idx * block_k_major
        # Calculate the absolute position of the last token in the Q block
        q_end = (q_block_idx + 1) * block_q - 1
        
        # If the start of K is after the end of Q, the entire block is masked out.
        # We only run if K starts before Q ends.
        return k_start <= q_end

    return causal_mask_fn, causal_block_mask_fn


def make_sliding_window_mask_fns(block_q, block_k, window_size):
    """
    Returns (mask_fn, block_mask_fn) for Sliding Window Attention.
    
    Args:
        block_q: Size of Query tile (e.g., 1024)
        block_k: Size of Key tile (e.g., 512)
        window_size: How many tokens back to look (e.g., 4096)
    """

    # 1. Element-wise Mask (The Exact Logic)
    #    Run inside the kernel for every pixel in the tile.
    def sliding_window_mask_fn(q_idx, k_idx):
        # Broadcast indices to (BlockQ, BlockK)
        # q_idx shape: (BlockQ,) -> (BlockQ, 1)
        # k_idx shape: (BlockK,) -> (1, BlockK)
        q_broadcast = q_idx[:, None]
        k_broadcast = k_idx[None, :]

        # Causal Check: Q must be after K
        # is_causal = q_broadcast >= k_broadcast

        # Window Check: Distance must be less than window size
        dist = q_broadcast - k_broadcast
        in_window = dist < window_size

        return in_window

    # 2. Block-wise Mask (The Optimization)
    #    Run by Pallas to decide whether to SKIP the tile entirely.
    def sliding_window_block_mask_fn(q_block_idx, k_block_idx):
        # Calculate global token positions for the boundaries of the blocks
        q_start = q_block_idx * block_q
        q_end   = (q_block_idx + 1) * block_q - 1
        
        k_start = k_block_idx * block_k
        k_end   = (k_block_idx + 1) * block_k - 1

        # Causal Skip: 
        # If the start of K is after the end of Q, it's fully in the future.
        # keep_causal = k_start <= q_end

        # Window Skip: 
        # If the END of K is too far behind the START of Q, it's out of window.
        # i.e., Min Distance > Window Size
        min_distance = q_start - k_end
        keep_window = min_distance < window_size

        return keep_window

    return sliding_window_mask_fn, sliding_window_block_mask_fn

# Assume we have a segmentation map, e.g., breaks at [0, 8192, 12000]
# In practice, you'd pass `segment_ids` (an array of ints like [0, 0, ..., 1, 1])

def make_jagged_mask_fns(block_q, block_k, doc_lengths):
    # 1. Pre-calculate boundaries
    boundaries = [0]
    cumsum = 0
    for l in doc_lengths:
        cumsum += l
        boundaries.append(cumsum)
    boundaries = tuple(boundaries)

    # Helper: Get Doc ID
    def get_doc_id(idx):
        doc_id = 0
        for b in boundaries[1:]:
            doc_id = doc_id + (idx >= b).astype(jnp.int32)
        return doc_id

    # 2. Element-wise Logic (Unchanged)
    def jagged_mask_fn(q_idx, k_idx):
        doc_q = get_doc_id(q_idx)
        doc_k = get_doc_id(k_idx)
        is_same_doc = doc_q[:, None] == doc_k[None, :]
        is_causal = q_idx[:, None] >= k_idx[None, :]
        return is_same_doc & is_causal

    # 3. Block-wise Logic (FIXED)
    def jagged_block_mask_fn(q_b, k_b):
        # Calculate start and end of the blocks
        q_start = q_b * block_q
        q_end   = (q_b + 1) * block_q - 1
        
        k_start = k_b * block_k
        k_end   = (k_b + 1) * block_k - 1
        
        # FIND THE RANGE OF DOC IDs IN THIS BLOCK
        # Since IDs are monotonic, we just check the ID of the start and end.
        q_doc_min = get_doc_id(q_start)
        q_doc_max = get_doc_id(q_end)
        
        k_doc_min = get_doc_id(k_start)
        k_doc_max = get_doc_id(k_end)
        
        # CHECK OVERLAP
        # Two ranges [A, B] and [C, D] overlap if max(A, C) <= min(B, D)
        overlap_start = jnp.maximum(q_doc_min, k_doc_min)
        overlap_end   = jnp.minimum(q_doc_max, k_doc_max)
        
        has_common_doc = overlap_start <= overlap_end
        
        # Causal Check
        is_causal = k_start <= q_end
        
        return has_common_doc & is_causal

    return jagged_mask_fn, jagged_block_mask_fn

def make_s2_mask_fns(block_q, block_k, window_size, stride):
    
    # 1. Element-wise
    def s2_mask_fn(q_idx, k_idx):
        # A. Sliding Window
        dist = q_idx[:, None] - k_idx[None, :]
        is_local = (dist >= 0) & (dist < window_size)
        
        # B. Sparse Stride (The "S2" trick)
        # Attend to key if it's a multiple of the stride
        is_strided = (k_idx % stride == 0)
        
        # Causal constraint for the stride
        is_causal = q_idx[:, None] >= k_idx[None,:]
        
        return is_local | (is_strided & is_causal)

    # 2. Block-wise
    def s2_block_mask_fn(q_b, k_b):
        q_start = q_b * block_q
        k_start = k_b * block_k
        k_end   = k_start + block_k - 1
        q_end   = q_start + block_q - 1

        # A. Window check (Optimized skip)
        is_causal = k_start <= q_end
        min_dist = q_start - k_end
        in_window = min_dist < window_size
        
        # B. Stride check
        # Does this K-block contain ANY strided tokens?
        # If block_k >= stride, it *must* contain a strided token.
        has_stride = True # Simplified if block_size > stride
        
        return (is_causal & in_window) | (is_causal & has_stride)

    return s2_mask_fn, s2_block_mask_fn