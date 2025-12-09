import jax
import jax.numpy as jnp
from util import make_jax_score_fn

def mha_reference(
    q,
    k,
    v,
    ab: jax.Array | None = None,
    *,
    sm_scale: float = 1.0,
    save_residuals: bool = False,
    score_fn=None,
    causal: bool = False,
    window_size: int | None = None,
    segment_ids: jax.Array | None = None,  # <--- Added for Document Masking
    s2_stride: int | None = None,          # <--- Added for S2 Attention
    alibi_slope: float | None=None
):
    batch, heads, q_len, dim = q.shape
    k_len = k.shape[2]

    # 1. Compute Scores (Logits)
    if score_fn is None:
        logits = jnp.einsum("bhqc,bhkc->bhqk", q, k) * sm_scale
    else:
        # vmap over Batch (0) and Heads (1)
        # The score_fn expects (Q_len, D) and (K_len, D)
        # So we map over the first two dimensions of Q and K
        if alibi_slope is not None:
        # Calculate distance: |i - j|
        # Note: Standard ALiBi usually penalizes the past (i - j). 
        # We use abs(i - j) to handle symmetric cases if needed, 
        # or just (q - k) for strictly causal.
            q_idx = jnp.arange(q_len)[:, None]%1024      # (Q, 1)
            k_idx = jnp.arange(k_len)[None, :]%512
            dist = jnp.abs(q_idx - k_idx)
            
            # Bias = -slope * distance
            # Broadcasts to (Batch, Heads, Q, K) automatically
            logits = jnp.einsum("bhqc,bhkc->bhqk", q, k) * sm_scale
            logits = logits - (alibi_slope * dist)
        else:
            score_fn_batched = jax.vmap(
                jax.vmap(score_fn, in_axes=(0, 0)), 
                in_axes=(0, 0)
            )
            logits = score_fn_batched(q, k)

    if ab is not None:
        logits += ab

    # 2. Prepare Indices for Masking
    #    Shapes: q_idx (Q, 1), k_idx (1, K)
    q_idx = jnp.arange(q_len)[:, None]
    k_idx = jnp.arange(k_len)[None, :]

    # 3. Apply Causal Mask
    if causal:
        # Mask where Key is in the future relative to Query
        causal_mask = k_idx > q_idx
        logits = jnp.where(causal_mask, -1e9, logits)

    # 4. Apply Document Masking (Jagged Attention)
    if segment_ids is not None:
        # Assume segment_ids shape is (Batch, SeqLen) or (SeqLen,)
        # Broadcast to (Batch, 1, Q, 1) and (Batch, 1, 1, K)
        if segment_ids.ndim == 1:
            ids_q = segment_ids[None, None, :, None]
            ids_k = segment_ids[None, None, None, :]
        else:
            ids_q = segment_ids[:, None, :, None]
            ids_k = segment_ids[:, None, None, :]
            
        # Mask where documents do not match
        doc_mask = ids_q != ids_k
        logits = jnp.where(doc_mask, -1e9, logits)

    # 5. Apply Sliding Window / S2 Mask
    if window_size is not None:
        dist = q_idx - k_idx
        
        # Base Window Logic: Mask if too far in the past
        # (dist >= window_size) means "Too Old"
        should_mask = dist >= window_size
        
        # S2 Attention Logic: "Sparse" tokens are exempt from the window limit
        if s2_stride is not None:
            # If K is a stride token, we KEEP it even if it's old.
            # So we only mask if it is old AND it is NOT a stride token.
            is_strided_token = (k_idx % s2_stride == 0)
            should_mask = should_mask & (~is_strided_token)
            
        logits = jnp.where(should_mask, -1e9, logits)

    # -------------------------
    # Softmax & Output
    # -------------------------
    m = logits.max(axis=-1)
    unnormalized = jnp.exp(logits - m[..., None])
    l = unnormalized.sum(axis=-1)
    
    # Safety for purely masked rows (prevent NaN)
    l_safe = jnp.where(l == 0, 1.0, l)
    weights = unnormalized / l_safe[..., None]
    
    out = jnp.einsum("bhqk,bhkc->bhqc", weights, v)

    return (out, l, m) if save_residuals else out


def mha_bwd_reference(
    q,
    k,
    v,
    o,
    do,
    l,
    m,
    ab: jax.Array | None = None,
    *,
    causal: bool = False,
    mask_value: float | None = None,   # kept for API compatibility, not used
    sm_scale: float = 1.0,
    save_residuals: bool = True,       # kept for API compatibility, not used
    score_fn=None,
    window_size: int | None = None,
    segment_ids: jax.Array | None = None,
    s2_stride: int | None = None,
    alibi_slope: float | None = None,
):
    """
    Backward pass that exactly mirrors `mha_reference`.
    Returns gradients w.r.t q, k, v.
    """

    batch, heads, q_len, dim = q.shape
    k_len = k.shape[2]

    # -------------------------
    # Rebuild logits exactly as in mha_reference
    # -------------------------
    if score_fn is None or alibi_slope is not None:
        # Dot-product path (also used when ALiBi is active, even if score_fn is provided)
        logits = jnp.einsum("bhqc,bhkc->bhqk", q, k) * sm_scale

        if alibi_slope is not None:
            # Same ALiBi logic as in forward
            q_idx_alibi = jnp.arange(q_len)[:, None] % 1024
            k_idx_alibi = jnp.arange(k_len)[None, :] % 512
            dist = jnp.abs(q_idx_alibi - k_idx_alibi)
            logits = logits - (alibi_slope * dist)

        batched_score = None
    else:
        # score_fn path (no sm_scale in forward)
        score_fn_batched = jax.vmap(
            jax.vmap(score_fn, in_axes=(0, 0)),
            in_axes=(0, 0),
        )
        logits = score_fn_batched(q, k)
        batched_score = score_fn_batched

    if ab is not None:
        logits = logits + ab

    # Prepare indices for masking (same as forward)
    q_idx = jnp.arange(q_len)[:, None]  # (Q, 1)
    k_idx = jnp.arange(k_len)[None, :]  # (1, K)

    # Causal mask
    if causal:
        causal_mask = k_idx > q_idx
        logits = jnp.where(causal_mask, -1e9, logits)

    # Document (segment) mask
    if segment_ids is not None:
        if segment_ids.ndim == 1:
            ids_q = segment_ids[None, None, :, None]  # (1, 1, Q, 1)
            ids_k = segment_ids[None, None, None, :]  # (1, 1, 1, K)
        else:
            ids_q = segment_ids[:, None, :, None]     # (B, 1, Q, 1)
            ids_k = segment_ids[:, None, None, :]     # (B, 1, 1, K)

        doc_mask = ids_q != ids_k
        logits = jnp.where(doc_mask, -1e9, logits)

    # Sliding window / S2 mask
    if window_size is not None:
        dist = q_idx - k_idx
        should_mask = dist >= window_size

        if s2_stride is not None:
            is_strided_token = (k_idx % s2_stride == 0)
            should_mask = should_mask & (~is_strided_token)

        logits = jnp.where(should_mask, -1e9, logits)

    # -------------------------
    # Reconstruct softmax probabilities from (logits, m, l)
    # -------------------------
    # forward: m = max(logits), l = sum(exp(logits - m))
    unnormalized = jnp.exp(logits - m[..., None])  # [B, H, Q, K]

    # handle fully-masked rows (l == 0) the same way as forward
    l_safe = jnp.where(l == 0, 1.0, l)
    p = unnormalized / l_safe[..., None]           # [B, H, Q, K]

    # -------------------------
    # Backprop through softmax + V
    # -------------------------
    # dv = P^T * dO
    dv = jnp.einsum("bhqk,bhqc->bhkc", p, do)

    # dp = dO * V^T
    dp = jnp.einsum("bhqc,bhkc->bhqk", do, v)

    # softmax backward:
    # sum_d = Σ_k p_k * dp_k = (dO · O) per (B,H,Q)
    sum_d = jnp.sum(do * o, axis=-1)              # [B, H, Q]
    ds = p * (dp - sum_d[..., None])              # dL/d(logits)

    # -------------------------
    # Backprop to q, k
    # -------------------------
    if score_fn is None or alibi_slope is not None:
        # logits = sm_scale * <q, k> (+ ALiBi bias)
        # ∂logits/∂q = sm_scale * k; ∂logits/∂k = sm_scale * q
        ds_scaled = ds * sm_scale
        dq = jnp.einsum("bhqk,bhkc->bhqc", ds_scaled, k)
        dk = jnp.einsum("bhqk,bhqc->bhkc", ds_scaled, q)
    else:
        # score_fn path: use VJP of the batched score function
        _, vjp_fn = jax.vjp(batched_score, q, k)
        dq, dk = vjp_fn(ds)

    return dq, dk, dv
