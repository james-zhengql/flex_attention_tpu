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
    mask_value: float = None,
    sm_scale: float = 1.0,
    save_residuals: bool = True,
    score_fn = None
):
    batched_score = jax.vmap(
            jax.vmap(score_fn, in_axes=(0, 0)), 
            in_axes=(0, 0)
        )
    
    if score_fn is None:
        logits = jnp.einsum("bhqc,bhkc->bhqk", q, k)
    else:
        logits = batched_score(q, k)
    print(f"DEBUG: logits shape: {logits.shape}")
    if ab is not None:
        logits += ab
    if sm_scale != 1.0:
        logits *= sm_scale

    # # no causal masking
    # mask = None
    # logits = logits if mask is None else logits + jnp.where(mask, 0.0, -1e9)

    # m = logits.max(axis=-1)
    unnormalized = jnp.exp(logits - m)
    # l = unnormalized.sum(axis=-1)
    # weights = unnormalized / l[..., None]
    # o = jnp.einsum("bhkq,bhkc->bhqc", weights, v)
    print(f"DEBUG: m shape: {m.shape}")
    print(f"DEBUG: unnormalized shape: {unnormalized.shape}")
    p = unnormalized / l
    print(f"DEBUG: l shape: {l.shape}")
    print(f"DEBUG: p shape: {p.shape}")

    # dv = P^T * do
    dv = jnp.einsum("bhqk,bhqc->bhkc", p, do)

    # dp = do * V^T
    dp = jnp.einsum("bhqc,bhkc->bhqk", do, v)

    # software backward
    sum_d = jnp.sum(do*o, axis=-1)
    ds = p * (dp - sum_d[..., None])

    if score_fn is None:
        dq = jnp.einsum("bhqk,bhkc->bhqc", ds, k)
        dk = jnp.einsum("bhqk,bhqc->bhkc", ds, q)
    else:
        _, vjp_fn = jax.vjp(batched_score, q, k)
        dq, dk = vjp_fn(ds)


    return dq, dk, dv
