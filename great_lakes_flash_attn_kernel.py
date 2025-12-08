import jax
import jax.numpy as jnp
import time


def causal_mask(q_len, k_len):
    i = jnp.arange(q_len)[:, None]
    j = jnp.arange(k_len)[None, :]
    return i >= j


def flash_attention_forward(q, k, v, causal=False):
    b, h, q_len, d = q.shape
    _, _, k_len, _ = k.shape

    scale = 1.0 / jnp.sqrt(d)
    logits = jnp.einsum("bhqd,bhkd->bhqk", q, k) * scale

    if causal:
        mask = causal_mask(q_len, k_len)
        logits = logits + jnp.where(mask, 0.0, -1e9)

    attn = jax.nn.softmax(logits, axis=-1)
    out = jnp.einsum("bhqk,bhkd->bhqd", attn, v)

    return out, (attn, q, k, v)


def flash_attention_backward(ctx, dout, causal=False):
    attn, q, k, v = ctx
    b, h, q_len, d = q.shape

    dv = jnp.einsum("bhqk,bhqd->bhkd", attn, dout)

    datt = jnp.einsum("bhqd,bhkd->bhqk", dout, v)

    attn_dot = jnp.sum(attn * datt, axis=-1, keepdims=True)
    dlogits = attn * (datt - attn_dot)

    dq = jnp.einsum("bhqk,bhkd->bhqd", dlogits, k)
    dk = jnp.einsum("bhqk,bhqd->bhkd", dlogits, q)

    scale = 1.0 / jnp.sqrt(d)
    dq = dq * scale
    dk = dk * scale

    return dq, dk, dv


@jax.custom_vjp
def flash_attention(q, k, v, causal=False):
    out, _ = flash_attention_forward(q, k, v, causal)
    return out


def flash_attention_fwd(q, k, v, causal):
    out, ctx = flash_attention_forward(q, k, v, causal)
    return out, (ctx, causal)


def flash_attention_bwd(res, dout):
    ctx, causal = res
    dq, dk, dv = flash_attention_backward(ctx, dout, causal)
    return dq, dk, dv, None


flash_attention.defvjp(flash_attention_fwd, flash_attention_bwd)


if __name__ == "__main__":
    print("Flash kernel loaded.")
