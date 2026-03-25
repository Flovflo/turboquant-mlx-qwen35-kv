from __future__ import annotations

from typing import Optional

import mlx.core as mx
from mlx.utils import tree_map
from mlx_lm.models.base import quantized_scaled_dot_product_attention

from .cache import TurboQuantKVCache
from .packing import unpack_sign_bits
from .projection import apply_rotation


def turboquant_scaled_dot_product_attention(
    queries: mx.array,
    key_state,
    value_state,
    cache: TurboQuantKVCache,
    scale: float,
    mask: Optional[mx.array],
):
    q_keys, residual_bits, residual_rms = key_state
    B, n_q_heads, _, D = queries.shape
    n_kv_heads = q_keys[0].shape[-3]
    n_repeats = n_q_heads // n_kv_heads
    q = queries * scale
    if n_repeats > 1:
        q = mx.reshape(q, (B, n_kv_heads, n_repeats, q.shape[-2], D))
        q_keys = tree_map(lambda x: mx.expand_dims(x, axis=-3), q_keys)
        residual_bits = mx.expand_dims(residual_bits, axis=-3)
        residual_rms = mx.expand_dims(residual_rms, axis=-3)
        if cache.config.quantize_values:
            value_state = tree_map(lambda x: mx.expand_dims(x, axis=-3), value_state)
        else:
            value_state = mx.expand_dims(value_state, axis=-3)
    q_rot = apply_rotation(q, cache.projection)
    scores = mx.quantized_matmul(
        q_rot,
        *q_keys,
        transpose=True,
        group_size=cache.group_size,
        bits=cache.config.bits,
    )
    q_proj = mx.take(q_rot.astype(mx.float32), cache.projection.sketch_idx, axis=-1)
    q_proj = q_proj * cache.projection.sketch_signs
    k_sign = unpack_sign_bits(residual_bits, cache.config.sketch_dim)
    residual = mx.matmul(q_proj, mx.swapaxes(k_sign * residual_rms, -1, -2))
    scores = scores + cache.config.residual_scale * residual
    if mask is not None:
        if isinstance(mask, str):
            qL, kL = scores.shape[-2:]
            mask = mx.arange(kL - qL, kL)[:, None] >= mx.arange(kL)[None]
        scores = mx.where(mask, scores, mx.finfo(scores.dtype).min) if mask.dtype == mx.bool_ else scores + mask
    weights = mx.softmax(scores, axis=-1, precise=True)
    if cache.config.quantize_values:
        out = mx.quantized_matmul(
            weights,
            *value_state,
            transpose=False,
            group_size=cache.value_group_size,
            bits=cache.config.value_bits,
        )
    else:
        out = mx.matmul(weights, value_state)
    out = out.astype(queries.dtype)
    if n_repeats > 1:
        out = mx.reshape(out, (B, n_q_heads, out.shape[-2], out.shape[-1]))
    return out


def dispatch_attention(previous):
    def _patched(queries, keys, values, cache, scale, mask, sinks=None):
        if isinstance(cache, TurboQuantKVCache):
            if sinks is not None:
                raise ValueError("Attention sinks are not supported for TurboQuantKVCache.")
            return turboquant_scaled_dot_product_attention(queries, keys, values, cache, scale, mask)
        return previous(queries, keys, values, cache, scale=scale, mask=mask, sinks=sinks)

    return _patched
