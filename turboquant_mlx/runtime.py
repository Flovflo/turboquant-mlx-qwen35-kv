from __future__ import annotations

import mlx_lm.models.base as base
import mlx_lm.models.qwen3_next as qwen3_next

from .attention import dispatch_attention


_PATCHED = False


def patch_attention_dispatch():
    global _PATCHED
    if _PATCHED:
        return
    patched = dispatch_attention(base.scaled_dot_product_attention)
    base.scaled_dot_product_attention = patched
    qwen3_next.scaled_dot_product_attention = patched
    _PATCHED = True
