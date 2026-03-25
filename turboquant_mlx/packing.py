from __future__ import annotations

import numpy as np
import mlx.core as mx


def pack_sign_bits(signs: mx.array) -> mx.array:
    packed = np.packbits(np.array(signs, dtype=np.uint8), axis=-1, bitorder="little")
    return mx.array(packed, dtype=mx.uint8)


def unpack_sign_bits(packed: mx.array, nbits: int) -> mx.array:
    shifts = mx.arange(8, dtype=mx.uint8)
    expanded = mx.expand_dims(packed, axis=-1)
    one = mx.array(1, dtype=mx.uint8)
    bits = mx.bitwise_and(mx.right_shift(expanded, shifts), one)
    bits = bits.reshape(*bits.shape[:-2], bits.shape[-2] * bits.shape[-1])[..., :nbits]
    return bits.astype(mx.float32) * 2.0 - 1.0
