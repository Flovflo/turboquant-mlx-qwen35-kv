from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
from mlx.utils import tree_map
from mlx_lm.models.cache import _BaseCache, create_attention_mask

from .projection import ProjectionSpec, apply_rotation, apply_sketch, make_projection


@dataclass(slots=True)
class TurboQuantConfig:
    bits: int = 4
    group_size: int = 64
    value_bits: int = 4
    quantize_values: bool = True
    sketch_dim: int = 4
    residual_scale: float = 1.0
    seed: int = 0


class TurboQuantKVCache(_BaseCache):
    def __init__(self, config: TurboQuantConfig):
        self.config = config
        self.offset = 0
        self.projection: Optional[ProjectionSpec] = None
        self.keys_main = None
        self.values_main = None
        self.residual_t = None
        self.value_dtype = None
        self.bits = config.bits
        self.group_size = config.group_size
        self.value_group_size = config.group_size

    def _init_params(self, dim: int):
        if self.projection is None:
            self.projection = make_projection(dim, self.config.sketch_dim, self.config.seed)
            self.group_size = self._effective_group_size(dim)

    def _append_tuple(self, current, update):
        if current is None:
            return update
        return tuple(mx.concatenate([c, u], axis=2) for c, u in zip(current, update))

    def _append_array(self, current, update):
        return update if current is None else mx.concatenate([current, update], axis=2)

    def _append_residual(self, current, update):
        return update if current is None else mx.concatenate([current, update], axis=-1)

    def _quantize_keys(self, keys: mx.array):
        rotated = apply_rotation(keys.astype(mx.bfloat16), self.projection)
        q_keys = mx.quantize(
            rotated,
            group_size=self.group_size,
            bits=self.config.bits,
            mode="affine",
        )
        dequant = mx.dequantize(
            *q_keys,
            group_size=self.group_size,
            bits=self.config.bits,
        )
        residual = rotated.astype(mx.float32) - dequant.astype(mx.float32)
        proj = apply_sketch(residual, self.projection).astype(mx.bfloat16)
        signs = mx.where(proj >= 0, 1.0, -1.0).astype(mx.bfloat16)
        rms = mx.sqrt(mx.mean(mx.square(residual), axis=-1, keepdims=True)).astype(mx.bfloat16)
        residual_t = mx.swapaxes(signs * rms, -1, -2)
        return q_keys, residual_t

    def _quantize_values(self, values: mx.array):
        self.value_group_size = self._effective_group_size(values.shape[-1])
        if self.config.quantize_values:
            return mx.quantize(
                values.astype(mx.bfloat16),
                group_size=self.value_group_size,
                bits=self.config.value_bits,
                mode="affine",
            )
        return values

    def _effective_group_size(self, dim: int) -> int:
        candidates = [128, 64, 32, 16, 8, 4, 2, 1]
        limit = min(dim, self.config.group_size)
        for candidate in candidates:
            if candidate <= limit and dim % candidate == 0:
                return candidate
        return 1

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        self._init_params(keys.shape[-1])
        self.value_dtype = values.dtype
        q_keys, residual_t = self._quantize_keys(keys)
        q_values = self._quantize_values(values)
        self.keys_main = self._append_tuple(self.keys_main, q_keys)
        self.values_main = self._append_tuple(self.values_main, q_values) if self.config.quantize_values else self._append_array(self.values_main, q_values)
        self.residual_t = self._append_residual(self.residual_t, residual_t)
        self.offset += keys.shape[2]
        return self.key_state, self.value_state

    @property
    def key_state(self):
        return self.keys_main, self.residual_t

    @property
    def value_state(self):
        return self.values_main

    @property
    def state(self):
        return [self.keys_main, self.values_main, self.residual_t]

    @state.setter
    def state(self, v):
        self.keys_main, self.values_main, self.residual_t = v

    @property
    def meta_state(self):
        return tuple(
            map(
                str,
                (
                    self.offset,
                    self.config.bits,
                    self.config.group_size,
                    self.config.value_bits,
                    int(self.config.quantize_values),
                    self.config.sketch_dim,
                    self.config.seed,
                ),
            )
        )

    @meta_state.setter
    def meta_state(self, v):
        offset, bits, group_size, value_bits, qv, sketch_dim, seed = map(int, v)
        self.offset = offset
        self.config = TurboQuantConfig(bits, group_size, value_bits, bool(qv), sketch_dim, 1.0, seed)
        self.bits = bits
        self.group_size = group_size

    def make_mask(self, *args, **kwargs):
        return create_attention_mask(*args, offset=self.offset, **kwargs)

    def size(self):
        return self.offset

    def empty(self):
        return self.keys_main is None

    def is_trimmable(self):
        return True

    def trim(self, n: int):
        n = min(self.offset, n)
        self.offset -= n
        return n

    @property
    def nbytes(self):
        pieces = [self.residual_t]
        if self.keys_main is not None:
            pieces.extend(self.keys_main)
        if self.values_main is not None:
            if self.config.quantize_values:
                pieces.extend(self.values_main)
            else:
                pieces.append(self.values_main)
        return sum(p.nbytes for p in pieces if p is not None)

    @classmethod
    def from_kvcache(cls, cache, config: TurboQuantConfig):
        tq = cls(config)
        if cache.keys is not None:
            tq.update_and_fetch(cache.keys[..., : cache.offset, :], cache.values[..., : cache.offset, :])
        return tq
