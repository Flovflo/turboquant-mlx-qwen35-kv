from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import mlx.core as mx


@dataclass(slots=True)
class ProjectionSpec:
    signs: mx.array
    perm: mx.array
    sketch_idx: mx.array
    sketch_signs: mx.array


def make_projection(dim: int, sketch_dim: int, seed: int) -> ProjectionSpec:
    rng = np.random.default_rng(seed)
    signs = mx.array(rng.choice([-1.0, 1.0], size=(dim,)).astype(np.float32))
    perm = mx.array(rng.permutation(dim).astype(np.int32))
    sketch_dim = min(sketch_dim, dim)
    sketch_idx = mx.array(rng.choice(dim, size=(sketch_dim,), replace=False).astype(np.int32))
    sketch_signs = mx.array(rng.choice([-1.0, 1.0], size=(sketch_dim,)).astype(np.float32))
    return ProjectionSpec(signs=signs, perm=perm, sketch_idx=sketch_idx, sketch_signs=sketch_signs)


def apply_rotation(x: mx.array, spec: ProjectionSpec) -> mx.array:
    flipped = x.astype(mx.float32) * spec.signs
    return mx.take(flipped, spec.perm, axis=-1)


def apply_sketch(x: mx.array, spec: ProjectionSpec) -> mx.array:
    projected = mx.take(x.astype(mx.float32), spec.sketch_idx, axis=-1)
    return projected * spec.sketch_signs
