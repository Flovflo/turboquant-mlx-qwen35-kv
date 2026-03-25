from __future__ import annotations

from dataclasses import asdict

import mlx.core as mx

from .cache import TurboQuantConfig
from .generation import generate_tokens


def synthetic_prompt(vocab_size: int, prompt_tokens: int, seed: int = 0):
    mx.random.seed(seed)
    return mx.random.randint(0, vocab_size, (prompt_tokens,), dtype=mx.int32)


def run_benchmark(model, config, prompt_tokens: int, generation_tokens: int, kv_bits: int, kv_group_size: int):
    vocab_size = config.get("vocab_size") or config["text_config"]["vocab_size"]
    prompt = synthetic_prompt(vocab_size, prompt_tokens)
    results = []
    for backend in ("baseline", "mlx_quant", "turboquant"):
        turbo = TurboQuantConfig(bits=kv_bits, group_size=kv_group_size)
        _, _, stats = generate_tokens(
            model,
            prompt,
            max_tokens=generation_tokens,
            backend=backend,
            kv_bits=kv_bits,
            kv_group_size=kv_group_size,
            turbo_config=turbo,
        )
        results.append(asdict(stats))
    return results
