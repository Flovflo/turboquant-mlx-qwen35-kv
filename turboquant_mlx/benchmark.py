from __future__ import annotations

from dataclasses import asdict

import mlx.core as mx

from .cache import TurboQuantConfig
from .generation import generate_tokens


def synthetic_prompt(vocab_size: int, prompt_tokens: int, seed: int = 0):
    mx.random.seed(seed)
    return mx.random.randint(0, vocab_size, (prompt_tokens,), dtype=mx.int32)


def run_benchmark(
    model,
    config,
    prompt_tokens: int,
    generation_tokens: int,
    kv_bits: int,
    kv_group_size: int,
    backends=("baseline", "mlx_quant", "turboquant"),
    trials: int = 1,
):
    vocab_size = config.get("vocab_size") or config["text_config"]["vocab_size"]
    results = []
    for backend in backends:
        trial_rows = []
        for trial in range(trials):
            prompt = synthetic_prompt(vocab_size, prompt_tokens, seed=trial)
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
            trial_rows.append(asdict(stats))
        if trials == 1:
            results.extend(trial_rows)
            continue
        avg = dict(trial_rows[0])
        avg["prompt_tps"] = sum(r["prompt_tps"] for r in trial_rows) / trials
        avg["generation_tps"] = sum(r["generation_tps"] for r in trial_rows) / trials
        avg["generation_wall_time_s"] = (
            sum(r["generation_wall_time_s"] for r in trial_rows) / trials
        )
        avg["peak_memory_gb"] = sum(r["peak_memory_gb"] for r in trial_rows) / trials
        avg["cache_bytes"] = round(sum(r["cache_bytes"] for r in trial_rows) / trials)
        avg["trials"] = trials
        avg["runs"] = trial_rows
        results.append(avg)
    return results
