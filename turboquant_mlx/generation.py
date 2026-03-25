from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal, Optional

import mlx.core as mx
from mlx_lm.models.cache import KVCache, make_prompt_cache

from .cache import TurboQuantConfig, TurboQuantKVCache
from .runtime import patch_attention_dispatch


Backend = Literal["baseline", "mlx_quant", "turboquant"]


@dataclass(slots=True)
class GenerationStats:
    prompt_tokens: int
    generation_tokens: int
    prompt_tps: float
    generation_tps: float
    peak_memory_gb: float
    cache_bytes: int
    backend: str


def maybe_convert_cache(cache_list, backend: Backend, start: int, bits: int, group_size: int, turbo: TurboQuantConfig):
    if backend == "baseline":
        return cache_list
    for idx, cache in enumerate(cache_list):
        if not isinstance(cache, KVCache) or cache.offset < start:
            continue
        if backend == "mlx_quant":
            cache_list[idx] = cache.to_quantized(group_size=group_size, bits=bits)
        elif backend == "turboquant":
            cache_list[idx] = TurboQuantKVCache.from_kvcache(cache, turbo)
    return cache_list


def _step(model, tokens, prompt_cache):
    logits = model(tokens[None], cache=prompt_cache)
    logits = logits[:, -1, :]
    logprobs = logits - mx.logsumexp(logits, keepdims=True)
    next_token = mx.argmax(logprobs, axis=-1)
    return next_token, logprobs.squeeze(0)


def _tuple_nbytes(value) -> int:
    if value is None:
        return 0
    if isinstance(value, tuple):
        return sum(_tuple_nbytes(v) for v in value)
    if isinstance(value, list):
        return sum(_tuple_nbytes(v) for v in value)
    return int(value.nbytes)


def cache_nbytes(prompt_cache) -> int:
    total = 0
    for cache in prompt_cache:
        try:
            total += int(cache.nbytes)
        except Exception:
            if hasattr(cache, "state"):
                total += _tuple_nbytes(cache.state)
    return total


def generate_tokens(model, prompt_tokens, *, max_tokens=16, backend: Backend = "baseline", kv_bits=4, kv_group_size=64, quantized_kv_start=0, turbo_config: Optional[TurboQuantConfig] = None, prefill_step_size=2048):
    patch_attention_dispatch()
    turbo_config = turbo_config or TurboQuantConfig(bits=kv_bits, group_size=kv_group_size)
    prompt_cache = make_prompt_cache(model)
    prompt = prompt_tokens
    t0 = time.perf_counter()
    processed = 0
    while prompt.size - processed > 1:
        remaining = (prompt.size - processed) - 1
        n_to_process = min(prefill_step_size, remaining)
        model(prompt[processed : processed + n_to_process][None], cache=prompt_cache)
        maybe_convert_cache(prompt_cache, backend, quantized_kv_start, kv_bits, kv_group_size, turbo_config)
        mx.eval([c.state for c in prompt_cache])
        processed += n_to_process
    prompt_tps = max(prompt.size, 1) / max(time.perf_counter() - t0, 1e-6)
    next_token, logprobs = _step(model, prompt[processed:], prompt_cache)
    maybe_convert_cache(prompt_cache, backend, quantized_kv_start, kv_bits, kv_group_size, turbo_config)
    generated = [int(next_token.item())]
    decode_start = time.perf_counter()
    for _ in range(max_tokens - 1):
        next_token, logprobs = _step(model, next_token, prompt_cache)
        maybe_convert_cache(prompt_cache, backend, quantized_kv_start, kv_bits, kv_group_size, turbo_config)
        generated.append(int(next_token.item()))
    decode_tps = len(generated) / max(time.perf_counter() - decode_start, 1e-6)
    cache_bytes = cache_nbytes(prompt_cache)
    stats = GenerationStats(prompt.size, len(generated), prompt_tps, decode_tps, mx.get_peak_memory() / 1e9, cache_bytes, backend)
    return mx.array(generated), logprobs, stats


def generate_text(model, tokenizer, prompt: str, **kwargs):
    prompt_tokens = tokenizer.encode(prompt, return_tensors="mlx")[0]
    tokens, _, stats = generate_tokens(model, prompt_tokens, **kwargs)
    return tokenizer.decode(tokens.tolist()), stats
