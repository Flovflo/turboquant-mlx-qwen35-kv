---
license: apache-2.0
library_name: mlx
base_model: mlx-community/Qwen3.5-35B-A3B-4bit
tags:
  - mlx
  - qwen3_5_moe
  - kv-cache
  - quantization
  - turboquant
---

# TurboQuant MLX Prototype

Experimental KV-cache prototype for MLX models with three backends:

- `baseline`: regular KV cache
- `mlx_quant`: existing MLX affine KV quantization
- `turboquant`: TurboQuant-inspired rotated key cache plus residual sketch correction

This project targets `mlx-community/Qwen3.5-35B-A3B-4bit` without changing its existing MLX 4-bit weights.

## RAM safety

The exact target model is large. On a machine with around 30 GB unified memory:

- prefer short prompts for smoke runs
- do not launch multiple benchmarks in parallel
- keep `generation_tokens` small when validating the exact model
- start with `baseline` or `mlx_quant` before `turboquant`

## Mandatory sources inspected

- Model card: [mlx-community/Qwen3.5-35B-A3B-4bit](https://huggingface.co/mlx-community/Qwen3.5-35B-A3B-4bit)
- TurboQuant blog: [Google Research TurboQuant](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/?utm_source=twitter&utm_medium=social&utm_campaign=social_post&utm_content=gr-acct)

## Repo layout

```text
turboquant_mlx/
  attention.py
  benchmark.py
  cache.py
  cli.py
  generation.py
  loaders.py
  packing.py
  projection.py
  runtime.py
tests/
benchmarks/
```

## Setup

```bash
python3 -m venv .venv
./.venv/bin/pip install --upgrade pip setuptools wheel
./.venv/bin/pip install -e '.[dev]'
```

## Smoke tests

```bash
./.venv/bin/pytest -q
```

Result on this machine:

```text
2 passed, 2 warnings in 2.01s
```

## CLI

Help:

```bash
./.venv/bin/tqkv --help
```

Text generation on the exact target model:

```bash
./.venv/bin/tqkv generate \
  mlx-community/Qwen3.5-35B-A3B-4bit \
  'Hi, what can you help me with?' \
  --backend baseline \
  --max-tokens 1
```

TurboQuant-inspired generation:

```bash
./.venv/bin/tqkv generate \
  mlx-community/Qwen3.5-35B-A3B-4bit \
  'Hi, what can you help me with?' \
  --backend turboquant \
  --max-tokens 2
```

## Benchmarks

32 prompt tokens, 4 generation tokens:

```bash
./.venv/bin/tqkv benchmark \
  mlx-community/Qwen3.5-35B-A3B-4bit \
  --prompt-tokens 32 \
  --generation-tokens 4 \
  --output benchmarks/benchmark_32_4.json
```

Observed result:

```text
backend      prompt_tps   generation_tps   peak_memory_gb   cache_bytes
baseline     11.51        22.07            19.594           38174720
mlx_quant    101.24       25.70            19.595           33133440
turboquant   109.21       9.80             19.993           62779096
```

128 prompt tokens, 8 generation tokens:

```bash
./.venv/bin/tqkv benchmark \
  mlx-community/Qwen3.5-35B-A3B-4bit \
  --prompt-tokens 128 \
  --generation-tokens 8 \
  --output benchmarks/benchmark_128_8.json
```

Observed result:

```text
backend      prompt_tps   generation_tps   peak_memory_gb   cache_bytes
baseline     52.27        27.96            19.750           38174720
mlx_quant    236.25       37.59            19.750           33709440
turboquant   217.10       10.15            20.015           63375096
```

## What the prototype implements

- Reuses `mlx-lm` for model loading and Qwen3.5 text generation.
- Patches Qwen3.5 attention dispatch at runtime so a custom cache backend can be consumed without forking `mlx-lm`.
- Implements `TurboQuantKVCache` as a real backend, not a stub.
- Applies a fixed random orthogonal transform to keys before the main quantizer.
- Uses affine MLX quantization for the main compressed key representation.
- Stores a 1-bit residual sign sketch plus a residual RMS term for score correction.
- Optionally quantizes values with the same MLX affine path.
- Benchmarks `baseline`, `mlx_quant`, and `turboquant` on the exact model.

## Deviations from paper-faithful TurboQuant

- The main quantizer is MLX affine quantization, not PolarQuant.
- The residual correction is a simple packed sign sketch with RMS scaling, not a faithful QJL estimator.
- The implementation patches only the Qwen3.5 attention path loaded via `mlx-lm`, not every model family in the MLX ecosystem.
- The residual sketch currently improves prefill throughput in this prototype but hurts decode throughput.
- The current experimental backend does not beat MLX's built-in KV quantization on memory or decode speed.
- Cache bytes are reported honestly from the live cache state. For short prompts they can stay flat because upstream `KVCache` grows in 256-token blocks.

## Status

This repo is runnable and loads the exact target model end-to-end. The experimental backend is real, benchmarkable, and clearly labeled `TurboQuant-inspired`, but it is not yet a performance win over the existing MLX KV quantization path.
