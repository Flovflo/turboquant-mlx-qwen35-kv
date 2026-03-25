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

Recommended low-RAM benchmark flow: one backend at a time.

Baseline:

```bash
./.venv/bin/tqkv benchmark \
  mlx-community/Qwen3.5-35B-A3B-4bit \
  --backend baseline \
  --prompt-tokens 128 \
  --generation-tokens 8 \
  --output benchmarks/baseline_128_8.json
```

MLX quantized KV:

```bash
./.venv/bin/tqkv benchmark \
  mlx-community/Qwen3.5-35B-A3B-4bit \
  --backend mlx_quant \
  --prompt-tokens 128 \
  --generation-tokens 8 \
  --output benchmarks/mlx_quant_128_8.json
```

TurboQuant-inspired KV:

```bash
./.venv/bin/tqkv benchmark \
  mlx-community/Qwen3.5-35B-A3B-4bit \
  --backend turboquant \
  --prompt-tokens 128 \
  --generation-tokens 8 \
  --output benchmarks/turboquant_128_8.json
```

Observed sequential results on this machine:

```text
backend      prompt_tps   generation_tps   peak_memory_gb   cache_bytes
baseline     46.51        38.18            19.750           38174720
mlx_quant    65.42        36.97            19.750           33709440
turboquant   50.87        30.73            19.709           33717540
```

Saved artifacts:

- `benchmarks/baseline_128_8.json`
- `benchmarks/mlx_quant_128_8.json`
- `benchmarks/turboquant_128_8.json`
- `benchmarks/summary_128_8.json`

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
- The residual sketch is deliberately lightweight for MLX and is not a faithful QJL estimator.
- The current experimental backend is now close to `mlx_quant` in memory footprint, but still trails it in decode throughput.
- Cache bytes are reported honestly from the live cache state. For short prompts they can stay flat because upstream `KVCache` grows in 256-token blocks.

## Status

This repo is runnable and loads the exact target model end-to-end. The experimental backend is real, benchmarkable, and clearly labeled `TurboQuant-inspired`, but it is not yet a performance win over the existing MLX KV quantization path.
