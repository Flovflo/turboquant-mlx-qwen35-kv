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

# TurboQuant MLX for Qwen3.5

TurboQuant-inspired KV-cache compression for the exact MLX model [`mlx-community/Qwen3.5-35B-A3B-4bit`](https://huggingface.co/mlx-community/Qwen3.5-35B-A3B-4bit).

If you want something simple to try:

- install it
- load the real Qwen3.5 MLX model
- compare `baseline`, `mlx_quant`, and `turboquant`
- measure actual Apple Silicon results instead of CUDA assumptions

This repo focuses on the runtime KV cache path only. It does not touch the model’s existing MLX 4-bit weights.

## Why test this repo

- Exact target model, not a toy checkpoint.
- Real CLI, real benchmarks, real cache backend.
- Direct inspiration from Google Research TurboQuant.
- Clean side-by-side comparison against current MLX KV quantization.

## Quick results

Same exact model. Same machine class. One backend at a time. Safe for a ~30 GB Apple Silicon machine.

### Best result today

`2048 prompt / 8 gen`, `3 trials`, exact model `mlx-community/Qwen3.5-35B-A3B-4bit`

```text
backend      prompt_tps   generation_tps   gen_wall_s   cache
baseline     514.34       35.67            5.67         80.12 MB
mlx_quant    516.13       38.30            5.16         44.77 MB
turboquant   679.14       44.83            4.20         45.10 MB
```

TurboQuant vs baseline on this benchmark:

- `+32.0%` prompt throughput
- `+25.7%` decode throughput
- `-26.0%` generation wall time
- `-43.7%` KV cache size

TurboQuant vs current MLX KV quantization on this benchmark:

- `+31.6%` prompt throughput
- `+17.1%` decode throughput
- `-18.5%` generation wall time
- cache size within `+0.73%`

### Short-context reality check

`128 prompt / 8 gen`, `3 trials`

```text
backend      prompt_tps   generation_tps   gen_wall_s   cache
baseline     270.47       54.32            0.93         38.17 MB
mlx_quant    274.53       52.20            0.87         33.71 MB
turboquant   266.08       52.65            1.09         33.73 MB
```

At short context, TurboQuant is mainly a memory optimization. At longer context, it starts to behave like the use case described in the Google TurboQuant announcement.

### Longer-context checkpoint

`1024 prompt / 8 gen`, `1 trial`

```text
backend      prompt_tps   generation_tps   gen_wall_s   cache
baseline     378.00       28.98            6.29         59.15 MB
mlx_quant    471.34       49.89            2.57         38.87 MB
turboquant   490.29       50.65            2.43         39.04 MB
```

## Install and try

```bash
python3 -m venv .venv
./.venv/bin/pip install --upgrade pip setuptools wheel
./.venv/bin/pip install -e '.[dev]'
```

Smoke test:

```bash
./.venv/bin/pytest -q
```

Run the exact model:

```bash
./.venv/bin/tqkv generate \
  mlx-community/Qwen3.5-35B-A3B-4bit \
  'Hi, what can you help me with?' \
  --backend baseline \
  --max-tokens 1
```

Run the TurboQuant-inspired backend:

```bash
./.venv/bin/tqkv generate \
  mlx-community/Qwen3.5-35B-A3B-4bit \
  'Hi, what can you help me with?' \
  --backend turboquant \
  --max-tokens 2
```

Run low-RAM benchmarks one backend at a time:

```bash
./.venv/bin/tqkv benchmark \
  mlx-community/Qwen3.5-35B-A3B-4bit \
  --backend baseline \
  --prompt-tokens 128 \
  --generation-tokens 8 \
  --output benchmarks/baseline_128_8.json
```

```bash
./.venv/bin/tqkv benchmark \
  mlx-community/Qwen3.5-35B-A3B-4bit \
  --backend mlx_quant \
  --prompt-tokens 128 \
  --generation-tokens 8 \
  --output benchmarks/mlx_quant_128_8.json
```

```bash
./.venv/bin/tqkv benchmark \
  mlx-community/Qwen3.5-35B-A3B-4bit \
  --backend turboquant \
  --prompt-tokens 128 \
  --generation-tokens 8 \
  --output benchmarks/turboquant_128_8.json
```

Longer-context sanity check:

```bash
./.venv/bin/tqkv benchmark \
  mlx-community/Qwen3.5-35B-A3B-4bit \
  --backend turboquant \
  --prompt-tokens 1024 \
  --generation-tokens 8 \
  --output benchmarks/turboquant_1024_8.json
```

## Backends

- `baseline`: regular KV cache
- `mlx_quant`: existing MLX affine KV quantization
- `turboquant`: TurboQuant-inspired rotated key cache plus residual sketch correction

## Why this repo exists

In the Google Research TurboQuant announcement, Google positions TurboQuant as a runtime compression method for high-dimensional vectors, with KV-cache compression as a primary target. The blog highlights a two-stage design:

1. a high-quality main quantizer after random rotation, based on PolarQuant
2. a 1-bit residual correction based on QJL to remove attention-score bias

According to the Google blog, TurboQuant can:

- compress KV cache very aggressively
- preserve model quality on long-context benchmarks
- reach around 3-bit KV cache compression without fine-tuning
- reduce KV memory by at least 6x on some reported tasks
- improve attention-logit throughput on H100-class GPUs

Source: [Google Research TurboQuant](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/?utm_source=twitter&utm_medium=social&utm_campaign=social_post&utm_content=gr-acct)

This repository is the MLX / Apple Silicon translation of that idea:

- same target problem: KV cache bottlenecks
- different runtime reality: MLX kernels, Apple unified memory, Qwen3.5 mixed full-attention + linear-attention stack
- honest goal: build a runnable TurboQuant-inspired prototype first, then close the gap to the paper

## Why Google shows larger speedups

Google's blog reports speedups for attention-logit computation on long contexts, on H100 GPUs, with a production-grade TurboQuant stack built around PolarQuant and QJL.

This repo is different in four important ways:

- it runs on Apple Silicon with MLX, not CUDA / H100
- it targets the exact `mlx-community/Qwen3.5-35B-A3B-4bit` stack, which mixes full-attention and linear-attention layers
- only the full-attention layers benefit from KV-cache acceleration here
- it is still a `TurboQuant-inspired` prototype, not a faithful PolarQuant + QJL kernel implementation

That means the right comparison is not "why isn't `128` tokens 8x faster?", but "does the backend improve once KV-cache work starts to dominate?".

On this repo today, the answer is:

- short context: mostly memory savings
- longer context: real throughput gains appear
- the remaining gap to the paper is mostly kernel quality and algorithm fidelity, not the high-level direction

What the repo already provides is still valuable:

- a real experimental `TurboQuantKVCache`
- integration with the exact MLX Qwen3.5 model
- reproducible baseline vs MLX-quantized vs TurboQuant-inspired comparisons
- measured Apple Silicon results instead of CUDA/H100 assumptions

## Model target

The exact model integrated here is [`mlx-community/Qwen3.5-35B-A3B-4bit`](https://huggingface.co/mlx-community/Qwen3.5-35B-A3B-4bit).

That matters because this model is not a plain dense decoder:

- it uses Qwen3.5 MoE text architecture
- it mixes linear-attention layers and full-attention layers
- only full-attention layers use the KV cache path targeted by this project

This means the upside from KV compression is real, but structurally smaller than on a model where every layer is a standard full-attention KV layer.

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

## CLI

Help:

```bash
./.venv/bin/tqkv --help
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

Relative to baseline on this run:

- `mlx_quant` reduces KV bytes by about `11.7%`
- `turboquant` reduces KV bytes by about `11.7%`
- `mlx_quant` improves prompt throughput by about `40.7%`
- `turboquant` improves prompt throughput by about `9.4%`
- `turboquant` still trails baseline decode throughput by about `19.5%`

Relative to current MLX KV quantization on this run:

- `turboquant` matches memory almost exactly
- `turboquant` is slower in decode
- `turboquant` still needs kernel- and estimator-level work before it can challenge `mlx_quant`

Saved artifacts:

- `benchmarks/baseline_128_8.json`
- `benchmarks/mlx_quant_128_8.json`
- `benchmarks/turboquant_128_8.json`
- `benchmarks/summary_128_8.json`
- `benchmarks/baseline_128_8_trials3.json`
- `benchmarks/mlx_quant_128_8_trials3.json`
- `benchmarks/turboquant_128_8_trials3.json`

## What the prototype implements

- Reuses `mlx-lm` for model loading and Qwen3.5 text generation.
- Patches Qwen3.5 attention dispatch at runtime so a custom cache backend can be consumed without forking `mlx-lm`.
- Implements `TurboQuantKVCache` as a real backend, not a stub.
- Applies a lightweight structured random transform to keys before the main quantizer.
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

## Roadmap

The next meaningful steps are:

1. reduce decode overhead in `turboquant` so it can beat baseline on Apple Silicon
2. improve the residual estimator so it gets closer to the QJL spirit without breaking MLX efficiency
3. push beyond current MLX KV quantization rather than merely matching its memory footprint
4. extend validation to longer prompts in a RAM-safe way on machines around 30 GB unified memory

## Status

This repo is runnable and loads the exact target model end-to-end. The experimental backend is real, benchmarkable, and clearly labeled `TurboQuant-inspired`.

Today, the project should be read as:

- a serious MLX adaptation of the TurboQuant direction
- not yet a faithful reproduction of the Google algorithm
- not yet better than the existing MLX KV quantization path
- already useful as a clean benchmark and implementation base for pushing TurboQuant-style KV compression further on Apple Silicon
