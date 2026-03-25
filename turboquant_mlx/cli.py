from __future__ import annotations

import json
import time
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .benchmark import run_benchmark
from .generation import generate_text
from .loaders import load_model

app = typer.Typer(no_args_is_help=True)
console = Console()


@app.command()
def generate(model: str, prompt: str, backend: str = "baseline", max_tokens: int = 16, kv_bits: int = 4, kv_group_size: int = 64):
    loaded_model, tokenizer, _ = load_model(model)
    text, stats = generate_text(
        loaded_model,
        tokenizer,
        prompt,
        backend=backend,
        max_tokens=max_tokens,
        kv_bits=kv_bits,
        kv_group_size=kv_group_size,
    )
    console.print(text)
    console.print(stats)


@app.command()
def benchmark(model: str, prompt_tokens: int = 512, generation_tokens: int = 32, kv_bits: int = 4, kv_group_size: int = 64, backend: str = "all", trials: int = 1, output: Path | None = None):
    load_start = time.perf_counter()
    loaded_model, _, config = load_model(model)
    load_time_s = time.perf_counter() - load_start
    backends = ("baseline", "mlx_quant", "turboquant") if backend == "all" else (backend,)
    results = run_benchmark(
        loaded_model,
        config,
        prompt_tokens,
        generation_tokens,
        kv_bits,
        kv_group_size,
        backends=backends,
        trials=trials,
    )
    for row in results:
        row["model_load_time_s"] = load_time_s
    table = Table("backend", "prompt_tps", "generation_tps", "gen_wall_s", "load_s", "peak_memory_gb", "cache_bytes")
    for row in results:
        table.add_row(
            row["backend"],
            f'{row["prompt_tps"]:.2f}',
            f'{row["generation_tps"]:.2f}',
            f'{row["generation_wall_time_s"]:.2f}',
            f'{row["model_load_time_s"]:.2f}',
            f'{row["peak_memory_gb"]:.3f}',
            str(row["cache_bytes"]),
        )
    console.print(table)
    if output:
        output.write_text(json.dumps(results, indent=2))
        console.print(f"Saved results to {output}")
