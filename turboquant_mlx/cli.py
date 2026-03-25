from __future__ import annotations

import json
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
def benchmark(model: str, prompt_tokens: int = 512, generation_tokens: int = 32, kv_bits: int = 4, kv_group_size: int = 64, backend: str = "all", output: Path | None = None):
    loaded_model, _, config = load_model(model)
    backends = ("baseline", "mlx_quant", "turboquant") if backend == "all" else (backend,)
    results = run_benchmark(loaded_model, config, prompt_tokens, generation_tokens, kv_bits, kv_group_size, backends=backends)
    table = Table("backend", "prompt_tps", "generation_tps", "peak_memory_gb", "cache_bytes")
    for row in results:
        table.add_row(row["backend"], f'{row["prompt_tps"]:.2f}', f'{row["generation_tps"]:.2f}', f'{row["peak_memory_gb"]:.3f}', str(row["cache_bytes"]))
    console.print(table)
    if output:
        output.write_text(json.dumps(results, indent=2))
        console.print(f"Saved results to {output}")
