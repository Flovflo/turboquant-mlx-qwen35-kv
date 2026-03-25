from __future__ import annotations

from mlx_lm import load as load_lm


def load_model(model_id: str, lazy: bool = False):
    return load_lm(
        model_id,
        lazy=lazy,
        return_config=True,
        tokenizer_config={"trust_remote_code": True},
    )
