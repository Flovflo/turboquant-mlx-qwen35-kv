import mlx.core as mx
from mlx_lm.models import qwen3_5

from turboquant_mlx.generation import generate_tokens


def tiny_model():
    args = qwen3_5.TextModelArgs(
        model_type="qwen3_5",
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=1,
        vocab_size=256,
        head_dim=32,
        full_attention_interval=2,
        num_experts=0,
        max_position_embeddings=512,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
    )
    return qwen3_5.TextModel(args)


def test_generate_baseline_and_turboquant():
    model = tiny_model()
    prompt = mx.array([1, 2, 3, 4, 5], dtype=mx.int32)
    base_tokens, _, base_stats = generate_tokens(model, prompt, max_tokens=2, backend="baseline")
    tq_tokens, _, tq_stats = generate_tokens(model, prompt, max_tokens=2, backend="turboquant")
    assert base_tokens.shape == tq_tokens.shape == (2,)
    assert base_stats.cache_bytes > 0
    assert tq_stats.cache_bytes > 0
