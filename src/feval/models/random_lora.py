from __future__ import annotations

from pathlib import Path
from typing import Any

from ..core.config import NetworkConfig
from ..core.constants import BASE_MODEL, MAX_LORA_RANK
from ..utils.jsonutil import write_json


DEFAULT_RANDOM_TARGET_MODULES = ("q_proj", "v_proj")


def _nested_configs(model_config: Any) -> list[Any]:
    configs = [model_config]
    get_text_config = getattr(model_config, "get_text_config", None)
    if callable(get_text_config):
        try:
            configs.append(get_text_config())
        except TypeError:
            pass
    for name in ("text_config", "language_config", "llm_config", "model_config", "decoder_config"):
        nested = getattr(model_config, name, None)
        if nested is not None:
            configs.append(nested)
    return configs


def _get_config_int(configs: list[Any], names: tuple[str, ...], *, default: int | None = None) -> int:
    for config in configs:
        for name in names:
            value = getattr(config, name, None)
            if value is not None:
                return int(value)
            if isinstance(config, dict) and config.get(name) is not None:
                return int(config[name])
    if default is not None:
        return default
    raise ValueError(f"base model config is missing one of: {', '.join(names)}")


def _base_dimensions(config: NetworkConfig) -> dict[str, int]:
    try:
        from transformers import AutoConfig
    except ImportError as exc:
        raise RuntimeError("random LoRA generation requires the 'transformers' package") from exc
    model_config = AutoConfig.from_pretrained(
        config.base_model,
        revision=config.base_revision,
        trust_remote_code=False,
    )
    configs = _nested_configs(model_config)
    hidden = _get_config_int(configs, ("hidden_size", "d_model", "n_embd"))
    intermediate = _get_config_int(configs, ("intermediate_size", "ffn_hidden_size", "n_inner"))
    layers = _get_config_int(configs, ("num_hidden_layers", "num_layers", "n_layer"))
    attention_heads = _get_config_int(configs, ("num_attention_heads", "n_head"))
    key_value_heads = _get_config_int(
        configs,
        ("num_key_value_heads", "num_kv_heads", "n_head_kv"),
        default=attention_heads,
    )
    head_dim = _get_config_int(configs, ("head_dim", "attention_head_dim"), default=hidden // attention_heads)
    return {
        "hidden_size": hidden,
        "attention_size": attention_heads * head_dim,
        "intermediate_size": intermediate,
        "num_hidden_layers": layers,
        "key_value_size": key_value_heads * head_dim,
    }


def _module_shape(module: str, dims: dict[str, int]) -> tuple[int, int]:
    hidden = dims["hidden_size"]
    intermediate = dims["intermediate_size"]
    key_value = dims["key_value_size"]
    attention = dims["attention_size"]
    if module == "q_proj":
        return attention, hidden
    if module in {"k_proj", "v_proj"}:
        return key_value, hidden
    if module == "o_proj":
        return hidden, attention
    if module in {"gate_proj", "up_proj"}:
        return intermediate, hidden
    if module == "down_proj":
        return hidden, intermediate
    raise ValueError(f"random LoRA generation does not know module shape for {module!r}")


def _module_prefix(layer: int, module: str) -> str:
    block = "self_attn" if module in {"q_proj", "k_proj", "v_proj", "o_proj"} else "mlp"
    return f"base_model.model.model.layers.{layer}.{block}.{module}"


def write_random_lora(
    *,
    out_dir: str | Path,
    config: NetworkConfig,
    rank: int = 4,
    alpha: float | None = None,
    target_modules: tuple[str, ...] = DEFAULT_RANDOM_TARGET_MODULES,
    layers: int | None = None,
    seed: int = 0,
    scale: float = 0.01,
) -> dict[str, Any]:
    if rank <= 0 or rank > min(config.max_lora_rank, MAX_LORA_RANK):
        raise ValueError(f"rank must be between 1 and {min(config.max_lora_rank, MAX_LORA_RANK)}")
    if scale <= 0.0:
        raise ValueError("scale must be positive")
    if not target_modules:
        raise ValueError("at least one target module is required")
    dims = _base_dimensions(config)
    layer_count = dims["num_hidden_layers"] if layers is None else int(layers)
    if layer_count <= 0 or layer_count > dims["num_hidden_layers"]:
        raise ValueError(f"layers must be between 1 and {dims['num_hidden_layers']}")

    try:
        import torch
        from safetensors.torch import save_file
    except ImportError as exc:
        raise RuntimeError("random LoRA generation requires 'torch' and 'safetensors'") from exc

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    tensors = {}
    for layer in range(layer_count):
        for module in target_modules:
            out_features, in_features = _module_shape(module, dims)
            prefix = _module_prefix(layer, module)
            tensors[f"{prefix}.lora_A.weight"] = torch.randn(
                (rank, in_features),
                generator=generator,
                dtype=torch.float16,
            ) * float(scale)
            # Standard zero-impact LoRA initialization: delta = B @ A, so a
            # zero B makes the adapter exactly equivalent to the base model.
            # A stays seeded-random to keep this a deterministic test adapter.
            tensors[f"{prefix}.lora_B.weight"] = torch.zeros(
                (out_features, rank),
                dtype=torch.float16,
            )

    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    adapter_config = {
        "base_model_name_or_path": BASE_MODEL,
        "revision": config.base_revision,
        "bias": "none",
        "inference_mode": True,
        "lora_alpha": float(alpha if alpha is not None else rank),
        "lora_dropout": 0.0,
        "peft_type": "LORA",
        "r": int(rank),
        "target_modules": list(target_modules),
        "task_type": "CAUSAL_LM",
    }
    write_json(root / "adapter_config.json", adapter_config)
    save_file(tensors, root / "adapter_model.safetensors")
    return {
        "out_dir": str(root),
        "rank": rank,
        "alpha": adapter_config["lora_alpha"],
        "target_modules": list(target_modules),
        "layers": layer_count,
        "tensors": len(tensors),
        "zero_effect": True,
    }

