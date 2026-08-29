from __future__ import annotations

import hashlib
import json
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

from ..core.config import NetworkConfig
from ..core.constants import (
    ALLOWED_LORA_TARGET_MODULES,
    BASE_ATTENTION_SIZE,
    BASE_HIDDEN_SIZE,
    BASE_INTERMEDIATE_SIZE,
    BASE_KEY_VALUE_SIZE,
    BASE_MODEL,
    BASE_NUM_HIDDEN_LAYERS,
    MAX_ADAPTER_BYTES,
    MAX_ADAPTER_ELEMENTS,
    MAX_ADAPTER_CONFIG_BYTES,
    MAX_ABS_LORA_VALUE,
    MAX_LORA_ALPHA,
    MAX_LORA_RANK,
    MAX_MANIFEST_BYTES,
    MAX_OUTPUT_TOKENS,
    MAX_ROLLOUT_BYTES,
    MAX_ROLLOUT_LINE_BYTES,
    MAX_TENSOR_DIMENSION,
    MODEL_FILES,
    PROTOCOL_MODEL_COMMITMENT,
    PROTOCOL_MODEL_MANIFEST,
    PROTOCOL_ROLLOUT_MANIFEST,
)
from ..utils.crypto import hash_file
from ..utils.jsonutil import canonical_json_bytes, write_json


_REPO_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}/[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$")
_REVISION = re.compile(r"^[0-9a-f]{40}$")
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_ROW_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,191}$")


def _lora_rank_dimension(shape: tuple[int, int], side: str) -> int:
    if side == "A":
        return shape[0]
    if side == "B":
        return shape[1]
    raise ValueError("LoRA tensor side must be A or B")


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is forbidden: {value}")


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key is forbidden: {key!r}")
        result[key] = value
    return result


def strict_json_bytes(data: bytes, *, max_bytes: int) -> Any:
    if len(data) > max_bytes:
        raise ValueError(f"JSON artifact exceeds {max_bytes} bytes")
    try:
        text = data.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError("artifact is not valid UTF-8") from exc
    return json.loads(
        text,
        parse_constant=_reject_constant,
        object_pairs_hook=_strict_object,
    )


def validate_repo_id(value: str) -> str:
    if not _REPO_ID.fullmatch(value):
        raise ValueError(f"invalid Hugging Face repository id: {value!r}")
    return value


def validate_revision(value: str) -> str:
    value = value.lower()
    if not _REVISION.fullmatch(value):
        raise ValueError("revision must be a full 40-character lowercase commit SHA")
    return value


def validate_digest(value: str) -> str:
    value = value.lower()
    if not _DIGEST.fullmatch(value):
        raise ValueError("digest must be a 64-character lowercase SHA-256")
    return value


@dataclass(frozen=True)
class ModelCommitment:
    model_repo: str
    model_revision: str
    model_digest: str
    rollout_repo: str
    protocol: str = PROTOCOL_MODEL_COMMITMENT

    def validate(self) -> None:
        if self.protocol != PROTOCOL_MODEL_COMMITMENT:
            raise ValueError("unsupported model commitment protocol")
        validate_repo_id(self.model_repo)
        validate_revision(self.model_revision)
        validate_digest(self.model_digest)
        validate_repo_id(self.rollout_repo)

    def compact_dict(self) -> dict[str, str]:
        self.validate()
        return {
            "p": self.protocol,
            "m": self.model_repo,
            "r": self.model_revision,
            "h": self.model_digest,
            "d": self.rollout_repo,
        }

    def to_chain_bytes(self) -> bytes:
        data = canonical_json_bytes(self.compact_dict())
        if len(data) > 512:
            raise ValueError("model commitment exceeds the chain BigRaw limit")
        return data

    @classmethod
    def from_chain_value(cls, value: str | bytes | dict[str, Any]) -> "ModelCommitment":
        if isinstance(value, bytes):
            raw = strict_json_bytes(value, max_bytes=512)
        elif isinstance(value, str):
            raw = strict_json_bytes(value.encode("utf-8"), max_bytes=512)
        elif isinstance(value, dict):
            raw = value
        else:
            raise ValueError("unsupported chain commitment value")
        expected = {"p", "m", "r", "h", "d"}
        if not isinstance(raw, dict) or set(raw) != expected:
            raise ValueError("model commitment has unknown or missing fields")
        result = cls(
            protocol=str(raw["p"]),
            model_repo=str(raw["m"]),
            model_revision=str(raw["r"]),
            model_digest=str(raw["h"]),
            rollout_repo=str(raw["d"]),
        )
        result.validate()
        return result


@dataclass(frozen=True)
class RolloutManifest:
    miner_hotkey: str
    model_repo: str
    model_revision: str
    model_digest: str
    dataset_window: int
    evaluation_seed: str
    evaluation_root: str
    row_count: int
    rows_sha256: str
    base_model: str
    base_revision: str
    max_output_tokens: int
    protocol: str = PROTOCOL_ROLLOUT_MANIFEST

    def validate(self, config: NetworkConfig) -> None:
        if self.protocol != PROTOCOL_ROLLOUT_MANIFEST:
            raise ValueError(
                f"unsupported rollout manifest protocol {self.protocol!r}; "
                f"expected {PROTOCOL_ROLLOUT_MANIFEST!r}"
            )
        if not self.miner_hotkey or len(self.miner_hotkey) > 128:
            raise ValueError("invalid miner hotkey")
        validate_repo_id(self.model_repo)
        validate_revision(self.model_revision)
        validate_digest(self.model_digest)
        validate_digest(self.evaluation_seed)
        validate_digest(self.evaluation_root)
        validate_digest(self.rows_sha256)
        if self.base_model != config.base_model or self.base_revision != config.base_revision:
            raise ValueError("rollout base model does not match network config")
        if self.row_count != config.evaluation_rows:
            raise ValueError(f"rollout must contain exactly {config.evaluation_rows} rows")
        if self.max_output_tokens != config.max_output_tokens:
            raise ValueError(
                f"rollout generation limit must equal {config.max_output_tokens}"
            )
        if self.dataset_window < 0:
            raise ValueError("dataset window must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RolloutManifest":
        value = dict(value)
        known = {field.name for field in cls.__dataclass_fields__.values()}
        if set(value) != known:
            raise ValueError(
                f"rollout manifest has unknown or missing fields: {sorted(set(value) ^ known)}"
            )
        return cls(**value)


def _read_bounded(path: str | Path, max_bytes: int) -> bytes:
    source = Path(path)
    size = source.stat().st_size
    if size > max_bytes:
        raise ValueError(f"{source.name} exceeds the {max_bytes}-byte limit")
    with source.open("rb") as stream:
        data = stream.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ValueError(f"{source.name} exceeds the {max_bytes}-byte limit")
    return data


def validate_adapter_config(path: str | Path, config: NetworkConfig) -> dict[str, Any]:
    raw = strict_json_bytes(_read_bounded(path, MAX_ADAPTER_CONFIG_BYTES), max_bytes=MAX_ADAPTER_CONFIG_BYTES)
    if not isinstance(raw, dict):
        raise ValueError("adapter_config.json must be an object")
    allowed_keys = {
        "alpha_pattern",
        "auto_mapping",
        "base_model_name_or_path",
        "bias",
        "corda_config",
        "eva_config",
        "exclude_modules",
        "fan_in_fan_out",
        "inference_mode",
        "init_lora_weights",
        "layer_replication",
        "layers_pattern",
        "layers_to_transform",
        "loftq_config",
        "lora_alpha",
        "lora_bias",
        "lora_dropout",
        "megatron_config",
        "megatron_core",
        "modules_to_save",
        "peft_type",
        "qalora_group_size",
        "r",
        "rank_pattern",
        "revision",
        "target_modules",
        "task_type",
        "trainable_token_indices",
        "use_dora",
        "use_qalora",
        "use_rslora",
    }
    unknown = set(raw) - allowed_keys
    if unknown:
        raise ValueError(f"adapter config contains unsupported fields: {sorted(unknown)}")
    if raw.get("auto_mapping") not in (None, {}):
        raise ValueError("adapter auto_mapping is forbidden")
    if str(raw.get("peft_type", "LORA")).upper() != "LORA":
        raise ValueError("only LoRA adapters are accepted")
    if str(raw.get("task_type", "CAUSAL_LM")).upper() != "CAUSAL_LM":
        raise ValueError("adapter task_type must be CAUSAL_LM")
    base = raw.get("base_model_name_or_path")
    if str(base or "") != BASE_MODEL:
        raise ValueError("adapter was created for a different base model")
    if raw.get("revision") not in (None, config.base_revision):
        raise ValueError("adapter revision does not match the fixed base revision")
    rank = raw.get("r")
    if isinstance(rank, bool) or not isinstance(rank, int) or not 1 <= rank <= min(config.max_lora_rank, MAX_LORA_RANK):
        raise ValueError("adapter rank is missing or exceeds the network limit")
    targets = raw.get("target_modules")
    if (
        not isinstance(targets, list)
        or not targets
        or any(not isinstance(item, str) or not item for item in targets)
        or len(targets) != len(set(targets))
    ):
        raise ValueError("adapter target_modules must be a non-empty list")
    if any(item != item.rsplit(".", 1)[-1] for item in targets):
        raise ValueError("adapter target_modules must use protocol-owned module suffixes")
    normalized_targets = set(targets)
    forbidden = normalized_targets - ALLOWED_LORA_TARGET_MODULES
    if forbidden:
        raise ValueError(f"adapter contains forbidden target modules: {sorted(forbidden)}")
    if raw.get("modules_to_save") not in (None, []):
        raise ValueError("modules_to_save is forbidden; submit LoRA weights only")
    if str(raw.get("bias", "none")).lower() != "none":
        raise ValueError("LoRA bias weights are forbidden")
    if raw.get("lora_bias") not in (None, False):
        raise ValueError("lora_bias is forbidden")
    if raw.get("fan_in_fan_out") not in (None, False):
        raise ValueError("fan_in_fan_out is forbidden")
    if raw.get("use_rslora") not in (None, False):
        raise ValueError("RSLoRA is not supported")
    if raw.get("exclude_modules") not in (None, []):
        raise ValueError("exclude_modules is forbidden")
    if raw.get("use_dora") not in (None, False) or raw.get("use_qalora") not in (None, False):
        raise ValueError("DoRA and QALoRA adapters are not supported")
    if raw.get("rank_pattern") not in (None, {}) or raw.get("alpha_pattern") not in (None, {}):
        raise ValueError("per-module rank and alpha patterns are forbidden")
    if raw.get("layers_to_transform") not in (None, []) or raw.get("layers_pattern") not in (None, []):
        raise ValueError("layer transformation selectors are forbidden")
    if raw.get("layer_replication") not in (None, []):
        raise ValueError("layer replication is forbidden")
    if raw.get("trainable_token_indices") not in (None, []):
        raise ValueError("trainable token indices are forbidden")
    for name in ("megatron_config", "corda_config", "eva_config"):
        if raw.get(name) not in (None, {}):
            raise ValueError(f"{name} is forbidden")
    if raw.get("loftq_config") not in (None, {}):
        raise ValueError("LoftQ configuration is forbidden")
    dropout = raw.get("lora_dropout", 0.0)
    if isinstance(dropout, bool) or not isinstance(dropout, (int, float)) or float(dropout) != 0.0:
        raise ValueError("lora_dropout must be zero")
    alpha = raw.get("lora_alpha", rank)
    if (
        isinstance(alpha, bool)
        or not isinstance(alpha, (int, float))
        or not 0 < float(alpha) <= MAX_LORA_ALPHA
    ):
        raise ValueError("lora_alpha is invalid or exceeds the protocol limit")
    return raw


def validate_safetensors(
    path: str | Path,
    *,
    max_rank: int = MAX_LORA_RANK,
    expected_rank: int | None = None,
    expected_targets: set[str] | None = None,
    semantic_hasher: Any | None = None,
) -> dict[str, Any]:
    source = Path(path)
    size = source.stat().st_size
    if size <= 0 or size > MAX_ADAPTER_BYTES:
        raise ValueError("adapter_model.safetensors is empty or too large")
    try:
        from safetensors import safe_open
        import torch
    except ImportError as exc:
        raise RuntimeError("SafeTensors validation requires the 'safetensors' package") from exc
    tensor_count = 0
    total_elements = 0
    pairs: dict[str, dict[str, Any]] = {}
    seen_targets: set[str] = set()
    allowed_dtypes = {"F16", "BF16", "F32"}
    with safe_open(source, framework="pt", device="cpu") as tensors:
        keys = list(tensors.keys())
        if not keys or len(keys) > 1_024:
            raise ValueError("invalid number of tensors in adapter")
        if any(not isinstance(key, str) or len(key) > 512 or ".." in key for key in keys):
            raise ValueError("invalid tensor name")
        canonical_keys = [
            (
                re.sub(
                    r"\.lora_([AB])(?:\.[A-Za-z0-9_-]+)?\.weight$",
                    r".lora_\1.weight",
                    key,
                ),
                key,
            )
            for key in keys
        ]
        if len({canonical for canonical, _ in canonical_keys}) != len(canonical_keys):
            raise ValueError("adapter contains duplicate canonical LoRA tensor names")
        for canonical_key, key in sorted(canonical_keys):
            view = tensors.get_slice(key)
            shape = tuple(int(item) for item in view.get_shape())
            dtype = str(view.get_dtype())
            if len(shape) != 2 or any(item <= 0 or item > MAX_TENSOR_DIMENSION for item in shape):
                raise ValueError(f"forbidden tensor shape for {key!r}: {shape}")
            match = re.fullmatch(
                r"(?:base_model\.model\.)?model\.layers\.(\d+)\."
                r"(self_attn|mlp)\.([A-Za-z0-9_]+)\.lora_([AB])"
                r"(?:\.[A-Za-z0-9_-]+)?\.weight",
                key,
            )
            if not match or match.group(3) not in ALLOWED_LORA_TARGET_MODULES:
                raise ValueError(f"forbidden LoRA tensor name: {key!r}")
            layer, block, target, side = (
                int(match.group(1)),
                match.group(2),
                match.group(3),
                match.group(4),
            )
            if layer >= BASE_NUM_HIDDEN_LAYERS:
                raise ValueError(f"LoRA tensor layer is outside the base model: {key!r}")
            expected_block = "self_attn" if target in {"q_proj", "k_proj", "v_proj", "o_proj"} else "mlp"
            if block != expected_block:
                raise ValueError(f"LoRA tensor uses the wrong base-model block: {key!r}")
            seen_targets.add(target)
            pair_name = re.sub(
                r"\.lora_[AB](?:\.[A-Za-z0-9_-]+)?\.weight$",
                "",
                key,
            )
            pair = pairs.setdefault(pair_name, {"target": target})
            if side in pair:
                raise ValueError(f"duplicate LoRA {side} tensor for {pair_name!r}")
            pair[side] = shape
            rank_dimension = _lora_rank_dimension(shape, side)
            if rank_dimension > max_rank:
                raise ValueError(f"LoRA tensor {key!r} exceeds the rank limit")
            if expected_rank is not None and rank_dimension != expected_rank:
                raise ValueError(f"LoRA tensor {key!r} does not match adapter rank {expected_rank}")
            if dtype not in allowed_dtypes:
                raise ValueError(f"forbidden tensor dtype for {key!r}: {dtype}")
            # SafeTensors prevents code execution, but NaN/Inf payloads can
            # still poison kernels or make inference verification undefined.
            # Reading at most the protocol's 256 MiB limit on CPU is an
            # intentional validation cost before any miner tensor reaches a GPU.
            tensor = tensors.get_tensor(key)
            if not bool(tensor.isfinite().all().item()):
                raise ValueError(f"LoRA tensor {key!r} contains a non-finite value")
            if float(tensor.detach().abs().max().item()) > MAX_ABS_LORA_VALUE:
                raise ValueError(f"LoRA tensor {key!r} exceeds the absolute value limit")
            if semantic_hasher is not None:
                semantic_hasher.update(b"tensor\0" + canonical_key.encode("utf-8") + b"\0")
                semantic_hasher.update(dtype.encode("ascii") + b"\0")
                semantic_hasher.update(
                    (",".join(str(dimension) for dimension in shape)).encode("ascii") + b"\0"
                )
                # Viewing as uint8 preserves BF16/F16/F32 value bits without
                # conversion; the tensor came from the inert SafeTensors reader.
                value_bytes = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
                semantic_hasher.update(str(len(value_bytes)).encode("ascii") + b"\0")
                semantic_hasher.update(value_bytes)
            elements = 1
            for dimension in shape:
                elements *= dimension
            total_elements += elements
            if total_elements > MAX_ADAPTER_ELEMENTS:
                raise ValueError("adapter tensor element count exceeds the protocol limit")
            tensor_count += 1
    module_dimensions = {
        "q_proj": (BASE_ATTENTION_SIZE, BASE_HIDDEN_SIZE),
        "k_proj": (BASE_KEY_VALUE_SIZE, BASE_HIDDEN_SIZE),
        "v_proj": (BASE_KEY_VALUE_SIZE, BASE_HIDDEN_SIZE),
        "o_proj": (BASE_HIDDEN_SIZE, BASE_ATTENTION_SIZE),
        "gate_proj": (BASE_INTERMEDIATE_SIZE, BASE_HIDDEN_SIZE),
        "up_proj": (BASE_INTERMEDIATE_SIZE, BASE_HIDDEN_SIZE),
        "down_proj": (BASE_HIDDEN_SIZE, BASE_INTERMEDIATE_SIZE),
    }
    for pair_name, pair in pairs.items():
        if {name for name in pair if name in {"A", "B"}} != {"A", "B"}:
            raise ValueError(f"LoRA module {pair_name!r} must contain exactly one A/B tensor pair")
        if pair["A"][0] != pair["B"][1]:
            raise ValueError(f"LoRA module {pair_name!r} has inconsistent A/B rank dimensions")
        out_features, in_features = module_dimensions[str(pair["target"])]
        rank = pair["A"][0]
        if pair["A"] != (rank, in_features) or pair["B"] != (out_features, rank):
            raise ValueError(
                f"LoRA module {pair_name!r} does not match the pinned base-model shape"
            )
    if expected_targets is not None and seen_targets != expected_targets:
        raise ValueError(
            "LoRA tensor targets do not match adapter_config.json: "
            f"expected {sorted(expected_targets)}, received {sorted(seen_targets)}"
        )
    return {"tensor_count": tensor_count, "total_elements": total_elements, "bytes": size}


def prepare_runtime_adapter(
    source_dir: str | Path,
    runtime_dir: str | Path,
    config: NetworkConfig,
) -> Path:
    """Create the only LoRA directory that is allowed to reach vLLM.

    The miner JSON is validation input, never runtime configuration. This
    protocol-owned minimal config prevents optional PEFT fields from changing
    loader behavior while retaining the committed tensor bytes.
    """

    source = Path(source_dir)
    target = Path(runtime_dir)
    validated = validate_adapter_config(source / MODEL_FILES[0], config)
    validate_safetensors(
        source / MODEL_FILES[1],
        max_rank=config.max_lora_rank,
        expected_rank=int(validated["r"]),
        expected_targets={str(item).rsplit(".", 1)[-1] for item in validated["target_modules"]},
    )
    target.mkdir(parents=True, exist_ok=True)
    weights = source / MODEL_FILES[1]
    if weights.is_symlink() or not weights.is_file():
        raise ValueError("adapter weights must be a regular file")
    shutil.copyfile(weights, target / MODEL_FILES[1])
    runtime_config = runtime_adapter_config(validated, config)
    write_json(target / MODEL_FILES[0], runtime_config)
    return target


def runtime_adapter_config(validated: dict[str, Any], config: NetworkConfig) -> dict[str, Any]:
    """Canonical semantic LoRA configuration used for inference and identity."""

    return {
        "base_model_name_or_path": config.base_model,
        "bias": "none",
        "inference_mode": True,
        "lora_alpha": float(validated.get("lora_alpha", validated["r"])),
        "lora_dropout": 0.0,
        "peft_type": "LORA",
        "r": int(validated["r"]),
        "target_modules": sorted(validated["target_modules"]),
        "task_type": "CAUSAL_LM",
        "use_rslora": False,
    }


def adapter_digest(directory: str | Path, config: NetworkConfig) -> str:
    root = Path(directory)
    validated = validate_adapter_config(root / MODEL_FILES[0], config)
    hasher = hashlib.sha256()
    hasher.update(b"feval/adapter/v3\0")
    semantic_config = canonical_json_bytes(runtime_adapter_config(validated, config))
    hasher.update(MODEL_FILES[0].encode("ascii") + b"\0")
    hasher.update(str(len(semantic_config)).encode("ascii") + b"\0" + semantic_config)
    hasher.update(MODEL_FILES[1].encode("ascii") + b"\0")
    validate_safetensors(
        root / MODEL_FILES[1],
        max_rank=config.max_lora_rank,
        expected_rank=int(validated["r"]),
        expected_targets={str(item).rsplit(".", 1)[-1] for item in validated["target_modules"]},
        semantic_hasher=hasher,
    )
    return hasher.hexdigest()


def model_manifest(model_repo: str, model_digest: str, config: NetworkConfig) -> dict[str, Any]:
    validate_repo_id(model_repo)
    validate_digest(model_digest)
    return {
        "protocol": PROTOCOL_MODEL_MANIFEST,
        "base_model": config.base_model,
        "base_revision": config.base_revision,
        "model_repo": model_repo,
        "model_digest": model_digest,
        "max_lora_rank": config.max_lora_rank,
        "files": list(MODEL_FILES),
    }


def _validate_rollout_row(value: Any, *, max_output_tokens: int, vocab_size: int | None) -> dict[str, Any]:
    minimal = {"row_id", "tokens"}
    legacy = minimal | {"finish_reason", "stop_reason", "max_tokens"}
    if not isinstance(value, dict) or frozenset(value) not in {frozenset(minimal), frozenset(legacy)}:
        raise ValueError("rollout row has unknown or missing fields")
    row_id = value["row_id"]
    if not isinstance(row_id, str) or not _ROW_ID.fullmatch(row_id):
        raise ValueError("invalid rollout row_id")
    tokens = value["tokens"]
    if not isinstance(tokens, list):
        raise ValueError(f"rollout {row_id!r} tokens must be a list")
    # The miner does not define its own generation or termination limit. Both
    # scoring and auditing use this same protocol-owned 2K prefix.
    tokens = tokens[:max_output_tokens]
    for token in tokens:
        if isinstance(token, bool) or not isinstance(token, int) or token < 0:
            raise ValueError(f"rollout {row_id!r} has an invalid token id")
        if vocab_size is not None and token >= vocab_size:
            raise ValueError(f"rollout {row_id!r} has an out-of-vocabulary token")
    return {"row_id": row_id, "tokens": tokens}


def iter_rollouts(
    path: str | Path,
    *,
    max_output_tokens: int = MAX_OUTPUT_TOKENS,
    vocab_size: int | None = None,
) -> Iterator[dict[str, Any]]:
    source = Path(path)
    if source.stat().st_size > MAX_ROLLOUT_BYTES:
        raise ValueError("rollouts.jsonl exceeds the protocol size limit")
    with source.open("rb") as stream:
        for line_number, line in enumerate(stream, start=1):
            if len(line) > MAX_ROLLOUT_LINE_BYTES:
                raise ValueError(f"rollout line {line_number} exceeds the size limit")
            if not line.strip():
                raise ValueError(f"blank rollout line at {line_number}")
            value = strict_json_bytes(line, max_bytes=MAX_ROLLOUT_LINE_BYTES)
            yield _validate_rollout_row(
                value,
                max_output_tokens=max_output_tokens,
                vocab_size=vocab_size,
            )


def load_rollouts_strict(
    path: str | Path,
    *,
    expected_row_ids: Iterable[str],
    max_output_tokens: int,
    vocab_size: int | None = None,
) -> list[dict[str, Any]]:
    expected = list(expected_row_ids)
    if len(expected) != len(set(expected)):
        raise ValueError("evaluation set contains duplicate row IDs")
    rows = list(
        iter_rollouts(
            path,
            max_output_tokens=max_output_tokens,
            vocab_size=vocab_size,
        )
    )
    actual = [row["row_id"] for row in rows]
    if actual != expected:
        if len(actual) != len(expected):
            raise ValueError(f"expected {len(expected)} rollout rows, received {len(actual)}")
        raise ValueError("rollout row IDs or order do not exactly match the evaluation set")
    return rows


def write_rollouts(path: str | Path, rows: Iterable[dict[str, Any]]) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    hasher = hashlib.sha256()
    with target.open("wb") as stream:
        for row in rows:
            normalized = _validate_rollout_row(
                row,
                max_output_tokens=MAX_OUTPUT_TOKENS,
                vocab_size=None,
            )
            line = canonical_json_bytes(normalized) + b"\n"
            if len(line) > MAX_ROLLOUT_LINE_BYTES:
                raise ValueError("rollout row exceeds the serialized size limit")
            hasher.update(line)
            stream.write(line)
    if target.stat().st_size > MAX_ROLLOUT_BYTES:
        raise ValueError("rollout artifact exceeds the protocol size limit")
    return hasher.hexdigest()


def load_rollout_manifest(path: str | Path, config: NetworkConfig) -> RolloutManifest:
    value = strict_json_bytes(_read_bounded(path, MAX_MANIFEST_BYTES), max_bytes=MAX_MANIFEST_BYTES)
    if not isinstance(value, dict):
        raise ValueError("rollout manifest must be an object")
    manifest = RolloutManifest.from_dict(value)
    manifest.validate(config)
    return manifest


def validate_rollout_bundle(
    directory: str | Path,
    *,
    config: NetworkConfig,
    expected_row_ids: Iterable[str],
    vocab_size: int | None = None,
) -> tuple[RolloutManifest, list[dict[str, Any]]]:
    root = Path(directory)
    manifest = load_rollout_manifest(root / "manifest.json", config)
    rollout_path = root / "rollouts.jsonl"
    if hash_file(rollout_path) != manifest.rows_sha256:
        raise ValueError("rollouts.jsonl hash does not match manifest")
    rows = load_rollouts_strict(
        rollout_path,
        expected_row_ids=expected_row_ids,
        max_output_tokens=manifest.max_output_tokens,
        vocab_size=vocab_size,
    )
    return manifest, rows

