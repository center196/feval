from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .constants import (
    AUDIT_DETECTION_CONFIDENCE,
    AUDIT_DELAY_BLOCKS,
    AUDIT_MAX_TOKEN_RANK,
    AUDIT_MIN_EXACT_ARGMAX_RATIO,
    AUDIT_MIN_FAKE_ROW_FRACTION,
    AUDIT_MIN_RELATIVE_PROBABILITY,
    AUDIT_ROWS_PER_ROUND,
    AUDIT_TOTAL_ROUNDS,
    BASE_MODEL,
    BASE_MODEL_REVISION,
    BLACKLIST_ENABLED,
    BLACKLIST_DURATION_BLOCKS,
    BURN_SHARE,
    CHALLENGER_SHARE,
    CHAMPION_COUNT,
    MINER_SHARE,
    DATASET_WINDOW_BLOCKS,
    EVALUATION_ROWS,
    INVALID_ROUNDS_BEFORE_BLACKLIST,
    INSTRUCTION_DATASET,
    INSTRUCTION_DATASET_REVISION,
    INSTRUCTION_ROWS,
    MATH_DATASET,
    MATH_DATASET_REVISION,
    MATH_ROWS,
    MAX_CONTEXT_TOKENS,
    MAX_LORA_RANK,
    MAX_OUTPUT_TOKENS,
    MAX_PROMPT_CHARS,
    PROTOCOL_NETWORK,
    PROMOTION_CONFIDENCE_Z,
    PROMOTION_DELTA_MIN,
    SUBNET_NETUID,
    WEIGHT_INTERVAL_BLOCKS,
)
from ..utils.jsonutil import load_json, write_json


@dataclass(frozen=True)
class NetworkConfig:
    protocol: str = PROTOCOL_NETWORK
    netuid: int = SUBNET_NETUID
    base_model: str = BASE_MODEL
    base_revision: str = BASE_MODEL_REVISION
    math_dataset: str = MATH_DATASET
    math_revision: str = MATH_DATASET_REVISION
    instruction_dataset: str = INSTRUCTION_DATASET
    instruction_revision: str = INSTRUCTION_DATASET_REVISION
    split: str = "train"
    evaluation_rows: int = EVALUATION_ROWS
    math_rows: int = MATH_ROWS
    instruction_rows: int = INSTRUCTION_ROWS
    dataset_window_blocks: int = DATASET_WINDOW_BLOCKS
    weight_interval_blocks: int = WEIGHT_INTERVAL_BLOCKS
    audit_delay_blocks: int = AUDIT_DELAY_BLOCKS
    audit_rows_per_round: int = AUDIT_ROWS_PER_ROUND
    audit_min_fake_row_fraction: float = AUDIT_MIN_FAKE_ROW_FRACTION
    audit_detection_confidence: float = AUDIT_DETECTION_CONFIDENCE
    audit_total_rounds: int = AUDIT_TOTAL_ROUNDS
    audit_max_token_rank: int = AUDIT_MAX_TOKEN_RANK
    audit_min_relative_probability: float = AUDIT_MIN_RELATIVE_PROBABILITY
    audit_min_exact_argmax_ratio: float = AUDIT_MIN_EXACT_ARGMAX_RATIO
    audit_interval_seconds: int = 420
    max_prompt_chars: int = MAX_PROMPT_CHARS
    max_output_tokens: int = MAX_OUTPUT_TOKENS
    max_context_tokens: int = MAX_CONTEXT_TOKENS
    max_lora_rank: int = MAX_LORA_RANK
    tensor_parallel_size: int = 1
    champion_count: int = CHAMPION_COUNT
    miner_share: float = MINER_SHARE
    burn_share: float = BURN_SHARE
    challenger_share: float = CHALLENGER_SHARE
    invalid_rounds_before_blacklist: int = INVALID_ROUNDS_BEFORE_BLACKLIST
    blacklist_duration_blocks: int = BLACKLIST_DURATION_BLOCKS
    blacklist_enabled: bool = BLACKLIST_ENABLED
    promotion_delta_min: float = PROMOTION_DELTA_MIN
    promotion_confidence_z: float = PROMOTION_CONFIDENCE_Z
    bootstrap_min_score: float = 0.0
    mechanism_id: int = 0
    weights_version_key: int = 0

    def validate(self, *, production: bool = False) -> None:
        if self.protocol != PROTOCOL_NETWORK:
            raise ValueError(f"unsupported network protocol: {self.protocol!r}")
        if self.evaluation_rows != self.math_rows + self.instruction_rows:
            raise ValueError("math_rows plus instruction_rows must equal evaluation_rows")
        if self.evaluation_rows != EVALUATION_ROWS:
            raise ValueError(f"this protocol requires exactly {EVALUATION_ROWS} rows")
        if self.base_model != BASE_MODEL or self.base_revision != BASE_MODEL_REVISION:
            raise ValueError("base model or revision differs from the subnet protocol")
        if self.math_dataset != MATH_DATASET or self.math_revision != MATH_DATASET_REVISION:
            raise ValueError("math dataset or revision differs from the subnet protocol")
        if (
            self.instruction_dataset != INSTRUCTION_DATASET
            or self.instruction_revision != INSTRUCTION_DATASET_REVISION
        ):
            raise ValueError("instruction dataset or revision differs from the subnet protocol")
        if self.split != "train":
            raise ValueError("this protocol requires the train dataset split")
        if self.math_rows != MATH_ROWS or self.instruction_rows != INSTRUCTION_ROWS:
            raise ValueError(f"this protocol requires {MATH_ROWS} math and {INSTRUCTION_ROWS} instruction rows")
        if self.max_lora_rank > MAX_LORA_RANK or self.max_lora_rank <= 0:
            raise ValueError("invalid max_lora_rank")
        if self.max_output_tokens != MAX_OUTPUT_TOKENS:
            raise ValueError(f"this protocol requires max_output_tokens={MAX_OUTPUT_TOKENS}")
        if self.max_context_tokens != MAX_CONTEXT_TOKENS:
            raise ValueError(f"this protocol requires max_context_tokens={MAX_CONTEXT_TOKENS}")
        if self.dataset_window_blocks != DATASET_WINDOW_BLOCKS:
            raise ValueError(f"this protocol requires {DATASET_WINDOW_BLOCKS}-block dataset windows")
        if self.weight_interval_blocks != WEIGHT_INTERVAL_BLOCKS:
            raise ValueError(f"this protocol requires {WEIGHT_INTERVAL_BLOCKS}-block weight intervals")
        if self.audit_delay_blocks != AUDIT_DELAY_BLOCKS:
            raise ValueError(f"this protocol requires audit_delay_blocks={AUDIT_DELAY_BLOCKS}")
        if self.audit_rows_per_round != AUDIT_ROWS_PER_ROUND:
            raise ValueError(f"this protocol requires audit_rows_per_round={AUDIT_ROWS_PER_ROUND}")
        if self.audit_min_fake_row_fraction != AUDIT_MIN_FAKE_ROW_FRACTION:
            raise ValueError(
                "this protocol requires "
                f"audit_min_fake_row_fraction={AUDIT_MIN_FAKE_ROW_FRACTION}"
            )
        if self.audit_detection_confidence != AUDIT_DETECTION_CONFIDENCE:
            raise ValueError(
                "this protocol requires "
                f"audit_detection_confidence={AUDIT_DETECTION_CONFIDENCE}"
            )
        if self.audit_total_rounds != AUDIT_TOTAL_ROUNDS:
            raise ValueError(
                f"this protocol requires audit_total_rounds={AUDIT_TOTAL_ROUNDS}"
            )
        if (
            isinstance(self.audit_max_token_rank, bool)
            or not isinstance(self.audit_max_token_rank, int)
            or self.audit_max_token_rank != AUDIT_MAX_TOKEN_RANK
        ):
            raise ValueError(
                f"this protocol requires audit_max_token_rank={AUDIT_MAX_TOKEN_RANK}"
            )
        if self.audit_min_relative_probability != AUDIT_MIN_RELATIVE_PROBABILITY:
            raise ValueError(
                "this protocol requires audit_min_relative_probability="
                f"{AUDIT_MIN_RELATIVE_PROBABILITY}"
            )
        if self.audit_min_exact_argmax_ratio != AUDIT_MIN_EXACT_ARGMAX_RATIO:
            raise ValueError(
                "this protocol requires audit_min_exact_argmax_ratio="
                f"{AUDIT_MIN_EXACT_ARGMAX_RATIO}"
            )
        if self.max_prompt_chars != MAX_PROMPT_CHARS:
            raise ValueError(f"this protocol requires max_prompt_chars={MAX_PROMPT_CHARS}")
        for name in (
            "dataset_window_blocks",
            "weight_interval_blocks",
            "audit_delay_blocks",
            "audit_rows_per_round",
            "audit_total_rounds",
            "audit_max_token_rank",
            "max_output_tokens",
            "max_context_tokens",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        revisions = (self.math_revision, self.instruction_revision)
        if production and any(
            len(value) != 40 or any(character not in "0123456789abcdef" for character in value)
            for value in revisions
        ):
            raise ValueError(
                "production requires immutable 40-character Hugging Face dataset revisions; "
                "replace both dataset revisions in the network config"
            )
        if self.champion_count != CHAMPION_COUNT:
            raise ValueError(f"this protocol requires exactly {CHAMPION_COUNT} champions")
        if self.miner_share != MINER_SHARE:
            raise ValueError(f"this protocol requires miner_share={MINER_SHARE}")
        if self.burn_share != BURN_SHARE:
            raise ValueError(f"this protocol requires burn_share={BURN_SHARE}")
        if self.challenger_share != CHALLENGER_SHARE:
            raise ValueError(f"this protocol requires challenger_share={CHALLENGER_SHARE}")
        if self.invalid_rounds_before_blacklist != INVALID_ROUNDS_BEFORE_BLACKLIST:
            raise ValueError(
                "this protocol requires invalid_rounds_before_blacklist="
                f"{INVALID_ROUNDS_BEFORE_BLACKLIST}"
            )
        if self.blacklist_duration_blocks != BLACKLIST_DURATION_BLOCKS:
            raise ValueError(
                f"this protocol requires blacklist_duration_blocks={BLACKLIST_DURATION_BLOCKS}"
            )
        if self.blacklist_enabled is not BLACKLIST_ENABLED:
            raise ValueError(f"this protocol requires blacklist_enabled={BLACKLIST_ENABLED}")
        if self.promotion_delta_min != PROMOTION_DELTA_MIN:
            raise ValueError(f"this protocol requires promotion_delta_min={PROMOTION_DELTA_MIN}")
        if self.promotion_confidence_z != PROMOTION_CONFIDENCE_Z:
            raise ValueError(
                f"this protocol requires promotion_confidence_z={PROMOTION_CONFIDENCE_Z}"
            )
        if self.bootstrap_min_score != 0.0:
            raise ValueError("this protocol requires bootstrap_min_score=0.0")
        if self.mechanism_id != 0 or self.weights_version_key != 0:
            raise ValueError("this protocol requires mechanism_id=0 and weights_version_key=0")
        if self.netuid != SUBNET_NETUID:
            raise ValueError(f"this protocol is for subnet netuid {SUBNET_NETUID}")
        if production and self.netuid <= 0:
            raise ValueError("production requires a positive mainnet netuid")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "NetworkConfig":
        value = dict(value)
        if value.get("protocol") in {
            "feval-network-v5",
            "feval-network-v6",
            "feval-network-v7",
            "feval-network-v8",
            "feval-network-v9",
            "feval-network-v10",
            "feval-network-v11",
            "feval-network-v12",
            "feval-network-v13",
            "feval-network-v14",
            "feval-network-v15",
            "feval-network-v16",
            "feval-network-v17",
            "feval-network-v18",
            "feval-network-v19",
            "feval-network-v20",
            "feval-network-v21",
            "feval-network-v22",
            "feval-network-v23",
            "feval-network-v24",
            "feval-network-v25",
            "feval-network-v26",
            "feval-network-v27",
            "feval-network-v28",
            "feval-network-v29",
            "feval-network-v30",
            "feval-network-v31",
            "feval-network-v32",
        }:
            # Operational compatibility for previously sealed configurations. All
            # code-pinned evaluation and audit-size fields take their current
            # canonical values.
            value["protocol"] = PROTOCOL_NETWORK
            value["base_model"] = BASE_MODEL
            value["base_revision"] = BASE_MODEL_REVISION
            value.pop("audit_max_logprob_gap", None)
            value.pop("audit_required_rounds", None)
            value.pop("audit_min_exact_match_ratio", None)
            value["audit_rows_per_round"] = AUDIT_ROWS_PER_ROUND
            value["audit_min_fake_row_fraction"] = AUDIT_MIN_FAKE_ROW_FRACTION
            value["audit_detection_confidence"] = AUDIT_DETECTION_CONFIDENCE
            value["audit_total_rounds"] = AUDIT_TOTAL_ROUNDS
            value["audit_max_token_rank"] = AUDIT_MAX_TOKEN_RANK
            value.pop("canonical_max_candidates", None)
            value.pop("canonical_min_relative_probability", None)
            value["audit_min_relative_probability"] = AUDIT_MIN_RELATIVE_PROBABILITY
            value["audit_min_exact_argmax_ratio"] = AUDIT_MIN_EXACT_ARGMAX_RATIO
            value["challenger_share"] = CHALLENGER_SHARE
            value["champion_count"] = CHAMPION_COUNT
            value.pop("champion_shares", None)
            value["miner_share"] = MINER_SHARE
            value["burn_share"] = BURN_SHARE
            value["invalid_rounds_before_blacklist"] = INVALID_ROUNDS_BEFORE_BLACKLIST
            value["blacklist_duration_blocks"] = BLACKLIST_DURATION_BLOCKS
            value["blacklist_enabled"] = BLACKLIST_ENABLED
            value["dataset_window_blocks"] = DATASET_WINDOW_BLOCKS
            value["evaluation_rows"] = EVALUATION_ROWS
            value["math_rows"] = MATH_ROWS
            value["instruction_rows"] = INSTRUCTION_ROWS
            value["max_output_tokens"] = MAX_OUTPUT_TOKENS
            value["max_context_tokens"] = MAX_CONTEXT_TOKENS
            value.pop("candidate_pool_root", None)
            value.pop("history_start_block", None)
            value.pop("history_batch_blocks", None)
        known = {field.name for field in cls.__dataclass_fields__.values()}
        unknown = set(value) - known
        if unknown:
            raise ValueError(f"unknown network config fields: {sorted(unknown)}")
        result = cls(**value)
        result.validate()
        return result


def default_network_config_path() -> Path:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "network.json"
        if (parent / "pyproject.toml").exists() and candidate.exists():
            return candidate
    return Path("network.json")


def _is_default_network_config_request(path: str | Path) -> bool:
    requested = Path(path)
    return requested.name == "network.json" and requested.parent in (Path("."), Path(""))


def _resolve_network_config_path(path: str | Path) -> Path:
    requested = Path(path)
    if requested.exists() or requested.is_absolute():
        return requested
    if _is_default_network_config_request(requested):
        default_path = default_network_config_path()
        if default_path.exists():
            return default_path
    return requested


def load_network_config(path: str | Path, *, production: bool = False) -> NetworkConfig:
    resolved = _resolve_network_config_path(path)
    if not resolved.exists() and _is_default_network_config_request(path):
        config = NetworkConfig()
    else:
        config = NetworkConfig.from_dict(load_json(resolved))
    config.validate(production=production)
    return config


def write_network_config(path: str | Path, *, netuid: int = SUBNET_NETUID) -> NetworkConfig:
    config = NetworkConfig(netuid=netuid)
    write_json(path, config.to_dict())
    return config

