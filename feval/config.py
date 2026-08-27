from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .constants import (
    AUDIT_DETECTION_CONFIDENCE,
    AUDIT_DELAY_BLOCKS,
    AUDIT_MIN_FAKE_ROW_FRACTION,
    AUDIT_ROWS_PER_ROUND,
    BASE_MODEL,
    BASE_MODEL_REVISION,
    CANONICAL_MAX_CANDIDATES,
    CANONICAL_MIN_RELATIVE_PROBABILITY,
    CHALLENGER_SHARE,
    CHAMPION_COUNT,
    CHAMPION_SHARES,
    DATASET_WINDOW_BLOCKS,
    EVALUATION_ROWS,
    HISTORY_BATCH_BLOCKS,
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
from .jsonutil import load_json, write_json


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
    canonical_max_candidates: int = CANONICAL_MAX_CANDIDATES
    canonical_min_relative_probability: float = CANONICAL_MIN_RELATIVE_PROBABILITY
    audit_interval_seconds: int = 420
    max_prompt_chars: int = MAX_PROMPT_CHARS
    max_output_tokens: int = MAX_OUTPUT_TOKENS
    max_context_tokens: int = MAX_CONTEXT_TOKENS
    max_lora_rank: int = MAX_LORA_RANK
    tensor_parallel_size: int = 1
    # These values bind dataset consensus and commitment history to the
    # canonical network configuration.
    candidate_pool_root: str | None = None
    history_start_block: int | None = None
    history_batch_blocks: int = HISTORY_BATCH_BLOCKS
    champion_count: int = CHAMPION_COUNT
    champion_shares: tuple[float, ...] = CHAMPION_SHARES
    challenger_share: float = CHALLENGER_SHARE
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
        if self.canonical_max_candidates != CANONICAL_MAX_CANDIDATES:
            raise ValueError(
                f"this protocol requires canonical_max_candidates={CANONICAL_MAX_CANDIDATES}"
            )
        if self.canonical_min_relative_probability != CANONICAL_MIN_RELATIVE_PROBABILITY:
            raise ValueError(
                "this protocol requires canonical_min_relative_probability="
                f"{CANONICAL_MIN_RELATIVE_PROBABILITY}"
            )
        if self.max_prompt_chars != MAX_PROMPT_CHARS:
            raise ValueError(f"this protocol requires max_prompt_chars={MAX_PROMPT_CHARS}")
        for name in (
            "dataset_window_blocks",
            "weight_interval_blocks",
            "audit_delay_blocks",
            "audit_rows_per_round",
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
        if self.history_batch_blocks <= 0:
            raise ValueError("history_batch_blocks must be positive")
        if self.champion_count != CHAMPION_COUNT:
            raise ValueError(f"this protocol requires exactly {CHAMPION_COUNT} champions")
        if tuple(self.champion_shares) != CHAMPION_SHARES:
            raise ValueError(f"this protocol requires champion_shares={CHAMPION_SHARES}")
        if self.challenger_share != CHALLENGER_SHARE:
            raise ValueError(f"this protocol requires challenger_share={CHALLENGER_SHARE}")
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
        if production:
            root = self.candidate_pool_root or ""
            if len(root) != 64 or any(character not in "0123456789abcdef" for character in root):
                raise ValueError(
                    "validator production mode requires a sealed network.json with a "
                    "64-character candidate_pool_root. Protocol constants are code-pinned; "
                    "the sealed root is an optional launch hardening value. Use the "
                    "canonical sealed subnet config, or create it during launch with "
                    "feval dataset candidate-pool "
                    "--sealed-config-out network.sealed.json --history-start-block <block>."
                )
            if self.history_start_block is None or self.history_start_block <= 0:
                raise ValueError(
                    "validator production mode requires a sealed network.json with a "
                    "positive history_start_block."
                )
            if self.netuid <= 0:
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
        }:
            value["protocol"] = PROTOCOL_NETWORK
            value.pop("audit_max_logprob_gap", None)
            value.pop("audit_required_rounds", None)
            value.pop("audit_min_exact_match_ratio", None)
            value["audit_min_fake_row_fraction"] = AUDIT_MIN_FAKE_ROW_FRACTION
            value["audit_detection_confidence"] = AUDIT_DETECTION_CONFIDENCE
            value["canonical_max_candidates"] = CANONICAL_MAX_CANDIDATES
            value["canonical_min_relative_probability"] = CANONICAL_MIN_RELATIVE_PROBABILITY
            value["challenger_share"] = CHALLENGER_SHARE
            value["evaluation_rows"] = EVALUATION_ROWS
            value["math_rows"] = MATH_ROWS
            value["instruction_rows"] = INSTRUCTION_ROWS
        known = {field.name for field in cls.__dataclass_fields__.values()}
        unknown = set(value) - known
        if unknown:
            raise ValueError(f"unknown network config fields: {sorted(unknown)}")
        result = cls(**value)
        result.validate()
        return result


def default_network_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "network.json"


def _resolve_network_config_path(path: str | Path) -> Path:
    requested = Path(path)
    if requested.exists() or requested.is_absolute():
        return requested
    if requested.name == "network.json" and requested.parent in (Path("."), Path("")):
        default_path = default_network_config_path()
        if default_path.exists():
            return default_path
    return requested


def load_network_config(path: str | Path, *, production: bool = False) -> NetworkConfig:
    config = NetworkConfig.from_dict(load_json(_resolve_network_config_path(path)))
    config.validate(production=production)
    return config


def write_network_config(path: str | Path, *, netuid: int = SUBNET_NETUID) -> NetworkConfig:
    config = NetworkConfig(netuid=netuid)
    write_json(path, config.to_dict())
    return config


def seal_network_config(
    path: str | Path,
    *,
    candidate_pool_root: str,
    history_start_block: int,
) -> NetworkConfig:
    """Seal the consensus roots and launch block into a network config."""

    value = load_json(path)
    existing_root = value.get("candidate_pool_root")
    existing_start = value.get("history_start_block")
    if existing_root is not None and existing_root != candidate_pool_root:
        raise ValueError("network config is already sealed with a different candidate pool root")
    if existing_start is not None and int(existing_start) != int(history_start_block):
        raise ValueError("network config is already sealed with a different history start block")
    value["candidate_pool_root"] = candidate_pool_root
    value["history_start_block"] = int(history_start_block)
    config = NetworkConfig.from_dict(value)
    config.validate(production=True)
    write_json(path, config.to_dict())
    return config
