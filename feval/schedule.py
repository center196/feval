from __future__ import annotations

import hashlib
import math
import random
from typing import Iterable

from .jsonutil import canonical_json_bytes


def dataset_window(block: int, window_blocks: int = 3_600) -> int:
    """Return the cross-machine dataset window.

    Integer division is deliberately used instead of Python ``round``. Python
    uses bankers' rounding, which is surprising at exact half-window boundaries
    and differs from implementations in other languages.
    """

    if block < 0:
        raise ValueError("block must be non-negative")
    if window_blocks <= 0:
        raise ValueError("window_blocks must be positive")
    return block // window_blocks


def evaluation_seed(netuid: int, window: int) -> str:
    return hashlib.sha256(
        canonical_json_bytes(
            {"domain": "feval/evaluation/v2", "netuid": netuid, "window": window}
        )
    ).hexdigest()


def audit_seed(
    *,
    netuid: int,
    block_hash: str,
    hotkey: str,
    model_digest: str,
    rollout_revision: str,
    round_number: int,
) -> str:
    return hashlib.sha256(
        canonical_json_bytes(
            {
                "domain": "feval/audit/v2",
                "netuid": netuid,
                "block_hash": block_hash,
                "hotkey": hotkey,
                "model_digest": model_digest,
                "rollout_revision": rollout_revision,
                "round": round_number,
            }
        )
    ).hexdigest()


def choose_audit_ids(
    rows: list[dict],
    *,
    seed: str,
    count: int,
    already_audited: Iterable[str] = (),
) -> list[str]:
    """Choose uniformly from the remaining rows without replacement."""

    if count <= 0:
        return []
    excluded = set(already_audited)
    available = [row for row in rows if str(row["row_id"]) not in excluded]
    rng = random.Random(seed)
    rng.shuffle(available)
    return [str(row["row_id"]) for row in available[:count]]


def required_audit_rounds(
    *,
    population: int,
    rows_per_round: int,
    min_fake_fraction: float,
    confidence: float,
) -> int:
    """Return conservative rounds needed to detect the target forged fraction.

    The bound assumes sampling with replacement, while the protocol samples
    distinct rows. It is therefore conservative. Small test sets are capped at
    full coverage.
    """

    if population <= 0 or rows_per_round <= 0:
        raise ValueError("population and rows_per_round must be positive")
    if not 0.0 < min_fake_fraction < 1.0:
        raise ValueError("min_fake_fraction must be in (0, 1)")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    samples = math.ceil(math.log1p(-confidence) / math.log1p(-min_fake_fraction))
    statistical_rounds = math.ceil(samples / rows_per_round)
    full_coverage_rounds = math.ceil(population / rows_per_round)
    return min(statistical_rounds, full_coverage_rounds)
