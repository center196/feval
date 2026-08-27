from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable

from .jsonutil import canonical_json_bytes


HASH_PRIME = 2_147_483_647


def canonical_max_logprob_gap(min_relative_probability: float) -> float:
    if not 0.0 < min_relative_probability <= 1.0:
        raise ValueError("min_relative_probability must be in (0, 1]")
    return -math.log(min_relative_probability)


def canonical_row_coefficients(
    *, evaluation_seed: str, model_digest: str, row_id: str
) -> tuple[int, int]:
    digest = hashlib.sha256(
        canonical_json_bytes(
            {
                "domain": "feval/canonical-token/v1",
                "evaluation_seed": evaluation_seed,
                "model_digest": model_digest,
                "row_id": row_id,
            }
        )
    ).digest()
    # A non-zero multiplier makes the ordering depend on the token id. Both
    # values fit safely in signed int64 arithmetic when multiplied by vocab IDs.
    multiplier = int.from_bytes(digest[:8], "big") % (HASH_PRIME - 1) + 1
    offset = int.from_bytes(digest[8:16], "big") % HASH_PRIME
    return multiplier, offset


def canonical_token_choice(
    candidate_token_ids: Iterable[int],
    *,
    evaluation_seed: str,
    model_digest: str,
    row_id: str,
) -> int:
    candidates = sorted({int(token_id) for token_id in candidate_token_ids})
    if not candidates:
        raise ValueError("canonical token choice requires at least one candidate")
    multiplier, offset = canonical_row_coefficients(
        evaluation_seed=evaluation_seed,
        model_digest=model_digest,
        row_id=row_id,
    )
    return min(
        candidates,
        key=lambda token_id: ((token_id * multiplier + offset) % HASH_PRIME, token_id),
    )
