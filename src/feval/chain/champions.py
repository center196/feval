from __future__ import annotations

import base64
import math
from statistics import NormalDist
from typing import Any

from ..models.artifacts import ModelCommitment
from ..core.config import NetworkConfig


def encode_reward_bits(rewards: list[int]) -> str:
    if any(value not in (0, 1) for value in rewards):
        raise ValueError("reward vector must contain only zero and one")
    packed = bytearray((len(rewards) + 7) // 8)
    for index, value in enumerate(rewards):
        if value:
            packed[index // 8] |= 1 << (index % 8)
    return base64.b64encode(bytes(packed)).decode("ascii")


def decode_reward_bits(value: str, length: int) -> list[int]:
    if length < 0:
        raise ValueError("reward vector length cannot be negative")
    try:
        packed = base64.b64decode(value, validate=True)
    except Exception as exc:
        raise ValueError("invalid reward bit vector") from exc
    expected = (length + 7) // 8
    if len(packed) != expected:
        raise ValueError("reward bit vector has the wrong byte length")
    if length % 8 and packed and packed[-1] >> (length % 8):
        raise ValueError("reward bit vector has non-zero padding")
    return [(packed[index // 8] >> (index % 8)) & 1 for index in range(length)]


def paired_lcb(candidate: list[int], king: list[int], z: float) -> tuple[float, float]:
    if len(candidate) != len(king) or not candidate:
        raise ValueError("paired reward vectors must have the same non-zero length")
    differences = [float(left) - float(right) for left, right in zip(candidate, king)]
    mean = sum(differences) / len(differences)
    if len(differences) == 1:
        return mean, mean
    variance = sum((value - mean) ** 2 for value in differences) / (len(differences) - 1)
    return mean, mean - z * math.sqrt(variance / len(differences))


def unpaired_lcb(candidate_score: float, king_score: float, rows: int, z: float) -> tuple[float, float]:
    """Conservative fallback when an old king misses the current window."""

    if rows <= 0 or not 0.0 <= candidate_score <= 1.0 or not 0.0 <= king_score <= 1.0:
        raise ValueError("invalid unpaired score comparison")
    delta = candidate_score - king_score
    variance = (
        candidate_score * (1.0 - candidate_score) / rows
        + king_score * (1.0 - king_score) / rows
    )
    return delta, delta - z * math.sqrt(variance)


def familywise_z(base_z: float, comparisons: int) -> float:
    """Bonferroni-adjust a one-sided normal threshold for challenger selection.

    Validators choose the strongest result among all challengers. Testing each
    one at the unadjusted threshold would inflate the chance of replacing the
    king merely because many miners entered the window.
    """

    if not math.isfinite(base_z) or base_z <= 0.0:
        raise ValueError("base confidence z must be positive and finite")
    if comparisons <= 0:
        raise ValueError("comparison count must be positive")
    normal = NormalDist()
    base_alpha = 1.0 - normal.cdf(base_z)
    adjusted_alpha = base_alpha / comparisons
    return normal.inv_cdf(1.0 - adjusted_alpha)


def _valid_result(result: dict[str, Any], rows: int) -> bool:
    return bool(
        result.get("valid")
        and float(result.get("score", 0.0)) > 0.0
        and result.get("model_digest")
        and int(result.get("rows", -1)) == rows
        and isinstance(result.get("reward_bits"), str)
    )


def update_champions(
    state: dict[str, Any],
    *,
    config: NetworkConfig,
    current_block: int,
) -> dict[str, Any]:
    """Deterministically promote at most one model using a paired 99% LCB."""

    results: dict[str, dict[str, Any]] = state.get("results", {})
    valid = [
        (hotkey, result)
        for hotkey, result in results.items()
        if _valid_result(result, config.evaluation_rows)
    ]
    champions = list(state.setdefault("champions", []))
    for champion in champions:
        hotkey = str(champion.get("miner_hotkey"))
        result = results.get(hotkey, {})
        if (
            _valid_result(result, config.evaluation_rows)
            and result.get("model_digest") == champion.get("model_digest")
        ):
            champion["last_score"] = float(result["score"])
            champion["last_window"] = state.get("window")
    champion_digests = {item.get("model_digest") for item in champions}
    candidates = [item for item in valid if item[1]["model_digest"] not in champion_digests]
    decision: dict[str, Any] = {
        "block": current_block,
        "promoted": False,
        "reason": "no eligible challenger",
        "delta_min": config.promotion_delta_min,
        "confidence_z": config.promotion_confidence_z,
    }

    promoted: tuple[str, dict[str, Any], float | None, float | None] | None = None
    if not champions:
        if candidates:
            hotkey, candidate = max(
                candidates,
                key=lambda item: (
                    float(item[1]["score"]),
                    -int(item[1].get("commit_block", 2**63 - 1)),
                    item[0],
                ),
            )
            if float(candidate["score"]) > max(0.0, config.bootstrap_min_score):
                promoted = (hotkey, candidate, None, None)
                decision["reason"] = "first champion"
            else:
                decision["reason"] = "bootstrap score below threshold"
    else:
        king = champions[0]
        comparison_z = familywise_z(
            config.promotion_confidence_z,
            max(1, len(candidates)),
        )
        decision["candidate_comparisons"] = len(candidates)
        decision["comparison_confidence_z"] = comparison_z
        king_result = results.get(str(king.get("miner_hotkey")), {})
        if (
            _valid_result(king_result, config.evaluation_rows)
            and king_result.get("model_digest") == king.get("model_digest")
        ):
            king_rewards = decode_reward_bits(king_result["reward_bits"], config.evaluation_rows)
            comparisons: list[tuple[float, float, str, dict[str, Any]]] = []
            for hotkey, candidate in candidates:
                candidate_rewards = decode_reward_bits(
                    candidate["reward_bits"], config.evaluation_rows
                )
                delta, lower_bound = paired_lcb(
                    candidate_rewards, king_rewards, comparison_z
                )
                comparisons.append((lower_bound, delta, hotkey, candidate))
            if comparisons:
                lower_bound, delta, hotkey, candidate = max(
                    comparisons,
                    key=lambda item: (
                        item[0],
                        item[1],
                        -int(item[3].get("commit_block", 2**63 - 1)),
                        item[2],
                    ),
                )
                decision.update({"delta": delta, "lower_confidence_bound": lower_bound})
                if lower_bound >= config.promotion_delta_min:
                    promoted = (hotkey, candidate, delta, lower_bound)
                    decision["reason"] = "clear paired improvement"
                else:
                    decision["reason"] = "improvement below paired confidence threshold"
        else:
            # Preserve liveness if a king disappears. Windows are fixed-size
            # samples from the same deterministic pool, so use a more conservative
            # independent-proportions bound against the king's last valid
            # score. Same-window paired comparison remains the preferred path.
            reference = king.get("last_score", king.get("score_at_promotion"))
            comparisons = []
            if reference is not None:
                for hotkey, candidate in candidates:
                    delta, lower_bound = unpaired_lcb(
                        float(candidate["score"]),
                        float(reference),
                        config.evaluation_rows,
                        comparison_z,
                    )
                    comparisons.append((lower_bound, delta, hotkey, candidate))
            if comparisons:
                lower_bound, delta, hotkey, candidate = max(
                    comparisons,
                    key=lambda item: (
                        item[0],
                        item[1],
                        -int(item[3].get("commit_block", 2**63 - 1)),
                        item[2],
                    ),
                )
                decision.update({"delta": delta, "lower_confidence_bound": lower_bound})
                if lower_bound >= config.promotion_delta_min:
                    promoted = (hotkey, candidate, delta, lower_bound)
                    decision["reason"] = "clear unpaired improvement over inactive king"
                else:
                    decision["reason"] = "inactive-king improvement below confidence threshold"
            else:
                decision["reason"] = "current king has no reference score"

    if promoted is not None:
        hotkey, result, delta, lower_bound = promoted
        champion = {
            "miner_hotkey": hotkey,
            "model_digest": result["model_digest"],
            "model_revision": result.get("model_revision"),
            "score_at_promotion": float(result["score"]),
            "last_score": float(result["score"]),
            "last_window": state.get("window"),
            "promoted_block": current_block,
            "parent_digest": champions[0].get("model_digest") if champions else None,
            "delta": delta,
            "lower_confidence_bound": lower_bound,
        }
        champions = [champion] + [
            old for old in champions if old.get("model_digest") != champion["model_digest"]
        ]
        state["champions"] = champions[: config.champion_count]
        decision.update(
            {
                "promoted": True,
                "candidate_hotkey": hotkey,
                "candidate_model_digest": result["model_digest"],
            }
        )
    state["last_promotion_decision"] = decision
    return decision


def champion_weight_mapping(
    state: dict[str, Any],
    *,
    config: NetworkConfig,
    commitments: dict[str, dict[str, Any]],
    copies: dict[str, str],
) -> dict[int, float]:
    """Apply the configured burn and active-champion allocation."""

    by_hotkey: dict[str, float] = {}
    for share, champion in zip(config.champion_shares, state.get("champions", [])):
        hotkey = str(champion.get("miner_hotkey"))
        current = commitments.get(hotkey)
        result = state.get("results", {}).get(hotkey, {})
        if not (
            isinstance(result, dict)
            and result.get("valid")
            and float(result.get("score", 0.0)) > 0.0
            and result.get("model_digest") == champion.get("model_digest")
        ):
            # Window transitions do not interrupt an already verified king's
            # emission. Carryover is accepted only for that exact model digest
            # and only until the king validates again or a challenger replaces
            # it with a statistically qualified current-window result.
            result = state.get("carryover_results", {}).get(hotkey, {})
        if (
            current is None
            or hotkey in copies
            or not result.get("valid")
            or float(result.get("score", 0.0)) <= 0.0
        ):
            continue
        commitment: ModelCommitment = current["commitment"]
        if commitment.model_digest != champion.get("model_digest"):
            continue
        if result.get("model_digest") != commitment.model_digest:
            continue
        by_hotkey[hotkey] = by_hotkey.get(hotkey, 0.0) + float(share)

    uid_weights: dict[int, float] = {0: float(config.burn_share)}
    allocated = 0.0
    for hotkey, value in by_hotkey.items():
        uid = commitments[hotkey].get("uid")
        if uid is not None:
            uid_weights[int(uid)] = uid_weights.get(int(uid), 0.0) + value
            allocated += value
    uid_weights[0] += sum(float(value) for value in config.champion_shares) - allocated
    total = sum(uid_weights.values())
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"protocol weights must sum to one, received {total}")
    return dict(sorted(uid_weights.items()))

