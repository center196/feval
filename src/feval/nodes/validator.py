from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any

from ..utils.crypto import hash_file, sha256_hex, verify_signature
from ..utils.jsonutil import canonical_json_bytes, load_json, load_jsonl, write_json
from ..protocol.merkle import root_for_values
from ..models.mock_model import verify_rollout
from ..datasets.rewards import reward_for_row


def _payload_without_signature(submission: dict[str, Any]) -> dict[str, Any]:
    payload = dict(submission)
    payload.pop("signature", None)
    return payload


def verify_submission_structure(config: dict[str, Any], eval_rows: list[dict[str, Any]], submission: dict[str, Any], keyring: dict[str, Any] | None = None) -> list[str]:
    errors: list[str] = []
    payload = _payload_without_signature(submission)
    if submission.get("base_hash") != config.get("base_hash"):
        errors.append("base_hash mismatch")
    if submission.get("evaluation_root") != root_for_values(eval_rows):
        errors.append("evaluation_root mismatch")
    expected_ids = [str(row.get("row_id")) for row in eval_rows]
    submitted_ids = [str(row.get("row_id")) for row in submission.get("rows", [])]
    if submitted_ids != expected_ids:
        errors.append("submission rows must match the evaluation row IDs exactly and in order")
    answer_values = [{"row_id": row["row_id"], "answer": row["answer"], "reward": row["reward_claimed"]} for row in submission.get("rows", [])]
    rollout_values = [{"row_id": row["row_id"], "tokens": row["tokens"]} for row in submission.get("rows", [])]
    if submission.get("answer_root") != root_for_values(answer_values):
        errors.append("answer_root mismatch")
    if submission.get("rollout_root") != root_for_values(rollout_values):
        errors.append("rollout_root mismatch")
    adapter_path = Path(str(submission.get("adapter_path", "")))
    if not adapter_path.is_absolute():
        adapter_path = Path.cwd() / adapter_path
    if adapter_path.exists():
        if submission.get("adapter_hash") != hash_file(adapter_path):
            errors.append("adapter_hash mismatch")
    else:
        errors.append(f"adapter file not found: {adapter_path}")
    if keyring is not None and not verify_signature(payload, submission.get("signature", {}), keyring):
        errors.append("signature invalid")
    return errors


def score_submission(config_path: str | Path, eval_path: str | Path, submission_path: str | Path, out_path: str | Path, keyring_path: str | Path | None = None) -> dict[str, Any]:
    config = load_json(config_path)
    eval_rows = load_jsonl(eval_path)
    submission = load_json(submission_path)
    keyring = load_json(keyring_path) if keyring_path else None
    errors = verify_submission_structure(config, eval_rows, submission, keyring)
    eval_by_id = {row["row_id"]: row for row in eval_rows}
    rewards: list[int] = []
    row_reports: list[dict[str, Any]] = []
    tolerance = config.get("reward", {}).get("numeric_tolerance", 1e-6)
    for row in submission.get("rows", []):
        reward = reward_for_row(row.get("answer", ""), eval_by_id.get(row["row_id"], {}), tolerance)
        rewards.append(reward)
        row_reports.append({"row_id": row["row_id"], "reward": reward, "claimed": row.get("reward_claimed")})
        if reward != row.get("reward_claimed"):
            errors.append(f"reward mismatch for {row['row_id']}")
    score = sum(rewards) / max(1, len(eval_rows))
    report = {
        "protocol": "feval-score-v1",
        "miner_hotkey": submission.get("miner_hotkey"),
        "adapter_hash": submission.get("adapter_hash"),
        "submission": str(submission_path),
        "valid": not errors,
        "errors": errors,
        "score": score,
        "rewards": rewards,
        "rows": row_reports,
    }
    write_json(out_path, report)
    return report


def choose_audit_rows(submission: dict[str, Any], seed: str, count: int, positive_ratio: float = 0.8) -> list[str]:
    rows = list(submission.get("rows", []))
    positives = [row for row in rows if int(row.get("reward_claimed", 0)) > 0]
    rng = random.Random(sha256_hex(canonical_json_bytes({
        "seed": seed,
        "miner_hotkey": submission.get("miner_hotkey"),
        "adapter_hash": submission.get("adapter_hash"),
        "answer_root": submission.get("answer_root"),
        "rollout_root": submission.get("rollout_root"),
    })))
    selected: list[str] = []
    selected_set: set[str] = set()
    positive_budget = min(count, round(count * positive_ratio))
    for pool, budget in ((positives, positive_budget), (rows, count - positive_budget)):
        shuffled = pool[:]
        rng.shuffle(shuffled)
        for row in shuffled:
            if len(selected) >= count:
                break
            row_id = row["row_id"]
            if row_id in selected_set:
                continue
            selected.append(row_id)
            selected_set.add(row_id)
        if len(selected) >= count:
            break
    if len(selected) < count:
        shuffled = rows[:]
        rng.shuffle(shuffled)
        for row in shuffled:
            row_id = row["row_id"]
            if row_id not in selected_set:
                selected.append(row_id)
                selected_set.add(row_id)
            if len(selected) >= count:
                break
    return selected


def audit_submission(config_path: str | Path, eval_path: str | Path, submission_path: str | Path, out_path: str | Path, seed: str, rows: int, keyring_path: str | Path | None = None) -> dict[str, Any]:
    config = load_json(config_path)
    eval_rows = load_jsonl(eval_path)
    submission = load_json(submission_path)
    keyring = load_json(keyring_path) if keyring_path else None
    errors = verify_submission_structure(config, eval_rows, submission, keyring)
    selected = choose_audit_rows(submission, seed, rows)
    prompt_by_id = {row["row_id"]: row["prompt"] for row in eval_rows}
    submitted_by_id = {row["row_id"]: row for row in submission.get("rows", [])}
    adapter_path = submission.get("adapter_path")
    failures: list[dict[str, Any]] = []
    adapter = load_json(adapter_path)
    for row_id in selected:
        row = submitted_by_id[row_id]
        if not verify_rollout(adapter, prompt_by_id[row_id], list(row["tokens"])):
            failures.append({"row_id": row_id, "error": "rollout tokens do not match adapter output"})
    report = {
        "protocol": "feval-audit-v1",
        "miner_hotkey": submission.get("miner_hotkey"),
        "adapter_hash": submission.get("adapter_hash"),
        "submission": str(submission_path),
        "seed": seed,
        "audited_rows": selected,
        "valid": not errors and not failures,
        "errors": errors,
        "failures": failures,
    }
    write_json(out_path, report)
    return report


def paired_lcb(candidate_rewards: list[int], king_rewards: list[int], z: float) -> tuple[float, float]:
    if len(candidate_rewards) != len(king_rewards):
        raise ValueError("candidate and king reward vectors must have same length")
    diffs = [float(a) - float(b) for a, b in zip(candidate_rewards, king_rewards)]
    if not diffs:
        return 0.0, 0.0
    mean = sum(diffs) / len(diffs)
    if len(diffs) == 1:
        return mean, mean
    variance = sum((d - mean) ** 2 for d in diffs) / (len(diffs) - 1)
    se = math.sqrt(variance / len(diffs))
    return mean, mean - z * se


def promote_candidate(state_path: str | Path, candidate_score_path: str | Path, candidate_audit_path: str | Path, out_path: str | Path, king_score_path: str | Path | None = None, delta_min: float = 0.01, confidence_z: float = 2.326347874) -> dict[str, Any]:
    state_file = Path(state_path)
    state = load_json(state_file) if state_file.exists() else {"protocol": "feval-champions-v1", "champions": []}
    score = load_json(candidate_score_path)
    audit = load_json(candidate_audit_path)
    promoted = False
    reason = "candidate failed scoring or audit"
    delta = None
    lcb = None
    if score.get("valid") and audit.get("valid"):
        if king_score_path:
            king = load_json(king_score_path)
            delta, lcb = paired_lcb(list(score["rewards"]), list(king["rewards"]), confidence_z)
            promoted = lcb >= delta_min
            reason = "clear improvement" if promoted else "improvement below threshold"
        elif not state.get("champions"):
            promoted = True
            reason = "first champion"
        else:
            current = state["champions"][0]
            delta = float(score["score"]) - float(current.get("score", 0.0))
            lcb = delta
            promoted = lcb >= delta_min
            reason = "clear improvement" if promoted else "improvement below threshold"
    if promoted:
        champion_id = sha256_hex(canonical_json_bytes({
            "hotkey": score["miner_hotkey"],
            "adapter_hash": score["adapter_hash"],
            "score": score["score"],
            "parent": state.get("champions", [{}])[0].get("champion_id") if state.get("champions") else None,
        }))[:16]
        champion = {
            "champion_id": champion_id,
            "miner_hotkey": score["miner_hotkey"],
            "adapter_hash": score["adapter_hash"],
            "score": score["score"],
            "delta": delta,
            "lower_confidence_bound": lcb,
            "reason": reason,
        }
        champions = [champion] + [old for old in state.get("champions", []) if old.get("adapter_hash") != score["adapter_hash"]]
        state["champions"] = champions[:1]
    state["last_promotion_decision"] = {
        "candidate_hotkey": score.get("miner_hotkey"),
        "candidate_adapter_hash": score.get("adapter_hash"),
        "promoted": promoted,
        "reason": reason,
        "delta": delta,
        "lower_confidence_bound": lcb,
        "delta_min": delta_min,
        "confidence_z": confidence_z,
    }
    write_json(out_path, state)
    return state


def write_weights(state_path: str | Path, score_paths: list[str], audit_paths: list[str], out_path: str | Path, uid_map_path: str | Path | None = None) -> dict[str, Any]:
    state = load_json(state_path)
    scores = [load_json(path) for path in score_paths]
    audit_reports = [load_json(path) for path in audit_paths]
    audits = {report.get("adapter_hash"): report for report in audit_reports}
    scores_by_hash = {score.get("adapter_hash"): score for score in scores}
    weights: dict[str, float] = {}
    champion_splits = [0.10]
    for split, champion in zip(champion_splits, state.get("champions", [])):
        adapter_hash = champion.get("adapter_hash")
        score = scores_by_hash.get(adapter_hash, {})
        audit = audits.get(adapter_hash, {})
        if (
            not score.get("valid")
            or not audit.get("valid")
            or float(score.get("score", 0.0)) <= 0.0
            or score.get("miner_hotkey") != champion.get("miner_hotkey")
        ):
            continue
        hotkey = champion["miner_hotkey"]
        weights[hotkey] = weights.get(hotkey, 0.0) + split
    weights = dict(sorted(weights.items()))
    uid_weights = None
    if uid_map_path:
        uid_map = load_json(uid_map_path)
        uid_weights = {"0": 0.90}
        assigned = 0.0
        for hotkey, weight in weights.items():
            if hotkey not in uid_map:
                continue
            uid = str(uid_map[hotkey])
            uid_weights[uid] = uid_weights.get(uid, 0.0) + weight
            assigned += weight
        uid_weights["0"] += 0.10 - assigned
    report = {"protocol": "feval-weights-v1", "weights": weights, "uid_weights": uid_weights, "champions": state.get("champions", [])}
    write_json(out_path, report)
    return report



