from __future__ import annotations

from pathlib import Path
from typing import Any

from ..utils.crypto import hash_file, hash_json, sign_payload
from ..core.constants import (
    BASE_MODEL,
    BASE_MODEL_REVISION,
    BURN_SHARE,
    CHALLENGER_SHARE,
    MINER_SHARE,
)
from ..utils.jsonutil import load_json, load_jsonl, write_json, write_jsonl
from .merkle import leaf_hash, root_for_values
from ..models.mock_model import generate_rollout
from ..datasets.rewards import reward_for_row


PROTOCOL_CONFIG = "feval-config-v1"
PROTOCOL_ADAPTER = "feval-adapter-v1"
PROTOCOL_SUBMISSION = "feval-submission-v1"


def create_demo_files(out_dir: str | Path) -> dict[str, str]:
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    config = {
        "protocol": PROTOCOL_CONFIG,
        "netuid": 47,
        "base_model": BASE_MODEL,
        "base_hash": hash_json({"base": BASE_MODEL, "revision": BASE_MODEL_REVISION}),
        "reward": {"type": "row_verifier", "numeric_tolerance": 1e-6},
        "promotion": {"delta_min": 0.01, "confidence_z": 2.326347874},
        "emissions": {
            "burn_uid_0": BURN_SHARE,
            "miners": MINER_SHARE,
            "challengers": CHALLENGER_SHARE,
        },
    }
    train_rows = [
        {"row_id": "train-1", "prompt": "2 + 2 =", "expected": ["4"], "task_type": "math", "verifier": "exact_or_numeric"},
        {"row_id": "train-2", "prompt": "Answer with exactly 2 words and no comma.", "expected": ["Paris wins"], "task_type": "instruction_follow", "verifier": "instruction_constraints", "constraints": [{"id": "length_constraints:number_words", "kwargs": {"N": 2}}, {"id": "punctuation:no_comma", "kwargs": {}}]},
        {"row_id": "train-3", "prompt": "10 / 2 =", "expected": ["5"], "task_type": "math", "verifier": "exact_or_numeric"},
        {"row_id": "train-4", "prompt": "Answer with a title in double angle brackets.", "expected": ["<<blue sky>>"], "task_type": "instruction_follow", "verifier": "instruction_constraints", "constraints": [{"id": "detectable_format:title", "kwargs": {}}]},
    ]
    eval_rows = [
        {"row_id": "eval-1", "prompt": "2 + 2 =", "expected": ["4"], "task_type": "math", "verifier": "exact_or_numeric"},
        {"row_id": "eval-2", "prompt": "Answer with exactly 2 words and no comma.", "expected": ["Paris wins"], "task_type": "instruction_follow", "verifier": "instruction_constraints", "constraints": [{"id": "length_constraints:number_words", "kwargs": {"N": 2}}, {"id": "punctuation:no_comma", "kwargs": {}}]},
        {"row_id": "eval-3", "prompt": "10 / 2 =", "expected": ["5"], "task_type": "math", "verifier": "exact_or_numeric"},
        {"row_id": "eval-4", "prompt": "Answer with a title in double angle brackets.", "expected": ["<<blue sky>>"], "task_type": "instruction_follow", "verifier": "instruction_constraints", "constraints": [{"id": "detectable_format:title", "kwargs": {}}]},
        {"row_id": "eval-5", "prompt": "7 * 6 =", "expected": ["42"], "task_type": "math", "verifier": "exact_or_numeric"},
    ]
    write_json(root / "subnet.json", config)
    write_jsonl(root / "train.jsonl", train_rows)
    write_jsonl(root / "eval.jsonl", eval_rows)
    write_json(root / "champions.json", {"protocol": "feval-champions-v1", "champions": []})
    return {
        "config": str(root / "subnet.json"),
        "train": str(root / "train.jsonl"),
        "eval": str(root / "eval.jsonl"),
        "champions": str(root / "champions.json"),
    }


def train_mock_adapter(config_path: str | Path, train_path: str | Path, key_path: str | Path, out_path: str | Path, parent: str | None = None) -> dict[str, Any]:
    config = load_json(config_path)
    key = load_json(key_path)
    train_rows = load_jsonl(train_path)
    adapter = {
        "protocol": PROTOCOL_ADAPTER,
        "hotkey": key["hotkey"],
        "base_hash": config["base_hash"],
        "parent_champion_id": parent,
        "learned_answers": {row["prompt"]: str(row["expected"][0] if isinstance(row["expected"], list) else row["expected"]) for row in train_rows},
        "fallback_answer": "",
        "training_root": root_for_values(train_rows),
    }
    write_json(out_path, adapter)
    return adapter


def build_submission(config_path: str | Path, eval_path: str | Path, adapter_path: str | Path, key_path: str | Path, out_path: str | Path, epoch: int = 0) -> dict[str, Any]:
    config = load_json(config_path)
    adapter = load_json(adapter_path)
    key = load_json(key_path)
    eval_rows = load_jsonl(eval_path)
    rows: list[dict[str, Any]] = []
    for row in eval_rows:
        rollout = generate_rollout(adapter, row["prompt"])
        reward = reward_for_row(rollout["answer"], row, config["reward"].get("numeric_tolerance", 1e-6))
        answer_record = {"row_id": row["row_id"], "answer": rollout["answer"], "reward": reward}
        rollout_record = {"row_id": row["row_id"], "tokens": rollout["tokens"]}
        rows.append({
            "row_id": row["row_id"],
            "answer": rollout["answer"],
            "tokens": rollout["tokens"],
            "reward_claimed": reward,
            "answer_leaf": leaf_hash(answer_record),
            "rollout_leaf": leaf_hash(rollout_record),
        })
    score = sum(row["reward_claimed"] for row in rows) / max(1, len(rows))
    payload = {
        "protocol": PROTOCOL_SUBMISSION,
        "miner_hotkey": key["hotkey"],
        "epoch": epoch,
        "base_hash": config["base_hash"],
        "adapter_path": str(Path(adapter_path)),
        "adapter_hash": hash_file(adapter_path),
        "evaluation_root": root_for_values(eval_rows),
        "answer_root": root_for_values([{"row_id": row["row_id"], "answer": row["answer"], "reward": row["reward_claimed"]} for row in rows]),
        "rollout_root": root_for_values([{"row_id": row["row_id"], "tokens": row["tokens"]} for row in rows]),
        "score_claimed": score,
        "rows": rows,
    }
    signed = dict(payload)
    signed["signature"] = sign_payload(payload, key)
    write_json(out_path, signed)
    return signed



