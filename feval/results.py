from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from .jsonutil import load_json, write_json, write_jsonl
from .merkle import root_for_values


PROTOCOL_RESULTS_MANIFEST = "feval-results-v1"


def _load_jsonl_exact(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"results line {line_number} is not an object")
            rows.append(value)
    return rows


def _status(valid: bool, final_weight: float) -> str:
    if not valid:
        return "invalid"
    if final_weight > 0.0:
        return "rewarded"
    return "valid"


def result_rows_from_state(state: dict[str, Any]) -> list[dict[str, Any]]:
    results = state.get("results", {})
    if not isinstance(results, dict):
        raise ValueError("validator state results must be an object")
    last_weights = state.get("last_weights", {})
    if not isinstance(last_weights, dict):
        last_weights = {}
    rows: list[dict[str, Any]] = []
    for hotkey, result in sorted(results.items()):
        if not isinstance(result, dict):
            continue
        uid = result.get("uid")
        uid_key = str(uid) if uid is not None else None
        final_weight = float(last_weights.get(uid_key, 0.0)) if uid_key is not None else 0.0
        valid = bool(result.get("valid"))
        audit_status = result.get("audit_status")
        status = (
            str(audit_status)
            if audit_status in {"auditing", "retrying"}
            else _status(valid, final_weight)
        )
        row = {
            "hotkey": str(hotkey),
            "uid": uid,
            "valid": valid,
            "status": status,
            "score": float(result.get("score", 0.0)),
            "final_weight": final_weight,
            "correct": result.get("correct"),
            "rows": result.get("rows"),
            "model_digest": result.get("model_digest"),
            "model_revision": result.get("model_revision"),
            "commit_block": result.get("commit_block"),
            "rollout_revision": result.get("rollout_revision"),
            "audit_block": result.get("audit_block"),
            "audited_count": len(result.get("audited_rows", [])),
            "audit_round": result.get("audit_round"),
            "audit_required_rounds": result.get("audit_required_rounds"),
            "audit_exact_match_ratio": result.get("audit_cumulative_exact_match_ratio"),
            "error": result.get("error"),
        }
        rows.append(row)
    return rows


def leaderboard_from_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(
        rows,
        key=lambda row: (
            -float(row.get("final_weight") or 0.0),
            -float(row.get("score") or 0.0),
            str(row.get("hotkey") or ""),
        ),
    )
    return [
        {
            "rank": index,
            "hotkey": row["hotkey"],
            "uid": row.get("uid"),
            "status": row.get("status"),
            "valid": bool(row.get("valid")),
            "score": float(row.get("score") or 0.0),
            "final_weight": float(row.get("final_weight") or 0.0),
            "model_digest": row.get("model_digest"),
            "error": row.get("error"),
        }
        for index, row in enumerate(ordered, start=1)
    ]


def export_results_bundle(
    *,
    state_path: str | Path,
    out_dir: str | Path,
    validator_hotkey: str | None = None,
) -> dict[str, Any]:
    state = load_json(state_path)
    rows = result_rows_from_state(state)
    leaderboard = leaderboard_from_rows(rows)
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    results_path = root / "summary.jsonl"
    leaderboard_path = root / "leaderboard.json"
    write_jsonl(results_path, rows)
    write_json(leaderboard_path, leaderboard)
    manifest = {
        "protocol": PROTOCOL_RESULTS_MANIFEST,
        "validator_hotkey": validator_hotkey,
        "window": state.get("window"),
        "round": state.get("round"),
        "last_weight_block": state.get("last_weight_block"),
        "generated_at": time.time(),
        "summary_file": results_path.name,
        "leaderboard_file": leaderboard_path.name,
        "rows": len(rows),
        "valid_rows": sum(1 for row in rows if row["valid"]),
        "summary_root": root_for_values(rows),
        "leaderboard_root": root_for_values(leaderboard),
    }
    write_json(root / "manifest.json", manifest)
    return manifest


def verify_results_bundle(bundle_dir: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    root = Path(bundle_dir)
    manifest = load_json(root / "manifest.json")
    if not isinstance(manifest, dict) or manifest.get("protocol") != PROTOCOL_RESULTS_MANIFEST:
        raise ValueError("results manifest has an unsupported protocol")
    results_file = manifest.get("summary_file")
    leaderboard_file = manifest.get("leaderboard_file")
    if not isinstance(results_file, str) or not isinstance(leaderboard_file, str):
        raise ValueError("results manifest is missing file names")
    if results_file != "summary.jsonl" or leaderboard_file != "leaderboard.json":
        raise ValueError("results manifest contains forbidden file names")
    rows = _load_jsonl_exact(root / results_file)
    leaderboard = load_json(root / leaderboard_file)
    if not isinstance(leaderboard, list):
        raise ValueError("leaderboard file must contain a list")
    if manifest.get("rows") != len(rows):
        raise ValueError("results row count does not match manifest")
    if manifest.get("summary_root") != root_for_values(rows):
        raise ValueError("summary root does not match manifest")
    if manifest.get("leaderboard_root") != root_for_values(leaderboard):
        raise ValueError("leaderboard root does not match manifest")
    return manifest, rows, leaderboard


def miner_result(bundle_dir: str | Path, hotkey: str) -> dict[str, Any]:
    _manifest, rows, _leaderboard = verify_results_bundle(bundle_dir)
    for row in rows:
        if row.get("hotkey") == hotkey:
            return row
    raise ValueError(f"miner hotkey not found in results: {hotkey}")


def leaderboard(bundle_dir: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    _manifest, _rows, rows = verify_results_bundle(bundle_dir)
    if limit is None:
        return rows
    if limit <= 0:
        raise ValueError("leaderboard limit must be positive")
    return rows[:limit]


def log_results_to_wandb(
    *,
    bundle_dir: str | Path,
    project: str | None = None,
    entity: str | None = None,
    run_name: str | None = None,
) -> dict[str, Any]:
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("W&B logging requires the 'wandb' package") from exc
    project = project or os.environ.get("WANDB_PROJECT")
    entity = entity or os.environ.get("WANDB_ENTITY") or None
    if not project:
        raise ValueError("set WANDB_PROJECT or pass --wandb-project")
    manifest, rows, board = verify_results_bundle(bundle_dir)
    root = Path(bundle_dir)
    run = wandb.init(
        project=project,
        entity=entity,
        name=run_name or os.environ.get("WANDB_RUN_NAME") or f"feval-window-{manifest.get('window')}",
        job_type="validator-results",
        config=manifest,
    )
    table = wandb.Table(
        columns=["rank", "hotkey", "uid", "status", "valid", "score", "final_weight", "error"],
        data=[
            [
                row.get("rank"),
                row.get("hotkey"),
                row.get("uid"),
                row.get("status"),
                row.get("valid"),
                row.get("score"),
                row.get("final_weight"),
                row.get("error"),
            ]
            for row in board
        ],
    )
    run.log(
        {
            "window": manifest.get("window"),
            "miners": len(rows),
            "valid_miners": manifest.get("valid_rows"),
            "leaderboard": table,
        }
    )
    artifact_name = f"feval-results-window-{manifest.get('window')}"
    artifact = wandb.Artifact(artifact_name, type="feval-results", metadata=manifest)
    artifact.add_file(str(root / "manifest.json"))
    artifact.add_file(str(root / manifest["summary_file"]))
    artifact.add_file(str(root / manifest["leaderboard_file"]))
    run.log_artifact(artifact, aliases=["latest", f"window-{manifest.get('window')}"])
    artifact_ref = f"{project}/{artifact_name}:latest"
    if entity:
        artifact_ref = f"{entity}/{artifact_ref}"
    run.finish()
    return {"project": project, "entity": entity, "artifact": artifact_ref}


def download_wandb_results(*, artifact: str, out_dir: str | Path) -> dict[str, Any]:
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("W&B result download requires the 'wandb' package") from exc
    api = wandb.Api()
    downloaded = api.artifact(artifact, type="feval-results").download(root=str(out_dir))
    manifest, _rows, _leaderboard = verify_results_bundle(downloaded)
    return {"out_dir": str(downloaded), "window": manifest.get("window"), "rows": manifest.get("rows")}
