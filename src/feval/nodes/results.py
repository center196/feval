from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from ..core.constants import PUBLIC_WANDB_ENTITY, PUBLIC_WANDB_PROJECT
from ..utils.jsonutil import load_json, write_json, write_jsonl
from ..protocol.merkle import root_for_values


PROTOCOL_RESULTS_MANIFEST = "feval-results-v3"
SUPPORTED_RESULTS_MANIFESTS = {
    "feval-results-v1",
    "feval-results-v2",
    PROTOCOL_RESULTS_MANIFEST,
}


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


def _public_invalid_reason(result: dict[str, Any], status: str) -> str | None:
    """Return a stable audit-gate reason without exposing internal exceptions."""

    if "invalid_reason" in result:
        reason = result.get("invalid_reason")
        return reason if isinstance(reason, str) and reason else None
    copy_kind = result.get("copy_kind")
    copy_source = result.get("copy_source")
    if copy_kind in {"model", "rollout"}:
        suffix = f" of {copy_source}" if isinstance(copy_source, str) and copy_source else ""
        return f"{copy_kind} copy{suffix}"
    if status in {"invalid", "blacklisted"}:
        return "invalid rollout"
    return None


def result_rows_from_state(state: dict[str, Any]) -> list[dict[str, Any]]:
    results = state.get("results", {})
    if not isinstance(results, dict):
        raise ValueError("validator state results must be an object")
    carryover = state.get("carryover_results", {})
    if not isinstance(carryover, dict):
        raise ValueError("validator state carryover_results must be an object")
    combined = dict(carryover)
    # A current-window result always supersedes the public carryover snapshot,
    # even while it is still auditing or has failed this particular revision.
    combined.update(results)
    last_weights = state.get("last_weights", {})
    if not isinstance(last_weights, dict):
        last_weights = {}
    rows: list[dict[str, Any]] = []
    for hotkey, result in sorted(combined.items()):
        if not isinstance(result, dict):
            continue
        uid = result.get("uid")
        uid_key = str(uid) if uid is not None else None
        final_weight = float(last_weights.get(uid_key, 0.0)) if uid_key is not None else 0.0
        valid = bool(result.get("valid"))
        audit_status = result.get("audit_status")
        status = (
            "carried"
            if result.get("carryover")
            else (
                _status(valid, final_weight)
                if valid
                else (
                    str(audit_status)
                    if audit_status in {"auditing", "retrying", "blacklisted"}
                    else _status(valid, final_weight)
                )
            )
        )
        invalid_reason = _public_invalid_reason(result, status)
        row = {
            "hotkey": str(hotkey),
            "uid": uid,
            "valid": valid,
            "status": status,
            "audit_status": audit_status,
            "score": float(result.get("score", 0.0)),
            "raw_score": result.get("raw_score"),
            "category_scores": result.get("category_scores"),
            "category_rows": result.get("category_rows"),
            "final_weight": final_weight,
            "correct": result.get("correct"),
            "rows": result.get("rows"),
            "model_digest": result.get("model_digest"),
            "model_repo": result.get("model_repo"),
            "model_revision": result.get("model_revision"),
            "commit_block": result.get("commit_block"),
            "rollout_repo": result.get("rollout_repo"),
            "rollout_revision": result.get("rollout_revision"),
            "audit_block": result.get("audit_block"),
            "audited_count": len(result.get("audited_rows", [])),
            "audit_round": result.get("audit_round"),
            "audit_required_rounds": result.get("audit_required_rounds"),
            "audit_total_rounds": result.get("audit_total_rounds"),
            "audit_exact_match_ratio": result.get("audit_cumulative_exact_match_ratio"),
            "blacklisted_until_block": result.get("blacklisted_until_block"),
            "source_window": result.get("source_window"),
            "invalid_reason": invalid_reason,
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
            "model_repo": row.get("model_repo"),
            "rollout_repo": row.get("rollout_repo"),
            "audit_status": row.get("audit_status"),
            "audit_round": row.get("audit_round"),
            "audit_required_rounds": row.get("audit_required_rounds"),
            "audit_total_rounds": row.get("audit_total_rounds"),
            "audited_count": row.get("audited_count"),
            "invalid_reason": row.get("invalid_reason"),
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
    if not isinstance(manifest, dict) or manifest.get("protocol") not in SUPPORTED_RESULTS_MANIFESTS:
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


def start_wandb_results_run(
    *,
    bundle_dir: str | Path,
    run_name: str | None = None,
    run_id: str | None = None,
) -> Any:
    """Start or resume one continuous validator/window W&B run."""

    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("W&B logging requires the 'wandb' package") from exc
    manifest, _rows, _board = verify_results_bundle(bundle_dir)
    options: dict[str, Any] = {
        "project": PUBLIC_WANDB_PROJECT,
        "entity": PUBLIC_WANDB_ENTITY,
        "mode": "online",
        "name": run_name
        or os.environ.get("WANDB_RUN_NAME")
        or f"feval-window-{manifest.get('window')}",
        "job_type": "validator-results",
        "config": manifest,
    }
    if run_id:
        options.update({"id": run_id, "resume": "allow"})
    return wandb.init(**options)


def log_results_to_wandb(
    *,
    bundle_dir: str | Path,
    run_name: str | None = None,
    run_id: str | None = None,
    run: Any | None = None,
) -> dict[str, Any]:
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("W&B logging requires the 'wandb' package") from exc
    manifest, rows, board = verify_results_bundle(bundle_dir)
    root = Path(bundle_dir)
    owns_run = run is None
    if run is None:
        run = start_wandb_results_run(
            bundle_dir=bundle_dir,
            run_name=run_name,
            run_id=run_id,
        )
    table = wandb.Table(
        columns=[
            "rank",
            "hotkey",
            "uid",
            "status",
            "audit_status",
            "audit_round",
            "audit_required_rounds",
            "audit_total_rounds",
            "audited_count",
            "valid",
            "score",
            "final_weight",
            "model_repo",
            "rollout_repo",
            "invalid_reason",
        ],
        data=[
            [
                row.get("rank"),
                row.get("hotkey"),
                row.get("uid"),
                row.get("status"),
                row.get("audit_status"),
                row.get("audit_round"),
                row.get("audit_required_rounds"),
                row.get("audit_total_rounds"),
                row.get("audited_count"),
                row.get("valid"),
                row.get("score"),
                row.get("final_weight"),
                row.get("model_repo"),
                row.get("rollout_repo"),
                row.get("invalid_reason"),
            ]
            for row in board
        ],
    )
    run.log(
        {
            "window": manifest.get("window"),
            "miners": len(rows),
            "valid_miners": manifest.get("valid_rows"),
            "audit_round": max(
                (int(row.get("audit_round") or 0) for row in rows),
                default=0,
            ),
            "summary_root": manifest.get("summary_root"),
            "leaderboard": table,
        }
    )
    validator = str(manifest.get("validator_hotkey") or "unknown-validator")
    validator_slug = "".join(
        character if character.isalnum() or character in {"-", "_"} else "-"
        for character in validator
    )
    artifact_name = f"feval-results-{validator_slug}-window-{manifest.get('window')}"
    artifact = wandb.Artifact(artifact_name, type="feval-results", metadata=manifest)
    artifact.add_file(str(root / "manifest.json"))
    artifact.add_file(str(root / manifest["summary_file"]))
    artifact.add_file(str(root / manifest["leaderboard_file"]))
    logged_artifact = run.log_artifact(
        artifact,
        aliases=["latest", f"window-{manifest.get('window')}"],
    )
    wait_for_upload = getattr(logged_artifact, "wait", None)
    if callable(wait_for_upload):
        wait_for_upload()
    artifact_ref = (
        f"{PUBLIC_WANDB_ENTITY}/{PUBLIC_WANDB_PROJECT}/{artifact_name}:latest"
    )
    if owns_run:
        run.finish()
    return {
        "project": PUBLIC_WANDB_PROJECT,
        "entity": PUBLIC_WANDB_ENTITY,
        "artifact": artifact_ref,
        "run_id": getattr(run, "id", run_id),
        "run_url": getattr(run, "url", None),
    }


def download_wandb_results(*, artifact: str, out_dir: str | Path) -> dict[str, Any]:
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("W&B result download requires the 'wandb' package") from exc
    api = wandb.Api()
    downloaded = api.artifact(artifact, type="feval-results").download(root=str(out_dir))
    manifest, _rows, _leaderboard = verify_results_bundle(downloaded)
    return {"out_dir": str(downloaded), "window": manifest.get("window"), "rows": manifest.get("rows")}


def discover_running_wandb_results() -> list[str]:
    """Return the newest result artifact from every running validator job."""

    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("W&B result discovery requires the 'wandb' package") from exc
    api = wandb.Api()
    runs = api.runs(
        f"{PUBLIC_WANDB_ENTITY}/{PUBLIC_WANDB_PROJECT}",
        filters={"state": "running"},
    )
    references: list[str] = []
    for run in runs:
        if str(getattr(run, "job_type", "")) != "validator-results":
            continue
        artifacts = [
            artifact
            for artifact in run.logged_artifacts()
            if str(getattr(artifact, "type", "")) == "feval-results"
        ]
        if not artifacts:
            continue
        latest = [
            artifact
            for artifact in artifacts
            if "latest" in list(getattr(artifact, "aliases", []) or [])
        ]
        selected = (latest or artifacts)[-1]
        name = str(getattr(selected, "name", ""))
        if not name:
            continue
        references.append(
            name
            if name.count("/") >= 2
            else f"{PUBLIC_WANDB_ENTITY}/{PUBLIC_WANDB_PROJECT}/{name}"
        )
    return sorted(set(references))

