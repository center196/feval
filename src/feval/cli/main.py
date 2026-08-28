from __future__ import annotations

import argparse
import functools
import http.server
import os
import shutil
import socketserver
import urllib.request
from pathlib import Path

from ..chain import publish_commitment, serve_axon, set_weights
from ..core.config import NetworkConfig, load_network_config
from ..core.constants import DEFAULT_ROLLOUT_BATCH_SIZE, MAX_PROMPT_CHARS
from ..utils.crypto import write_key
from ..datasets.dataset import (
    DEFAULT_SPLIT,
    INSTRUCTION_DATASET,
    MATH_DATASET,
    prepare_combined_eval,
    prepare_candidate_pool_from_config,
)
from ..utils.jsonutil import write_json
from ..utils.ops import check_health
from ..protocol import build_submission, create_demo_files, train_mock_adapter
from ..nodes.results import (
    download_wandb_results,
    discover_running_wandb_results,
    export_results_bundle,
    log_results_to_wandb,
    miner_result,
    verify_results_bundle,
)
from ..nodes import MinerRolloutRunner, ValidatorRunner, publish_miner_model, publish_miner_rollouts
from ..models.random_lora import DEFAULT_RANDOM_TARGET_MODULES, write_random_lora
from ..utils.ui import FevalHelpFormatter, banner, fail, print_rows_table, print_table
from ..nodes.validator import audit_submission, promote_candidate, score_submission, write_weights


def _print_result(title: str, value: dict) -> None:
    print_table(title, {key: item for key, item in value.items() if isinstance(item, (str, int, float, bool)) or item is None})


def _load_env_file(path: str | Path = ".env") -> None:
    source = Path(path)
    if not source.exists():
        return
    for line in source.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or key in os.environ:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ[key] = value


def _hf_token_from_env() -> str | None:
    token_path = os.environ.get("HF_TOKEN_PATH")
    if token_path:
        value = Path(token_path).read_text(encoding="utf-8").strip()
        return value or None
    value = os.environ.get("HF_TOKEN")
    return value or None


def cmd_init(args: argparse.Namespace) -> None:
    paths = create_demo_files(args.out)
    _print_result("created demo subnet files", paths)


def cmd_keygen(args: argparse.Namespace) -> None:
    key = write_key(args.out, args.hotkey, args.keyring)
    _print_result("created miner key", {"hotkey": key["hotkey"], "out": args.out, "keyring": args.keyring})


def cmd_dataset_prepare(args: argparse.Namespace) -> None:
    try:
        manifest = prepare_combined_eval(
            out_path=args.out,
            manifest_path=args.manifest,
            math_input_file=args.math_input_file,
            instruction_input_file=args.instruction_input_file,
            math_dataset=args.math_dataset,
            instruction_dataset=args.instruction_dataset,
            split=args.split,
            scan_limit=args.scan_limit,
            max_rows=args.max_rows,
            math_rows=args.math_rows,
            instruction_rows=args.instruction_rows,
            max_prompt_chars=args.max_prompt_chars,
        )
    except Exception as exc:
        fail(str(exc))
    _print_result("prepared evaluation set", {
        "base_model": manifest["base_model"],
        "math_dataset": manifest["datasets"]["math"],
        "instruction_dataset": manifest["datasets"]["instruction_follow"],
        "rows": manifest["rows"],
        "math": manifest["tasks"]["math"],
        "instruction_follow": manifest["tasks"]["instruction_follow"],
        "evaluation_root": manifest["evaluation_root"],
        "out": args.out,
        "manifest": args.manifest,
    })


def cmd_dataset_candidate_pool(args: argparse.Namespace) -> None:
    try:
        config = load_network_config(args.config, production=False)
        manifest = prepare_candidate_pool_from_config(
            config,
            out_path=args.out,
            manifest_path=args.manifest,
            math_input_file=args.math_input_file,
            instruction_input_file=args.instruction_input_file,
            scan_limit=args.scan_limit,
        )
        result = {
            "candidate_pool_root": manifest["evaluation_root"],
            "rows": manifest["rows"],
            "math": manifest["tasks"]["math"],
            "instruction_follow": manifest["tasks"]["instruction_follow"],
            "out": args.out,
            "manifest": args.manifest,
        }
        _print_result("prepared candidate pool", result)
    except Exception as exc:
        fail(str(exc))


def cmd_miner_train(args: argparse.Namespace) -> None:
    adapter = train_mock_adapter(args.config, args.train, args.key, args.out, args.parent)
    _print_result("trained mock adapter", {"hotkey": adapter["hotkey"], "base_hash": adapter["base_hash"], "out": args.out})


def cmd_miner_evaluate(args: argparse.Namespace) -> None:
    submission = build_submission(args.config, args.eval, args.adapter, args.key, args.out, args.epoch)
    _print_result("wrote miner submission", {
        "miner_hotkey": submission["miner_hotkey"],
        "score_claimed": submission["score_claimed"],
        "answer_root": submission["answer_root"],
        "rollout_root": submission["rollout_root"],
        "out": args.out,
    })


def cmd_miner_serve(args: argparse.Namespace) -> None:
    directory = Path(args.dir).resolve()
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(directory))
    with socketserver.ThreadingTCPServer((args.host, args.port), handler) as server:
        print(f"hosting miner artifacts from {directory} at http://{args.host}:{args.port}")
        server.serve_forever()


def cmd_miner_publish_model(args: argparse.Namespace) -> None:
    try:
        result = publish_miner_model(
            config_path=args.config,
            adapter_dir=args.adapter_dir,
            model_repo=args.model_repo,
            rollout_repo=args.rollout_repo,
            wallet=args.wallet,
            hotkey=args.wallet_hotkey,
            network=args.network,
            wallet_path=args.wallet_path,
            private=args.private,
            create_repos=not args.no_create_repos,
            dry_run_chain=args.dry_run_chain,
            hf_token=_hf_token_from_env(),
        )
    except Exception as exc:
        fail(str(exc))
    commitment = result["commitment"]
    _print_result(
        "published model",
        {
            "model_repo": commitment["m"],
            "model_revision": commitment["r"],
            "model_digest": commitment["h"],
            "rollout_repo": commitment["d"],
            "chain_dry_run": args.dry_run_chain,
        },
    )


def cmd_miner_rollout(args: argparse.Namespace) -> None:
    try:
        result = publish_miner_rollouts(
            config_path=args.config,
            adapter_dir=args.adapter_dir,
            wallet=args.wallet,
            hotkey=args.wallet_hotkey,
            network=args.network,
            wallet_path=args.wallet_path,
            work_dir=args.work_dir,
            batch_size=args.batch_size,
            hf_token=_hf_token_from_env(),
        )
    except Exception as exc:
        fail(str(exc))
    _print_result("published rollout window", result)


def cmd_miner_watch_rollouts(args: argparse.Namespace) -> None:
    try:
        if args.force and not args.once:
            fail("--force is only allowed with --once, to avoid repeated same-window uploads")
        runner = MinerRolloutRunner(
            config_path=args.config,
            adapter_dir=args.adapter_dir,
            wallet=args.wallet,
            hotkey=args.wallet_hotkey,
            network=args.network,
            wallet_path=args.wallet_path,
            work_dir=args.work_dir,
            batch_size=args.batch_size,
            poll_seconds=args.poll_seconds,
            hf_token=_hf_token_from_env(),
        )
        if args.once:
            with runner.process_lock():
                report = runner.cycle(force=args.force)
            _print_result("miner rollout watcher", report)
        else:
            runner.run_forever()
    except KeyboardInterrupt:
        print("miner rollout watcher stopped")
    except Exception as exc:
        fail(str(exc))


def cmd_miner_random_lora(args: argparse.Namespace) -> None:
    try:
        config = load_network_config(args.config) if args.config else NetworkConfig()
        result = write_random_lora(
            out_dir=args.out_dir,
            config=config,
            rank=args.rank,
            alpha=args.alpha,
            target_modules=tuple(args.target_modules),
            layers=args.layers,
            seed=args.seed,
            scale=args.scale,
        )
    except Exception as exc:
        fail(str(exc))
    _print_result("created random LoRA", result)


def _results_dir_from_args(args: argparse.Namespace) -> str:
    results_dir = getattr(args, "results_dir", None)
    artifact = getattr(args, "wandb_artifact", None)
    if results_dir and artifact:
        fail("pass either --results-dir or --wandb-artifact, not both")
    if results_dir:
        return str(results_dir)
    if artifact:
        cache_dir = getattr(args, "cache_dir", None) or ".feval-results"
        try:
            report = download_wandb_results(artifact=artifact, out_dir=cache_dir)
        except Exception as exc:
            fail(str(exc))
        return str(report["out_dir"])
    fail("pass --results-dir or --wandb-artifact")


def _check_expected_summary_root(args: argparse.Namespace, results_dir: str) -> None:
    expected = getattr(args, "expected_summary_root", None)
    if not expected:
        return
    try:
        manifest, _rows, _leaderboard = verify_results_bundle(results_dir)
    except Exception as exc:
        fail(str(exc))
    if manifest.get("summary_root") != expected:
        fail("results summary root does not match --expected-summary-root")


def cmd_miner_status(args: argparse.Namespace) -> None:
    try:
        results_dir = _results_dir_from_args(args)
        _check_expected_summary_root(args, results_dir)
        result = miner_result(results_dir, args.hotkey)
    except Exception as exc:
        fail(str(exc))
    _print_result(
        "miner result",
        {
            "hotkey": result["hotkey"],
            "uid": result.get("uid"),
            "status": result.get("status"),
            "audit_status": result.get("audit_status"),
            "valid": result.get("valid"),
            "score": result.get("score"),
            "final_weight": result.get("final_weight"),
            "model_digest": result.get("model_digest"),
            "rollout_revision": result.get("rollout_revision"),
            "audited_count": result.get("audited_count"),
            "audit_round": result.get("audit_round"),
            "audit_required_rounds": result.get("audit_required_rounds"),
            "audit_total_rounds": result.get("audit_total_rounds"),
            "audit_exact_match_ratio": result.get("audit_exact_match_ratio"),
            "audit_block": result.get("audit_block"),
            "error": result.get("error"),
        },
    )


def cmd_miner_leaderboard(args: argparse.Namespace) -> None:
    try:
        result_dirs = list(args.results_dir or [])
        artifacts = list(args.wandb_artifact or [])
        if result_dirs and artifacts:
            fail("pass --results-dir or --wandb-artifact, not both")
        if not result_dirs and not artifacts:
            artifacts = discover_running_wandb_results(
                project=args.wandb_project,
                entity=args.wandb_entity,
            )
            if not artifacts:
                fail("no running Feval validator result jobs were found in W&B")
        if artifacts:
            cache_root = Path(args.cache_dir or ".feval-results")
            result_dirs = []
            for index, artifact in enumerate(artifacts, start=1):
                report = download_wandb_results(
                    artifact=artifact,
                    out_dir=cache_root / f"validator-{index}",
                )
                result_dirs.append(str(report["out_dir"]))
        expected_roots = list(args.expected_summary_root or [])
        if expected_roots and len(expected_roots) != len(result_dirs):
            fail("pass one --expected-summary-root for each validator result source")
        views = []
        used_labels: set[str] = set()
        for index, result_dir in enumerate(result_dirs, start=1):
            manifest, rows, _board = verify_results_bundle(result_dir)
            if expected_roots and manifest.get("summary_root") != expected_roots[index - 1]:
                fail(f"validator {index} summary root does not match --expected-summary-root")
            base_label = str(manifest.get("validator_hotkey") or f"validator-{index}")
            label = base_label if len(base_label) <= 14 else f"{base_label[:7]}...{base_label[-4:]}"
            if label in used_labels:
                label = f"{label}#{index}"
            used_labels.add(label)
            views.append((label, {str(row["hotkey"]): row for row in rows}))
    except Exception as exc:
        fail(str(exc))

    def result_cell(row: dict | None) -> str:
        if row is None:
            return "-"
        score = float(row.get("score") or 0.0)
        if float(row.get("final_weight") or 0.0) > 0.0:
            return f"KING {score:.4f}"
        if row.get("valid"):
            return f"VALID {score:.4f}"
        status = str(row.get("audit_status") or row.get("status") or "invalid").lower()
        if status == "auditing":
            return f"AUDIT {int(row.get('audit_round') or 0)}/{int(row.get('audit_total_rounds') or 20)}"
        if status == "retrying":
            return f"RETRY {int(row.get('audit_round') or 0)}/{int(row.get('audit_total_rounds') or 20)}"
        if status == "blacklisted":
            return "BLACKLIST"
        return "INVALID"

    miners = sorted({hotkey for _label, rows in views for hotkey in rows})

    def order_key(hotkey: str):
        miner_rows = [rows.get(hotkey) for _label, rows in views]
        king_count = sum(
            float(row.get("final_weight") or 0.0) > 0.0
            for row in miner_rows
            if row is not None
        )
        valid_rows = [row for row in miner_rows if row is not None and row.get("valid")]
        mean_score = (
            sum(float(row.get("score") or 0.0) for row in valid_rows) / len(valid_rows)
            if valid_rows
            else 0.0
        )
        return (-king_count, -len(valid_rows), -mean_score, hotkey)

    miners.sort(key=order_key)
    miners = miners[: args.limit]
    table_rows = []
    for rank, hotkey in enumerate(miners, start=1):
        available = [rows[hotkey] for _label, rows in views if hotkey in rows]
        uid = next((row.get("uid") for row in available if row.get("uid") is not None), None)
        table_row = {
            "rank": rank,
            "miner": hotkey if len(hotkey) <= 20 else f"{hotkey[:8]}...{hotkey[-6:]}",
            "uid": uid,
        }
        for index, (_label, rows) in enumerate(views):
            table_row[f"validator_{index}"] = result_cell(rows.get(hotkey))
        table_rows.append(table_row)
    columns = [
        ("rank", "RANK", "right"),
        ("uid", "UID", "right"),
        ("miner", "MINER", "left"),
        *[
            (f"validator_{index}", label, "left")
            for index, (label, _rows) in enumerate(views)
        ],
    ]
    print_rows_table("Feval validator leaderboard", columns, table_rows)
    print("  KING=rewarded  VALID=eligible  AUDIT=collecting rounds  INVALID=failed")


def cmd_validator_fetch(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(args.url, timeout=args.timeout) as response:
        out.write_bytes(response.read())
    _print_result("fetched miner artifact", {"url": args.url, "out": str(out)})


def cmd_validator_score(args: argparse.Namespace) -> None:
    report = score_submission(args.config, args.eval, args.submission, args.out, args.keyring)
    _print_result("wrote score report", {"miner_hotkey": report["miner_hotkey"], "valid": report["valid"], "score": report["score"], "out": args.out})


def cmd_validator_audit(args: argparse.Namespace) -> None:
    report = audit_submission(args.config, args.eval, args.submission, args.out, args.seed, args.rows, args.keyring)
    _print_result("wrote audit report", {"miner_hotkey": report["miner_hotkey"], "valid": report["valid"], "audited": len(report["audited_rows"]), "out": args.out})


def cmd_validator_promote(args: argparse.Namespace) -> None:
    state = promote_candidate(args.state, args.candidate_score, args.candidate_audit, args.out, args.king_score, args.delta_min, args.confidence_z)
    decision = state["last_promotion_decision"]
    _print_result("updated champion state", {
        "promoted": decision["promoted"],
        "reason": decision["reason"],
        "current_king": state.get("champions", [{}])[0].get("miner_hotkey"),
        "out": args.out,
    })


def cmd_validator_weights(args: argparse.Namespace) -> None:
    try:
        report = write_weights(args.state, args.score_reports, args.audit_reports, args.out, args.uid_map)
    except Exception as exc:
        fail(str(exc))
    _print_result("wrote validator weights", {"miners": len(report["weights"]), "out": args.out})


def cmd_validator_run(args: argparse.Namespace) -> None:
    runner = None
    try:
        runner = ValidatorRunner(
            config_path=args.config,
            wallet=args.wallet,
            hotkey=args.wallet_hotkey,
            network=args.network,
            wallet_path=args.wallet_path,
            work_dir=args.work_dir,
            state_path=args.state,
            dry_run_weights=args.dry_run_weights,
            poll_seconds=args.poll_seconds,
        )
        if args.once:
            with runner.process_lock():
                report = runner.cycle()
                runner._heartbeat(healthy=True, report=report)
            _print_result("validator cycle", report)
        else:
            runner.run_forever()
    except KeyboardInterrupt:
        print("validator stopped")
    except Exception as exc:
        fail(str(exc))
    finally:
        if runner is not None:
            runner.close()


def cmd_health(args: argparse.Namespace) -> None:
    try:
        report = check_health(args.state, max_age_seconds=args.max_age_seconds)
    except Exception as exc:
        fail(str(exc))
    _print_result("validator health", report)


def cmd_validator_export_results(args: argparse.Namespace) -> None:
    try:
        manifest = export_results_bundle(
            state_path=args.state,
            out_dir=args.out_dir,
            validator_hotkey=args.validator_hotkey,
        )
    except Exception as exc:
        fail(str(exc))
    _print_result(
        "exported validator results",
        {
            "window": manifest.get("window"),
            "rows": manifest.get("rows"),
            "valid_rows": manifest.get("valid_rows"),
            "summary_root": manifest.get("summary_root"),
            "leaderboard_root": manifest.get("leaderboard_root"),
            "out_dir": args.out_dir,
        },
    )


def cmd_validator_log_wandb(args: argparse.Namespace) -> None:
    try:
        report = log_results_to_wandb(
            bundle_dir=args.results_dir,
            project=args.wandb_project,
            entity=args.wandb_entity,
            run_name=args.wandb_run_name,
        )
    except Exception as exc:
        fail(str(exc))
    _print_result("logged validator results to wandb", report)


def cmd_chain_serve_axon(args: argparse.Namespace) -> None:
    try:
        result = serve_axon(args.netuid, args.ip, args.port, args.wallet, args.wallet_hotkey, args.network, args.wallet_path, args.dry_run)
    except Exception as exc:
        fail(str(exc))
    _print_result("chain serve-axon", result)


def cmd_chain_commit(args: argparse.Namespace) -> None:
    try:
        result = publish_commitment(args.netuid, args.submission, args.wallet, args.wallet_hotkey, args.network, args.wallet_path, args.dry_run)
    except Exception as exc:
        fail(str(exc))
    _print_result("chain commitment", result)


def cmd_chain_set_weights(args: argparse.Namespace) -> None:
    try:
        result = set_weights(args.netuid, args.weights, args.wallet, args.wallet_hotkey, args.network, args.wallet_path, args.dry_run, args.mechid, args.version_key)
    except Exception as exc:
        fail(str(exc))
    _print_result("chain set-weights", result)


def cmd_demo(args: argparse.Namespace) -> None:
    out = Path(args.out)
    if out.exists() and args.clean:
        shutil.rmtree(out)
    create_demo_files(out)
    write_key(out / "miner-1-key.json", "miner-1", out / "keyring.json")
    train_mock_adapter(out / "subnet.json", out / "train.jsonl", out / "miner-1-key.json", out / "miner-1-adapter.json")
    build_submission(out / "subnet.json", out / "eval.jsonl", out / "miner-1-adapter.json", out / "miner-1-key.json", out / "miner-1-submission.json")
    score_submission(out / "subnet.json", out / "eval.jsonl", out / "miner-1-submission.json", out / "miner-1-score.json", out / "keyring.json")
    audit_submission(out / "subnet.json", out / "eval.jsonl", out / "miner-1-submission.json", out / "miner-1-audit.json", "demo-future-block-randomness", 4, out / "keyring.json")
    promote_candidate(out / "champions.json", out / "miner-1-score.json", out / "miner-1-audit.json", out / "champions.json")
    write_json(out / "uid-map.json", {"miner-1": 0})
    write_weights(out / "champions.json", [str(out / "miner-1-score.json")], [str(out / "miner-1-audit.json")], out / "weights.json", out / "uid-map.json")
    write_json(out / "COMMANDS.json", {
        "miner_train": f"feval miner train --config {out / 'subnet.json'} --train {out / 'train.jsonl'} --key {out / 'miner-1-key.json'} --out {out / 'miner-1-adapter.json'}",
        "miner_evaluate": f"feval miner evaluate --config {out / 'subnet.json'} --eval {out / 'eval.jsonl'} --adapter {out / 'miner-1-adapter.json'} --key {out / 'miner-1-key.json'} --out {out / 'miner-1-submission.json'}",
        "validator_score": f"feval validator score --config {out / 'subnet.json'} --eval {out / 'eval.jsonl'} --submission {out / 'miner-1-submission.json'} --keyring {out / 'keyring.json'} --out {out / 'miner-1-score.json'}",
        "validator_audit": f"feval validator audit --config {out / 'subnet.json'} --eval {out / 'eval.jsonl'} --submission {out / 'miner-1-submission.json'} --keyring {out / 'keyring.json'} --seed demo-future-block-randomness --rows 4 --out {out / 'miner-1-audit.json'}",
        "validator_weights": f"feval validator weights --state {out / 'champions.json'} --score-reports {out / 'miner-1-score.json'} --audit-reports {out / 'miner-1-audit.json'} --uid-map {out / 'uid-map.json'} --out {out / 'weights.json'}",
    })
    print(f"demo complete: {out}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="feval",
        description=banner(),
        formatter_class=FevalHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    health = sub.add_parser(
        "health",
        help="Check that a validator is healthy and its heartbeat is fresh.",
        formatter_class=FevalHelpFormatter,
    )
    health.add_argument("--state", required=True)
    health.add_argument("--max-age-seconds", type=int, default=900)
    health.set_defaults(func=cmd_health)

    dataset = sub.add_parser("dataset", help="Prepare cheap-verifiable evaluation sets.", formatter_class=FevalHelpFormatter)
    dataset_sub = dataset.add_subparsers(dest="dataset_command", required=True)
    prepare = dataset_sub.add_parser("prepare", help="Build a combined NVIDIA math + instruction-following eval set.", formatter_class=FevalHelpFormatter)
    prepare.add_argument("--math-dataset", default=MATH_DATASET)
    prepare.add_argument("--instruction-dataset", default=INSTRUCTION_DATASET)
    prepare.add_argument("--split", default=DEFAULT_SPLIT)
    prepare.add_argument("--math-input-file", help="Local JSONL/JSON/CSV/parquet export for the math dataset.")
    prepare.add_argument("--instruction-input-file", help="Local JSONL/JSON/CSV/parquet export for the instruction-following dataset.")
    prepare.add_argument("--scan-limit", type=int, help="Maximum raw rows to scan before filtering.")
    prepare.add_argument("--max-rows", type=int, default=10_000)
    prepare.add_argument("--math-rows", type=int, help="Math row budget. Defaults to half of --max-rows.")
    prepare.add_argument("--instruction-rows", type=int, help="Instruction-following row budget. Defaults to remaining rows.")
    prepare.add_argument("--max-prompt-chars", type=int, default=MAX_PROMPT_CHARS)
    prepare.add_argument("--out", required=True)
    prepare.add_argument("--manifest", required=True)
    prepare.set_defaults(func=cmd_dataset_prepare)
    candidate_pool = dataset_sub.add_parser(
        "candidate-pool",
        help="Independently derive the deterministic candidate pool and print its root.",
        formatter_class=FevalHelpFormatter,
    )
    candidate_pool.add_argument("--config", default="network.json")
    candidate_pool.add_argument("--math-input-file", help="Local JSONL/JSON/CSV/parquet export for the math dataset.")
    candidate_pool.add_argument("--instruction-input-file", help="Local JSONL/JSON/CSV/parquet export for the instruction-following dataset.")
    candidate_pool.add_argument("--scan-limit", type=int, help="Maximum raw rows to scan before filtering.")
    candidate_pool.add_argument("--out", required=True)
    candidate_pool.add_argument("--manifest", required=True)
    candidate_pool.set_defaults(func=cmd_dataset_candidate_pool)

    miner = sub.add_parser("miner", help="Miner commands.", formatter_class=FevalHelpFormatter)
    miner_sub = miner.add_subparsers(dest="miner_command", required=True)
    status = miner_sub.add_parser(
        "status",
        help="Read this miner's public validator result summary.",
        formatter_class=FevalHelpFormatter,
    )
    status.add_argument("--hotkey", required=True)
    status.add_argument("--results-dir")
    status.add_argument("--wandb-artifact")
    status.add_argument("--cache-dir")
    status.add_argument("--expected-summary-root")
    status.set_defaults(func=cmd_miner_status)
    board = miner_sub.add_parser(
        "leaderboard",
        help="Read the public validator leaderboard summary.",
        formatter_class=FevalHelpFormatter,
    )
    board.add_argument("--results-dir", action="append")
    board.add_argument("--wandb-artifact", action="append")
    board.add_argument("--cache-dir")
    board.add_argument(
        "--wandb-project",
        default=os.environ.get("WANDB_PROJECT", "feval-subnet-47"),
        help="W&B project used for automatic running-validator discovery.",
    )
    board.add_argument(
        "--wandb-entity",
        default=os.environ.get("WANDB_ENTITY") or None,
        help="W&B entity used for discovery. Defaults to the logged-in account.",
    )
    board.add_argument("--limit", type=int, default=20)
    board.add_argument("--expected-summary-root", action="append")
    board.set_defaults(func=cmd_miner_leaderboard)
    publish_model = miner_sub.add_parser(
        "publish-model",
        help="Upload a validated LoRA and commit its immutable Hub revision on chain.",
        formatter_class=FevalHelpFormatter,
    )
    publish_model.add_argument(
        "--config",
        default="network.json",
        help="Network config path. Defaults to ./network.json or the bundled subnet-47 default.",
    )
    publish_model.add_argument("--adapter-dir", required=True)
    publish_model.add_argument("--model-repo", required=True, help="Hugging Face model repo, for example org/miner-lora.")
    publish_model.add_argument("--rollout-repo", required=True, help="Hugging Face dataset repo updated each window.")
    publish_model.add_argument("--private", action="store_true")
    publish_model.add_argument("--no-create-repos", action="store_true")
    publish_model.add_argument("--dry-run-chain", action="store_true", help="Upload to Hub but only plan the chain commitment.")
    _add_runtime_wallet_args(publish_model)
    publish_model.set_defaults(func=cmd_miner_publish_model)
    random_lora = miner_sub.add_parser(
        "random-lora",
        help="Create a seeded zero-effect LoRA adapter for submit-path testing.",
        formatter_class=FevalHelpFormatter,
    )
    random_lora.add_argument("--config", help="Optional network config. Defaults to Feval subnet 47 constants.")
    random_lora.add_argument("--out-dir", required=True)
    random_lora.add_argument("--rank", type=int, default=4)
    random_lora.add_argument("--alpha", type=float)
    random_lora.add_argument("--target-modules", nargs="+", default=list(DEFAULT_RANDOM_TARGET_MODULES))
    random_lora.add_argument("--layers", type=int, help="Defaults to all base-model layers.")
    random_lora.add_argument("--seed", type=int, default=0)
    random_lora.add_argument(
        "--scale",
        type=float,
        default=0.01,
        help="Scale for seeded random A tensors; B is zero, so model output is unchanged.",
    )
    random_lora.set_defaults(func=cmd_miner_random_lora)
    rollout = miner_sub.add_parser(
        "rollout",
        help=(
            "Run greedy vLLM inference for the current 10,000-row window and update "
            "the Hugging Face rollout dataset repo from this hotkey's on-chain model commitment."
        ),
        formatter_class=FevalHelpFormatter,
    )
    rollout.add_argument(
        "--config",
        default="network.json",
        help="Network config path. Defaults to code-pinned subnet-47 constants.",
    )
    rollout.add_argument("--adapter-dir", required=True)
    rollout.add_argument("--work-dir", required=True)
    rollout.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_ROLLOUT_BATCH_SIZE,
        help="Maximum active vLLM sequences. Raise on large GPUs, lower if memory is tight.",
    )
    _add_runtime_wallet_args(rollout)
    rollout.set_defaults(func=cmd_miner_rollout)
    watch_rollouts = miner_sub.add_parser(
        "watch-rollouts",
        help=(
            "Continuously publish rollouts to the Hugging Face dataset repo from this "
            "hotkey's on-chain model commitment whenever the dataset window or model changes."
        ),
        formatter_class=FevalHelpFormatter,
    )
    watch_rollouts.add_argument(
        "--config",
        default="network.json",
        help="Network config path. Defaults to code-pinned subnet-47 constants.",
    )
    watch_rollouts.add_argument("--adapter-dir", required=True)
    watch_rollouts.add_argument("--work-dir", required=True)
    watch_rollouts.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_ROLLOUT_BATCH_SIZE,
        help="Maximum active vLLM sequences. Raise on large GPUs, lower if memory is tight.",
    )
    watch_rollouts.add_argument("--poll-seconds", type=int, default=60)
    watch_rollouts.add_argument("--once", action="store_true", help="Check the current window once and exit.")
    watch_rollouts.add_argument("--force", action="store_true", help="With --once, regenerate even if this model/window was already uploaded.")
    _add_runtime_wallet_args(watch_rollouts)
    watch_rollouts.set_defaults(func=cmd_miner_watch_rollouts)

    validator = sub.add_parser("validator", help="Validator commands.", formatter_class=FevalHelpFormatter)
    validator_sub = validator.add_subparsers(dest="validator_command", required=True)
    run = validator_sub.add_parser(
        "run",
        help="Discover miners, audit rollouts, score the full configured subset, and set weights.",
        formatter_class=FevalHelpFormatter,
    )
    run.add_argument(
        "--config",
        default="network.json",
        help="Network config path. Defaults to code-pinned subnet-47 constants.",
    )
    run.add_argument("--work-dir", required=True)
    run.add_argument("--state", required=True)
    run.add_argument("--once", action="store_true", help="Run one cycle and exit.")
    run.add_argument(
        "--poll-seconds",
        type=int,
        help="Seconds between validator cycles. Defaults to the protocol audit interval.",
    )
    run.add_argument("--dry-run-weights", action="store_true")
    _add_runtime_wallet_args(run)
    run.set_defaults(func=cmd_validator_run)
    export_results = validator_sub.add_parser(
        "export-results",
        help="Export public per-miner summary results and a leaderboard.",
        formatter_class=FevalHelpFormatter,
    )
    export_results.add_argument("--state", required=True)
    export_results.add_argument("--out-dir", required=True)
    export_results.add_argument("--validator-hotkey")
    export_results.set_defaults(func=cmd_validator_export_results)
    log_wandb = validator_sub.add_parser(
        "log-wandb",
        help="Log an exported public summary bundle to Weights & Biases.",
        formatter_class=FevalHelpFormatter,
    )
    log_wandb.add_argument("--results-dir", required=True)
    log_wandb.add_argument("--wandb-project")
    log_wandb.add_argument("--wandb-entity")
    log_wandb.add_argument("--wandb-run-name")
    log_wandb.set_defaults(func=cmd_validator_log_wandb)

    dev = sub.add_parser(
        "dev",
        help="Local mock protocol fixtures (never used by mainnet commands).",
        formatter_class=FevalHelpFormatter,
    )
    dev_sub = dev.add_subparsers(dest="dev_command", required=True)
    demo = dev_sub.add_parser(
        "demo", help="Run the local mock miner/validator smoke test.", formatter_class=FevalHelpFormatter
    )
    demo.add_argument("--out", required=True)
    demo.add_argument("--clean", action="store_true")
    demo.set_defaults(func=cmd_demo)
    return parser


def _add_chain_wallet_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--network", "-n", default="finney")
    parser.add_argument("--wallet", "-w", required=True)
    parser.add_argument("--wallet-hotkey", "-H")
    parser.add_argument("--wallet-path")
    parser.add_argument("--dry-run", action="store_true")


def _add_runtime_wallet_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--network", "-n", default="finney")
    parser.add_argument("--wallet", "-w", required=True)
    parser.add_argument("--wallet-hotkey", "-H")
    parser.add_argument("--wallet-path")


def main() -> None:
    _load_env_file()
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()



