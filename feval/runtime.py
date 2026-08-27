from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

from .artifacts import ModelCommitment, validate_rollout_bundle
from .chain import (
    block_hash,
    finalized_block,
    publish_model_commitment,
    read_model_commitments,
    scan_model_commitment_history,
    set_weight_mapping,
    wallet_hotkey_ss58,
)
from .champions import champion_weight_mapping, encode_reward_bits, update_champions
from .config import NetworkConfig, load_network_config
from .constants import PROTOCOL_MINER_ROLLOUT_STATE, PROTOCOL_VALIDATOR_STATE
from .dataset import prepare_candidate_pool_from_config, prepare_window_from_pool
from .hub import (
    resolve_rollout_revision,
    safe_download_model,
    safe_download_rollouts,
    upload_model_adapter,
    upload_rollout_bundle,
)
from .inference import (
    VllmAuditEngine,
    build_rollout_bundle_vllm,
    decode_rollout,
    load_protocol_tokenizer,
    protocol_rollout_tokens,
    tokenizer_vocab_size,
)
from .jsonutil import load_json, load_jsonl
from .merkle import root_for_values
from .ops import ProcessLock, health_path_for_state
from .rewards import reward_for_row
from .results import export_results_bundle, log_results_to_wandb
from .schedule import (
    audit_seed,
    choose_audit_ids,
    dataset_window,
    evaluation_seed,
    required_audit_rounds,
)


PENDING_AUDIT_POLL_SECONDS = 30


def _atomic_write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    if target.exists():
        backup = target.with_name(target.name + ".bak")
        shutil.copyfile(target, backup)
        with backup.open("r+b") as stream:
            os.fsync(stream.fileno())
    os.replace(temporary, target)
    if os.name != "nt":
        directory_fd = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)


def _initial_state() -> dict[str, Any]:
    return {
        "protocol": PROTOCOL_VALIDATOR_STATE,
        "window": None,
        "last_weight_block": None,
        "round": 0,
        "pending": {},
        "results": {},
        "audited": {},
        "first_seen": {},
        "history_cursor": None,
        "champions": [],
        "last_promotion_decision": None,
    }


def _normalize_state(state: Any) -> dict[str, Any]:
    if not isinstance(state, dict):
        raise ValueError("validator state is not an object")
    if state.get("protocol") in {
        "feval-validator-state-v2",
        "feval-validator-state-v3",
        "feval-validator-state-v4",
        "feval-validator-state-v5",
        "feval-validator-state-v6",
        "feval-validator-state-v7",
        "feval-validator-state-v8",
        "feval-validator-state-v9",
    }:
        state["protocol"] = PROTOCOL_VALIDATOR_STATE
        state["window"] = None
        state["last_weight_block"] = None
        state["pending"] = {}
        state["results"] = {}
        state["audited"] = {}
        state["first_seen"] = {}
        state["history_cursor"] = None
        state["champions"] = []
        state["last_promotion_decision"] = None
        state.pop("last_weights", None)
    if state.get("protocol") != PROTOCOL_VALIDATOR_STATE:
        raise ValueError("validator state has an unsupported protocol")
    for name, default in _initial_state().items():
        state.setdefault(name, default)
    for name in ("pending", "results", "audited", "first_seen"):
        if not isinstance(state[name], dict):
            raise ValueError(f"validator state field {name} must be an object")
    if not isinstance(state["champions"], list):
        raise ValueError("validator state champions must be a list")
    if isinstance(state["round"], bool) or not isinstance(state["round"], int) or state["round"] < 0:
        raise ValueError("validator state round is invalid")
    for name in ("window", "last_weight_block", "history_cursor"):
        value = state[name]
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
            raise ValueError(f"validator state field {name} is invalid")
    return state


def _load_state(path: str | Path) -> dict[str, Any]:
    state_path = Path(path)
    if not state_path.exists():
        return _initial_state()
    errors: list[Exception] = []
    for candidate in (state_path, state_path.with_name(state_path.name + ".bak")):
        if not candidate.exists():
            continue
        try:
            return _normalize_state(load_json(candidate))
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            errors.append(exc)
    if errors:
        raise ValueError(f"validator state and backup are unusable: {errors[-1]}") from errors[-1]
    return _initial_state()


def ensure_evaluation_files(
    config: NetworkConfig,
    *,
    window: int,
    directory: str | Path,
    math_input_file: str | Path | None = None,
    instruction_input_file: str | Path | None = None,
    scan_limit: int | None = None,
) -> tuple[Path, Path, dict[str, Any]]:
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    eval_path = root / f"eval-{window}.jsonl"
    manifest_path = root / f"eval-{window}.manifest.json"
    if eval_path.exists() and manifest_path.exists():
        manifest = load_json(manifest_path)
        if (
            manifest.get("dataset_window") == window
            and manifest.get("evaluation_seed") == evaluation_seed(config.netuid, window)
            and manifest.get("evaluation_root") == root_for_values(load_jsonl(eval_path))
            and manifest.get("rows") == config.evaluation_rows
        ):
            return eval_path, manifest_path, manifest
    pool_path = root / "candidate-pool.jsonl"
    pool_manifest_path = root / "candidate-pool.manifest.json"
    pool_valid = False
    if pool_path.exists() and pool_manifest_path.exists():
        try:
            pool_manifest = load_json(pool_manifest_path)
            pool_valid = (
                pool_manifest.get("kind") == "candidate_pool"
                and pool_manifest.get("dataset_revisions")
                == {
                    "math": config.math_revision,
                    "instruction_follow": config.instruction_revision,
                }
                and pool_manifest.get("evaluation_root") == root_for_values(load_jsonl(pool_path))
                and (
                    config.candidate_pool_root is None
                    or pool_manifest.get("evaluation_root") == config.candidate_pool_root
                )
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pool_valid = False
    if not pool_valid:
        prepare_candidate_pool_from_config(
            config,
            out_path=pool_path,
            manifest_path=pool_manifest_path,
            math_input_file=math_input_file,
            instruction_input_file=instruction_input_file,
            scan_limit=scan_limit,
        )
    pool_manifest = load_json(pool_manifest_path)
    if (
        config.candidate_pool_root is not None
        and pool_manifest.get("evaluation_root") != config.candidate_pool_root
    ):
        raise ValueError(
            "candidate pool root differs from the sealed network config; refusing to evaluate"
        )
    manifest = prepare_window_from_pool(
        config,
        window=window,
        pool_path=pool_path,
        pool_manifest_path=pool_manifest_path,
        out_path=eval_path,
        manifest_path=manifest_path,
    )
    return eval_path, manifest_path, manifest


def publish_miner_model(
    *,
    config_path: str | Path,
    adapter_dir: str | Path,
    model_repo: str,
    rollout_repo: str,
    wallet: str,
    hotkey: str | None,
    network: str,
    wallet_path: str | None,
    private: bool,
    create_repos: bool,
    dry_run_chain: bool,
    hf_token: str | bool | None = None,
) -> dict[str, Any]:
    # Publishing a model only needs subnet identity, base model, and LoRA
    # validation rules. The sealed candidate pool is required later for rollout
    # generation and validation, not for the first model commitment.
    config = load_network_config(config_path, production=False)
    commitment = upload_model_adapter(
        adapter_dir=adapter_dir,
        model_repo=model_repo,
        rollout_repo=rollout_repo,
        config=config,
        token=hf_token,
        private=private,
        create_repos=create_repos,
    )
    chain_result = publish_model_commitment(
        netuid=config.netuid,
        commitment=commitment,
        wallet=wallet,
        hotkey=hotkey,
        network=network,
        wallet_path=wallet_path,
        dry_run=dry_run_chain,
    )
    if not dry_run_chain and (chain_result.get("success") is False or chain_result.get("error")):
        raise RuntimeError(f"model commitment submission failed: {chain_result}")
    return {"commitment": commitment.compact_dict(), "chain": chain_result}


def publish_miner_rollouts(
    *,
    config_path: str | Path,
    adapter_dir: str | Path,
    wallet: str,
    hotkey: str | None,
    network: str,
    wallet_path: str | None,
    work_dir: str | Path,
    batch_size: int,
    scan_limit: int | None = None,
    math_input_file: str | Path | None = None,
    instruction_input_file: str | Path | None = None,
    hf_token: str | bool | None = None,
) -> dict[str, Any]:
    # Miners do not need a sealed candidate-pool root to generate rollouts.
    # The rollout manifest binds the evaluation root, and validators recompute
    # their own expected root before accepting any miner output.
    config = load_network_config(config_path, production=False)
    miner_hotkey = wallet_hotkey_ss58(wallet=wallet, hotkey=hotkey, wallet_path=wallet_path)
    block = finalized_block(network=network)
    rows = read_model_commitments(netuid=config.netuid, network=network, block=block)
    own = [row for row in rows if row["hotkey"] == miner_hotkey and row.get("uid") is not None]
    if not own:
        raise ValueError("this hotkey has no Feval model commitment on chain")
    current = max(own, key=lambda row: row["commit_block"])
    commitment: ModelCommitment = current["commitment"]
    window = dataset_window(block, config.dataset_window_blocks)
    work = Path(work_dir)
    eval_path, manifest_path, _ = ensure_evaluation_files(
        config,
        window=window,
        directory=work / "evaluation",
        math_input_file=math_input_file,
        instruction_input_file=instruction_input_file,
        scan_limit=scan_limit,
    )
    bundle = work / "rollouts" / str(window)
    manifest = build_rollout_bundle_vllm(
        config=config,
        eval_path=eval_path,
        eval_manifest_path=manifest_path,
        adapter_dir=adapter_dir,
        commitment=commitment,
        miner_hotkey=miner_hotkey,
        out_dir=bundle,
        batch_size=batch_size,
    )
    rollout_revision = upload_rollout_bundle(
        bundle_dir=bundle,
        rollout_repo=commitment.rollout_repo,
        token=hf_token,
    )
    return {
        "hotkey": miner_hotkey,
        "block": block,
        "dataset_window": window,
        "model_revision": commitment.model_revision,
        "model_digest": commitment.model_digest,
        "rollout_repo": commitment.rollout_repo,
        "rollout_revision": rollout_revision,
        "rows": manifest.row_count,
    }


def _miner_rollout_state_path(work_dir: str | Path) -> Path:
    return Path(work_dir) / "miner-rollouts.state.json"


def _load_miner_rollout_state(work_dir: str | Path) -> dict[str, Any]:
    path = _miner_rollout_state_path(work_dir)
    if not path.exists():
        return {"protocol": PROTOCOL_MINER_ROLLOUT_STATE, "last_success": None}
    state = load_json(path)
    if isinstance(state, dict) and state.get("protocol") in {
        "feval-miner-rollout-state-v1",
        "feval-miner-rollout-state-v2",
        "feval-miner-rollout-state-v3",
        "feval-miner-rollout-state-v4",
    }:
        return {"protocol": PROTOCOL_MINER_ROLLOUT_STATE, "last_success": None}
    if not isinstance(state, dict) or state.get("protocol") != PROTOCOL_MINER_ROLLOUT_STATE:
        raise ValueError("miner rollout state has an unsupported protocol")
    state.setdefault("last_success", None)
    return state


def _current_miner_model_commitment(
    *,
    config: NetworkConfig,
    wallet: str,
    hotkey: str | None,
    network: str,
    wallet_path: str | None,
    block: int,
) -> tuple[str, dict[str, Any]]:
    miner_hotkey = wallet_hotkey_ss58(wallet=wallet, hotkey=hotkey, wallet_path=wallet_path)
    rows = read_model_commitments(netuid=config.netuid, network=network, block=block)
    own = [row for row in rows if row["hotkey"] == miner_hotkey and row.get("uid") is not None]
    if not own:
        raise ValueError("this hotkey has no Feval model commitment on chain")
    return miner_hotkey, max(own, key=lambda row: row["commit_block"])


class MinerRolloutRunner:
    def __init__(
        self,
        *,
        config_path: str | Path,
        adapter_dir: str | Path,
        wallet: str,
        hotkey: str | None,
        network: str,
        wallet_path: str | None,
        work_dir: str | Path,
        batch_size: int,
        poll_seconds: int = 60,
        hf_token: str | bool | None = None,
        scan_limit: int | None = None,
        math_input_file: str | Path | None = None,
        instruction_input_file: str | Path | None = None,
    ) -> None:
        if poll_seconds < 5:
            raise ValueError("poll_seconds must be at least 5")
        # Validators reject any rollout whose evaluation root differs from the
        # root derived from the code-pinned protocol constants.
        self.config = load_network_config(config_path, production=False)
        self.adapter_dir = Path(adapter_dir)
        self.wallet = wallet
        self.hotkey = hotkey
        self.network = network
        self.wallet_path = wallet_path
        self.work_dir = Path(work_dir)
        self.batch_size = batch_size
        self.poll_seconds = poll_seconds
        self.hf_token = hf_token
        self.scan_limit = scan_limit
        self.math_input_file = math_input_file
        self.instruction_input_file = instruction_input_file
        self.state = _load_miner_rollout_state(self.work_dir)

    def process_lock(self) -> ProcessLock:
        state_path = _miner_rollout_state_path(self.work_dir)
        return ProcessLock(state_path.with_name(state_path.name + ".lock"))

    def cycle(self, *, force: bool = False) -> dict[str, Any]:
        block = finalized_block(network=self.network)
        window = dataset_window(block, self.config.dataset_window_blocks)
        miner_hotkey, current = _current_miner_model_commitment(
            config=self.config,
            wallet=self.wallet,
            hotkey=self.hotkey,
            network=self.network,
            wallet_path=self.wallet_path,
            block=block,
        )
        commitment: ModelCommitment = current["commitment"]
        target = {
            "hotkey": miner_hotkey,
            "dataset_window": window,
            "model_revision": commitment.model_revision,
            "model_digest": commitment.model_digest,
            "commit_block": current["commit_block"],
        }
        last = self.state.get("last_success")
        if not force and isinstance(last, dict) and all(last.get(key) == value for key, value in target.items()):
            return {
                "status": "current",
                "block": block,
                "hotkey": miner_hotkey,
                "dataset_window": window,
                "model_digest": commitment.model_digest,
                "model_revision": commitment.model_revision,
                "rollout_repo": commitment.rollout_repo,
                "rollout_revision": last.get("rollout_revision"),
                "rows": last.get("rows"),
            }
        eval_path, manifest_path, _ = ensure_evaluation_files(
            self.config,
            window=window,
            directory=self.work_dir / "evaluation",
            math_input_file=self.math_input_file,
            instruction_input_file=self.instruction_input_file,
            scan_limit=self.scan_limit,
        )
        bundle = self.work_dir / "rollouts" / str(window)
        manifest = build_rollout_bundle_vllm(
            config=self.config,
            eval_path=eval_path,
            eval_manifest_path=manifest_path,
            adapter_dir=self.adapter_dir,
            commitment=commitment,
            miner_hotkey=miner_hotkey,
            out_dir=bundle,
            batch_size=self.batch_size,
        )
        rollout_revision = upload_rollout_bundle(
            bundle_dir=bundle,
            rollout_repo=commitment.rollout_repo,
            token=self.hf_token,
        )
        success = {
            **target,
            "block": block,
            "rollout_repo": commitment.rollout_repo,
            "rollout_revision": rollout_revision,
            "rows": manifest.row_count,
            "updated_at": time.time(),
        }
        self.state = {"protocol": PROTOCOL_MINER_ROLLOUT_STATE, "last_success": success}
        _atomic_write_json(_miner_rollout_state_path(self.work_dir), self.state)
        return {"status": "uploaded", **success}

    def run_forever(self) -> None:
        failures = 0
        with self.process_lock():
            while True:
                try:
                    report = self.cycle()
                    failures = 0
                    print(json.dumps(report, sort_keys=True), flush=True)
                    delay = self.poll_seconds
                except KeyboardInterrupt:
                    raise
                except Exception as exc:
                    failures += 1
                    error = f"{type(exc).__name__}: {exc}"
                    print(json.dumps({"status": "unhealthy", "error": error}), flush=True)
                    delay = min(self.poll_seconds, 15 * (2 ** min(failures - 1, 5)))
                time.sleep(delay)


def _current_commitments(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        old = latest.get(row["hotkey"])
        if old is None or row["commit_block"] > old["commit_block"]:
            latest[row["hotkey"]] = row
    return list(latest.values())


def _copy_filtered(rows: list[dict[str, Any]], state: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    first_seen = state.setdefault("first_seen", {})
    for row in sorted(rows, key=lambda item: (item["commit_block"], item["hotkey"])):
        digest = row["commitment"].model_digest
        existing = first_seen.get(digest)
        candidate = {"hotkey": row["hotkey"], "block": row["commit_block"]}
        if existing is None or (candidate["block"], candidate["hotkey"]) < (existing["block"], existing["hotkey"]):
            first_seen[digest] = candidate
    eligible: list[dict[str, Any]] = []
    copies: dict[str, str] = {}
    for row in rows:
        owner = first_seen[row["commitment"].model_digest]["hotkey"]
        if row["hotkey"] == owner:
            eligible.append(row)
        else:
            copies[row["hotkey"]] = owner
    return eligible, copies


def _manifest_matches(
    manifest: Any,
    *,
    hotkey: str,
    commitment: ModelCommitment,
    window: int,
    eval_manifest: dict[str, Any],
) -> None:
    expected = {
        "miner_hotkey": hotkey,
        "model_repo": commitment.model_repo,
        "model_revision": commitment.model_revision,
        "model_digest": commitment.model_digest,
        "dataset_window": window,
        "evaluation_seed": eval_manifest["evaluation_seed"],
        "evaluation_root": eval_manifest["evaluation_root"],
    }
    for name, value in expected.items():
        if getattr(manifest, name) != value:
            raise ValueError(f"rollout manifest {name} does not match chain/evaluation state")


def score_rollouts(
    eval_rows: list[dict[str, Any]],
    rollout_rows: list[dict[str, Any]],
    tokenizer: Any,
    config: NetworkConfig | None = None,
) -> tuple[float, list[dict[str, Any]]]:
    if len(eval_rows) != len(rollout_rows):
        raise ValueError("evaluation and rollout row counts differ")
    scored: list[dict[str, Any]] = []
    protocol_config = config or NetworkConfig()
    for expected, rollout in zip(eval_rows, rollout_rows):
        if expected["row_id"] != rollout["row_id"]:
            raise ValueError("evaluation and rollout row ordering differs")
        # The answer is always derived from committed token IDs. Miner-provided
        # text is never accepted, parsed, rendered, imported, or executed.
        tokens, _prompt_ids = protocol_rollout_tokens(
            protocol_config,
            tokenizer,
            str(expected["prompt"]),
            list(rollout["tokens"]),
        )
        if not tokens:
            raise ValueError(f"rollout {rollout['row_id']!r} is empty")
        answer = decode_rollout(tokenizer, tokens)
        reward = reward_for_row(answer, expected)
        scored.append({**rollout, "tokens": tokens, "reward": reward})
    score = sum(row["reward"] for row in scored) / len(scored) if scored else 0.0
    return score, scored


def _scratch_root(work_dir: Path) -> Path:
    root = work_dir / "tmp"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _clear_audit_scratch(work_dir: Path) -> int:
    root = _scratch_root(work_dir)
    removed = 0
    resolved_root = root.resolve()
    for child in root.iterdir():
        if not child.name.startswith("audit-"):
            continue
        try:
            child.resolve().relative_to(resolved_root)
        except ValueError:
            continue
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
            removed += 1
        elif child.is_file() or child.is_symlink():
            child.unlink()
            removed += 1
    return removed


def _validator_sleep_seconds(report: dict[str, Any], base_delay: int) -> int:
    if report.get("status") == "waiting_for_audit" and report.get("pending"):
        return min(base_delay, PENDING_AUDIT_POLL_SECONDS)
    return base_delay


class ValidatorRunner:
    def __init__(
        self,
        *,
        config_path: str | Path,
        wallet: str,
        hotkey: str | None,
        network: str,
        wallet_path: str | None,
        work_dir: str | Path,
        state_path: str | Path,
        hf_token: str | bool | None = None,
        dry_run_weights: bool = False,
        scan_limit: int | None = None,
        poll_seconds: int | None = None,
        math_input_file: str | Path | None = None,
        instruction_input_file: str | Path | None = None,
    ) -> None:
        # Protocol constants are code-pinned. A sealed candidate_pool_root and
        # history_start_block are optional hardening/launch metadata; validators
        # can still run from defaults and verify miner rollout evaluation roots.
        self.config = load_network_config(config_path, production=False)
        self.wallet = wallet
        self.hotkey = hotkey
        self.network = network
        self.wallet_path = wallet_path
        self.work_dir = Path(work_dir)
        self.state_path = Path(state_path)
        self.hf_token = hf_token
        self.dry_run_weights = dry_run_weights
        self.scan_limit = scan_limit
        self.poll_seconds = poll_seconds if poll_seconds is not None else self.config.audit_interval_seconds
        if self.poll_seconds < 5:
            raise ValueError("validator poll interval must be at least 5 seconds")
        self.math_input_file = math_input_file
        self.instruction_input_file = instruction_input_file
        self.state = _load_state(self.state_path)
        self.tokenizer = None
        self.audit_engine = None
        self.report_wandb = os.environ.get("FEVAL_REPORT_WANDB", "1").strip().lower() not in {
            "0",
            "false",
            "no",
            "off",
        }
        self.wandb_project = os.environ.get("WANDB_PROJECT", "feval-subnet-47").strip()
        self.wandb_entity = os.environ.get("WANDB_ENTITY", "").strip() or None

    def process_lock(self) -> ProcessLock:
        return ProcessLock(self.state_path.with_name(self.state_path.name + ".lock"))

    def _heartbeat(self, *, healthy: bool, report: dict[str, Any] | None = None, error: str | None = None) -> None:
        previous: dict[str, Any] = {}
        path = health_path_for_state(self.state_path)
        if path.exists():
            try:
                previous = load_json(path)
            except Exception:
                previous = {}
        value = {
            "protocol": "feval-health-v1",
            "healthy": healthy,
            "pid": os.getpid(),
            "updated_at": time.time(),
            "last_success_at": time.time() if healthy else previous.get("last_success_at"),
            "report": report,
            "error": error,
        }
        _atomic_write_json(path, value)

    def _tokenizer(self):
        if self.tokenizer is None:
            self.tokenizer = load_protocol_tokenizer(self.config)
        return self.tokenizer

    def _audit_engine(self) -> VllmAuditEngine:
        if self.audit_engine is None:
            # Reuse the tokenizer already loaded for structural scoring instead
            # of initializing the same pinned tokenizer twice.
            self.audit_engine = VllmAuditEngine(self.config, tokenizer=self._tokenizer())
        return self.audit_engine

    def close(self) -> None:
        if self.audit_engine is not None:
            self.audit_engine.close()
            self.audit_engine = None

    def _discard_audit_engine(self) -> None:
        """Drop a failed GPU worker so the next retry starts from clean state."""

        engine, self.audit_engine = self.audit_engine, None
        if engine is not None:
            try:
                engine.close()
            except Exception:
                pass

    def _reset_window(self, window: int) -> None:
        self.state["window"] = window
        self.state["pending"] = {}
        self.state["results"] = {}
        self.state["audited"] = {}

    def _process_pending(
        self,
        *,
        current_block: int,
        window: int,
        commitments: dict[str, dict[str, Any]],
        eval_path: Path,
        eval_manifest: dict[str, Any],
    ) -> None:
        eval_rows = load_jsonl(eval_path)
        expected_ids = [str(row["row_id"]) for row in eval_rows]
        completed: set[str] = set()
        for hotkey, pending in list(self.state["pending"].items()):
            if int(pending["challenge_block"]) > current_block:
                continue
            current = commitments.get(hotkey)
            audit_started = time.monotonic()
            timings: dict[str, float] = {}
            try:
                if current is None:
                    raise ValueError("miner commitment disappeared")
                commitment: ModelCommitment = current["commitment"]
                if commitment.compact_dict() != pending["commitment"]:
                    raise ValueError("miner changed its model commitment before audit")
                rollout_revision = str(pending["rollout_revision"])
                scratch_parent = _scratch_root(self.work_dir)
                with tempfile.TemporaryDirectory(prefix="audit-", dir=scratch_parent) as temporary:
                    scratch = Path(temporary)
                    model_dir = scratch / "model"
                    rollout_dir = scratch / "rollout"
                    stage_started = time.monotonic()
                    runtime_model_dir = safe_download_model(
                        commitment,
                        config=self.config,
                        local_dir=model_dir,
                        token=self.hf_token,
                    )
                    timings["model_download_validate"] = round(time.monotonic() - stage_started, 3)
                    stage_started = time.monotonic()
                    safe_download_rollouts(
                        commitment.rollout_repo,
                        revision=rollout_revision,
                        local_dir=rollout_dir,
                        token=self.hf_token,
                    )
                    timings["rollout_download"] = round(time.monotonic() - stage_started, 3)
                    stage_started = time.monotonic()
                    tokenizer = self._tokenizer()
                    manifest, rollout_rows = validate_rollout_bundle(
                        rollout_dir,
                        config=self.config,
                        expected_row_ids=expected_ids,
                        vocab_size=tokenizer_vocab_size(tokenizer),
                    )
                    _manifest_matches(
                        manifest,
                        hotkey=hotkey,
                        commitment=commitment,
                        window=window,
                        eval_manifest=eval_manifest,
                    )
                    score, scored = score_rollouts(
                        eval_rows,
                        rollout_rows,
                        tokenizer,
                        self.config,
                    )
                    timings["local_validation_score"] = round(time.monotonic() - stage_started, 3)
                    history_key = (
                        f"{hotkey}:{commitment.model_digest}:{rollout_revision}:{window}"
                    )
                    progress = self.state["audited"].setdefault(
                        history_key,
                        {
                            "rows": [],
                            "rounds_passed": 0,
                            "exact_argmax_tokens": 0,
                            "tokens_checked": 0,
                        },
                    )
                    if not isinstance(progress, dict):
                        progress = {
                            "rows": [],
                            "rounds_passed": 0,
                            "exact_argmax_tokens": 0,
                            "tokens_checked": 0,
                        }
                        self.state["audited"][history_key] = progress
                    history = progress.setdefault("rows", [])
                    seed = audit_seed(
                        netuid=self.config.netuid,
                        block_hash=block_hash(network=self.network, block=int(pending["challenge_block"])),
                        hotkey=hotkey,
                        model_digest=commitment.model_digest,
                        rollout_revision=rollout_revision,
                        round_number=int(pending["round"]),
                    )
                    selected_ids = choose_audit_ids(
                        scored,
                        seed=seed,
                        count=self.config.audit_rows_per_round,
                        already_audited=history,
                    )
                    if not selected_ids:
                        raise ValueError("no unaudited rollout rows remain")
                    selected = set(selected_ids)
                    eval_by_id = {row["row_id"]: row for row in eval_rows}
                    audit_rows = [
                        {**row, "prompt": eval_by_id[row["row_id"]]["prompt"]}
                        for row in scored
                        if row["row_id"] in selected
                    ]
                    stage_started = time.monotonic()
                    try:
                        audit_reports = self._audit_engine().verify(
                            adapter_dir=runtime_model_dir,
                            model_digest=commitment.model_digest,
                            evaluation_seed=manifest.evaluation_seed,
                            rows=audit_rows,
                        )
                    except Exception as exc:
                        # Engine startup, CUDA, and forward-pass failures say
                        # nothing about miner validity. The outer recoverable
                        # path rebuilds the worker and retries this exact round.
                        raise RuntimeError("audit engine failed") from exc
                    timings["vllm_load_and_verify"] = round(time.monotonic() - stage_started, 3)
                    if len(audit_reports) != len(selected_ids):
                        raise RuntimeError("audit engine returned an incomplete report")
                    failed_audits = [report for report in audit_reports if not report["valid"]]
                    exact_tokens = sum(int(report.get("exact_argmax_tokens", 0)) for report in audit_reports)
                    tokens_checked = sum(int(report.get("tokens_checked", 0)) for report in audit_reports)
                    exact_match_ratio = exact_tokens / tokens_checked if tokens_checked else 0.0
                    audit_valid = all(report["valid"] for report in audit_reports)
                timings["total"] = round(time.monotonic() - audit_started, 3)
                history.extend(selected_ids)
                progress["exact_argmax_tokens"] = int(progress.get("exact_argmax_tokens", 0)) + exact_tokens
                progress["tokens_checked"] = int(progress.get("tokens_checked", 0)) + tokens_checked
                if audit_valid:
                    progress["rounds_passed"] = int(progress.get("rounds_passed", 0)) + 1
                required_rounds = required_audit_rounds(
                    population=len(scored),
                    rows_per_round=self.config.audit_rows_per_round,
                    min_fake_fraction=self.config.audit_min_fake_row_fraction,
                    confidence=self.config.audit_detection_confidence,
                )
                rounds_passed = int(progress.get("rounds_passed", 0))
                fully_validated = audit_valid and rounds_passed >= required_rounds
                failure_details = []
                for report in failed_audits[:3]:
                    detail = f"{report['row_id']}:{report.get('failure_reason') or 'unknown'}"
                    if report.get("failure_position") is not None:
                        detail += f"@{report['failure_position']}"
                    if report.get("failure_rank") is not None:
                        detail += f" rank={report['failure_rank']}"
                    if report.get("failure_logprob_gap") is not None:
                        detail += f" gap={float(report['failure_logprob_gap']):.6f}"
                    if report.get("failure_expected_token_id") is not None:
                        detail += f" expected={report['failure_expected_token_id']}"
                    failure_details.append(detail)
                failure_summary = "; ".join(failure_details)
                self.state["results"][hotkey] = {
                    "uid": current.get("uid"),
                    "valid": fully_validated,
                    "audit_status": (
                        "passed" if fully_validated else ("auditing" if audit_valid else "failed")
                    ),
                    "score": score if fully_validated else 0.0,
                    "correct": sum(row["reward"] for row in scored),
                    "rows": len(scored),
                    "reward_bits": encode_reward_bits([int(row["reward"]) for row in scored]),
                    "model_digest": commitment.model_digest,
                    "model_revision": commitment.model_revision,
                    "commit_block": current["commit_block"],
                    "rollout_revision": rollout_revision,
                    "audited_rows": list(history),
                    "audit_round": rounds_passed,
                    "audit_required_rounds": required_rounds,
                    "audit_reports": audit_reports,
                    "audit_exact_match_ratio": exact_match_ratio,
                    "audit_cumulative_exact_match_ratio": (
                        int(progress["exact_argmax_tokens"]) / int(progress["tokens_checked"])
                        if int(progress["tokens_checked"]) else 0.0
                    ),
                    "audit_response_tokens": sum(len(row["tokens"]) for row in audit_rows),
                    "audit_timings_seconds": timings,
                    "audit_block": int(pending["challenge_block"]),
                    "error": None
                    if audit_valid
                    else (
                        f"teacher-forced rollout mismatch ({len(failed_audits)}/{len(selected_ids)} "
                        f"audited rows failed; exact argmax ratio {exact_match_ratio:.6f} is diagnostic "
                        f"only): {failure_summary}"
                    ),
                }
                completed.add(hotkey)
            except ValueError as exc:
                timings["total"] = round(time.monotonic() - audit_started, 3)
                self.state["results"][hotkey] = {
                    "uid": current.get("uid") if current is not None else None,
                    "valid": False,
                    "score": 0.0,
                    "model_digest": pending["commitment"].get("h"),
                    "rollout_revision": pending.get("rollout_revision"),
                    "audit_status": "failed",
                    "audit_timings_seconds": timings,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                completed.add(hotkey)
            except Exception as exc:
                # Hub, chain, disk, and GPU-worker failures are operational,
                # not evidence that a miner cheated. Preserve the pending
                # audit, restart the GPU worker, and retry from clean scratch.
                timings["total"] = round(time.monotonic() - audit_started, 3)
                self._discard_audit_engine()
                pending["attempts"] = int(pending.get("attempts", 0)) + 1
                retry_blocks = min(
                    self.config.weight_interval_blocks,
                    self.config.audit_delay_blocks * (2 ** min(pending["attempts"] - 1, 5)),
                )
                pending["challenge_block"] = current_block + max(1, retry_blocks)
                self.state["results"][hotkey] = {
                    "uid": current.get("uid") if current is not None else None,
                    "valid": False,
                    "score": 0.0,
                    "model_digest": pending["commitment"].get("h"),
                    "rollout_revision": pending.get("rollout_revision"),
                    "audit_status": "retrying",
                    "audit_timings_seconds": timings,
                    "error": f"recoverable {type(exc).__name__}: {exc}",
                }
        for hotkey in completed:
            self.state["pending"].pop(hotkey, None)

    def _schedule_latest(
        self,
        *,
        current_block: int,
        eligible: list[dict[str, Any]],
    ) -> None:
        for row in eligible:
            hotkey = row["hotkey"]
            commitment: ModelCommitment = row["commitment"]
            try:
                revision = resolve_rollout_revision(commitment.rollout_repo, token=self.hf_token)
            except Exception as exc:
                self.state["results"][hotkey] = {
                    "uid": row.get("uid"),
                    "valid": False,
                    "score": 0.0,
                    "model_digest": commitment.model_digest,
                    "audit_status": "retrying",
                    "error": f"recoverable rollout pin failure: {type(exc).__name__}: {exc}",
                }
                continue
            pending = self.state["pending"].get(hotkey)
            if (
                isinstance(pending, dict)
                and pending.get("commitment") == commitment.compact_dict()
                and pending.get("rollout_revision") == revision
            ):
                continue
            if pending is not None:
                # Updating either committed model or rollout revision abandons
                # every previous audit round and starts again at round one.
                self.state["pending"].pop(hotkey, None)
            previous = self.state["results"].get(hotkey)
            previous_status = previous.get("audit_status") if isinstance(previous, dict) else None
            if previous_status is None and isinstance(previous, dict) and "valid" in previous:
                previous_status = "passed" if previous.get("valid") else "failed"
            if (
                isinstance(previous, dict)
                and previous.get("model_digest") == commitment.model_digest
                and previous.get("rollout_revision") == revision
                and previous_status in {"passed", "failed"}
            ):
                continue
            history_key = f"{hotkey}:{commitment.model_digest}:{revision}:{self.state.get('window')}"
            progress = self.state["audited"].get(history_key, {})
            next_round = int(progress.get("rounds_passed", 0)) + 1 if isinstance(progress, dict) else 1
            if not (
                isinstance(previous, dict)
                and previous.get("model_digest") == commitment.model_digest
                and previous.get("rollout_revision") == revision
            ):
                # Each revision is an independent statistical trial.
                prefix = f"{hotkey}:"
                self.state["audited"] = {
                    key: value
                    for key, value in self.state["audited"].items()
                    if not str(key).startswith(prefix)
                }
                self.state["results"].pop(hotkey, None)
                next_round = 1
            self.state["round"] = int(self.state.get("round", 0)) + 1
            self.state["pending"][hotkey] = {
                "commitment": commitment.compact_dict(),
                "commit_block": row["commit_block"],
                "rollout_revision": revision,
                "pinned_block": current_block,
                "challenge_block": current_block + self.config.audit_delay_blocks,
                "round": next_round,
                "attempts": 0,
            }

    def _maybe_set_weights(
        self,
        *,
        current_block: int,
        commitments: dict[str, dict[str, Any]],
        copies: dict[str, str],
    ) -> dict[str, Any] | None:
        last = self.state.get("last_weight_block")
        if last is not None and current_block - int(last) < self.config.weight_interval_blocks:
            return None
        update_champions(self.state, config=self.config, current_block=current_block)
        normalized = champion_weight_mapping(
            self.state,
            config=self.config,
            commitments=commitments,
            copies=copies,
        )
        if not normalized:
            # UID 0 is the protocol burn target. Never reward a zero-score
            # model merely because no positive champion exists yet.
            normalized = {0: 1.0}
        result = set_weight_mapping(
            netuid=self.config.netuid,
            uid_weights=normalized,
            wallet=self.wallet,
            hotkey=self.hotkey,
            network=self.network,
            wallet_path=self.wallet_path,
            dry_run=self.dry_run_weights,
            mechid=self.config.mechanism_id,
            version_key=self.config.weights_version_key,
        )
        if result.get("success") is False or result.get("error"):
            raise RuntimeError(f"weight submission failed: {result}")
        self.state["last_weight_block"] = current_block
        self.state["last_weights"] = {str(uid): value for uid, value in normalized.items()}
        return result

    def _maybe_report_results(self) -> dict[str, Any]:
        """Publish only the public summary bundle; reporting never affects consensus."""

        if not getattr(self, "report_wandb", False):
            return {"status": "disabled"}
        try:
            out_dir = self.work_dir / "results" / f"window-{self.state.get('window')}"
            manifest = export_results_bundle(
                state_path=self.state_path,
                out_dir=out_dir,
                validator_hotkey=self.hotkey,
            )
            summary_root = str(manifest["summary_root"])
            if self.state.get("last_wandb_summary_root") == summary_root:
                return {"status": "unchanged", "summary_root": summary_root}
            report = log_results_to_wandb(
                bundle_dir=out_dir,
                project=self.wandb_project,
                entity=self.wandb_entity,
            )
            self.state["last_wandb_summary_root"] = summary_root
            _atomic_write_json(self.state_path, self.state)
            return {"status": "published", "summary_root": summary_root, **report}
        except Exception as exc:
            # W&B is a public convenience mirror, never a validator dependency.
            return {"status": "unavailable", "error": f"{type(exc).__name__}: {exc}"}

    def _sync_history(self, *, current_block: int) -> tuple[int, bool]:
        if self.config.history_start_block is None:
            # Without a history boundary, begin indexing at the finalized block.
            # Active commitments remain subject to their on-chain commit block.
            self.state["history_cursor"] = current_block
            self.state.setdefault("history_start_block_assumed", current_block)
            return 0, True
        launch = int(self.config.history_start_block)
        cursor_value = self.state.get("history_cursor")
        cursor = launch - 1 if cursor_value is None else int(cursor_value)
        if cursor < launch - 1:
            raise ValueError("validator history cursor precedes the configured launch block")
        if cursor > current_block:
            raise ValueError("validator history cursor is ahead of the finalized chain")
        if cursor == current_block:
            return 0, True
        end = min(current_block, cursor + self.config.history_batch_blocks)
        rows = scan_model_commitment_history(
            netuid=self.config.netuid,
            network=self.network,
            start_block=cursor + 1,
            end_block=end,
        )
        _copy_filtered(rows, self.state)
        self.state["history_cursor"] = end
        return len(rows), end == current_block

    def sync_history_once(self) -> dict[str, Any]:
        current_block = finalized_block(network=self.network)
        records, ready = self._sync_history(current_block=current_block)
        _atomic_write_json(self.state_path, self.state)
        return {
            "status": "ready" if ready else "syncing_history",
            "history_cursor": self.state["history_cursor"],
            "history_target": current_block,
            "history_records": records,
            "ready": ready,
        }

    def cycle(self) -> dict[str, Any]:
        scratch_cleaned = _clear_audit_scratch(self.work_dir)
        current_block = finalized_block(network=self.network)
        history_rows, history_ready = self._sync_history(current_block=current_block)
        if not history_ready:
            _atomic_write_json(self.state_path, self.state)
            return {
                "block": current_block,
                "status": "syncing_history",
                "history_cursor": self.state["history_cursor"],
                "history_target": current_block,
                "history_records": history_rows,
                "scratch_cleaned": scratch_cleaned,
                "weights_submitted": False,
            }
        window = dataset_window(current_block, self.config.dataset_window_blocks)
        if self.state.get("window") != window:
            self._reset_window(window)
        eval_path, _, eval_manifest = ensure_evaluation_files(
            self.config,
            window=window,
            directory=self.work_dir / "evaluation",
            math_input_file=self.math_input_file,
            instruction_input_file=self.instruction_input_file,
            scan_limit=self.scan_limit,
        )
        rows = _current_commitments(
            [
                row
                for row in read_model_commitments(
                    netuid=self.config.netuid,
                    network=self.network,
                    block=current_block,
                )
                if row.get("uid") is not None
            ]
        )
        eligible, copies = _copy_filtered(rows, self.state)
        current = {row["hotkey"]: row for row in rows}
        for hotkey, owner in copies.items():
            self.state["results"][hotkey] = {
                "uid": current.get(hotkey, {}).get("uid"),
                "valid": False,
                "score": 0.0,
                "error": f"model digest was committed first by {owner}",
            }
        self._process_pending(
            current_block=current_block,
            window=window,
            commitments=current,
            eval_path=eval_path,
            eval_manifest=eval_manifest,
        )
        self._schedule_latest(current_block=current_block, eligible=eligible)
        weight_result = self._maybe_set_weights(
            current_block=current_block,
            commitments=current,
            copies=copies,
        )
        _atomic_write_json(self.state_path, self.state)
        wandb_report = self._maybe_report_results()
        pending_blocks = [
            int(item["challenge_block"])
            for item in self.state["pending"].values()
            if item.get("challenge_block") is not None
        ]
        evaluated = len(self.state["results"])
        valid_results = sum(1 for item in self.state["results"].values() if item.get("valid"))
        auditing_results = sum(
            1
            for item in self.state["results"].values()
            if item.get("audit_status") in {"auditing", "retrying"}
        )
        invalid_results = evaluated - valid_results - auditing_results
        errors = [str(item["error"]) for item in self.state["results"].values() if item.get("error")]
        timed_results = [
            (hotkey, item)
            for hotkey, item in self.state["results"].items()
            if isinstance(item.get("audit_timings_seconds"), dict)
        ]
        latest_timed = (
            max(timed_results, key=lambda pair: int(pair[1].get("audit_block") or -1))
            if timed_results
            else None
        )
        status = "waiting_for_audit" if pending_blocks else ("scored" if evaluated else "idle")
        if weight_result is not None:
            status = "weights_submitted"
        return {
            "block": current_block,
            "status": status,
            "window": window,
            "miners": len(rows),
            "eligible": len(eligible),
            "copies": len(copies),
            "pending": len(self.state["pending"]),
            "next_audit_block": min(pending_blocks) if pending_blocks else None,
            "audit_delay_blocks": self.config.audit_delay_blocks,
            "evaluated": evaluated,
            "valid_results": valid_results,
            "invalid_results": invalid_results,
            "auditing_results": auditing_results,
            "result_errors": len(errors),
            "result_error_sample": errors[0] if errors else None,
            "audit_timing_sample": (
                {
                    "hotkey": latest_timed[0],
                    "response_tokens": latest_timed[1].get("audit_response_tokens"),
                    "seconds": latest_timed[1]["audit_timings_seconds"],
                }
                if latest_timed is not None
                else None
            ),
            "weights_submitted": weight_result is not None,
            "history_cursor": self.state["history_cursor"],
            "champions": len(self.state.get("champions", [])),
            "scratch_cleaned": scratch_cleaned,
            "wandb_results": wandb_report,
        }

    def run_forever(self) -> None:
        failures = 0
        try:
            with self.process_lock():
                while True:
                    try:
                        report = self.cycle()
                        delay = _validator_sleep_seconds(report, self.poll_seconds)
                        report = {**report, "next_cycle_seconds": delay}
                        self._heartbeat(healthy=True, report=report)
                        failures = 0
                        print(json.dumps(report, sort_keys=True), flush=True)
                    except KeyboardInterrupt:
                        raise
                    except Exception as exc:
                        failures += 1
                        error = f"{type(exc).__name__}: {exc}"
                        self._heartbeat(healthy=False, error=error)
                        print(json.dumps({"status": "unhealthy", "error": error}), flush=True)
                        # Remain fail-closed and retry transient chain/Hub failures.
                        # No state or weights are advanced by a failed cycle.
                        delay = min(self.poll_seconds, 15 * (2 ** min(failures - 1, 5)))
                    time.sleep(delay)
        finally:
            self.close()
