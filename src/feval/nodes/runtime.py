from __future__ import annotations

import json
import hashlib
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from ..models.artifacts import (
    ModelCommitment,
    adapter_digest,
    prepare_runtime_adapter,
    validate_rollout_bundle,
)
from ..chain import (
    block_hash,
    finalized_block,
    publish_model_commitment,
    read_model_commitments,
    scan_model_commitment_history,
    set_weight_mapping,
    wallet_hotkey_ss58,
)
from ..chain.champions import champion_weight_mapping, encode_reward_bits, update_champions
from ..core.config import NetworkConfig, load_network_config
from ..core.constants import HISTORY_LOOKBACK_BLOCKS, PROTOCOL_MINER_ROLLOUT_STATE, PROTOCOL_VALIDATOR_STATE
from ..datasets.dataset import prepare_candidate_pool_from_config, prepare_window_from_pool
from ..models.hub import (
    resolve_rollout_revision,
    safe_download_model,
    safe_download_rollouts,
    upload_model_adapter,
    upload_rollout_bundle,
)
from ..models.inference import (
    VllmAuditEngine,
    build_rollout_bundle_vllm,
    decode_rollout,
    load_protocol_tokenizer,
    protocol_rollout_tokens,
    tokenizer_vocab_size,
)
from ..utils.jsonutil import load_json, load_jsonl
from ..protocol.merkle import root_for_values
from ..utils.ops import ProcessLock, health_path_for_state
from ..datasets.rewards import reward_for_row
from .results import export_results_bundle, log_results_to_wandb, start_wandb_results_run
from ..protocol.schedule import (
    audit_seed,
    choose_audit_ids,
    dataset_window,
    evaluation_seed,
    required_audit_rounds,
)
from ..utils.ui import print_rows_table


PENDING_AUDIT_POLL_SECONDS = 30


def _short_hotkey(value: str) -> str:
    return value if len(value) <= 20 else f"{value[:8]}...{value[-6:]}"


def _print_round_results(*, window: int, block: int, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    print_rows_table(
        f"Validator round results (window {window}, block {block})",
        [
            ("uid", "UID", "right"),
            ("miner", "MINER", "left"),
            ("score", "SCORE", "right"),
            ("correct", "CORRECT", "right"),
            ("round", "ROUND", "right"),
            ("audit_rows", "AUDIT", "right"),
            ("exact", "ARGMAX", "right"),
            ("outcome", "OUTCOME", "left"),
            ("seconds", "TIME", "right"),
        ],
        rows,
        stream=sys.stderr,
    )
    for row in rows:
        if row.get("error"):
            print(
                f"  {_short_hotkey(str(row['hotkey']))}: {row['error']}",
                file=sys.stderr,
                flush=True,
            )


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
        "rollout_first_seen": {},
        "history_cursor": None,
        "champions": [],
        "carryover_results": {},
        "last_promotion_decision": None,
        "invalid_strikes": {},
        "blacklist": {},
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
        "feval-validator-state-v10",
        "feval-validator-state-v11",
        "feval-validator-state-v12",
        "feval-validator-state-v13",
        "feval-validator-state-v14",
        "feval-validator-state-v15",
        "feval-validator-state-v16",
        "feval-validator-state-v17",
        "feval-validator-state-v18",
        "feval-validator-state-v19",
        "feval-validator-state-v20",
        "feval-validator-state-v21",
        "feval-validator-state-v22",
        "feval-validator-state-v23",
        "feval-validator-state-v24",
        "feval-validator-state-v25",
    }:
        # Audit semantics changed. Never carry old pass/fail decisions or
        # partial rounds into a different token-validity protocol.
        state["protocol"] = PROTOCOL_VALIDATOR_STATE
        state["window"] = None
        state["last_weight_block"] = None
        state["pending"] = {}
        state["results"] = {}
        state["audited"] = {}
        state["first_seen"] = {}
        state["rollout_first_seen"] = {}
        state["history_cursor"] = None
        state["champions"] = []
        state["carryover_results"] = {}
        state["last_promotion_decision"] = None
        state["invalid_strikes"] = {}
        state["blacklist"] = {}
        state.pop("last_weights", None)
    if state.get("protocol") != PROTOCOL_VALIDATOR_STATE:
        raise ValueError("validator state has an unsupported protocol")
    for name, default in _initial_state().items():
        state.setdefault(name, default)
    for name in (
        "pending",
        "results",
        "audited",
        "first_seen",
        "rollout_first_seen",
        "carryover_results",
        "invalid_strikes",
        "blacklist",
    ):
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


def _expire_blacklist(state: dict[str, Any], *, current_block: int) -> dict[str, dict[str, Any]]:
    """Remove expired entries and return the active validator-local blacklist."""

    active: dict[str, dict[str, Any]] = {}
    blacklist = state.setdefault("blacklist", {})
    for hotkey, raw in list(blacklist.items()):
        if not isinstance(raw, dict):
            raise ValueError("validator blacklist entry must be an object")
        until = raw.get("until_block")
        if isinstance(until, bool) or not isinstance(until, int) or until < 0:
            raise ValueError("validator blacklist expiry is invalid")
        if current_block >= until:
            blacklist.pop(hotkey, None)
        else:
            active[str(hotkey)] = raw
    return active


def _record_invalid_round(
    state: dict[str, Any],
    *,
    hotkey: str,
    failure_key: str,
    current_block: int,
    config: NetworkConfig,
) -> dict[str, Any] | None:
    """Record one unique deterministic failure and blacklist on strike three."""

    if not config.blacklist_enabled:
        state["invalid_strikes"] = {}
        state["blacklist"] = {}
        return None
    active = _expire_blacklist(state, current_block=current_block)
    if hotkey in active:
        return active[hotkey]
    strikes = state.setdefault("invalid_strikes", {})
    raw = strikes.get(hotkey)
    entry = raw if isinstance(raw, dict) else {"count": 0, "failure_keys": []}
    keys = entry.get("failure_keys", [])
    if not isinstance(keys, list) or any(not isinstance(value, str) for value in keys):
        raise ValueError("validator invalid-strike evidence is malformed")
    if failure_key in keys:
        return None
    keys.append(failure_key)
    count = int(entry.get("count", 0)) + 1
    strikes[hotkey] = {
        "count": count,
        "failure_keys": keys[-config.invalid_rounds_before_blacklist :],
        "last_failure_block": current_block,
    }
    if count < config.invalid_rounds_before_blacklist:
        return None
    blacklisted = {
        "since_block": current_block,
        "until_block": current_block + config.blacklist_duration_blocks,
        "invalid_rounds": count,
        "last_failure_key": failure_key,
    }
    state.setdefault("blacklist", {})[hotkey] = blacklisted
    strikes.pop(hotkey, None)
    state.setdefault("pending", {}).pop(hotkey, None)
    return blacklisted


def _active_blacklist(
    state: dict[str, Any], *, current_block: int, config: NetworkConfig
) -> dict[str, dict[str, Any]]:
    """Return active entries, or clear all blacklist state when policy is off."""

    if not config.blacklist_enabled:
        state["invalid_strikes"] = {}
        state["blacklist"] = {}
        return {}
    return _expire_blacklist(state, current_block=current_block)


def _counts_as_invalid_strike(exc: ValueError) -> bool:
    """Lifecycle races are not miner-integrity failures."""

    return str(exc) not in {
        "miner commitment disappeared",
        "miner changed its model commitment before audit",
        "no unaudited rollout rows remain",
    }


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
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pool_valid = False
    if not pool_valid:
        prepare_candidate_pool_from_config(
            config,
            out_path=pool_path,
            manifest_path=pool_manifest_path,
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
    # validation rules.
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
    hf_token: str | bool | None = None,
) -> dict[str, Any]:
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
        max_output_tokens=config.max_output_tokens,
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
        "max_output_tokens": manifest.max_output_tokens,
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
        "feval-miner-rollout-state-v5",
        "feval-miner-rollout-state-v6",
        "feval-miner-rollout-state-v7",
        "feval-miner-rollout-state-v8",
        "feval-miner-rollout-state-v9",
        "feval-miner-rollout-state-v10",
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
    ) -> None:
        if poll_seconds < 5:
            raise ValueError("poll_seconds must be at least 5")
        # Miners can run from code-pinned protocol constants. If their local
        # dataset window differs from validators, validators reject the rollout
        # by evaluation_root; the miner cannot use it to manipulate validation.
        self.config = load_network_config(config_path, production=False)
        self.adapter_dir = Path(adapter_dir)
        self.wallet = wallet
        self.hotkey = hotkey
        self.network = network
        self.wallet_path = wallet_path
        self.work_dir = Path(work_dir)
        self.batch_size = batch_size
        self.max_output_tokens = self.config.max_output_tokens
        self.poll_seconds = poll_seconds
        self.hf_token = hf_token
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
            "max_output_tokens": self.max_output_tokens,
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
                "max_output_tokens": last.get("max_output_tokens"),
            }
        eval_path, manifest_path, _ = ensure_evaluation_files(
            self.config,
            window=window,
            directory=self.work_dir / "evaluation",
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
            max_output_tokens=self.max_output_tokens,
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


def _copy_filtered(
    rows: list[dict[str, Any]],
    state: dict[str, Any],
    *,
    minimum_block: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    first_seen = state.setdefault("first_seen", {})
    for digest, entry in list(first_seen.items()):
        if not isinstance(entry, dict) or int(entry.get("block", -1)) < minimum_block:
            first_seen.pop(digest, None)
    recent_rows = [row for row in rows if int(row["commit_block"]) >= minimum_block]
    for row in sorted(recent_rows, key=lambda item: (item["commit_block"], item["hotkey"])):
        digest = row["commitment"].model_digest
        existing = first_seen.get(digest)
        candidate = {"hotkey": row["hotkey"], "block": row["commit_block"]}
        if existing is None or (candidate["block"], candidate["hotkey"]) < (existing["block"], existing["hotkey"]):
            first_seen[digest] = candidate
    eligible: list[dict[str, Any]] = []
    copies: dict[str, str] = {}
    for row in recent_rows:
        first_miner = first_seen[row["commitment"].model_digest]["hotkey"]
        if row["hotkey"] == first_miner:
            eligible.append(row)
        else:
            copies[row["hotkey"]] = first_miner
    return eligible, copies


def _record_rollout_fingerprint(
    state: dict[str, Any],
    *,
    hotkey: str,
    commit_block: int,
    window: int,
    rows_sha256: str,
) -> tuple[str | None, str | None]:
    """Return (copy owner, displaced later owner) for exact greedy behavior."""

    fingerprints = state.setdefault("rollout_first_seen", {})
    key = f"{window}:{rows_sha256}"
    candidate = {"hotkey": hotkey, "block": int(commit_block)}
    existing = fingerprints.get(key)
    if not isinstance(existing, dict) or (candidate["block"], hotkey) < (
        int(existing["block"]),
        str(existing["hotkey"]),
    ):
        displaced = str(existing["hotkey"]) if isinstance(existing, dict) else None
        fingerprints[key] = candidate
        return None, displaced
    owner = str(existing["hotkey"])
    return (None, None) if owner == hotkey else (owner, None)


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
    *,
    max_output_tokens: int | None = None,
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
            max_output_tokens=max_output_tokens,
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


def _cache_runtime_adapter(
    *,
    work_dir: Path,
    source_dir: str | Path,
    model_digest: str,
    config: NetworkConfig,
) -> Path:
    """Keep only the validated inert LoRA files alive for vLLM's lazy loader."""

    cache_root = work_dir / "runtime-adapters"
    cache_root.mkdir(parents=True, exist_ok=True)
    target = cache_root / model_digest
    try:
        if target.is_dir() and adapter_digest(target, config) == model_digest:
            return target
    except (OSError, ValueError):
        pass
    if target.exists():
        resolved_root = cache_root.resolve()
        try:
            target.resolve().relative_to(resolved_root)
        except ValueError as exc:
            raise RuntimeError("runtime adapter cache target escapes its root") from exc
        if target.is_symlink() or not target.is_dir():
            raise RuntimeError("runtime adapter cache target is not a regular directory")
        shutil.rmtree(target)
    with tempfile.TemporaryDirectory(prefix=".adapter-", dir=cache_root) as temporary:
        prepared = prepare_runtime_adapter(
            source_dir,
            Path(temporary) / "validated-runtime",
            config,
        )
        os.replace(prepared, target)
    if adapter_digest(target, config) != model_digest:
        raise RuntimeError("persistent runtime adapter digest changed after installation")
    return target


def _validator_sleep_seconds(report: dict[str, Any], base_delay: int) -> int:
    # Pending work controls cadence independently of what else happened during
    # the cycle (for example, a successful weight submission).
    if report.get("pending"):
        return min(base_delay, PENDING_AUDIT_POLL_SECONDS)
    return base_delay


def _audit_round_valid(
    reports: list[dict[str, Any]],
    *,
    exact_match_ratio: float,
    min_exact_argmax_ratio: float,
) -> bool:
    """Apply both independent rollout-authenticity conditions to one round."""

    return (
        bool(reports)
        and all(bool(report.get("valid")) for report in reports)
        and exact_match_ratio >= min_exact_argmax_ratio
    )


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
        poll_seconds: int | None = None,
    ) -> None:
        # Protocol rules are code-pinned. Each validator independently derives
        # evaluation data and reconstructs commitment history from the chain.
        self.config = load_network_config(config_path, production=False)
        self.wallet = wallet
        self.hotkey = hotkey
        self.network = network
        self.wallet_path = wallet_path
        self.work_dir = Path(work_dir)
        self.state_path = Path(state_path)
        self.hf_token = hf_token
        self.dry_run_weights = dry_run_weights
        self.poll_seconds = poll_seconds if poll_seconds is not None else self.config.audit_interval_seconds
        if self.poll_seconds < 5:
            raise ValueError("validator poll interval must be at least 5 seconds")
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
        self.wandb_run = None
        self.wandb_run_window = None
        self.validator_hotkey = None

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
        self._close_wandb_run()
        if self.audit_engine is not None:
            self.audit_engine.close()
            self.audit_engine = None

    def _close_wandb_run(self) -> None:
        run, self.wandb_run = getattr(self, "wandb_run", None), None
        self.wandb_run_window = None
        if run is not None:
            try:
                run.finish()
            except Exception:
                pass

    def _validator_hotkey(self) -> str:
        if self.validator_hotkey is None:
            try:
                self.validator_hotkey = wallet_hotkey_ss58(
                    wallet=self.wallet,
                    hotkey=self.hotkey,
                    wallet_path=self.wallet_path,
                )
            except Exception:
                # W&B is non-consensus reporting. Preserve validator operation
                # in dry/test environments that do not expose a wallet object.
                self.validator_hotkey = str(self.hotkey or self.wallet)
        return self.validator_hotkey

    def _discard_audit_engine(self) -> None:
        """Drop a failed GPU worker so the next retry starts from clean state."""

        engine, self.audit_engine = self.audit_engine, None
        if engine is not None:
            try:
                engine.close()
            except Exception:
                pass

    def _reset_window(self, window: int) -> None:
        if getattr(self, "wandb_run_window", None) not in {None, window}:
            self._close_wandb_run()
        # Continue rewarding the last verified king while the next public
        # evaluation window accumulates ten successful audit rounds. Replace
        # the carryover only with a newer fully valid result for that same
        # champion; otherwise preserve the previous verified snapshot.
        champions = self.state.get("champions", [])
        carryover = self.state.get("carryover_results", {})
        next_carryover: dict[str, dict[str, Any]] = {}
        if champions:
            champion = champions[0]
            hotkey = str(champion.get("miner_hotkey"))
            digest = champion.get("model_digest")
            current = self.state.get("results", {}).get(hotkey, {})
            previous = carryover.get(hotkey, {}) if isinstance(carryover, dict) else {}
            source = current if (
                isinstance(current, dict)
                and current.get("valid")
                and float(current.get("score", 0.0)) > 0.0
                and current.get("model_digest") == digest
            ) else previous
            if (
                isinstance(source, dict)
                and source.get("valid")
                and float(source.get("score", 0.0)) > 0.0
                and source.get("model_digest") == digest
            ):
                if source is current:
                    champion["last_score"] = float(source["score"])
                    champion["last_window"] = self.state.get("window")
                next_carryover[hotkey] = {
                    "valid": True,
                    "score": float(source["score"]),
                    "model_digest": str(digest),
                    "uid": source.get("uid"),
                    "carryover": True,
                    "source_window": (
                        source.get("source_window")
                        if source is previous
                        else self.state.get("window")
                    ),
                }
        self.state["carryover_results"] = next_carryover
        self.state["window"] = window
        self.state["pending"] = {}
        self.state["results"] = {}
        self.state["audited"] = {}
        self.state["rollout_first_seen"] = {
            key: value
            for key, value in self.state.get("rollout_first_seen", {}).items()
            if str(key).startswith(f"{window}:")
        }

    def _process_pending(
        self,
        *,
        current_block: int,
        window: int,
        commitments: dict[str, dict[str, Any]],
        eval_path: Path,
        eval_manifest: dict[str, Any],
    ) -> list[dict[str, Any]]:
        eval_rows = load_jsonl(eval_path)
        expected_ids = [str(row["row_id"]) for row in eval_rows]
        completed: set[str] = set()
        round_results: list[dict[str, Any]] = []
        for hotkey, pending in list(self.state["pending"].items()):
            if int(pending["challenge_block"]) > current_block:
                continue
            current = commitments.get(hotkey)
            audit_started = time.monotonic()
            timings: dict[str, float] = {}
            round_score: float | None = None
            round_correct: int | None = None
            round_row_count: int | None = None
            round_index = int(pending.get("round", 1))
            previous_result = self.state["results"].get(hotkey)
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
                    runtime_model_dir = _cache_runtime_adapter(
                        work_dir=self.work_dir,
                        source_dir=runtime_model_dir,
                        model_digest=commitment.model_digest,
                        config=self.config,
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
                    copy_owner, displaced = _record_rollout_fingerprint(
                        self.state,
                        hotkey=hotkey,
                        commit_block=int(current["commit_block"]),
                        window=window,
                        rows_sha256=manifest.rows_sha256,
                    )
                    if displaced is not None and displaced != hotkey:
                        self.state["pending"].pop(displaced, None)
                        previous_copy = self.state["results"].get(displaced, {})
                        self.state["results"][displaced] = {
                            "uid": previous_copy.get("uid") if isinstance(previous_copy, dict) else None,
                            "valid": False,
                            "score": 0.0,
                            "audit_status": "copied",
                            "error": f"identical greedy rollouts were committed first by {hotkey}",
                        }
                    if copy_owner is not None:
                        raise ValueError(
                            f"identical greedy rollouts were committed first by {copy_owner}"
                        )
                    score, scored = score_rollouts(
                        eval_rows,
                        rollout_rows,
                        tokenizer,
                        self.config,
                        max_output_tokens=manifest.max_output_tokens,
                    )
                    round_score = score
                    round_correct = sum(int(row["reward"]) for row in scored)
                    round_row_count = len(scored)
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
                            max_output_tokens=manifest.max_output_tokens,
                            rows=audit_rows,
                        )
                    except Exception as exc:
                        # Engine startup, CUDA, and forward-pass failures say
                        # nothing about miner validity. The outer recoverable
                        # path rebuilds the worker and retries this exact round.
                        detail = f"{type(exc).__name__}: {exc}"
                        print(
                            json.dumps(
                                {"status": "audit_engine_error", "error": detail},
                                sort_keys=True,
                            ),
                            flush=True,
                        )
                        raise RuntimeError(f"audit engine failed: {detail}") from exc
                    timings["vllm_load_and_verify"] = round(time.monotonic() - stage_started, 3)
                    if len(audit_reports) != len(selected_ids):
                        raise RuntimeError("audit engine returned an incomplete report")
                    failed_audits = [report for report in audit_reports if not report["valid"]]
                    exact_tokens = sum(int(report.get("exact_argmax_tokens", 0)) for report in audit_reports)
                    tokens_checked = sum(int(report.get("tokens_checked", 0)) for report in audit_reports)
                    exact_match_ratio = exact_tokens / tokens_checked if tokens_checked else 0.0
                    audit_valid = tokens_checked > 0 and _audit_round_valid(
                        audit_reports,
                        exact_match_ratio=exact_match_ratio,
                        min_exact_argmax_ratio=self.config.audit_min_exact_argmax_ratio,
                    )
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
                total_rounds = min(
                    self.config.audit_total_rounds,
                    (len(scored) + self.config.audit_rows_per_round - 1)
                    // self.config.audit_rows_per_round,
                )
                rounds_passed = int(progress.get("rounds_passed", 0))
                fully_validated = audit_valid and rounds_passed >= required_rounds
                audit_complete = audit_valid and rounds_passed >= total_rounds
                blacklisted = None
                if not audit_valid:
                    blacklisted = _record_invalid_round(
                        self.state,
                        hotkey=hotkey,
                        failure_key=(
                            f"{commitment.model_digest}:{rollout_revision}:{round_index}"
                        ),
                        current_block=current_block,
                        config=self.config,
                    )
                failure_details = []
                for report in failed_audits[:3]:
                    detail = f"{report['row_id']}:{report.get('failure_reason') or 'unknown'}"
                    if report.get("failure_position") is not None:
                        detail += f"@{report['failure_position']}"
                    if report.get("failure_rank") is not None:
                        detail += f" rank={report['failure_rank']}"
                    if report.get("failure_logprob_gap") is not None:
                        detail += f" gap={float(report['failure_logprob_gap']):.6f}"
                    if report.get("failure_accepted_token_ids"):
                        detail += f" accepted={report['failure_accepted_token_ids']}"
                    failure_details.append(detail)
                failure_summary = "; ".join(failure_details)
                failed_conditions = []
                if failed_audits:
                    failed_conditions.append(
                        f"{len(failed_audits)}/{len(selected_ids)} audited rows "
                        "failed exact greedy argmax verification"
                    )
                if exact_match_ratio < self.config.audit_min_exact_argmax_ratio:
                    failed_conditions.append(
                        f"exact argmax ratio {exact_match_ratio:.6f} must be >= "
                        f"{self.config.audit_min_exact_argmax_ratio:.6f}"
                    )
                self.state["results"][hotkey] = {
                    "uid": current.get("uid"),
                    "valid": fully_validated,
                    "audit_status": (
                        "blacklisted"
                        if blacklisted is not None
                        else (
                            "passed"
                            if audit_complete
                            else (
                                "monitoring"
                                if fully_validated
                                else ("auditing" if audit_valid else "failed")
                            )
                        )
                    ),
                    "score": score if fully_validated else 0.0,
                    "correct": round_correct,
                    "rows": len(scored),
                    "reward_bits": encode_reward_bits([int(row["reward"]) for row in scored]),
                    "model_digest": commitment.model_digest,
                    "model_revision": commitment.model_revision,
                    "commit_block": current["commit_block"],
                    "rollout_revision": rollout_revision,
                    "audited_rows": list(history),
                    "audit_round": rounds_passed,
                    "audit_required_rounds": required_rounds,
                    "audit_total_rounds": total_rounds,
                    "audit_reports": audit_reports,
                    "audit_exact_match_ratio": exact_match_ratio,
                    "audit_cumulative_exact_match_ratio": (
                        int(progress["exact_argmax_tokens"]) / int(progress["tokens_checked"])
                        if int(progress["tokens_checked"]) else 0.0
                    ),
                    "audit_response_tokens": sum(len(row["tokens"]) for row in audit_rows),
                    "audit_timings_seconds": timings,
                    "audit_block": int(pending["challenge_block"]),
                    "blacklisted_until_block": (
                        int(blacklisted["until_block"]) if blacklisted is not None else None
                    ),
                    "error": None
                    if audit_valid
                    else (
                        "teacher-forced rollout mismatch ("
                        + "; ".join(failed_conditions)
                        + ")"
                        + (f": {failure_summary}" if failure_summary else "")
                    ),
                }
                round_results.append(
                    {
                        "uid": current.get("uid"),
                        "hotkey": hotkey,
                        "miner": _short_hotkey(hotkey),
                        "score": f"{score:.4f}",
                        "correct": f"{round_correct}/{len(scored)}",
                        "round": f"{round_index}/{total_rounds}",
                        "audit_rows": len(selected_ids),
                        "exact": f"{exact_match_ratio:.2%}",
                        "outcome": (
                            "BLACKLISTED"
                            if blacklisted is not None
                            else (
                                "COMPLETE"
                                if audit_complete
                                else (
                                    "VALID+AUDIT"
                                    if fully_validated
                                    else ("AUDITING" if audit_valid else "INVALID")
                                )
                            )
                        ),
                        "seconds": f"{timings['total']:.2f}s",
                        "error": self.state["results"][hotkey].get("error"),
                    }
                )
                completed.add(hotkey)
            except ValueError as exc:
                timings["total"] = round(time.monotonic() - audit_started, 3)
                blacklisted = None
                if _counts_as_invalid_strike(exc):
                    blacklisted = _record_invalid_round(
                        self.state,
                        hotkey=hotkey,
                        failure_key=(
                            f"{pending['commitment'].get('h')}:{pending.get('rollout_revision')}:{round_index}"
                        ),
                        current_block=current_block,
                        config=self.config,
                    )
                self.state["results"][hotkey] = {
                    "uid": current.get("uid") if current is not None else None,
                    "valid": False,
                    "score": 0.0,
                    "model_digest": pending["commitment"].get("h"),
                    "rollout_revision": pending.get("rollout_revision"),
                    "audit_status": "blacklisted" if blacklisted is not None else "failed",
                    "audit_timings_seconds": timings,
                    "blacklisted_until_block": (
                        int(blacklisted["until_block"]) if blacklisted is not None else None
                    ),
                    "error": f"{type(exc).__name__}: {exc}",
                }
                round_results.append(
                    {
                        "uid": current.get("uid") if current is not None else None,
                        "hotkey": hotkey,
                        "miner": _short_hotkey(hotkey),
                        "score": f"{round_score:.4f}" if round_score is not None else "-",
                        "correct": (
                            f"{round_correct}/{round_row_count}"
                            if round_correct is not None and round_row_count is not None
                            else "-"
                        ),
                        "round": str(round_index),
                        "audit_rows": "-",
                        "exact": "-",
                        "outcome": "BLACKLISTED" if blacklisted is not None else "INVALID",
                        "seconds": f"{timings['total']:.2f}s",
                        "error": self.state["results"][hotkey]["error"],
                    }
                )
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
                same_valid_revision = bool(
                    isinstance(previous_result, dict)
                    and previous_result.get("valid")
                    and previous_result.get("model_digest")
                    == pending["commitment"].get("h")
                    and previous_result.get("rollout_revision")
                    == pending.get("rollout_revision")
                )
                self.state["results"][hotkey] = {
                    **(previous_result if same_valid_revision else {}),
                    "uid": current.get("uid") if current is not None else None,
                    "valid": same_valid_revision,
                    "score": (
                        float(previous_result.get("score", 0.0))
                        if same_valid_revision
                        else 0.0
                    ),
                    "model_digest": pending["commitment"].get("h"),
                    "rollout_revision": pending.get("rollout_revision"),
                    "audit_status": "retrying",
                    "audit_timings_seconds": timings,
                    "error": f"recoverable {type(exc).__name__}: {exc}",
                }
                round_results.append(
                    {
                        "uid": current.get("uid") if current is not None else None,
                        "hotkey": hotkey,
                        "miner": _short_hotkey(hotkey),
                        "score": f"{round_score:.4f}" if round_score is not None else "-",
                        "correct": (
                            f"{round_correct}/{round_row_count}"
                            if round_correct is not None and round_row_count is not None
                            else "-"
                        ),
                        "round": str(round_index),
                        "audit_rows": "-",
                        "exact": "-",
                        "outcome": "RETRY",
                        "seconds": f"{timings['total']:.2f}s",
                        "error": self.state["results"][hotkey]["error"],
                    }
                )
        for hotkey in completed:
            self.state["pending"].pop(hotkey, None)
        return round_results

    def _schedule_latest(
        self,
        *,
        current_block: int,
        eligible: list[dict[str, Any]],
    ) -> None:
        active_blacklist = _active_blacklist(
            self.state, current_block=current_block, config=self.config
        )
        for row in eligible:
            hotkey = row["hotkey"]
            if hotkey in active_blacklist:
                self.state["pending"].pop(hotkey, None)
                continue
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
            total_rounds = min(
                self.config.audit_total_rounds,
                (self.config.evaluation_rows + self.config.audit_rows_per_round - 1)
                // self.config.audit_rows_per_round,
            )
            previous_round = int(previous.get("audit_round", 0)) if isinstance(previous, dict) else 0
            if (
                isinstance(previous, dict)
                and previous.get("model_digest") == commitment.model_digest
                and previous.get("rollout_revision") == revision
                and (
                    previous_status == "failed"
                    or (previous_status == "passed" and previous_round >= total_rounds)
                )
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
                # A new revision is a new statistical trial. Delete older
                # revision histories for this hotkey so they cannot contribute.
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
        try:
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
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            self.state["last_weight_error"] = error
            return {"success": False, "error": error}
        if result.get("success") is False or result.get("error"):
            self.state["last_weight_error"] = f"weight submission failed: {result}"
            return result
        self.state["last_weight_block"] = current_block
        self.state.pop("last_weight_error", None)
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
                validator_hotkey=self._validator_hotkey(),
            )
            summary_root = str(manifest["summary_root"])
            if self.state.get("last_wandb_summary_root") == summary_root:
                return {"status": "unchanged", "summary_root": summary_root}
            window = int(self.state.get("window"))
            if self.wandb_run is None or self.wandb_run_window != window:
                self._close_wandb_run()
                validator_hotkey = self._validator_hotkey()
                identity = f"{self.config.netuid}:{validator_hotkey}:{window}"
                run_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
                short_validator = (
                    validator_hotkey
                    if len(validator_hotkey) <= 20
                    else f"{validator_hotkey[:8]}-{validator_hotkey[-6:]}"
                )
                self.wandb_run = start_wandb_results_run(
                    bundle_dir=out_dir,
                    project=self.wandb_project,
                    entity=self.wandb_entity,
                    run_name=f"feval-{short_validator}-window-{window}",
                    run_id=run_id,
                )
                self.wandb_run_window = window
            report = log_results_to_wandb(
                bundle_dir=out_dir,
                project=self.wandb_project,
                entity=self.wandb_entity,
                run=self.wandb_run,
            )
            self.state["last_wandb_summary_root"] = summary_root
            _atomic_write_json(self.state_path, self.state)
            return {"status": "published", "summary_root": summary_root, **report}
        except Exception as exc:
            # W&B is a public convenience mirror, never a validator dependency.
            self._close_wandb_run()
            return {"status": "unavailable", "error": f"{type(exc).__name__}: {exc}"}

    def _sync_history(self, *, current_block: int) -> tuple[int, bool]:
        start_block = max(0, current_block - HISTORY_LOOKBACK_BLOCKS + 1)
        cursor_value = self.state.get("history_cursor")
        cursor = start_block - 1 if cursor_value is None else int(cursor_value)
        if cursor < start_block - 1:
            # The rolling seven-day window advanced beyond the saved cursor.
            # Rebuild recent priority rather than retaining stale identities.
            self.state["first_seen"] = {}
            cursor = start_block - 1
        _copy_filtered([], self.state, minimum_block=start_block)
        for key, entry in list(self.state.setdefault("rollout_first_seen", {}).items()):
            if not isinstance(entry, dict) or int(entry.get("block", -1)) < start_block:
                self.state["rollout_first_seen"].pop(key, None)
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
        _copy_filtered(rows, self.state, minimum_block=start_block)
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
        )
        history_cutoff = max(0, current_block - HISTORY_LOOKBACK_BLOCKS + 1)
        rows = _current_commitments(
            [
                row
                for row in read_model_commitments(
                    netuid=self.config.netuid,
                    network=self.network,
                    block=current_block,
                )
                if row.get("uid") is not None and int(row["commit_block"]) >= history_cutoff
            ]
        )
        eligible, copies = _copy_filtered(
            rows,
            self.state,
            minimum_block=history_cutoff,
        )
        current = {row["hotkey"]: row for row in rows}
        for hotkey, first_miner in copies.items():
            self.state["results"][hotkey] = {
                "uid": current.get(hotkey, {}).get("uid"),
                "valid": False,
                "score": 0.0,
                "error": f"model digest was committed first by {first_miner}",
            }
        active_blacklist = _active_blacklist(
            self.state, current_block=current_block, config=self.config
        )
        if active_blacklist:
            eligible = [row for row in eligible if row["hotkey"] not in active_blacklist]
            for blacklisted_hotkey, entry in active_blacklist.items():
                self.state["pending"].pop(blacklisted_hotkey, None)
                self.state["results"][blacklisted_hotkey] = {
                    "uid": current.get(blacklisted_hotkey, {}).get("uid"),
                    "valid": False,
                    "score": 0.0,
                    "audit_status": "blacklisted",
                    "blacklisted_until_block": int(entry["until_block"]),
                    "error": (
                        "hotkey is blacklisted after three deterministic invalid audit rounds "
                        f"until block {entry['until_block']}"
                    ),
                }
        round_results = self._process_pending(
            current_block=current_block,
            window=window,
            commitments=current,
            eval_path=eval_path,
            eval_manifest=eval_manifest,
        )
        active_blacklist = _active_blacklist(
            self.state, current_block=current_block, config=self.config
        )
        _print_round_results(window=window, block=current_block, rows=round_results)
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
            if not item.get("valid") and item.get("audit_status") in {"auditing", "retrying"}
        )
        monitoring_results = sum(
            1
            for item in self.state["results"].values()
            if item.get("audit_status") == "monitoring"
            or (item.get("valid") and item.get("audit_status") == "retrying")
        )
        invalid_results = sum(
            1
            for item in self.state["results"].values()
            if item.get("audit_status") in {"failed", "blacklisted"}
            or (not item.get("valid") and item.get("audit_status") not in {"auditing", "retrying"})
        )
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
        weights_submitted = bool(
            weight_result is not None
            and weight_result.get("success") is not False
            and not weight_result.get("error")
        )
        if weights_submitted and not pending_blocks:
            status = "weights_submitted"
        return {
            "block": current_block,
            "status": status,
            "window": window,
            "miners": len(rows),
            "eligible": len(eligible),
            "copies": len(copies),
            "blacklisted": len(active_blacklist),
            "pending": len(self.state["pending"]),
            "next_audit_block": min(pending_blocks) if pending_blocks else None,
            "audit_delay_blocks": self.config.audit_delay_blocks,
            "evaluated": evaluated,
            "valid_results": valid_results,
            "invalid_results": invalid_results,
            "auditing_results": auditing_results,
            "monitoring_results": monitoring_results,
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
            "weights_submitted": weights_submitted,
            "weight_error": self.state.get("last_weight_error"),
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

