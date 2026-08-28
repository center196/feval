from __future__ import annotations

from pathlib import Path
from typing import Any

from ..models.artifacts import ModelCommitment
from ..utils.crypto import sha256_hex
from ..utils.jsonutil import canonical_json_bytes, load_json


def _import_bittensor():
    try:
        import bittensor as bt
    except ImportError as exc:
        raise RuntimeError(
            "Bittensor SDK is not installed in this Python environment. "
            "Install bittensor>=11 on Linux/WSL, then rerun this command."
        ) from exc
    return bt


def _wallet(bt: Any, wallet: str, hotkey: str | None = None, wallet_path: str | None = None):
    kwargs: dict[str, Any] = {"name": wallet}
    if hotkey:
        kwargs["hotkey"] = hotkey
    if wallet_path:
        kwargs["path"] = wallet_path
    return bt.Wallet(**kwargs)


def serve_axon(
    netuid: int,
    ip: str,
    port: int,
    wallet: str,
    hotkey: str | None,
    network: str,
    wallet_path: str | None,
    dry_run: bool,
) -> dict[str, Any]:
    bt = _import_bittensor()
    sub = bt.Subtensor(network)
    w = _wallet(bt, wallet, hotkey, wallet_path)
    intent = bt.ServeAxon(netuid=netuid, ip=ip, port=port)
    result = sub.plan(intent, w) if dry_run else sub.execute(intent, w)
    return _result_to_dict(result, dry_run)


def publish_commitment(
    netuid: int,
    submission_path: str | Path,
    wallet: str,
    hotkey: str | None,
    network: str,
    wallet_path: str | None,
    dry_run: bool,
) -> dict[str, Any]:
    submission = load_json(submission_path)
    commitment = {
        "protocol": "feval-commitment-v1",
        "miner_hotkey": submission.get("miner_hotkey"),
        "adapter_hash": submission.get("adapter_hash"),
        "evaluation_root": submission.get("evaluation_root"),
        "answer_root": submission.get("answer_root"),
        "rollout_root": submission.get("rollout_root"),
        "score_claimed": submission.get("score_claimed"),
    }
    commitment_digest = sha256_hex(canonical_json_bytes(commitment))
    if dry_run:
        return {
            "dry_run": True,
            "call": "Commitments.set_commitment",
            "netuid": netuid,
            "commitment_digest": commitment_digest,
        }
    bt = _import_bittensor()
    sub = bt.Subtensor(network)
    w = _wallet(bt, wallet, hotkey, wallet_path)
    call = bt.calls.Commitments.set_commitment(
        netuid=netuid,
        info={"fields": [{"Sha256": bytes.fromhex(commitment_digest)}]},
    )
    result = sub.submit_call(call, w, signer="hotkey")
    data = _result_to_dict(result, dry_run)
    data["commitment_digest"] = commitment_digest
    return data


def publish_model_commitment(
    *,
    netuid: int,
    commitment: ModelCommitment,
    wallet: str,
    hotkey: str | None,
    network: str,
    wallet_path: str | None,
    dry_run: bool,
) -> dict[str, Any]:
    """Publish discoverable model metadata; the chain block proves priority."""

    payload = commitment.to_chain_bytes()
    if dry_run:
        return {
            "dry_run": True,
            "call": "Commitments.set_commitment",
            "netuid": netuid,
            "bytes": len(payload),
            "metadata": commitment.compact_dict(),
        }
    bt = _import_bittensor()
    sub = bt.Subtensor(network)
    w = _wallet(bt, wallet, hotkey, wallet_path)
    call = bt.calls.Commitments.set_commitment(
        netuid=netuid,
        info={"fields": [{"BigRaw": payload}]},
    )
    result = sub.submit_call(call, w, signer="hotkey")
    data = _result_to_dict(result, dry_run=False)
    data["metadata"] = commitment.compact_dict()
    return data


def _field(value: Any, *names: str) -> Any:
    for name in names:
        if isinstance(value, dict) and name in value:
            return value[name]
        if hasattr(value, name):
            return getattr(value, name)
    return None


def _decode_commitment_content(value: Any) -> str | bytes | dict[str, Any]:
    if isinstance(value, (str, bytes)):
        return value
    if isinstance(value, dict):
        # Current v11 reads normally return decoded plaintext. These fallbacks
        # also accept the raw CommitmentInfo/Data shapes used by local nodes.
        if set(value) >= {"p", "m", "r", "h", "d"}:
            return value
        fields = value.get("fields")
        if isinstance(fields, list):
            for field in fields:
                if not isinstance(field, dict):
                    continue
                for variant in ("BigRaw", "Raw"):
                    if variant in field:
                        raw = field[variant]
                        if isinstance(raw, list):
                            return bytes(raw)
                        if isinstance(raw, str) and raw.startswith("0x"):
                            return bytes.fromhex(raw[2:])
                        return raw
        for name in ("commitment", "content", "value", "data"):
            if name in value and value[name] is not None:
                return _decode_commitment_content(value[name])
    # The typed v11 `subnets.commitments()` result retains raw fields. Read
    # them before `.value`: some SDK releases only concatenate `Raw` and omit
    # `BigRaw` from the convenience plaintext property.
    fields = getattr(value, "fields", None)
    if isinstance(fields, list):
        return _decode_commitment_content({"fields": fields})
    for name in ("commitment", "content", "value", "data"):
        nested = getattr(value, name, None)
        if nested is not None:
            return _decode_commitment_content(nested)
    raise ValueError("commitment row has no decodable content")


def _read_model_commitments_from_sub(
    sub: Any, *, netuid: int, block: int | None = None
) -> list[dict[str, Any]]:
    namespace = getattr(sub, "subnets", None)
    if namespace is None or not hasattr(namespace, "commitments"):
        namespace = getattr(sub, "identity", None)
    if namespace is None or not hasattr(namespace, "commitments"):
        raise RuntimeError("installed Bittensor v11 SDK has no commitments query")
    kwargs: dict[str, Any] = {"netuid": netuid}
    if block is not None:
        kwargs["block"] = block
    raw_rows = namespace.commitments(**kwargs)
    if isinstance(raw_rows, dict):
        raw_rows = list(raw_rows.values())
    results: list[dict[str, Any]] = []
    for raw in raw_rows or []:
        status = _field(raw, "status")
        if status not in (None, "revealed", "plain", "plaintext"):
            continue
        try:
            content = _decode_commitment_content(raw)
            commitment = ModelCommitment.from_chain_value(content)
        except (ValueError, TypeError):
            continue
        hotkey_value = _field(raw, "hotkey", "hotkey_ss58", "account")
        block_value = _field(raw, "block", "block_number", "updated_at")
        uid_value = _field(raw, "uid")
        if hotkey_value is None or block_value is None:
            continue
        results.append(
            {
                "hotkey": str(hotkey_value),
                "uid": int(uid_value) if uid_value is not None else None,
                "commit_block": int(block_value),
                "commitment": commitment,
            }
        )
    return sorted(results, key=lambda item: (item["commit_block"], item["hotkey"]))


def read_model_commitments(
    *, netuid: int, network: str, block: int | None = None
) -> list[dict[str, Any]]:
    bt = _import_bittensor()
    return _read_model_commitments_from_sub(bt.Subtensor(network), netuid=netuid, block=block)


def finalized_block(*, network: str) -> int:
    bt = _import_bittensor()
    sub = bt.Subtensor(network)
    value = getattr(sub, "finalized_block", None)
    if value is not None:
        return int(value() if callable(value) else value)
    # Bittensor v11 exposes finalized heads as a public iterator. Reading one
    # item waits for the next finalized header and avoids treating a reorgable
    # best-head block as audit randomness.
    blocks = getattr(sub, "blocks", None)
    if callable(blocks):
        stream = blocks(finalized=True)
        try:
            header = next(stream)
            number = _field(header, "number", "block")
            if number is not None:
                return int(number)
        finally:
            close = getattr(stream, "close", None)
            if callable(close):
                close()
    raise RuntimeError("installed Bittensor SDK cannot read finalized chain heads")


def block_hash(*, network: str, block: int) -> str:
    bt = _import_bittensor()
    sub = bt.Subtensor(network)
    info = sub.block_info(block)
    value = _field(info, "hash", "block_hash")
    if value is None:
        raise RuntimeError("Bittensor block_info returned no block hash")
    return str(value)


def wallet_hotkey_ss58(
    *,
    wallet: str,
    hotkey: str | None,
    wallet_path: str | None,
) -> str:
    bt = _import_bittensor()
    value = _wallet(bt, wallet, hotkey, wallet_path).hotkey.ss58_address
    return str(value)


def neuron_status(*, netuid: int, network: str, hotkey_ss58: str) -> dict[str, Any]:
    bt = _import_bittensor()
    sub = bt.Subtensor(network)
    for neuron in sub.neurons.all(netuid=netuid, lite=True):
        hotkey = _field(neuron, "hotkey", "hotkey_ss58")
        if str(hotkey) != hotkey_ss58:
            continue
        return {
            "uid": int(_field(neuron, "uid")),
            "active": bool(_field(neuron, "active")),
            "validator_permit": bool(_field(neuron, "validator_permit")),
        }
    raise ValueError(f"hotkey {hotkey_ss58} is not registered on subnet {netuid}")


def set_weights(
    netuid: int,
    weights_path: str | Path,
    wallet: str,
    hotkey: str | None,
    network: str,
    wallet_path: str | None,
    dry_run: bool,
    mechid: int = 0,
    version_key: int = 0,
) -> dict[str, Any]:
    bt = _import_bittensor()
    report = load_json(weights_path)
    uid_weights = report.get("uid_weights")
    weights = {int(uid): float(weight) for uid, weight in uid_weights.items()} if uid_weights else None
    if not weights:
        raise ValueError("weights report has no uid_weights; rerun 'feval validator weights' with --uid-map")
    sub = bt.Subtensor(network)
    w = _wallet(bt, wallet, hotkey, wallet_path)
    intent = bt.SetWeights(netuid=netuid, weights=weights, mechid=mechid, version_key=version_key)
    result = sub.plan(intent, w) if dry_run else sub.execute(intent, w)
    return _result_to_dict(result, dry_run)


def set_weight_mapping(
    *,
    netuid: int,
    uid_weights: dict[int, float],
    wallet: str,
    hotkey: str | None,
    network: str,
    wallet_path: str | None,
    dry_run: bool,
    mechid: int = 0,
    version_key: int = 0,
) -> dict[str, Any]:
    if not uid_weights:
        raise ValueError("cannot set an empty weight mapping")
    weights = {int(uid): float(value) for uid, value in uid_weights.items() if float(value) > 0}
    if not weights:
        raise ValueError("all calculated weights are zero")
    bt = _import_bittensor()
    sub = bt.Subtensor(network)
    w = _wallet(bt, wallet, hotkey, wallet_path)
    intent = bt.SetWeights(
        netuid=netuid,
        weights=weights,
        mechid=mechid,
        version_key=version_key,
    )
    result = sub.plan(intent, w) if dry_run else sub.execute(intent, w)
    return _result_to_dict(result, dry_run)


def _result_to_dict(result: Any, dry_run: bool) -> dict[str, Any]:
    if isinstance(result, dict):
        return result
    data = {"dry_run": dry_run}
    for name in ("success", "message", "fee", "block_hash", "extrinsic_id", "explorer_url"):
        if hasattr(result, name):
            value = getattr(result, name)
            data[name] = value if isinstance(value, (str, int, float, bool)) or value is None else str(value)
    error = getattr(result, "error", None)
    if error is not None:
        data["error"] = {
            "code": str(getattr(error, "code", "")),
            "name": str(getattr(error, "name", "")),
            "remediation": str(getattr(error, "remediation", "")),
        }
    return data



