from __future__ import annotations

import hashlib
import hmac
import secrets
from pathlib import Path
from typing import Any

from .jsonutil import canonical_json_bytes, load_json, write_json


DEV_SIGNATURE_SCHEME = "hmac-sha256-dev"


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_json(value: Any) -> str:
    return sha256_hex(canonical_json_bytes(value))


def make_dev_key(hotkey: str) -> dict[str, str]:
    return {
        "scheme": DEV_SIGNATURE_SCHEME,
        "hotkey": hotkey,
        "secret": secrets.token_hex(32),
    }


def sign_payload(payload: dict[str, Any], key: dict[str, str]) -> dict[str, str]:
    if key.get("scheme") != DEV_SIGNATURE_SCHEME:
        raise ValueError(f"unsupported key scheme: {key.get('scheme')}")
    signature = hmac.new(bytes.fromhex(key["secret"]), canonical_json_bytes(payload), hashlib.sha256).hexdigest()
    return {
        "scheme": DEV_SIGNATURE_SCHEME,
        "hotkey": key["hotkey"],
        "signature": signature,
    }


def verify_signature(payload: dict[str, Any], signature: dict[str, str], keyring: dict[str, Any]) -> bool:
    if signature.get("scheme") != DEV_SIGNATURE_SCHEME:
        raise ValueError(f"unsupported signature scheme: {signature.get('scheme')}")
    hotkey = signature.get("hotkey")
    secret = keyring.get(hotkey)
    if isinstance(secret, dict):
        secret = secret.get("secret")
    if not isinstance(secret, str):
        return False
    expected = hmac.new(bytes.fromhex(secret), canonical_json_bytes(payload), hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature.get("signature", ""))


def write_key(path: str | Path, hotkey: str, keyring_path: str | Path | None = None) -> dict[str, str]:
    key = make_dev_key(hotkey)
    write_json(path, key)
    if keyring_path:
        keyring_file = Path(keyring_path)
        keyring = load_json(keyring_file) if keyring_file.exists() else {}
        keyring[hotkey] = {"scheme": DEV_SIGNATURE_SCHEME, "secret": key["secret"]}
        write_json(keyring_file, keyring)
    return key


