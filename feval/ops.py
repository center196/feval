from __future__ import annotations

import importlib.metadata
import json
import os
import shutil
import time
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

from .jsonutil import load_json


MAINNET_DEPENDENCIES = {
    "feval": "1.0.0",
    "bittensor": "11.1.0",
    "datasets": "5.0.1",
    "huggingface-hub": "1.28.0",
    "safetensors": "0.8.0",
    "transformers": "5.15.0",
    "vllm": "0.27.1",
}


class ProcessLock(AbstractContextManager["ProcessLock"]):
    """A dependency-free single-process guard for validator state."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.acquired = False

    @staticmethod
    def _alive(pid: int) -> bool:
        if pid <= 0:
            return False
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError:
            return False

    def __enter__(self) -> "ProcessLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(2):
            try:
                descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            except FileExistsError:
                try:
                    value = json.loads(self.path.read_text(encoding="utf-8"))
                    pid = int(value.get("pid", -1))
                except (OSError, ValueError, TypeError, json.JSONDecodeError):
                    pid = -1
                if self._alive(pid):
                    raise RuntimeError(f"another Feval process is active (pid {pid})")
                self.path.unlink(missing_ok=True)
                continue
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump({"pid": os.getpid(), "created_at": time.time()}, stream)
                stream.flush()
                os.fsync(stream.fileno())
            self.acquired = True
            return self
        raise RuntimeError("could not acquire validator process lock")

    def __exit__(self, *_args: Any) -> None:
        if self.acquired:
            self.path.unlink(missing_ok=True)
            self.acquired = False


def health_path_for_state(state_path: str | Path) -> Path:
    target = Path(state_path)
    return target.with_name(target.name + ".health.json")


def check_health(state_path: str | Path, *, max_age_seconds: int) -> dict[str, Any]:
    path = health_path_for_state(state_path)
    if not path.exists():
        raise RuntimeError("validator has not written a health heartbeat")
    value = load_json(path)
    if not isinstance(value, dict):
        raise RuntimeError("validator heartbeat is malformed")
    age = time.time() - float(value.get("updated_at", 0))
    if age < 0 or age > max_age_seconds:
        raise RuntimeError(f"validator heartbeat is stale ({age:.0f}s old)")
    if value.get("healthy") is not True:
        raise RuntimeError(str(value.get("error") or "validator reports unhealthy"))
    return {**value, "age_seconds": round(age, 3)}


def dependency_versions() -> dict[str, str]:
    packages = ("feval", "bittensor", "datasets", "huggingface-hub", "safetensors", "transformers", "vllm")
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "missing"
    return versions


def disk_free_bytes(path: str | Path) -> int:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return int(shutil.disk_usage(target).free)
