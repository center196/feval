from __future__ import annotations

from pathlib import Path
from typing import Any

from .artifacts import (
    ModelCommitment,
    adapter_digest,
    model_manifest,
    prepare_runtime_adapter,
    validate_repo_id,
    validate_revision,
)
from .config import NetworkConfig
from .constants import (
    MAX_ADAPTER_BYTES,
    MAX_ADAPTER_CONFIG_BYTES,
    MAX_MANIFEST_BYTES,
    MAX_ROLLOUT_BYTES,
    MODEL_FILES,
    ROLLOUT_FILES,
)
from .jsonutil import canonical_json_bytes


def _hub_imports():
    try:
        from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
    except ImportError as exc:
        raise RuntimeError("Hugging Face operations require the 'huggingface_hub' package") from exc
    return CommitOperationAdd, HfApi, hf_hub_download


def _commit_sha(value: Any) -> str:
    for name in ("oid", "commit_hash", "sha"):
        candidate = getattr(value, name, None)
        if isinstance(candidate, str) and len(candidate) == 40:
            return validate_revision(candidate)
    raise RuntimeError("Hugging Face did not return a full commit SHA")


def upload_model_adapter(
    *,
    adapter_dir: str | Path,
    model_repo: str,
    rollout_repo: str,
    config: NetworkConfig,
    token: str | bool | None = None,
    private: bool = False,
    create_repos: bool = True,
) -> ModelCommitment:
    """Validate and atomically upload a LoRA adapter.

    Only two non-executable files are accepted. The exact returned Hub commit
    SHA is what must be placed on chain.
    """

    CommitOperationAdd, HfApi, _ = _hub_imports()
    model_repo = validate_repo_id(model_repo)
    rollout_repo = validate_repo_id(rollout_repo)
    root = Path(adapter_dir)
    digest = adapter_digest(root, config)
    api = HfApi(token=token)
    if create_repos:
        api.create_repo(model_repo, repo_type="model", private=private, exist_ok=True)
        api.create_repo(rollout_repo, repo_type="dataset", private=private, exist_ok=True)
    manifest_bytes = canonical_json_bytes(model_manifest(model_repo, digest, config)) + b"\n"
    result = api.create_commit(
        repo_id=model_repo,
        repo_type="model",
        commit_message=f"Feval adapter {digest[:12]}",
        operations=[
            CommitOperationAdd(path_in_repo=MODEL_FILES[0], path_or_fileobj=root / MODEL_FILES[0]),
            CommitOperationAdd(path_in_repo=MODEL_FILES[1], path_or_fileobj=root / MODEL_FILES[1]),
            CommitOperationAdd(path_in_repo="feval_model.json", path_or_fileobj=manifest_bytes),
        ],
    )
    return ModelCommitment(
        model_repo=model_repo,
        model_revision=_commit_sha(result),
        model_digest=digest,
        rollout_repo=rollout_repo,
    )


def upload_rollout_bundle(
    *,
    bundle_dir: str | Path,
    rollout_repo: str,
    token: str | bool | None = None,
) -> str:
    """Atomically replace the current rollout manifest and JSONL.

    This does not touch the model's on-chain commitment. The returned revision
    is immutable and can be pinned by validators before future randomness.
    """

    CommitOperationAdd, HfApi, _ = _hub_imports()
    rollout_repo = validate_repo_id(rollout_repo)
    root = Path(bundle_dir)
    manifest = root / ROLLOUT_FILES[0]
    rollouts = root / ROLLOUT_FILES[1]
    if manifest.stat().st_size > MAX_MANIFEST_BYTES:
        raise ValueError("manifest.json exceeds the size limit")
    if rollouts.stat().st_size > MAX_ROLLOUT_BYTES:
        raise ValueError("rollouts.jsonl exceeds the size limit")
    api = HfApi(token=token)
    parent = api.repo_info(rollout_repo, repo_type="dataset").sha
    result = api.create_commit(
        repo_id=rollout_repo,
        repo_type="dataset",
        parent_commit=parent,
        commit_message="Update Feval rollout window",
        operations=[
            CommitOperationAdd(path_in_repo=ROLLOUT_FILES[0], path_or_fileobj=manifest),
            CommitOperationAdd(path_in_repo=ROLLOUT_FILES[1], path_or_fileobj=rollouts),
        ],
    )
    return _commit_sha(result)


def _repo_file_sizes(info: Any) -> dict[str, int]:
    result: dict[str, int] = {}
    siblings = getattr(info, "siblings", []) or []
    if len(siblings) > 128:
        raise ValueError("repository contains too many files")
    for sibling in siblings:
        name = getattr(sibling, "rfilename", None)
        size = getattr(sibling, "size", None)
        if isinstance(name, str) and isinstance(size, int):
            result[name] = size
    return result


def _require_file_sizes(info: Any, limits: dict[str, int]) -> None:
    sizes = _repo_file_sizes(info)
    for filename, maximum in limits.items():
        if filename not in sizes:
            raise ValueError(f"repository is missing size metadata for {filename}")
        if sizes[filename] < 0 or sizes[filename] > maximum:
            raise ValueError(f"repository file {filename} exceeds its size limit")


def _require_local_regular_file(root: Path, filename: str) -> Path:
    path = root / filename
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"downloaded {filename} is not a regular file")
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"downloaded {filename} escapes the validator cache") from exc
    return path


def resolve_rollout_revision(rollout_repo: str, *, token: str | bool | None = None) -> str:
    _, HfApi, _ = _hub_imports()
    rollout_repo = validate_repo_id(rollout_repo)
    info = HfApi(token=token).repo_info(
        rollout_repo,
        repo_type="dataset",
        files_metadata=True,
    )
    _require_file_sizes(
        info,
        {"manifest.json": MAX_MANIFEST_BYTES, "rollouts.jsonl": MAX_ROLLOUT_BYTES},
    )
    return validate_revision(info.sha)


def safe_download_model(
    commitment: ModelCommitment,
    *,
    config: NetworkConfig,
    local_dir: str | Path,
    token: str | bool | None = None,
) -> Path:
    """Download only SafeTensors and its inert LoRA JSON configuration."""

    _, HfApi, hf_hub_download = _hub_imports()
    commitment.validate()
    root = Path(local_dir)
    root.mkdir(parents=True, exist_ok=True)
    api = HfApi(token=token)
    info = api.repo_info(
        commitment.model_repo,
        repo_type="model",
        revision=commitment.model_revision,
        files_metadata=True,
    )
    if validate_revision(info.sha) != commitment.model_revision:
        raise ValueError("model repository did not resolve to the committed revision")
    _require_file_sizes(
        info,
        {
            MODEL_FILES[0]: MAX_ADAPTER_CONFIG_BYTES,
            MODEL_FILES[1]: MAX_ADAPTER_BYTES,
        },
    )
    for filename in MODEL_FILES:
        hf_hub_download(
            repo_id=commitment.model_repo,
            repo_type="model",
            revision=commitment.model_revision,
            filename=filename,
            local_dir=root,
            token=token,
        )
        _require_local_regular_file(root, filename)
    actual_digest = adapter_digest(root, config)
    if actual_digest != commitment.model_digest:
        raise ValueError("downloaded adapter does not match the on-chain digest")
    return prepare_runtime_adapter(root, root / "validated-runtime", config)


def safe_download_rollouts(
    rollout_repo: str,
    *,
    revision: str,
    local_dir: str | Path,
    token: str | bool | None = None,
) -> Path:
    """Download exactly two inert data files at an immutable Hub revision."""

    _, HfApi, hf_hub_download = _hub_imports()
    rollout_repo = validate_repo_id(rollout_repo)
    revision = validate_revision(revision)
    root = Path(local_dir)
    root.mkdir(parents=True, exist_ok=True)
    info = HfApi(token=token).repo_info(
        rollout_repo,
        repo_type="dataset",
        revision=revision,
        files_metadata=True,
    )
    if validate_revision(info.sha) != revision:
        raise ValueError("rollout repository did not resolve to the pinned revision")
    _require_file_sizes(
        info,
        {"manifest.json": MAX_MANIFEST_BYTES, "rollouts.jsonl": MAX_ROLLOUT_BYTES},
    )
    for filename in ROLLOUT_FILES:
        hf_hub_download(
            repo_id=rollout_repo,
            repo_type="dataset",
            revision=revision,
            filename=filename,
            local_dir=root,
            token=token,
        )
        _require_local_regular_file(root, filename)
    return root
