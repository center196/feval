"""Deterministic partial sampling of immutable Hugging Face dataset revisions.

Feval never needs a whole source dataset. It needs a reproducible sample of
valid rows, and every miner and validator must independently arrive at the
identical sample from the same pinned revision.

Two access patterns are used, both of which read far less than the full data:

* Parquet sources expose a footer that lists row groups without transferring
  any values. A seeded permutation picks shards and row groups, and only the
  protocol-relevant columns of those groups are fetched.
* JSONL sources have no row index. A seeded permutation of byte offsets is
  read instead, and each block is trimmed to whole lines. This is biased
  toward longer rows, which is acceptable because the bias is identical for
  every node and never depends on miner-controlled input.

Nothing here executes dataset code, and no value is ever passed to ``eval``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Callable, Iterator

from ..utils.jsonutil import canonical_json_bytes


# Blocks read per JSONL seek. Large enough to amortise the request, small
# enough that a source with no valid rows cannot pull an unbounded amount.
JSONL_BLOCK_BYTES = 1 << 20
# Refuse absurd footers so a malformed or hostile revision cannot fan out.
MAX_FILES_PER_SOURCE = 512
MAX_ROW_GROUPS_PER_FILE = 4096


@dataclass(frozen=True)
class ParquetSource:
    """A pinned parquet dataset and the only columns Feval reads from it."""

    name: str
    repo: str
    revision: str
    files: tuple[str, ...]
    columns: tuple[str, ...]

    kind: str = "parquet"


@dataclass(frozen=True)
class JsonlSource:
    """A pinned JSONL dataset addressed by byte offset."""

    name: str
    repo: str
    revision: str
    files: tuple[str, ...]

    kind: str = "jsonl"


Source = ParquetSource | JsonlSource


def _filesystem(token: str | bool | None = False):
    try:
        from huggingface_hub import HfFileSystem
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "dataset sampling requires the 'huggingface_hub' package"
        ) from exc
    return HfFileSystem(token=token)


def _hf_path(source: Source, filename: str) -> str:
    # Pinning the revision in the path is what makes the sample reproducible.
    return f"datasets/{source.repo}@{source.revision}/{filename}"


def seeded_order(items: list[Any], *, seed: str, name: str, kind: str) -> list[Any]:
    """Order items by a keyed hash rather than by a PRNG shuffle.

    ``random.shuffle`` would work, but the standard library makes no promise
    that its algorithms stay fixed across Python versions, and consensus here
    depends on two nodes producing the same order. Sorting on SHA-256 is fully
    specified, stable across versions, and reimplementable in any language.
    """

    def key(item: Any) -> bytes:
        return hashlib.sha256(
            canonical_json_bytes(
                {
                    "domain": "feval/source-order/v1",
                    "seed": seed,
                    "source": name,
                    "kind": kind,
                    "item": item,
                }
            )
        ).digest()

    return sorted(items, key=key)


def iter_parquet_blocks(
    source: ParquetSource,
    *,
    seed: str,
    token: str | bool | None = False,
    progress: Callable[[str, int], None] | None = None,
) -> Iterator[list[dict[str, Any]]]:
    """Yield whole row groups from seeded row groups, reading only the pinned columns.

    Shards are visited in a seeded order and their footers are read lazily, so
    a source with hundreds of shards costs one footer read per shard actually
    used rather than one per shard that exists.
    """

    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("parquet sampling requires the 'pyarrow' package") from exc
    if not source.files or len(source.files) > MAX_FILES_PER_SOURCE:
        raise ValueError(f"source {source.name} has an unusable file list")
    filesystem = _filesystem(token)
    shards = seeded_order(list(source.files), seed=seed, name=source.name, kind="shard")
    emitted = 0
    for filename in shards:
        path = _hf_path(source, filename)
        with filesystem.open(path, "rb") as handle:
            parquet = pq.ParquetFile(handle)
            total_groups = parquet.metadata.num_row_groups
            if total_groups <= 0 or total_groups > MAX_ROW_GROUPS_PER_FILE:
                raise ValueError(f"parquet file {path} has an unusable row-group count")
            order = seeded_order(
                list(range(total_groups)),
                seed=seed,
                name=source.name,
                kind=f"row_group:{filename}",
            )
            for index in order:
                rows = parquet.read_row_group(index, columns=list(source.columns)).to_pylist()
                emitted += len(rows)
                if progress is not None:
                    progress(source.name, emitted)
                # A row group is the unit that was actually transferred, so it
                # is also the unit a caller may stop on. Yielding it whole lets
                # selection rank across every row it paid for instead of taking
                # a prefix in stored order.
                yield rows


def iter_jsonl_blocks(
    source: JsonlSource,
    *,
    seed: str,
    token: str | bool | None = False,
    progress: Callable[[str, int], None] | None = None,
) -> Iterator[list[dict[str, Any]]]:
    """Yield whole blocks of rows from seeded byte offsets of a pinned JSONL file.

    A JSONL revision exposes no row index, so offsets are the only cheap
    handle. Each block discards its leading partial line and its trailing
    partial line, leaving whole JSON objects.
    """

    if not source.files or len(source.files) > MAX_FILES_PER_SOURCE:
        raise ValueError(f"source {source.name} has an unusable file list")
    filesystem = _filesystem(token)
    targets = []
    sizes: dict[str, int] = {}
    for filename in source.files:
        path = _hf_path(source, filename)
        size = int(filesystem.info(path)["size"])
        if size <= 0:
            raise ValueError(f"JSONL source {path} is empty")
        sizes[path] = size
        blocks = max(1, (size + JSONL_BLOCK_BYTES - 1) // JSONL_BLOCK_BYTES)
        targets.extend([path, index] for index in range(blocks))
    targets = seeded_order(targets, seed=seed, name=source.name, kind="byte_block")
    emitted = 0
    handles: dict[str, Any] = {}
    try:
        for path, block in targets:
            handle = handles.get(path)
            if handle is None:
                handle = handles[path] = filesystem.open(path, "rb")
            offset = block * JSONL_BLOCK_BYTES
            starts_mid_line = False
            if offset:
                handle.seek(offset - 1)
                starts_mid_line = handle.read(1) != b"\n"
            else:
                handle.seek(0)
            blob = handle.read(JSONL_BLOCK_BYTES)
            lines = blob.split(b"\n")
            # Every block but the first opens mid-line. Non-final blocks also
            # end mid-line, while the final block may end with a complete JSON
            # object even when the file has no trailing newline.
            at_eof = block * JSONL_BLOCK_BYTES + len(blob) >= sizes[path]
            line_end = None if at_eof and not blob.endswith(b"\n") else -1
            rows: list[dict[str, Any]] = []
            for line in lines[(1 if starts_mid_line else 0):line_end]:
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                if isinstance(value, dict):
                    rows.append(value)
            emitted += len(rows)
            if progress is not None:
                progress(source.name, emitted)
            yield rows
    finally:
        for handle in handles.values():
            try:
                handle.close()
            except Exception:
                pass


def iter_source_blocks(
    source: Source,
    *,
    seed: str,
    token: str | bool | None = False,
    progress: Callable[[str, int], None] | None = None,
) -> Iterator[list[dict[str, Any]]]:
    if isinstance(source, ParquetSource):
        return iter_parquet_blocks(source, seed=seed, token=token, progress=progress)
    if isinstance(source, JsonlSource):
        return iter_jsonl_blocks(source, seed=seed, token=token, progress=progress)
    raise ValueError(f"unsupported source kind: {source!r}")


def describe_source(source: Source) -> dict[str, Any]:
    """Manifest-safe description; validators compare these directly."""

    value = {
        "name": source.name,
        "kind": source.kind,
        "repo": source.repo,
        "revision": source.revision,
        "files": list(source.files),
    }
    if isinstance(source, ParquetSource):
        value["columns"] = list(source.columns)
    return value
