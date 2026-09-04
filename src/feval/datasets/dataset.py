"""Deterministic evaluation-window construction from pinned public sources.

Every miner and validator derives byte-identical rows for a window from the
same immutable revisions, without downloading the sources. Parquet footers give
a row-group map for free, and a seed drawn from the window number chooses which
row groups (or, for JSONL, which byte blocks) to read. Rows are normalised into
one protocol-owned shape, hash-ranked, and ordered by a second hash so no
source contributes a contiguous block.

Selection is deliberately conservative: a row is kept only when its answer can
be settled by exact string comparison. Anything needing a model judge, symbolic
algebra, or code execution is dropped rather than approximated.
"""

from __future__ import annotations

import hashlib
import heapq
import re
from pathlib import Path
from typing import Any, Callable, Iterable

from ..core.config import NetworkConfig
from ..core.constants import (
    BASE_MODEL,
    EVALUATION_ROWS,
    EVALUATION_SOURCES,
    GUESS_RESISTANT_ROWS,
    MAX_PROMPT_CHARS,
    SUBNET_NETUID,
)
from ..utils.crypto import hash_json
from ..utils.jsonutil import write_json, write_jsonl
from ..protocol.merkle import root_for_values
from ..protocol.schedule import evaluation_seed
from .sources import JsonlSource, ParquetSource, Source, describe_source, iter_source_blocks
from .tasks import NORMALIZERS, REQUIRED_MCQA_OPTIONS

PROTOCOL_EVALUATION_MANIFEST = "feval-dataset-manifest-v4"

VERIFIERS = ("strict_numeric", "mcqa_letter", "json_output_exact")
GUESS_RESISTANT_VERIFIERS = frozenset({"strict_numeric", "json_output_exact"})

# A source that cannot fill its quota must fail loudly rather than silently
# shrink the window, but it must also never pull an unbounded amount of data.
MAX_BLOCKS_PER_SOURCE = 4_096


def suspicious_prompt(prompt: str) -> str | None:
    """Reject prompts that read as an attempt to steer the model off-task."""

    lowered = prompt.lower()
    patterns = {
        "url": r"https?://|www\.",
        "script": r"<script|</script>|javascript:",
        "role_injection": r"ignore (all )?(previous|above) instructions|developer message|system message",
        "secret_request": r"private key|api key|password|token",
        "file_request": r"/etc/passwd|\.ssh|powershell|cmd\.exe",
    }
    for name, pattern in patterns.items():
        if re.search(pattern, lowered):
            return name
    return None


def build_source(spec: dict[str, Any]) -> Source:
    if spec["kind"] == "parquet":
        return ParquetSource(
            name=spec["name"],
            repo=spec["repo"],
            revision=spec["revision"],
            files=tuple(spec["files"]),
            columns=tuple(spec["columns"]),
        )
    if spec["kind"] == "jsonl":
        return JsonlSource(
            name=spec["name"],
            repo=spec["repo"],
            revision=spec["revision"],
            files=tuple(spec["files"]),
        )
    raise ValueError(f"unsupported evaluation source kind: {spec['kind']!r}")


def _rank(seed: str, domain: str, row_id: str) -> int:
    return int.from_bytes(
        hashlib.sha256(f"feval/window/v3\0{seed}\0{domain}\0{row_id}".encode("utf-8")).digest(),
        "big",
    )


def select_from_source(
    spec: dict[str, Any],
    *,
    seed: str,
    max_prompt_chars: int = MAX_PROMPT_CHARS,
    token: str | bool | None = False,
    progress: Callable[[str, int, int], None] | None = None,
    seen: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Read seeded blocks of one source until its quota can be filled.

    The stream order, the filters, and the tie-break hash are all functions of
    the pinned revision and the window seed, so two nodes that run this reach
    the same rows without either of them reading the whole dataset.

    ``seen`` may be shared across sources. Two of the pinned sources overlap in
    content — knowledge-mcqa is a refined subset of OpenScienceReasoning-2 — so
    without a shared set the same question can enter a window twice, where it
    would be scored twice and rewarded twice for one memorised answer.
    """

    source = build_source(spec)
    normalize = NORMALIZERS[spec["name"]]
    quota = int(spec["rows"])
    pool: list[dict[str, Any]] = []
    if seen is None:
        seen = set()
    # De-duplicate candidates within this source without reserving them for
    # later sources until quota selection has actually kept them. A fetched
    # row-group can be much larger than the quota, and an unselected candidate
    # must not suppress the same question in another source.
    candidate_seen = set(seen)
    scanned = 0
    blocks = 0
    for block in iter_source_blocks(source, seed=seed, token=token):
        blocks += 1
        for raw in block:
            scanned += 1
            row = normalize(raw, scanned)
            if row is None:
                continue
            if row["verifier"] != spec["verifier"]:
                raise ValueError(
                    f"source {spec['name']} produced verifier {row['verifier']!r}, "
                    f"expected {spec['verifier']!r}"
                )
            prompt = str(row["prompt"])
            if not prompt or len(prompt) > max_prompt_chars or suspicious_prompt(prompt):
                continue
            # De-duplicate on prompt text as well as id. A source can carry the
            # same question under two identifiers, and a repeated row would be
            # scored twice and memorised once.
            row_id = str(row["row_id"])
            fingerprint = hashlib.sha256(
                re.sub(r"\s+", " ", prompt).strip().lower().encode("utf-8")
            ).hexdigest()
            if row_id in candidate_seen or fingerprint in candidate_seen:
                continue
            candidate_seen.update((row_id, fingerprint))
            pool.append(row)
        if progress is not None:
            progress(spec["name"], scanned, len(pool))
        # Stop only on a block boundary. Every row of a fetched block is already
        # paid for, so ranking across the whole block costs nothing and avoids
        # selecting a prefix in stored order.
        if len(pool) >= quota or blocks >= MAX_BLOCKS_PER_SOURCE:
            break
    if len(pool) < quota:
        raise ValueError(
            f"source {spec['name']} yielded {len(pool)} usable rows, needs {quota}"
        )
    ranked = heapq.nsmallest(
        quota, pool, key=lambda row: (_rank(seed, spec["name"], row["row_id"]), row["row_id"])
    )
    for row in ranked:
        prompt = str(row["prompt"])
        fingerprint = hashlib.sha256(
            re.sub(r"\s+", " ", prompt).strip().lower().encode("utf-8")
        ).hexdigest()
        seen.update((str(row["row_id"]), fingerprint))
    if progress is not None:
        progress(spec["name"], scanned, len(ranked))
    return ranked


def _require_unique_row_ids(rows: list[dict[str, Any]]) -> None:
    seen: set[str] = set()
    for row in rows:
        row_id = str(row["row_id"])
        if row_id in seen:
            raise ValueError(f"evaluation set contains duplicate row_id {row_id!r}")
        seen.add(row_id)


def build_evaluation_window(
    *,
    window: int,
    netuid: int = SUBNET_NETUID,
    sources: Iterable[dict[str, Any]] = EVALUATION_SOURCES,
    evaluation_rows: int = EVALUATION_ROWS,
    max_prompt_chars: int = MAX_PROMPT_CHARS,
    out_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
    token: str | bool | None = False,
    progress: Callable[[str, int, int], None] | None = None,
) -> dict[str, Any]:
    specs = list(sources)
    seed = evaluation_seed(netuid, window)
    kept: list[dict[str, Any]] = []
    # One set for the whole window, so a later source never repeats a question
    # an earlier one already claimed. Source order is fixed, so which of two
    # overlapping sources wins a shared question is deterministic.
    seen: set[str] = set()
    for spec in specs:
        kept.extend(
            select_from_source(
                spec,
                seed=seed,
                max_prompt_chars=max_prompt_chars,
                token=token,
                progress=progress,
                seen=seen,
            )
        )
    if len(kept) != evaluation_rows:
        raise ValueError(
            f"evaluation window holds {len(kept)} rows; the protocol requires {evaluation_rows}"
        )
    # Interleave the sources so a rollout cannot be timed or truncated by task.
    kept.sort(
        key=lambda row: hashlib.sha256(
            f"feval/order/v3\0{seed}\0{row['row_id']}".encode("utf-8")
        ).digest()
    )
    _require_unique_row_ids(kept)
    by_task = {}
    by_verifier = {}
    by_license = {}
    for row in kept:
        by_task[row["task_type"]] = by_task.get(row["task_type"], 0) + 1
        by_verifier[row["verifier"]] = by_verifier.get(row["verifier"], 0) + 1
        by_license[str(row.get("license"))] = by_license.get(str(row.get("license")), 0) + 1
    guess_resistant = sum(
        count for name, count in by_verifier.items() if name in GUESS_RESISTANT_VERIFIERS
    )
    if guess_resistant < GUESS_RESISTANT_ROWS:
        raise ValueError(
            f"window holds {guess_resistant} guess-resistant rows; "
            f"the protocol requires at least {GUESS_RESISTANT_ROWS}"
        )
    manifest = {
        "protocol": PROTOCOL_EVALUATION_MANIFEST,
        "kind": "evaluation_window",
        "candidate_source": "seeded_row_groups",
        "base_model": BASE_MODEL,
        "netuid": netuid,
        "dataset_window": window,
        "evaluation_seed": seed,
        "rows": len(kept),
        "tasks": by_task,
        "verifiers": by_verifier,
        "licenses": by_license,
        "guess_resistant_rows": guess_resistant,
        "max_prompt_chars": max_prompt_chars,
        "sources": [
            {**describe_source(build_source(spec)), "rows": int(spec["rows"]),
             "verifier": spec["verifier"], "license": spec["license"]}
            for spec in specs
        ],
        "evaluation_root": root_for_values(kept),
        "filter_hash": hash_json(
            {
                "base_model": BASE_MODEL,
                "max_prompt_chars": max_prompt_chars,
                "verifiers": list(VERIFIERS),
                "required_mcqa_options": REQUIRED_MCQA_OPTIONS,
                "prompt_rejection_patterns": [
                    "url",
                    "script",
                    "role_injection",
                    "secret_request",
                    "file_request",
                ],
                "sources": [describe_source(build_source(spec)) for spec in specs],
            }
        ),
    }
    if out_path is not None:
        write_jsonl(out_path, kept, ascii_only=True)
    if manifest_path is not None:
        write_json(manifest_path, manifest)
    return manifest


def prepare_window_from_config(
    config: NetworkConfig,
    *,
    window: int,
    out_path: str | Path,
    manifest_path: str | Path,
    token: str | bool | None = False,
    progress: Callable[[str, int, int], None] | None = None,
) -> dict[str, Any]:
    config.validate()
    for spec in EVALUATION_SOURCES:
        revision = str(spec["revision"])
        if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
            raise ValueError(
                f"source {spec['name']} needs an immutable 40-character revision"
            )
    return build_evaluation_window(
        window=window,
        netuid=config.netuid,
        evaluation_rows=config.evaluation_rows,
        max_prompt_chars=config.max_prompt_chars,
        out_path=out_path,
        manifest_path=manifest_path,
        token=token,
        progress=progress,
    )
