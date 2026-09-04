from __future__ import annotations

import csv
import hashlib
import heapq
import json
import re
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

from ..core.config import NetworkConfig
from ..core.constants import (
    BASE_MODEL,
    INSTRUCTION_DATASET,
    MATH_DATASET,
    MAX_PROMPT_CHARS,
    SUBNET_NETUID,
)
from ..utils.crypto import hash_json
from ..utils.jsonutil import write_json, write_jsonl
from ..protocol.merkle import root_for_values
from .rewards import canonical_numeric_answer
from ..protocol.schedule import evaluation_seed


DEFAULT_SPLIT = "train"
PROTOCOL_EVALUATION_MANIFEST = "feval-dataset-manifest-v3"

SUPPORTED_INSTRUCTION_IDS = {
    "change_case:english_capital",
    "change_case:english_lowercase",
    "count:count_unique",
    "detectable_content:postscript",
    "detectable_content:number_placeholders",
    "detectable_format:number_highlighted_sections",
    "detectable_format:number_bullet_lists",
    "detectable_format:title",
    "first_word:first_word_sent",
    "first_word:first_word_answer",
    "keywords:existence",
    "keywords:forbidden_words",
    "keywords:frequency",
    "keywords:no_adjacent_consecutive",
    "keywords:word_once",
    "last_word:last_word_answer",
    "last_word:last_word_sent",
    "letters:letter_counting",
    "letters:letter_counting2",
    "length_constraints:number_paragraphs",
    "length_constraints:number_sentences",
    "length_constraints:number_words",
    "length_constraints:nth_paragraph_first_word",
    "paragraphs:paragraphs",
    "paragraphs:paragraphs2",
    "punctuation:no_comma",
    "punctuation:punctuation_dot",
    "punctuation:punctuation_exclamation",
    "startend:end_checker",
    "startend:quotation",
}

REQUIRED_CONSTRAINT_FIELDS = {
    "change_case:english_capital": (),
    "change_case:english_lowercase": (),
    "count:count_unique": (),
    "detectable_content:postscript": ("postscript_marker",),
    "detectable_content:number_placeholders": ("num_placeholders", "N"),
    "detectable_format:number_highlighted_sections": ("num_highlights", "N"),
    "detectable_format:number_bullet_lists": ("num_bullets", "num_bullet_lists", "N"),
    "detectable_format:title": (),
    "first_word:first_word_sent": ("first_word", "word"),
    "first_word:first_word_answer": ("first_word", "word"),
    "keywords:existence": ("keywords", "keyword"),
    "keywords:forbidden_words": ("forbidden_words", "keywords", "keyword"),
    "keywords:frequency": ("keyword",),
    "keywords:no_adjacent_consecutive": (),
    "keywords:word_once": ("keyword",),
    "last_word:last_word_answer": ("last_word", "word"),
    "last_word:last_word_sent": ("last_word", "word"),
    "letters:letter_counting": ("N",),
    "letters:letter_counting2": ("letter", "let_frequency"),
    "length_constraints:number_paragraphs": ("num_paragraphs", "N"),
    "length_constraints:number_sentences": ("num_sentences", "N"),
    "length_constraints:number_words": ("num_words", "N"),
    "length_constraints:nth_paragraph_first_word": ("num_paragraphs", "nth_paragraph", "first_word"),
    "paragraphs:paragraphs": (),
    "paragraphs:paragraphs2": (),
    "punctuation:no_comma": (),
    "punctuation:punctuation_dot": (),
    "punctuation:punctuation_exclamation": (),
    "startend:end_checker": ("end_phrase", "phrase", "suffix"),
    "startend:quotation": (),
}

SOURCE_COLUMNS = {
    MATH_DATASET: ["uuid", "problem", "expected_answer", "source", "license", "subset"],
    INSTRUCTION_DATASET: [
        "id",
        "prompt",
        "instruction_id_list",
        "kwargs",
        "grading_mode",
        "responses_create_params",
        "verifier_metadata",
    ],
}

DIRECT_JSONL_FILES = {
    INSTRUCTION_DATASET: "instruction_following.jsonl",
}


def _project_source_row(row: dict[str, Any], columns: list[str] | None) -> dict[str, Any]:
    if not columns:
        return row
    return {name: row.get(name) for name in columns if name in row}


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = json.loads(stripped)
                if isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError:
                pass
        return [value]
    return [value]


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _load_local(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        with source.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        return rows
    if suffix == ".json":
        with source.open("r", encoding="utf-8") as f:
            value = json.load(f)
        if isinstance(value, list):
            return value
        if isinstance(value, dict):
            for key in ("rows", "data", "train"):
                if isinstance(value.get(key), list):
                    return value[key]
        raise ValueError(f"JSON file {source} does not contain a list of rows")
    if suffix == ".csv":
        with source.open("r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    if suffix == ".parquet":
        try:
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError("reading parquet requires pandas plus pyarrow or fastparquet") from exc
        return pd.read_parquet(source).to_dict(orient="records")
    raise ValueError(f"unsupported dataset file type: {source.suffix}")


def _load_huggingface(
    dataset_name: str,
    split: str,
    limit: int | None = None,
    revision: str | None = None,
) -> Iterator[dict[str, Any]]:
    if dataset_name in DIRECT_JSONL_FILES:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise RuntimeError(
                "Hugging Face loading requires the 'huggingface-hub' package. "
                "Install the pinned project dependencies on the miner/validator host."
            ) from exc
        local_path = hf_hub_download(
            repo_id=dataset_name,
            repo_type="dataset",
            revision=revision,
            filename=DIRECT_JSONL_FILES[dataset_name],
        )
        columns = SOURCE_COLUMNS.get(dataset_name)
        with Path(local_path).open("r", encoding="utf-8") as stream:
            for index, line in enumerate(stream):
                if limit is not None and index >= limit:
                    break
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"malformed JSONL row {index + 1} in {dataset_name}"
                    ) from exc
                if not isinstance(row, dict):
                    raise ValueError(f"JSONL row {index + 1} in {dataset_name} is not an object")
                yield _project_source_row(row, columns)
        return
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Hugging Face loading requires the optional 'datasets' package. "
            "Install the pinned project dependencies on the miner/validator host."
        ) from exc
    columns = SOURCE_COLUMNS.get(dataset_name)
    load_kwargs: dict[str, Any] = {
        "split": split,
        "streaming": True,
        "revision": revision,
    }
    if columns:
        # Parquet projection avoids downloading/decoding large nested columns
        # that are irrelevant for Feval row selection and have triggered
        # pyarrow list-offset errors in the wild.
        load_kwargs["columns"] = columns
    try:
        stream = load_dataset(dataset_name, **load_kwargs)
    except (TypeError, ValueError) as exc:
        if not columns or "columns" not in str(exc):
            raise
        # Some Hugging Face builders, notably JSON/JSONL, do not support the
        # `columns` keyword. Retry without remote projection, then project each
        # row locally so the rest of Feval still sees only protocol fields.
        load_kwargs.pop("columns", None)
        stream = load_dataset(dataset_name, **load_kwargs)
    for index, row in enumerate(stream):
        if limit is not None and index >= limit:
            break
        yield _project_source_row(dict(row), columns)


def suspicious_prompt(prompt: str) -> str | None:
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


def _message_prompt(row: dict[str, Any]) -> str:
    messages = _as_list(row.get("messages"))
    if messages:
        first = _as_dict(messages[0])
        content = first.get("content")
        if content:
            return str(content).strip()
    params = _as_dict(row.get("responses_create_params"))
    input_rows = _as_list(params.get("input"))
    if input_rows:
        first = _as_dict(input_rows[0])
        content = first.get("content")
        if content:
            return str(content).strip()
    return str(row.get("prompt") or row.get("problem") or row.get("question") or "").strip()


def normalize_math_row(row: dict[str, Any], index: int, max_prompt_chars: int) -> dict[str, Any] | None:
    prompt = str(row.get("problem") or _message_prompt(row)).strip()
    expected = canonical_numeric_answer(row.get("expected_answer") or row.get("answer") or "")
    suffix = "\n\nReturn only the final answer as an integer, decimal, or fraction. Do not include reasoning."
    if not prompt or not expected or len(prompt) + len(suffix) > max_prompt_chars:
        return None
    if suspicious_prompt(prompt):
        return None
    row_id = str(row.get("uuid") or row.get("id") or f"math-{index}")
    return {
        "row_id": f"math:{row_id}",
        "task_type": "math",
        "prompt": prompt + suffix,
        "expected": [expected],
        "verifier": "strict_numeric",
        "source_dataset": MATH_DATASET,
        "source": row.get("source"),
        "license": row.get("license"),
        "subset": row.get("subset"),
    }


def _instruction_metadata(row: dict[str, Any]) -> dict[str, Any]:
    """Read NVIDIA's current nested schema and its legacy top-level schema."""

    nested = _as_dict(row.get("verifier_metadata"))
    return {
        name: nested.get(name, row.get(name))
        for name in ("instruction_id_list", "prompt", "kwargs", "grading_mode")
    }


def _constraint_specs(row: dict[str, Any]) -> list[dict[str, Any]]:
    metadata = _instruction_metadata(row)
    ids = [str(item) for item in _as_list(metadata.get("instruction_id_list")) if item]
    kwargs_list = _as_list(metadata.get("kwargs"))
    specs: list[dict[str, Any]] = []
    for index, instruction_id in enumerate(ids):
        params = _as_dict(kwargs_list[index]) if index < len(kwargs_list) else {}
        if instruction_id not in SUPPORTED_INSTRUCTION_IDS:
            return []
        required = REQUIRED_CONSTRAINT_FIELDS.get(instruction_id, ())
        if required and not any(params.get(name) is not None for name in required):
            return []
        specs.append({"id": instruction_id, "kwargs": params})
    return specs


def normalize_instruction_row(row: dict[str, Any], index: int, max_prompt_chars: int) -> dict[str, Any] | None:
    metadata = _instruction_metadata(row)
    prompt = _message_prompt(row) or str(metadata.get("prompt") or "").strip()
    if not prompt or len(prompt) > max_prompt_chars:
        return None
    if suspicious_prompt(prompt):
        return None
    constraints = _constraint_specs(row)
    if not constraints:
        return None
    grading_mode = str(metadata.get("grading_mode") or "binary")
    if grading_mode != "binary":
        return None
    row_id = str(row.get("id") or row.get("uuid") or f"instruction-{index}")
    return {
        "row_id": f"instruction_follow:{row_id}",
        "task_type": "instruction_follow",
        "prompt": prompt,
        "constraints": constraints,
        "verifier": "instruction_constraints",
        "source_dataset": INSTRUCTION_DATASET,
        "instruction_id_list": [spec["id"] for spec in constraints],
    }


def _load_source(
    input_file: str | Path | None,
    dataset_name: str,
    split: str,
    scan_limit: int | None,
    revision: str | None,
) -> Iterable[dict[str, Any]]:
    return _load_local(input_file) if input_file else _load_huggingface(dataset_name, split, scan_limit, revision)


def _select_streaming(
    rows: Iterable[dict[str, Any]],
    *,
    normalize: Any,
    count: int,
    seed: str,
    domain: str,
    max_prompt_chars: int,
    progress: Callable[[str, int, int], None] | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """Keep the smallest seeded hashes without retaining the source dataset."""

    heap: list[tuple[int, str, dict[str, Any]]] = []
    rejected = 0
    selected_ids: set[str] = set()
    scanned = 0
    if progress is not None:
        progress(domain, scanned, 0)
    for index, source in enumerate(rows):
        scanned = index + 1
        if progress is not None and scanned % 10_000 == 0:
            progress(domain, scanned, len(heap))
        row = normalize(source, index, max_prompt_chars)
        if row is None:
            rejected += 1
            continue
        row_id = str(row["row_id"])
        if row_id in selected_ids:
            continue
        rank = int.from_bytes(
            hashlib.sha256(f"feval/window/v2\0{seed}\0{domain}\0{row_id}".encode("utf-8")).digest(),
            "big",
        )
        entry = (-rank, row_id, row)
        if len(heap) < count:
            heapq.heappush(heap, entry)
            selected_ids.add(row_id)
        elif entry > heap[0]:
            removed = heapq.heapreplace(heap, entry)
            selected_ids.discard(removed[1])
            selected_ids.add(row_id)
    if progress is not None and scanned % 10_000 != 0:
        progress(domain, scanned, len(heap))
    return [entry[2] for entry in sorted(heap, key=lambda item: (-item[0], item[1]))], rejected


def _require_unique_row_ids(rows: list[dict[str, Any]], label: str) -> None:
    seen: set[str] = set()
    for row in rows:
        row_id = str(row["row_id"])
        if row_id in seen:
            raise ValueError(f"{label} contains duplicate row_id {row_id!r}")
        seen.add(row_id)


def prepare_combined_eval(
    out_path: str | Path,
    manifest_path: str | Path | None = None,
    math_input_file: str | Path | None = None,
    instruction_input_file: str | Path | None = None,
    math_dataset: str = MATH_DATASET,
    instruction_dataset: str = INSTRUCTION_DATASET,
    math_revision: str | None = None,
    instruction_revision: str | None = None,
    split: str = DEFAULT_SPLIT,
    scan_limit: int | None = None,
    max_rows: int = 10_000,
    math_rows: int | None = None,
    instruction_rows: int | None = None,
    max_prompt_chars: int = MAX_PROMPT_CHARS,
    seed: str | None = None,
    window: int | None = None,
    netuid: int = SUBNET_NETUID,
    progress: Callable[[str, int, int], None] | None = None,
) -> dict[str, Any]:
    math_budget = math_rows if math_rows is not None else max_rows // 2
    instruction_budget = instruction_rows if instruction_rows is not None else max_rows - math_budget
    selection_seed = seed or evaluation_seed(netuid, int(window or 0))
    raw_math = _load_source(math_input_file, math_dataset, split, scan_limit, math_revision)
    raw_instruction = _load_source(
        instruction_input_file,
        instruction_dataset,
        split,
        scan_limit,
        instruction_revision,
    )

    math_kept, math_rejected = _select_streaming(
        raw_math,
        normalize=normalize_math_row,
        count=math_budget,
        seed=selection_seed,
        domain="math",
        max_prompt_chars=max_prompt_chars,
        progress=progress,
    )
    instruction_kept, instruction_rejected = _select_streaming(
        raw_instruction,
        normalize=normalize_instruction_row,
        count=instruction_budget,
        seed=selection_seed,
        domain="instruction_follow",
        max_prompt_chars=max_prompt_chars,
        progress=progress,
    )
    for row in math_kept:
        row["source_dataset"] = math_dataset
    for row in instruction_kept:
        row["source_dataset"] = instruction_dataset
    rejected = {"math": math_rejected, "instruction_follow": instruction_rejected}
    kept = math_kept + instruction_kept
    kept = sorted(
        kept[:max_rows],
        key=lambda row: hashlib.sha256(
            f"feval/order/v2\0{selection_seed}\0{row['row_id']}".encode("utf-8")
        ).digest(),
    )
    _require_unique_row_ids(kept, "evaluation set")
    by_task = {
        "math": sum(1 for row in kept if row["task_type"] == "math"),
        "instruction_follow": sum(1 for row in kept if row["task_type"] == "instruction_follow"),
    }
    write_jsonl(out_path, kept)
    manifest = {
        "protocol": "feval-dataset-manifest-v1",
        "base_model": BASE_MODEL,
        "datasets": {
            "math": math_dataset,
            "instruction_follow": instruction_dataset,
        },
        "dataset_revisions": {
            "math": math_revision,
            "instruction_follow": instruction_revision,
        },
        "split": split,
        "netuid": netuid,
        "dataset_window": window,
        "evaluation_seed": selection_seed,
        "rows": len(kept),
        "rejected": rejected,
        "tasks": by_task,
        "max_prompt_chars": max_prompt_chars,
        "evaluation_root": root_for_values(kept),
        "filter_hash": hash_json({
            "base_model": BASE_MODEL,
            "math_dataset": math_dataset,
            "instruction_dataset": instruction_dataset,
            "max_prompt_chars": max_prompt_chars,
            "supported_instruction_ids": sorted(SUPPORTED_INSTRUCTION_IDS),
            "verifiers": ["strict_numeric", "instruction_constraints"],
            "prompt_rejection_patterns": ["url", "script", "role_injection", "secret_request", "file_request"],
        }),
    }
    if manifest_path:
        write_json(manifest_path, manifest)
    return manifest


def prepare_window_from_config(
    config: NetworkConfig,
    *,
    window: int,
    out_path: str | Path,
    manifest_path: str | Path,
    math_input_file: str | Path | None = None,
    instruction_input_file: str | Path | None = None,
    scan_limit: int | None = None,
    progress: Callable[[str, int, int], None] | None = None,
) -> dict[str, Any]:
    # Every miner and validator derives the same window from immutable dataset
    # snapshots and code-pinned protocol rules.
    config.validate()
    for revision in (config.math_revision, config.instruction_revision):
        if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
            raise ValueError("evaluation windows require immutable dataset revisions")
    manifest = prepare_combined_eval(
        out_path=out_path,
        manifest_path=manifest_path,
        math_input_file=math_input_file,
        instruction_input_file=instruction_input_file,
        math_dataset=config.math_dataset,
        instruction_dataset=config.instruction_dataset,
        math_revision=config.math_revision,
        instruction_revision=config.instruction_revision,
        split=config.split,
        scan_limit=scan_limit,
        max_rows=config.evaluation_rows,
        math_rows=config.math_rows,
        instruction_rows=config.instruction_rows,
        max_prompt_chars=config.max_prompt_chars,
        seed=evaluation_seed(config.netuid, window),
        window=window,
        netuid=config.netuid,
        progress=progress,
    )
    manifest["protocol"] = PROTOCOL_EVALUATION_MANIFEST
    manifest["kind"] = "evaluation_window"
    manifest["candidate_source"] = "all_valid_rows"
    write_json(manifest_path, manifest)
    if manifest["rows"] != config.evaluation_rows:
        raise ValueError(
            f"dataset window contains {manifest['rows']} valid rows; "
            f"the protocol requires {config.evaluation_rows}"
        )
    return manifest



