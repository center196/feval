from __future__ import annotations

import atexit
import contextlib
import gc
import json
import math
import multiprocessing
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any

from .artifacts import (
    ModelCommitment,
    RolloutManifest,
    adapter_digest,
    prepare_runtime_adapter,
    write_rollouts,
)
from ..core.config import NetworkConfig
from ..core.constants import (
    AUDIT_GPU_MEMORY_UTILIZATION,
    AUDIT_MAX_NUM_BATCHED_TOKENS,
    AUDIT_MAX_NUM_SEQS,
    DEFAULT_ROLLOUT_BATCH_SIZE,
)
from ..utils.jsonutil import load_jsonl, write_json


_ACTIVE_VLLMS: list[Any] = []
_SIGNAL_HANDLERS_INSTALLED = False
_INTERRUPT_REQUESTED = False
_LORA_ID_LOCK = threading.Lock()
_LORA_ID_BY_DIGEST: dict[str, int] = {}
_LORA_DIGEST_BY_ID: dict[int, str] = {}
_MAX_VLLM_LORA_ID = (1 << 31) - 1


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def configure_vllm_environment() -> None:
    # Canonical miner decoding uses vLLM's V1 batch-level custom logits
    # processor. vLLM 0.27 selects V1 itself; setting the removed VLLM_USE_V1
    # variable only produces a warning. Cleanup below bounds EngineCore life.
    os.environ.pop("VLLM_USE_V1", None)
    # Newer vLLM releases always use an EngineCore process for supported model
    # families. Keep its own worker cleanup bounded as a second line of defence.
    os.environ.setdefault("VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS", "5")
    # Dense Qwen3 is supported by vLLM's batch-invariant kernels. Pin the mode
    # for miners and validators so request ordering and batch size do not alter
    # greedy token selection on supported GPUs.
    os.environ["VLLM_BATCH_INVARIANT"] = "1"


def fixed_prompt(prompt: str) -> str:
    """The protocol-owned text template; miner repos cannot replace it."""

    return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"


def load_protocol_tokenizer(config: NetworkConfig):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("tokenization requires the 'transformers' package") from exc
    # Remote custom Python and miner-provided tokenizers are never allowed.
    return AutoTokenizer.from_pretrained(
        config.base_model,
        revision=config.base_revision,
        trust_remote_code=False,
    )


def tokenizer_vocab_size(tokenizer: Any) -> int:
    # ``vocab_size`` is the base vocabulary size in Transformers and excludes
    # added tokens for some tokenizers. Qwen keeps protocol-relevant special
    # tokens such as <|im_end|> in that added range, so using only the base
    # value falsely rejects valid vLLM output. get_vocab() includes both base
    # and added tokens and also gives us the actual highest decodable ID.
    get_vocab = getattr(tokenizer, "get_vocab", None)
    if callable(get_vocab):
        vocabulary = get_vocab()
        token_ids = [
            int(token_id)
            for token_id in vocabulary.values()
            if isinstance(token_id, int) and not isinstance(token_id, bool) and token_id >= 0
        ]
        if token_ids:
            return max(token_ids) + 1

    sizes: list[int] = []
    value = getattr(tokenizer, "vocab_size", None)
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        sizes.append(int(value))
    try:
        length = len(tokenizer)
    except (TypeError, AttributeError):
        length = 0
    if isinstance(length, int) and length > 0:
        sizes.append(int(length))
    if not sizes:
        raise ValueError("protocol tokenizer does not expose a valid vocabulary")
    return max(sizes)


def generation_stop_token_ids(tokenizer: Any) -> list[int]:
    ids: list[int] = []
    eos = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos, int):
        ids.append(int(eos))
    # Qwen3's pinned tokenizer and model generation config use different EOS
    # declarations: <|im_end|> and <|endoftext|>, respectively. vLLM observes
    # the model EOS automatically, so both must be protocol terminal tokens.
    for special in ("<|im_end|>", "<|endoftext|>"):
        encoded = tokenizer.encode(special, add_special_tokens=False)
        if len(encoded) == 1:
            ids.append(int(encoded[0]))
    return list(dict.fromkeys(ids))


def decode_rollout(tokenizer: Any, tokens: list[int]) -> str:
    return str(
        tokenizer.decode(
            tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
    ).strip()


def _runtime_lora_identifier(model_digest: str) -> int:
    """Allocate a positive signed-int32 vLLM cache key for a model digest."""

    if len(model_digest) != 64 or any(character not in "0123456789abcdef" for character in model_digest):
        raise ValueError("model digest must be 64 lowercase hexadecimal characters")
    with _LORA_ID_LOCK:
        existing = _LORA_ID_BY_DIGEST.get(model_digest)
        if existing is not None:
            return existing
        candidate = (int(model_digest, 16) % _MAX_VLLM_LORA_ID) + 1
        while True:
            registered_digest = _LORA_DIGEST_BY_ID.get(candidate)
            if registered_digest is None or registered_digest == model_digest:
                _LORA_ID_BY_DIGEST[model_digest] = candidate
                _LORA_DIGEST_BY_ID[candidate] = model_digest
                return candidate
            candidate = 1 if candidate == _MAX_VLLM_LORA_ID else candidate + 1


def _lora_request(adapter_dir: str | Path, model_digest: str):
    configure_vllm_environment()
    try:
        from vllm.lora.request import LoRARequest
    except ImportError as exc:
        raise RuntimeError("LoRA inference requires vLLM") from exc
    identifier = _runtime_lora_identifier(model_digest)
    return LoRARequest(f"feval-{model_digest[:12]}", identifier, str(Path(adapter_dir).resolve()))


def _new_vllm(
    config: NetworkConfig,
    *,
    max_model_len: int | None = None,
    reserve_output_token: bool = False,
    max_num_seqs: int | None = None,
    max_num_batched_tokens: int | None = None,
    gpu_memory_utilization: float | None = None,
):
    if reserve_output_token:
        # vLLM requires one ignored generated token after the complete
        # teacher-forced audit sequence. It is not miner controlled and is not
        # included in any score or audit result.
        os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
    configure_vllm_environment()
    try:
        from vllm import LLM
    except ImportError as exc:
        raise RuntimeError("GPU inference requires vLLM on Linux with a supported GPU") from exc
    # Cover model loading as well as generation, then register_vllm() installs
    # our handler again in case vLLM replaced it during engine construction.
    _install_vllm_shutdown_handlers()
    protocol_max_model_len = int(max_model_len or config.max_context_tokens)
    if protocol_max_model_len <= 0 or protocol_max_model_len > config.max_context_tokens:
        raise ValueError("effective vLLM context length exceeds the protocol context limit")
    # vLLM requires at least one generated token even when Feval only requests
    # teacher-forced prompt log-probabilities. Auditors reserve that one
    # engine-only position beyond the submitted protocol prefix.
    engine_max_model_len = protocol_max_model_len + int(reserve_output_token)
    runtime_options: dict[str, Any] = {}
    if max_num_batched_tokens is not None:
        runtime_options["max_num_batched_tokens"] = int(max_num_batched_tokens)
        runtime_options["enable_chunked_prefill"] = True
    if gpu_memory_utilization is not None:
        runtime_options["gpu_memory_utilization"] = float(gpu_memory_utilization)
    return LLM(
        model=config.base_model,
        revision=config.base_revision,
        # Feval always supplies protocol-tokenized IDs, so vLLM does not need
        # its own tokenizer, detokenizer, chat renderer, or processor warmups.
        tokenizer=None,
        tokenizer_revision=None,
        skip_tokenizer_init=True,
        trust_remote_code=False,
        dtype="bfloat16",
        enable_lora=True,
        max_lora_rank=config.max_lora_rank,
        max_model_len=engine_max_model_len,
        max_num_seqs=max_num_seqs,
        tensor_parallel_size=config.tensor_parallel_size,
        seed=0,
        enforce_eager=True,
        # Keep the protocol text-only and avoid tokenizer/chat warmups.
        language_model_only=True,
        enable_prefix_caching=False,
        disable_log_stats=True,
        **runtime_options,
    )


def _call_if_present(component: Any, name: str) -> None:
    method = getattr(component, name, None)
    if callable(method):
        method()


def _shutdown_vllm_impl(llm: Any) -> None:
    engine = getattr(llm, "llm_engine", None) or getattr(llm, "engine", None)
    engine_core = getattr(engine, "engine_core", None)
    model_executor = getattr(engine, "model_executor", None)
    for component in (llm, engine_core, model_executor, engine):
        if component is None:
            continue
        for name in ("shutdown", "close", "cleanup"):
            with contextlib.suppress(Exception):
                _call_if_present(component, name)
    with contextlib.suppress(Exception):
        from vllm.distributed.parallel_state import (
            destroy_distributed_environment,
            destroy_model_parallel,
        )

        destroy_model_parallel()
        destroy_distributed_environment()
    with contextlib.suppress(Exception):
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    gc.collect()


def _terminate_vllm_children() -> None:
    """Terminate multiprocessing children owned by this Feval CLI process."""

    children = multiprocessing.active_children()
    for child in children:
        with contextlib.suppress(Exception):
            child.terminate()
    deadline = time.monotonic() + 2.0
    for child in children:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        with contextlib.suppress(Exception):
            child.join(remaining)
    for child in children:
        with contextlib.suppress(Exception):
            if child.is_alive():
                child.kill()
    for child in children:
        with contextlib.suppress(Exception):
            child.join(0.5)


def shutdown_vllm(llm: Any | None) -> None:
    """Bounded cleanup for embedded vLLM engines and CUDA resources."""

    if llm is None:
        return
    with contextlib.suppress(ValueError):
        _ACTIVE_VLLMS.remove(llm)
    timeout_text = os.environ.get("FEVAL_VLLM_SHUTDOWN_TIMEOUT_SECONDS", "10")
    try:
        timeout = max(0.1, float(timeout_text))
    except ValueError:
        timeout = 10.0
    worker = threading.Thread(
        target=_shutdown_vllm_impl,
        args=(llm,),
        name="FevalVllmShutdown",
        daemon=True,
    )
    worker.start()
    worker.join(timeout)
    if worker.is_alive():
        _rollout_progress("forcing_vllm_shutdown", timeout_seconds=timeout)
        _terminate_vllm_children()


def _shutdown_active_vllms() -> None:
    for llm in list(_ACTIVE_VLLMS):
        shutdown_vllm(llm)


def _handle_vllm_signal(signum: int, _frame: Any) -> None:
    global _INTERRUPT_REQUESTED
    if _INTERRUPT_REQUESTED:
        # Emergency path for a second Ctrl-C while graceful cleanup is stuck.
        _terminate_vllm_children()
        os._exit(130 if signum == getattr(signal, "SIGINT", None) else 128 + signum)
    _INTERRUPT_REQUESTED = True
    _rollout_progress("interrupt_received", signal=signum)
    # Never run vLLM cleanup in a Python signal handler: its IPC shutdown can
    # block, making Ctrl-C appear ignored. Normal finally blocks perform the
    # bounded cleanup after this exception unwinds the current operation.
    if signum == getattr(signal, "SIGINT", None):
        raise KeyboardInterrupt
    raise SystemExit(128 + signum)


def _install_vllm_shutdown_handlers() -> None:
    global _SIGNAL_HANDLERS_INSTALLED
    if not _SIGNAL_HANDLERS_INSTALLED:
        _SIGNAL_HANDLERS_INSTALLED = True
        atexit.register(_shutdown_active_vllms)
    # Reinstall every time an engine is created because vLLM may replace the
    # parent process handlers during its own multiprocessing initialization.
    for sig in (getattr(signal, "SIGINT", None), getattr(signal, "SIGTERM", None)):
        if sig is None:
            continue
        with contextlib.suppress(Exception):
            signal.signal(sig, _handle_vllm_signal)


def register_vllm(llm: Any) -> Any:
    _install_vllm_shutdown_handlers()
    _ACTIVE_VLLMS.append(llm)
    return llm


def _rollout_progress(event: str, **fields: Any) -> None:
    print(
        json.dumps({"status": event, **fields}, sort_keys=True),
        file=sys.stderr,
        flush=True,
    )


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def rollout_output_limit(config: NetworkConfig, requested: int | None = None) -> int:
    value = config.max_output_tokens if requested is None else requested
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("max_output_tokens must be an integer")
    if value <= 0 or value > config.max_output_tokens:
        raise ValueError(
            f"max_output_tokens must be between 1 and {config.max_output_tokens}"
        )
    return value


def rollout_context_limit(
    *,
    config: NetworkConfig,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    max_output_tokens: int | None = None,
) -> tuple[int, int]:
    output_limit = rollout_output_limit(config, max_output_tokens)
    max_prompt_tokens = 0
    for row in rows:
        prompt_ids = tokenizer.encode(fixed_prompt(str(row["prompt"])), add_special_tokens=False)
        max_prompt_tokens = max(max_prompt_tokens, len(prompt_ids))
    if max_prompt_tokens >= config.max_context_tokens:
        raise ValueError(
            f"selected evaluation window has a {max_prompt_tokens}-token prompt, which leaves no room "
            f"for generation under the protocol context limit {config.max_context_tokens}"
        )
    max_new_tokens = min(output_limit, config.max_context_tokens - max_prompt_tokens)
    required = max_prompt_tokens + max_new_tokens
    # Use the smallest aligned context that can hold the longest prompt plus
    # its prompt-adjusted generation budget.
    return min(config.max_context_tokens, max(2_048, _round_up(required, 512))), max_prompt_tokens


def protocol_rollout_tokens(
    config: NetworkConfig,
    tokenizer: Any,
    prompt: str,
    tokens: list[int],
    *,
    max_output_tokens: int | None = None,
) -> tuple[list[int], list[int]]:
    """Return the single protocol-defined rollout prefix used everywhere.

    Responses are capped by the miner-selected manifest limit, then further
    capped by the protocol context after the fixed prompt.
    Per-row miner-provided stop or length metadata is deliberately irrelevant.
    """

    prompt_ids = tokenizer.encode(fixed_prompt(prompt), add_special_tokens=False)
    output_limit = rollout_output_limit(config, max_output_tokens)
    available = min(output_limit, config.max_context_tokens - len(prompt_ids))
    if available <= 0:
        raise ValueError("rollout prompt leaves no room under the protocol context limit")
    if len(tokens) > available:
        raise ValueError(
            "rollout response exceeds the context remaining after its prompt"
        )
    bounded = list(tokens)
    stop_ids = set(generation_stop_token_ids(tokenizer))
    if any(token_id in stop_ids for token_id in bounded[:-1]):
        raise ValueError("rollout continues after a protocol terminal token")
    if len(bounded) < available:
        if not bounded or bounded[-1] not in stop_ids:
            raise ValueError(
                "rollout ends before its protocol token budget without a terminal token"
            )
    return bounded, list(prompt_ids)


def generate_rollouts_vllm(
    *,
    config: NetworkConfig,
    eval_path: str | Path,
    adapter_dir: str | Path,
    model_digest: str,
    max_output_tokens: int | None = None,
    batch_size: int = DEFAULT_ROLLOUT_BATCH_SIZE,
    runtime_adapter_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    try:
        configure_vllm_environment()
        from vllm import SamplingParams
    except ImportError as exc:
        raise RuntimeError("GPU inference requires vLLM") from exc
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    rows = load_jsonl(eval_path)
    if len(rows) != config.evaluation_rows:
        raise ValueError(f"evaluation set must contain exactly {config.evaluation_rows} rows")
    actual_digest = adapter_digest(adapter_dir, config)
    if actual_digest != model_digest:
        raise ValueError("local adapter does not match the committed model digest")
    tokenizer = load_protocol_tokenizer(config)
    output_limit = rollout_output_limit(config, max_output_tokens)
    effective_max_model_len, max_prompt_tokens = rollout_context_limit(
        config=config,
        tokenizer=tokenizer,
        rows=rows,
        max_output_tokens=output_limit,
    )
    _rollout_progress(
        "loading_vllm",
        model=config.base_model,
        protocol_max_model_len=config.max_context_tokens,
        max_model_len=effective_max_model_len,
        max_prompt_tokens=max_prompt_tokens,
        rows=len(rows),
        batch_size=batch_size,
    )
    llm = None
    try:
        llm = register_vllm(
            _new_vllm(
                config,
                max_model_len=effective_max_model_len,
                max_num_seqs=batch_size,
            )
        )
        _rollout_progress(
            "vllm_ready",
            rows=len(rows),
            batch_size=batch_size,
            max_model_len=effective_max_model_len,
        )
        stop_token_ids = generation_stop_token_ids(tokenizer)
        lora = _lora_request(runtime_adapter_dir or adapter_dir, model_digest)
        if _env_bool("FEVAL_VLLM_LORA_WARMUP", True):
            warmup_count = min(batch_size, 8)
            warmup_prompt_ids = tokenizer.encode(
                fixed_prompt("Feval LoRA warmup."),
                add_special_tokens=False,
            )
            warmup_max_tokens = min(output_limit, config.max_context_tokens - len(warmup_prompt_ids))
            warmup_params = SamplingParams(
                temperature=0.0,
                top_p=1.0,
                seed=0,
                max_tokens=max(1, min(1, warmup_max_tokens)),
                stop_token_ids=stop_token_ids,
                detokenize=False,
                skip_special_tokens=False,
            )
            _rollout_progress("warming_up_lora", requests=warmup_count)
            llm.generate(
                [{"prompt_token_ids": warmup_prompt_ids} for _ in range(warmup_count)],
                sampling_params=warmup_params,
                lora_request=lora,
                use_tqdm=False,
            )
            _rollout_progress("lora_warmup_ready", requests=warmup_count)
        # Queue the complete evaluation window in one call. max_num_seqs above
        # remains the active-sequence limit, so vLLM continuously backfills a
        # slot as soon as a shorter rollout finishes instead of waiting for the
        # longest request in an application-level chunk.
        prompts = [
            {
                "prompt_token_ids": tokenizer.encode(
                    fixed_prompt(str(row["prompt"])),
                    add_special_tokens=False,
                )
            }
            for row in rows
        ]
        request_max_tokens = [
            min(
                output_limit,
                config.max_context_tokens - len(prompt["prompt_token_ids"]),
            )
            for prompt in prompts
        ]
        if any(value <= 0 for value in request_max_tokens):
            raise ValueError("a rollout prompt leaves no room for generation under the context limit")
        params = [
            SamplingParams(
                temperature=0.0,
                top_p=1.0,
                seed=0,
                max_tokens=request_max_tokens[index],
                stop_token_ids=stop_token_ids,
                detokenize=False,
                skip_special_tokens=False,
            )
            for index in range(len(rows))
        ]
        _rollout_progress(
            "generating_rollouts",
            queued=len(rows),
            max_active_sequences=min(batch_size, len(rows)),
            max_tokens=max(request_max_tokens),
        )
        outputs = llm.generate(
            prompts,
            sampling_params=params,
            lora_request=lora,
            use_tqdm=True,
        )
        if len(outputs) != len(rows):
            raise RuntimeError("vLLM returned a different number of outputs than prompts")
        generated: list[dict[str, Any]] = []
        for request_index, (source, request_output) in enumerate(zip(rows, outputs)):
            completion = request_output.outputs[0]
            token_ids = [int(item) for item in completion.token_ids]
            token_budget = request_max_tokens[request_index]
            finish_reason = getattr(completion, "finish_reason", None)
            stop_reason = getattr(completion, "stop_reason", None)
            if finish_reason == "stop":
                if isinstance(stop_reason, int) and not isinstance(stop_reason, bool):
                    if not token_ids or token_ids[-1] != int(stop_reason):
                        token_ids.append(int(stop_reason))
                elif not token_ids or token_ids[-1] not in stop_token_ids:
                    raise RuntimeError("vLLM stopped without exposing its terminal token")
            elif finish_reason == "length":
                if len(token_ids) != token_budget:
                    raise RuntimeError("vLLM length finish returned an incomplete rollout")
            else:
                raise RuntimeError(f"vLLM returned unsupported finish reason {finish_reason!r}")
            if len(token_ids) > token_budget:
                raise RuntimeError("vLLM returned more tokens than the protocol budget")
            generated.append(
                {
                    "row_id": str(source["row_id"]),
                    "tokens": token_ids,
                }
            )
        _rollout_progress("generated_rollouts", rows=len(generated))
        return generated
    finally:
        if llm is not None:
            _rollout_progress("shutting_down_vllm")
            shutdown_vllm(llm)


def build_rollout_bundle_vllm(
    *,
    config: NetworkConfig,
    eval_path: str | Path,
    eval_manifest_path: str | Path,
    adapter_dir: str | Path,
    commitment: ModelCommitment,
    miner_hotkey: str,
    out_dir: str | Path,
    max_output_tokens: int | None = None,
    batch_size: int = DEFAULT_ROLLOUT_BATCH_SIZE,
) -> RolloutManifest:
    eval_manifest = json.loads(Path(eval_manifest_path).read_text(encoding="utf-8"))
    runtime_adapter = prepare_runtime_adapter(
        adapter_dir,
        Path(out_dir) / "validated-runtime",
        config,
    )
    output_limit = rollout_output_limit(config, max_output_tokens)
    rows = generate_rollouts_vllm(
        config=config,
        eval_path=eval_path,
        adapter_dir=adapter_dir,
        model_digest=commitment.model_digest,
        max_output_tokens=output_limit,
        batch_size=batch_size,
        runtime_adapter_dir=runtime_adapter,
    )
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    rows_sha256 = write_rollouts(
        root / "rollouts.jsonl",
        rows,
        max_output_tokens=output_limit,
    )
    manifest = RolloutManifest(
        miner_hotkey=miner_hotkey,
        model_repo=commitment.model_repo,
        model_revision=commitment.model_revision,
        model_digest=commitment.model_digest,
        dataset_window=int(eval_manifest["dataset_window"]),
        evaluation_seed=str(eval_manifest["evaluation_seed"]),
        evaluation_root=str(eval_manifest["evaluation_root"]),
        row_count=len(rows),
        rows_sha256=rows_sha256,
        base_model=config.base_model,
        base_revision=config.base_revision,
        max_output_tokens=output_limit,
    )
    manifest.validate(config)
    write_json(root / "manifest.json", manifest.to_dict())
    return manifest


def _logprob_value(item: Any) -> float | None:
    value = getattr(item, "logprob", None)
    if value is None and isinstance(item, dict):
        value = item.get("logprob")
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _logprob_rank(item: Any) -> int | None:
    value = getattr(item, "rank", None)
    if value is None and isinstance(item, dict):
        value = item.get("rank")
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return int(value)


def _token_audit(
    position: Any,
    token_id: int,
    *,
    min_relative_probability: float,
) -> dict[str, Any]:
    """Accept only rank-one greedy tokens at the configured probability floor."""

    invalid = {
        "valid": False,
        "rank": None,
        "logprob_gap": None,
        "accepted_token_ids": [],
    }
    if position is None:
        return invalid
    try:
        candidates = [
            (int(candidate_id), logprob, _logprob_rank(item))
            for candidate_id, item in position.items()
            if (logprob := _logprob_value(item)) is not None
        ]
    except (TypeError, AttributeError, ValueError):
        return invalid
    if not candidates:
        return invalid
    candidates.sort(key=lambda candidate: (-candidate[1], candidate[0]))
    top = candidates[0][1]
    gap_limit = -math.log(min_relative_probability)
    accepted = [
        candidate_id
        for candidate_id, logprob, _rank in candidates
        if top - logprob <= gap_limit
    ]
    submitted = next((item for item in candidates if item[0] == token_id), None)
    rank = None if submitted is None else submitted[2]
    if submitted is not None and rank is None:
        rank = candidates.index(submitted) + 1
    gap = None if submitted is None else max(0.0, top - submitted[1])
    return {
        "valid": token_id in accepted,
        "rank": rank,
        "logprob_gap": gap,
        "accepted_token_ids": accepted,
    }


class VllmAuditEngine:
    """Shared-base, teacher-forced rollout verifier using vLLM prompt logprobs."""

    def __init__(self, config: NetworkConfig, *, tokenizer: Any | None = None):
        self.config = config
        started = time.monotonic()
        _rollout_progress(
            "loading_vllm_audit",
            model=config.base_model,
            protocol_max_model_len=config.max_context_tokens,
            engine_max_model_len=config.max_context_tokens + 1,
            max_num_seqs=AUDIT_MAX_NUM_SEQS,
            max_num_batched_tokens=AUDIT_MAX_NUM_BATCHED_TOKENS,
            gpu_memory_utilization=AUDIT_GPU_MEMORY_UTILIZATION,
            language_model_only=True,
        )
        self.tokenizer = tokenizer if tokenizer is not None else load_protocol_tokenizer(config)
        self.llm = register_vllm(
            _new_vllm(
                config,
                reserve_output_token=True,
                max_num_seqs=AUDIT_MAX_NUM_SEQS,
                max_num_batched_tokens=AUDIT_MAX_NUM_BATCHED_TOKENS,
                gpu_memory_utilization=AUDIT_GPU_MEMORY_UTILIZATION,
            )
        )
        _rollout_progress(
            "vllm_audit_ready",
            elapsed_seconds=round(time.monotonic() - started, 3),
        )

    def close(self) -> None:
        shutdown_vllm(self.llm)
        self.llm = None

    def __enter__(self) -> "VllmAuditEngine":
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        self.close()

    def verify(
        self,
        *,
        adapter_dir: str | Path,
        model_digest: str,
        max_output_tokens: int,
        rows: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        try:
            configure_vllm_environment()
            from vllm import SamplingParams
        except ImportError as exc:
            raise RuntimeError("GPU auditing requires vLLM") from exc
        inputs: list[dict[str, list[int]]] = []
        prompt_lengths: list[int] = []
        normalized_rows: list[dict[str, Any]] = []
        for row in rows:
            response_ids, prompt_ids = protocol_rollout_tokens(
                self.config,
                self.tokenizer,
                str(row["prompt"]),
                list(row["tokens"]),
                max_output_tokens=max_output_tokens,
            )
            if not response_ids:
                raise ValueError(f"rollout {row['row_id']!r} is empty")
            prompt_lengths.append(len(prompt_ids))
            # Keep every submitted token, including the final one, in the same
            # teacher-forced prefill path. The engine has one additional,
            # non-protocol position because vLLM requires one generated token.
            inputs.append({"prompt_token_ids": prompt_ids + response_ids})
            normalized_rows.append({**row, "tokens": response_ids})
        params = SamplingParams(
            temperature=0.0,
            max_tokens=1,
            # vLLM returns the top candidate and the actual prompt token. That
            # is sufficient to compare their probabilities without a rank cap.
            prompt_logprobs=1,
            detokenize=False,
            skip_special_tokens=False,
        )
        if self.llm is None:
            raise RuntimeError("vLLM audit engine is already closed")
        total_input_tokens = sum(len(item["prompt_token_ids"]) for item in inputs)
        started = time.monotonic()
        _rollout_progress(
            "verifying_rollouts",
            rows=len(rows),
            input_tokens=total_input_tokens,
        )
        lora_request = _lora_request(adapter_dir, model_digest)
        outputs = []
        for start in range(0, len(inputs), AUDIT_MAX_NUM_SEQS):
            microbatch = inputs[start : start + AUDIT_MAX_NUM_SEQS]
            _rollout_progress(
                "verifying_rollout_microbatch",
                start=start,
                end=start + len(microbatch),
                rows=len(inputs),
                input_tokens=sum(len(item["prompt_token_ids"]) for item in microbatch),
            )
            outputs.extend(
                self.llm.generate(
                    microbatch,
                    sampling_params=params,
                    lora_request=lora_request,
                    use_tqdm=False,
                )
            )
        elapsed = time.monotonic() - started
        _rollout_progress(
            "verified_rollouts",
            rows=len(rows),
            input_tokens=total_input_tokens,
            elapsed_seconds=round(elapsed, 3),
            tokens_per_second=round(total_input_tokens / elapsed, 1) if elapsed > 0 else None,
        )
        reports: list[dict[str, Any]] = []
        for source, output, prompt_length in zip(normalized_rows, outputs, prompt_lengths):
            failure_position = None
            failure_rank = None
            failure_logprob_gap = None
            exact_argmax_tokens = 0
            near_top_tokens = 0
            failure_accepted_token_ids: list[int] = []
            prompt_logprobs = output.prompt_logprobs
            for offset, token_id in enumerate(source["tokens"]):
                position = prompt_length + offset
                position_data = (
                    prompt_logprobs[position]
                    if position < len(prompt_logprobs)
                    else None
                )
                verdict = (
                    _token_audit(
                        position_data,
                        token_id,
                        min_relative_probability=self.config.audit_min_relative_probability,
                    )
                )
                if verdict["rank"] == 1:
                    exact_argmax_tokens += 1
                elif verdict["valid"]:
                    near_top_tokens += 1
                if not verdict["valid"] and failure_position is None:
                    failure_position = offset
                    failure_rank = verdict["rank"]
                    failure_logprob_gap = verdict["logprob_gap"]
                    failure_accepted_token_ids = verdict["accepted_token_ids"]
            tokens_checked = len(source["tokens"])
            reports.append(
                {
                    "row_id": source["row_id"],
                    "valid": failure_position is None,
                    "failure_position": failure_position,
                    "failure_reason": "token_not_greedy_argmax" if failure_position is not None else None,
                    "failure_rank": failure_rank,
                    "failure_logprob_gap": failure_logprob_gap,
                    "failure_accepted_token_ids": failure_accepted_token_ids,
                    "tokens_checked": tokens_checked,
                    "exact_argmax_tokens": exact_argmax_tokens,
                    "exact_argmax_match_ratio": exact_argmax_tokens / tokens_checked,
                    "near_top_tokens": near_top_tokens,
                }
            )
        return reports

