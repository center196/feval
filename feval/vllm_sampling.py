from __future__ import annotations

from typing import Any

import torch
from vllm import SamplingParams
from vllm.v1.sample.logits_processor import BatchUpdate, LogitsProcessor, process_dict_updates

from .canonical import HASH_PRIME, canonical_max_logprob_gap, canonical_row_coefficients
from .constants import CANONICAL_MAX_CANDIDATES, CANONICAL_MIN_RELATIVE_PROBABILITY


EXTRA_ARG = "feval_canonical"


class CanonicalNearMaxLogitsProcessor(LogitsProcessor):
    """Vectorized protocol token selector for miner autoregressive decoding."""

    def __init__(self, _vllm_config: Any, device: torch.device, _is_pin_memory: bool):
        # LogitsProcessor is an interface in vLLM 0.27.1; its __init__ raises
        # NotImplementedError and must not be called by direct implementations.
        self.device = device
        self._requests: dict[int, tuple[int, int]] = {}
        self._indices = torch.empty(0, dtype=torch.long, device=device)
        self._multipliers = torch.empty(0, dtype=torch.long, device=device)
        self._offsets = torch.empty(0, dtype=torch.long, device=device)
        self._gap = canonical_max_logprob_gap(CANONICAL_MIN_RELATIVE_PROBABILITY)

    @classmethod
    def validate_params(cls, sampling_params: SamplingParams) -> None:
        value = (sampling_params.extra_args or {}).get(EXTRA_ARG)
        if value is None:
            return
        if not isinstance(value, dict) or set(value) != {
            "evaluation_seed",
            "model_digest",
            "row_id",
        }:
            raise ValueError(f"{EXTRA_ARG} must contain evaluation_seed, model_digest, and row_id")
        if not all(isinstance(value[key], str) and value[key] for key in value):
            raise ValueError(f"all {EXTRA_ARG} values must be non-empty strings")

    def is_argmax_invariant(self) -> bool:
        return False

    @staticmethod
    def _new_state(
        params: SamplingParams, _prompt_ids: list[int] | None, _output_ids: list[int]
    ) -> tuple[int, int] | None:
        value = (params.extra_args or {}).get(EXTRA_ARG)
        if value is None:
            return None
        return canonical_row_coefficients(
            evaluation_seed=value["evaluation_seed"],
            model_digest=value["model_digest"],
            row_id=value["row_id"],
        )

    def update_state(self, batch_update: BatchUpdate | None) -> None:
        changed = process_dict_updates(self._requests, batch_update, self._new_state)
        if not changed:
            return
        ordered = sorted(self._requests.items())
        self._indices = torch.tensor(
            [index for index, _ in ordered], dtype=torch.long, device=self.device
        )
        self._multipliers = torch.tensor(
            [state[0] for _, state in ordered], dtype=torch.long, device=self.device
        ).unsqueeze(1)
        self._offsets = torch.tensor(
            [state[1] for _, state in ordered], dtype=torch.long, device=self.device
        ).unsqueeze(1)

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if self._indices.numel() == 0:
            return logits
        selected_logits = logits.index_select(0, self._indices)
        candidate_count = min(CANONICAL_MAX_CANDIDATES, selected_logits.shape[1])
        top_values, top_ids = torch.topk(selected_logits, k=candidate_count, dim=-1)
        eligible = (top_values[:, :1] - top_values) <= self._gap
        scores = torch.remainder(
            top_ids.to(torch.long) * self._multipliers + self._offsets,
            HASH_PRIME,
        )
        scores.masked_fill_(~eligible, HASH_PRIME)
        selected_columns = scores.argmin(dim=-1, keepdim=True)
        selected_ids = top_ids.gather(1, selected_columns).squeeze(1)
        logits[self._indices] = -float("inf")
        logits[self._indices, selected_ids] = 0.0
        return logits
