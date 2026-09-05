from __future__ import annotations

import json
import unittest
from pathlib import Path

from feval.core.config import NetworkConfig, load_network_config
from feval.core.constants import (
    EVALUATION_ROWS,
    EVALUATION_SOURCES,
    EVALUATION_SOURCES_DIGEST,
    REASONING_BUDGET_LEVELS,
)
from feval.datasets.dataset import reasoning_budget_for_row, source_quotas
from feval.datasets.sources import partial_block_count
from feval.models.inference import fixed_prompt, reasoning_budget_bounds, split_reasoning_tokens
from feval.nodes.runtime import _category_scores, _model_precedes_window, score_rollouts
from feval.protocol.schedule import audit_detection_probability, evaluation_seed


class WindowProtocolTests(unittest.TestCase):
    def test_future_seed_assigns_random_fixed_reasoning_levels(self) -> None:
        first = "1" * 64
        second = "2" * 64
        first_levels = [reasoning_budget_for_row(first, f"row-{index}") for index in range(200)]
        second_levels = [reasoning_budget_for_row(second, f"row-{index}") for index in range(200)]
        self.assertEqual(set(first_levels), set(REASONING_BUDGET_LEVELS))
        self.assertNotEqual(first_levels, second_levels)
        self.assertEqual(first_levels, [reasoning_budget_for_row(first, f"row-{index}") for index in range(200)])

    def test_thinking_prompt_and_token_range_are_protocol_owned(self) -> None:
        prompt = fixed_prompt("Solve this.", 1_024)
        self.assertIn("<|im_start|>system", prompt)
        self.assertIn("<think>", prompt)
        self.assertIn("</think>", prompt)
        self.assertIn("1024 tokenizer tokens", prompt)
        self.assertEqual(reasoning_budget_bounds(1_024), (922, 1_126))

    def test_only_tokens_inside_first_think_block_count(self) -> None:
        class MarkerTokenizer:
            def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
                del add_special_tokens
                return {"<think>": [1], "</think>": [2]}[text]

        tokenizer = MarkerTokenizer()
        reasoning, answer = split_reasoning_tokens(
            tokenizer,
            [1] + [9] * 922 + [2, 7, 8, 9],
            1_024,
        )
        self.assertEqual(len(reasoning), 922)
        self.assertEqual(answer, [7, 8, 9])
        with self.assertRaises(ValueError):
            split_reasoning_tokens(tokenizer, [1] + [9] * 921 + [2, 7], 1_024)
        with self.assertRaises(ValueError):
            split_reasoning_tokens(tokenizer, [9] * 1_024 + [2, 7], 1_024)

    def test_scoring_uses_only_the_post_think_answer(self) -> None:
        class ScoreTokenizer:
            eos_token_id = 99

            def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
                del add_special_tokens
                if text == "<think>":
                    return [1]
                if text == "</think>":
                    return [2]
                if text == "<|im_end|>":
                    return [98]
                if text == "<|endoftext|>":
                    return [99]
                return [10] * 20

            def decode(self, tokens: list[int], **_kwargs: object) -> str:
                return "".join("42" if token == 7 else "41" if token == 8 else "" for token in tokens)

        evaluation = [{
            "row_id": "row-1",
            "prompt": "Solve.",
            "expected": ["42"],
            "verifier": "math_exact",
            "category": "math",
            "reasoning_budget_tokens": 1_024,
        }]
        valid_tokens = [1] + [9] * 922 + [2, 7, 99]
        score, scored = score_rollouts(
            evaluation,
            [{"row_id": "row-1", "tokens": valid_tokens}],
            ScoreTokenizer(),
            NetworkConfig(),
        )
        self.assertEqual(score, 1.0)
        self.assertTrue(scored[0]["reasoning_valid"])
        wrong_after_think = [1] + [7] + [9] * 921 + [2, 8, 99]
        score, _ = score_rollouts(
            evaluation,
            [{"row_id": "row-1", "tokens": wrong_after_think}],
            ScoreTokenizer(),
            NetworkConfig(),
        )
        self.assertEqual(score, 0.0)

    def test_future_boundary_hash_changes_the_window_seed(self) -> None:
        first = evaluation_seed(47, 9, "1" * 64)
        self.assertEqual(first, evaluation_seed(47, 9, "1" * 64))
        self.assertEqual(first, evaluation_seed(47, 9, "0x" + "1" * 64))
        self.assertNotEqual(first, evaluation_seed(47, 9, "2" * 64))
        with self.assertRaises(ValueError):
            evaluation_seed(47, 9, "")

    def test_model_must_precede_the_seed_reveal_block(self) -> None:
        self.assertTrue(
            _model_precedes_window(commit_block=35_999, window=10, window_blocks=3_600)
        )
        self.assertFalse(
            _model_precedes_window(commit_block=36_000, window=10, window_blocks=3_600)
        )

    def test_source_mix_is_proportional_to_pinned_source_sizes(self) -> None:
        specs = list(EVALUATION_SOURCES)
        quotas = source_quotas(specs, EVALUATION_ROWS)
        self.assertEqual(
            quotas,
            {
                "open_math_reasoning": 56_382,
                "crossthink_math": 1_759,
                "numina_math_1_5": 15_785,
                "knowledge_mcqa": 10_868,
                "open_science": 14_138,
                "code_understanding": 1_068,
            },
        )
        self.assertEqual(sum(quotas.values()), EVALUATION_ROWS)

    def test_largest_remainder_allocation_is_exact_and_deterministic(self) -> None:
        specs = [
            {"name": "small", "source_rows": 1},
            {"name": "medium", "source_rows": 2},
            {"name": "large", "source_rows": 7},
        ]
        self.assertEqual(
            source_quotas(specs, 100),
            {"small": 10, "medium": 20, "large": 70},
        )

    def test_source_scan_never_reaches_every_block_when_partitioned(self) -> None:
        self.assertEqual(partial_block_count(1), 1)
        self.assertEqual(partial_block_count(2), 1)
        self.assertEqual(partial_block_count(10), 9)
        self.assertLess(partial_block_count(1_000), 1_000)

    def test_category_scores_are_diagnostics_only(self) -> None:
        scored = (
            [{"category": "math", "reward": int(index < 8)} for index in range(10)]
            + [{"category": "mcqa", "reward": 1} for _ in range(4)]
            + [{"category": "code", "reward": int(index < 3)} for index in range(4)]
        )
        scores = _category_scores(scored)
        self.assertAlmostEqual(scores["math"], 0.8)
        self.assertAlmostEqual(scores["mcqa"], 1.0)
        self.assertAlmostEqual(scores["code"], 0.75)
        self.assertAlmostEqual(
            sum(int(row["reward"]) for row in scored) / len(scored),
            15 / 18,
        )

    def test_30_round_gate_exceeds_9999_percent_detection_at_one_percent(self) -> None:
        probability = audit_detection_probability(
            population=EVALUATION_ROWS,
            forged_rows=EVALUATION_ROWS // 100,
            samples=30 * 32,
        )
        self.assertGreater(probability, 0.9999)

    def test_checked_in_network_config_matches_code_pins(self) -> None:
        config = load_network_config(Path("network.json"))
        self.assertEqual(config, NetworkConfig())
        self.assertEqual(config.sources_digest, EVALUATION_SOURCES_DIGEST)

    def test_v40_config_migrates_to_current_protocol(self) -> None:
        value = json.loads(Path("network.json").read_text(encoding="utf-8"))
        value["protocol"] = "feval-network-v40"
        value.pop("audit_required_rounds")
        migrated = NetworkConfig.from_dict(value)
        self.assertEqual(migrated, NetworkConfig())


if __name__ == "__main__":
    unittest.main()
