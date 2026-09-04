from __future__ import annotations

import json
import string
import unittest

from feval.datasets.rewards import reward_answer
from feval.datasets.tasks import (
    CODE_OUTPUT_SUFFIX,
    normalize_code_understanding,
    normalize_crossthink_math,
    normalize_knowledge_mcqa,
    normalize_open_math_reasoning,
    normalize_open_science,
)


LETTERS = "ABCDEFGHIJ"


def option_prompt(separator: str = ":") -> str:
    return "Choose one.\n" + "\n".join(
        f"{letter}{separator} option {letter}" for letter in LETTERS
    )


def declared_options(letters: str = LETTERS) -> list[dict[str, str | None]]:
    return [
        {
            candidate: f"option {candidate}" if candidate == letter else None
            for candidate in string.ascii_uppercase
        }
        for letter in letters
    ]


class DatasetNormalizerContractTests(unittest.TestCase):
    def test_open_math_uses_only_extracted_simple_numeric_answer(self) -> None:
        accepted = normalize_open_math_reasoning(
            {
                "problem_type": "has_answer_extracted",
                "problem": "What is one half?",
                "expected_answer": r"\(\frac{1}{2}\)",
            },
            1,
        )
        self.assertIsNotNone(accepted)
        self.assertEqual(accepted["expected"], ["1/2"])

        self.assertIsNone(
            normalize_open_math_reasoning(
                {
                    "problem_type": "no_answer_extracted",
                    "problem": "What is one half?",
                    "expected_answer": "1/2",
                },
                2,
            )
        )
        self.assertIsNone(
            normalize_open_math_reasoning(
                {
                    "problem_type": "has_answer_extracted",
                    "problem": "Find x.",
                    "expected_answer": r"\sqrt{2}",
                },
                3,
            )
        )

    def test_crossthink_rejects_ambiguous_multi_answer_prompts(self) -> None:
        base = {"reward_model": {"style": "rule", "ground_truth": "2"}}
        accepted = normalize_crossthink_math(
            {**base, "meta_data": {"index": 1, "question": "What is 1 + 1?"}},
            1,
        )
        self.assertIsNotNone(accepted)
        self.assertEqual(accepted["expected"], ["2"])

        self.assertIsNone(
            normalize_crossthink_math(
                {
                    **base,
                    "meta_data": {
                        "index": 2,
                        "question": "What is 1 + 1? What is 2 + 2?",
                    },
                },
                2,
            )
        )
        self.assertIsNone(
            normalize_crossthink_math(
                {
                    **base,
                    "meta_data": {
                        "index": 3,
                        "question": "1. Compute x.\n2. What is y?",
                    },
                },
                3,
            )
        )

    def test_knowledge_mcqa_accepts_source_declared_a_paren_options(self) -> None:
        row = {
            "responses_create_params": {
                "input": [{"role": "user", "content": option_prompt(")")}]
            },
            "expected_answer": "J",
            "uuid": "knowledge-1",
            # This source-provided pattern is data only and must never be compiled.
            "template_metadata": {"output_regex": "(?R)(a+)+$"},
            "options": declared_options(),
        }
        normalized = normalize_knowledge_mcqa(row, 1)
        self.assertIsNotNone(normalized)
        self.assertEqual(normalized["expected"], ["J"])

    def test_knowledge_mcqa_requires_prompt_and_declared_options_to_agree(self) -> None:
        row = {
            "responses_create_params": {
                "input": [{"role": "user", "content": option_prompt()}]
            },
            "expected_answer": "I",
            "uuid": "knowledge-2",
            "template_metadata": {},
            "options": declared_options("ABCDEFGHI"),
        }
        self.assertIsNone(normalize_knowledge_mcqa(row, 1))

    def test_open_science_accepts_static_a_paren_layout(self) -> None:
        normalized = normalize_open_science(
            {"input": option_prompt(")"), "expected_answer": r"\boxed{C}"},
            1,
        )
        self.assertIsNotNone(normalized)
        self.assertEqual(normalized["expected"], ["C"])

    def test_code_ground_truth_is_read_without_evaluating_expressions(self) -> None:
        accepted = normalize_code_understanding(
            {
                "problem_id": "vcu-safe",
                "prompt": "Predict the output.",
                "verification_info": r"{'ground_truth': 'a\n b'}",
            },
            1,
        )
        self.assertIsNotNone(accepted)
        self.assertEqual(accepted["expected"], ["a\n b"])
        self.assertTrue(accepted["prompt"].endswith(CODE_OUTPUT_SUFFIX))

        self.assertIsNone(
            normalize_code_understanding(
                {
                    "problem_id": "vcu-expression",
                    "prompt": "Predict the output.",
                    "verification_info": "{'ground_truth': (1 / 0)}",
                },
                2,
            )
        )
        self.assertIsNone(
            normalize_code_understanding(
                {
                    "problem_id": "vcu-duplicate",
                    "prompt": "Predict the output.",
                    "verification_info": "{'ground_truth': 'a', 'ground_truth': 'b'}",
                },
                3,
            )
        )


class RewardContractTests(unittest.TestCase):
    def test_strict_numeric_is_exact_rational_equality(self) -> None:
        self.assertEqual(reward_answer("0.5", ["1/2"], verifier="strict_numeric"), 1)
        self.assertEqual(reward_answer("0.333", ["1/3"], verifier="strict_numeric"), 0)
        self.assertEqual(reward_answer("about 2", ["2"], verifier="strict_numeric"), 0)

    def test_mcqa_uses_static_source_compatible_wrappers(self) -> None:
        self.assertEqual(reward_answer("C", ["C"], verifier="mcqa_letter"), 1)
        self.assertEqual(reward_answer(r"\boxed{C}", ["C"], verifier="mcqa_letter"), 1)
        self.assertEqual(reward_answer("Answer: C", ["C"], verifier="mcqa_letter"), 1)
        self.assertEqual(reward_answer("The choice is C", ["C"], verifier="mcqa_letter"), 0)

    def test_code_output_requires_one_complete_json_object_and_exact_value(self) -> None:
        expected = [" a\n b "]
        response = json.dumps({"output": expected[0]})
        self.assertEqual(reward_answer(response, expected, verifier="json_output_exact"), 1)
        self.assertEqual(reward_answer(expected[0], expected, verifier="json_output_exact"), 0)
        self.assertEqual(
            reward_answer(f"reasoning\n{response}", expected, verifier="json_output_exact"),
            0,
        )
        self.assertEqual(
            reward_answer(
                json.dumps({"output": expected[0], "extra": True}),
                expected,
                verifier="json_output_exact",
            ),
            0,
        )
        self.assertEqual(
            reward_answer(json.dumps({"output": "a\n b"}), expected, verifier="json_output_exact"),
            0,
        )


if __name__ == "__main__":
    unittest.main()
