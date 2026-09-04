from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from feval.core.constants import PROTOCOL_MINER_ROLLOUT_STATE, PROTOCOL_VALIDATOR_STATE
from feval.nodes.runtime import (
    _correct_scored_rows,
    _load_miner_rollout_state,
    _normalize_state,
)
from feval.nodes.validator import choose_audit_rows


class CorrectAuditSamplingTests(unittest.TestCase):
    def test_production_pool_contains_only_locally_correct_rows(self) -> None:
        scored = [
            {"row_id": "correct-a", "reward": 1},
            {"row_id": "incorrect", "reward": 0},
            {"row_id": "correct-b", "reward": 1},
        ]

        self.assertEqual(
            [row["row_id"] for row in _correct_scored_rows(scored)],
            ["correct-a", "correct-b"],
        )

    def test_legacy_audit_does_not_fill_from_incorrect_rows(self) -> None:
        submission = {
            "miner_hotkey": "miner",
            "adapter_hash": "adapter",
            "answer_root": "answers",
            "rollout_root": "rollouts",
            "rows": [
                {"row_id": "correct"},
                {"row_id": "incorrect-a"},
                {"row_id": "incorrect-b"},
            ],
        }

        selected = choose_audit_rows(
            submission,
            "seed",
            32,
            correct_row_ids={"correct"},
        )

        self.assertEqual(selected, ["correct"])

    def test_legacy_audit_returns_empty_for_zero_correct_rows(self) -> None:
        submission = {
            "miner_hotkey": "miner",
            "adapter_hash": "adapter",
            "answer_root": "answers",
            "rollout_root": "rollouts",
            "rows": [{"row_id": "incorrect"}],
        }

        self.assertEqual(
            choose_audit_rows(
                submission,
                "seed",
                32,
                correct_row_ids=set(),
            ),
            [],
        )

    def test_dataset_upgrade_clears_prior_validator_results(self) -> None:
        state = _normalize_state(
            {
                "protocol": "feval-validator-state-v33",
                "window": 7,
                "pending": {"miner": {}},
                "results": {"miner": {"score": 1.0}},
                "audited": {"miner": {}},
            }
        )

        self.assertEqual(state["protocol"], PROTOCOL_VALIDATOR_STATE)
        self.assertIsNone(state["window"])
        self.assertEqual(state["pending"], {})
        self.assertEqual(state["results"], {})
        self.assertEqual(state["audited"], {})

    def test_dataset_upgrade_clears_prior_miner_rollout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / "miner-rollouts.state.json"
            state_path.write_text(
                json.dumps(
                    {
                        "protocol": "feval-miner-rollout-state-v16",
                        "last_success": {"dataset_window": 7},
                    }
                ),
                encoding="utf-8",
            )

            state = _load_miner_rollout_state(directory)

        self.assertEqual(state["protocol"], PROTOCOL_MINER_ROLLOUT_STATE)
        self.assertIsNone(state["last_success"])


if __name__ == "__main__":
    unittest.main()
