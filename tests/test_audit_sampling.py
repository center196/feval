from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from feval.core.config import NetworkConfig
from feval.core.constants import (
    PROTOCOL_MINER_ROLLOUT_STATE,
    PROTOCOL_VALIDATOR_STATE,
)
from feval.models.artifacts import ModelCommitment
from feval.nodes.runtime import (
    ValidatorRunner,
    _correct_scored_rows,
    _load_miner_rollout_state,
    _normalize_state,
)
from feval.protocol.schedule import choose_audit_ids
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
                "protocol": "feval-validator-state-v38",
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
                        "protocol": "feval-miner-rollout-state-v21",
                        "last_success": {"dataset_window": 7},
                    }
                ),
                encoding="utf-8",
            )

            state = _load_miner_rollout_state(directory)

        self.assertEqual(state["protocol"], PROTOCOL_MINER_ROLLOUT_STATE)
        self.assertIsNone(state["last_success"])

    def test_audits_are_deterministic_uniform_and_distinct(self) -> None:
        rows = [
            {"row_id": f"row-{index}", "category": "any"}
            for index in range(300)
        ]
        first = choose_audit_ids(
            rows,
            seed="round-1",
            count=32,
        )
        self.assertEqual(
            first,
            choose_audit_ids(
                rows,
                seed="round-1",
                count=32,
            ),
        )
        second = choose_audit_ids(
            rows,
            seed="round-2",
            count=32,
            already_audited=first,
        )
        self.assertFalse(set(first) & set(second))
        self.assertEqual(len(first), 32)
        self.assertEqual(len(second), 32)

    def test_monitoring_waits_for_every_participant_to_reach_round_30(self) -> None:
        config = NetworkConfig()
        runner = ValidatorRunner.__new__(ValidatorRunner)
        runner.config = config
        runner.hf_token = False
        runner.state = {
            "window": 7,
            "round": 0,
            "pending": {},
            "results": {},
            "audited": {},
            "invalid_strikes": {},
            "blacklist": {},
        }

        def commitment(name: str) -> ModelCommitment:
            return ModelCommitment(
                model_repo=f"owner/{name}-model",
                model_revision="a" * 40,
                model_digest=("1" if name == "a" else "2") * 64,
                rollout_repo=f"owner/{name}-rollouts",
            )

        a = commitment("a")
        b = commitment("b")
        eligible = [
            {"hotkey": "a", "uid": 1, "commit_block": 1, "commitment": a},
            {"hotkey": "b", "uid": 2, "commit_block": 2, "commitment": b},
        ]
        revisions = {a.rollout_repo: "a" * 40, b.rollout_repo: "b" * 40}
        for hotkey, model, round_number in (("a", a, 30), ("b", b, 29)):
            revision = revisions[model.rollout_repo]
            runner.state["results"][hotkey] = {
                "model_digest": model.model_digest,
                "rollout_revision": revision,
                "audit_status": "monitoring" if round_number == 30 else "auditing",
                "audit_round": round_number,
            }
            runner.state["audited"][
                f"{hotkey}:{model.model_digest}:{revision}:7"
            ] = {"rounds_passed": round_number}

        with patch(
            "feval.nodes.runtime.resolve_rollout_revision",
            side_effect=lambda repo, token: revisions[repo],
        ):
            runner._schedule_latest(current_block=1_000, eligible=eligible)
        self.assertNotIn("a", runner.state["pending"])
        self.assertEqual(runner.state["pending"]["b"]["round"], 30)

        runner.state["pending"] = {}
        runner.state["results"]["b"].update(
            {"audit_status": "monitoring", "audit_round": 30}
        )
        b_key = f"b:{b.model_digest}:{revisions[b.rollout_repo]}:7"
        runner.state["audited"][b_key]["rounds_passed"] = 30
        with patch(
            "feval.nodes.runtime.resolve_rollout_revision",
            side_effect=lambda repo, token: revisions[repo],
        ):
            runner._schedule_latest(current_block=1_001, eligible=eligible)
        self.assertEqual(runner.state["pending"]["a"]["round"], 31)
        self.assertEqual(runner.state["pending"]["b"]["round"], 31)


if __name__ == "__main__":
    unittest.main()
