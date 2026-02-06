"""Tests for WP10b: Deterministic replay engine."""

import json
import os
import tempfile
import unittest

from src.trace.replay import (
    ReplayMismatch,
    ReplayReport,
    load_run_artifacts,
    replay_decisions,
)


class TestReplayReport(unittest.TestCase):
    def test_to_dict(self):
        r = ReplayReport(run_id="test-123", total_steps=5, steps_replayed=3, action_match_rate=1.0)
        d = r.to_dict()
        self.assertEqual(d["run_id"], "test-123")
        self.assertEqual(d["total_steps"], 5)


class TestLoadRunArtifacts(unittest.TestCase):
    def test_empty_dir(self):
        with tempfile.TemporaryDirectory() as td:
            artifacts = load_run_artifacts(td)
            self.assertEqual(artifacts["steps"], {})
            self.assertEqual(artifacts["decisions"], {})

    def test_load_steps_and_decisions(self):
        with tempfile.TemporaryDirectory() as td:
            steps_dir = os.path.join(td, "steps")
            os.makedirs(steps_dir)
            # Step artifact
            with open(os.path.join(steps_dir, "step_0.json"), "w") as f:
                json.dump({"action": {"kind": "initial", "config": {"A": "1"}}}, f)
            # Decision artifact
            with open(os.path.join(steps_dir, "step_1_decision.json"), "w") as f:
                json.dump({"chosen_action": {"kind": "hypothesis", "config": {"A": "2"}}}, f)
            artifacts = load_run_artifacts(td)
            self.assertIn("0", artifacts["steps"])
            self.assertIn("1", artifacts["decisions"])

    def test_load_convergence(self):
        with tempfile.TemporaryDirectory() as td:
            postrun_dir = os.path.join(td, "postrun")
            os.makedirs(postrun_dir)
            with open(os.path.join(postrun_dir, "convergence.json"), "w") as f:
                json.dump({"converged": True}, f)
            artifacts = load_run_artifacts(td)
            self.assertTrue(artifacts["convergence"]["converged"])


class TestReplayDecisions(unittest.TestCase):
    def test_no_steps(self):
        artifacts = {"steps": {}, "decisions": {}, "run_context": {"run_id": "r1"}, "events": []}
        report = replay_decisions(artifacts)
        self.assertEqual(report.total_steps, 0)
        self.assertEqual(report.run_id, "r1")

    def test_matching_decisions(self):
        artifacts = {
            "steps": {
                "1": {"action": {"kind": "hypothesis", "config": {"A": "2"}}},
            },
            "decisions": {
                "1": {"chosen_action": {"kind": "hypothesis", "config": {"A": "2"}}},
            },
            "run_context": {"run_id": "r1"},
            "events": [],
        }
        report = replay_decisions(artifacts)
        self.assertEqual(report.steps_replayed, 1)
        self.assertAlmostEqual(report.action_match_rate, 1.0)
        self.assertAlmostEqual(report.config_match_rate, 1.0)
        self.assertEqual(len(report.mismatches), 0)

    def test_action_mismatch(self):
        artifacts = {
            "steps": {
                "1": {"action": {"kind": "hypothesis", "config": {"A": "2"}}},
            },
            "decisions": {
                "1": {"chosen_action": {"kind": "numeric", "config": {"A": "2"}}},
            },
            "run_context": {"run_id": "r1"},
            "events": [],
        }
        report = replay_decisions(artifacts)
        self.assertAlmostEqual(report.action_match_rate, 0.0)
        self.assertEqual(len(report.mismatches), 1)
        self.assertEqual(report.mismatches[0].field, "action.kind")

    def test_config_mismatch(self):
        artifacts = {
            "steps": {
                "1": {"action": {"kind": "hypothesis", "config": {"A": "2"}}},
            },
            "decisions": {
                "1": {"chosen_action": {"kind": "hypothesis", "config": {"A": "3"}}},
            },
            "run_context": {"run_id": "r1"},
            "events": [],
        }
        report = replay_decisions(artifacts)
        self.assertAlmostEqual(report.action_match_rate, 1.0)
        self.assertAlmostEqual(report.config_match_rate, 0.0)
        self.assertEqual(len(report.mismatches), 1)
        self.assertEqual(report.mismatches[0].field, "action.config")

    def test_missing_decision_noted(self):
        artifacts = {
            "steps": {
                "0": {"action": {"kind": "initial", "config": {}}},
            },
            "decisions": {},
            "run_context": {"run_id": "r1"},
            "events": [],
        }
        report = replay_decisions(artifacts)
        self.assertEqual(report.steps_replayed, 0)
        self.assertTrue(len(report.notes) >= 1)
        self.assertTrue(any("no decision record" in n for n in report.notes))


if __name__ == "__main__":
    unittest.main()
