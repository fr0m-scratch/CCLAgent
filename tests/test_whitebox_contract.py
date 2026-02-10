import json
import tempfile
import unittest
from pathlib import Path

from src.trace.validator import validate_whitebox_contract


class TestWhiteboxContract(unittest.TestCase):
    def _write_trace(self, run_dir: Path, refs):
        trace_dir = run_dir / "trace"
        trace_dir.mkdir(parents=True, exist_ok=True)
        event = {
            "schema_version": "2.0",
            "event_id": "evt-1",
            "ts": 0.0,
            "run_id": "run-1",
            "phase": "online",
            "step": 1,
            "actor": "agent",
            "type": "decision.select_action",
            "payload": {},
            "refs": refs,
            "causal_refs": [],
            "quality_flags": [],
            "status": "ok",
        }
        (trace_dir / "events.jsonl").write_text(json.dumps(event) + "\n", encoding="utf-8")

    def test_unresolved_evidence_ref_violates_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            self._write_trace(run_dir, ["evidence:abc"])
            violations = validate_whitebox_contract(run_dir)
            self.assertTrue(any(item.startswith("unresolved_evidence_ref") for item in violations))

    def test_non_evidence_refs_are_allowed_during_migration(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            self._write_trace(run_dir, ["metric:0:primary"])
            violations = validate_whitebox_contract(run_dir)
            self.assertEqual(violations, [])

    def test_resolved_evidence_ref_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            self._write_trace(run_dir, ["evidence:abc"])
            whitebox_dir = run_dir / "whitebox"
            whitebox_dir.mkdir(parents=True, exist_ok=True)
            evidence = {
                "id": "abc",
                "kind": "metric",
                "source": "test",
                "payload": {"x": 1},
                "created_at": "2026-01-01T00:00:00+00:00",
            }
            (whitebox_dir / "evidence.jsonl").write_text(json.dumps(evidence) + "\n", encoding="utf-8")
            violations = validate_whitebox_contract(run_dir)
            self.assertEqual(violations, [])


if __name__ == "__main__":
    unittest.main()
