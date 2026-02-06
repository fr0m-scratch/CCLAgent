import unittest

from src.agent.decision_bundle import build_decision_bundle, validate_decision_bundle


class TestDecisionBundleRefs(unittest.TestCase):
    def test_builder_fills_refs_and_counterfactuals(self):
        payload = build_decision_bundle(
            step=5,
            chosen_action={"kind": "numeric", "rationale": "numeric search"},
            why_selected=["selected best predicted time"],
            why_rejected=["higher predicted time"],
            context_ref="steps/step_5_context_pack.json",
            constraints_snapshot={"risk_max": 0.7, "sla_max_iteration_time": None, "budget_remaining_steps": 4},
            rollback_plan={"last_known_good_ref": "metric:4:primary"},
            refs_fallback=["metric:4:primary"],
            candidates_considered=[
                {
                    "candidate_ref": "candidate:5:0",
                    "score_breakdown": {"pred_time_ms": 100.0, "uncertainty": 4.0, "risk_score": 0.1},
                    "status": "selected",
                    "refs": [],
                },
                {
                    "candidate_ref": "candidate:5:1",
                    "score_breakdown": {"pred_time_ms": 110.0, "uncertainty": 6.0, "risk_score": 0.15},
                    "status": "rejected",
                    "reject_reason": "dominated",
                    "refs": [],
                },
                {
                    "candidate_ref": "candidate:5:2",
                    "score_breakdown": {"pred_time_ms": 120.0, "uncertainty": 8.0, "risk_score": 0.2},
                    "status": "rejected",
                    "reject_reason": "dominated",
                    "refs": [],
                },
            ],
        )
        errors = validate_decision_bundle(payload)
        self.assertEqual(errors, [])
        self.assertTrue(payload["why_selected"][0]["refs"])
        self.assertTrue(payload["why_rejected"][0]["refs"])
        self.assertGreaterEqual(len(payload["counterfactuals"]), 2)

    def test_validator_rejects_missing_claim_refs(self):
        payload = {
            "schema_version": "2.0",
            "step": 1,
            "context_ref": "steps/step_1_context_pack.json",
            "chosen_action": {"kind": "numeric", "rationale": "x", "call_chain": ["metric:0:primary"]},
            "candidates_considered": [],
            "why_selected": [{"claim": "x", "refs": []}],
            "why_rejected": [],
            "counterfactuals": [],
            "constraints_snapshot": {},
            "rollback_plan": {},
            "quality_flags": [],
        }
        errors = validate_decision_bundle(payload)
        self.assertIn("missing_refs:why_selected[0]", errors)


if __name__ == "__main__":
    unittest.main()
