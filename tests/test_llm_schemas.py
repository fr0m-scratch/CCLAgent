import unittest

from src.llm.schemas import validate_offline_plan, validate_online_decision_support, validate_postrun_rules
from src.types import ParameterSpace, ParameterSpec


class TestLLMSchemas(unittest.TestCase):
    def setUp(self):
        self.ps = ParameterSpace.from_list(
            [
                ParameterSpec(name="NCCL_ALGO", kind="enum", choices=["RING", "TREE"], default="RING"),
                ParameterSpec(name="NCCL_PROTO", kind="enum", choices=["LL", "SIMPLE"], default="SIMPLE"),
            ]
        )

    def test_validate_offline_plan(self):
        plan = {
            "warm_start_program": {
                "mode": "single",
                "candidates": [
                    {"id": "WS0", "patch": {"NCCL_ALGO": "RING"}}
                ],
            },
            "baseline_patch": {"NCCL_PROTO": "LL"},
            "pruning_guidance": [],
            "subspace_priors": [],
            "hypothesis_playbook": [{"patch_template": {"NCCL_PROTO": "LL"}}],
            "tool_triggers": [],
        }
        errors = validate_offline_plan(plan, self.ps)
        self.assertEqual(errors, [])

    def test_validate_online_decision_support(self):
        output = {
            "hypotheses": [{"patch": {"NCCL_ALGO": "TREE"}}],
            "numeric_guidance": {},
            "tool_request": {"name": "none"},
            "action_preference": "numeric",
            "convergence": {"decision": "continue", "confidence": 0.8},
        }
        errors = validate_online_decision_support(output, self.ps)
        self.assertEqual(errors, [])

    def test_validate_online_decision_support_missing_required_key(self):
        output = {
            "hypotheses": [{"patch": {"NCCL_ALGO": "TREE"}}],
            "tool_request": {"name": "none"},
            "action_preference": "numeric",
            "convergence": {"decision": "continue", "confidence": 0.8},
        }
        errors = validate_online_decision_support(output, self.ps)
        self.assertIn("missing:numeric_guidance", errors)

    def test_validate_postrun_rules(self):
        rules = [
            {"action": {"set": {"NCCL_PROTO": "LL"}}},
        ]
        errors = validate_postrun_rules(rules, self.ps)
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
