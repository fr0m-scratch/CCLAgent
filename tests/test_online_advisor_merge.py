import unittest

from src.agent.analyzer import TuningAnalyzer
from src.types import (
    AgentConfig,
    ExecutionConfig,
    LLMSettings,
    MemoryConfig,
    MetricsConfig,
    MicrobenchSettings,
    NumericSearchSettings,
    RagConfig,
    SafetyConfig,
    SurrogateConfig,
    TuningBudget,
    WarmStartSettings,
    ParameterSpace,
    ParameterSpec,
)


class DummyHypothesisGen:
    def propose_portfolio(self, plan, context, base_config, last_metrics, max_hypotheses=3):
        return []


class DummyNumeric:
    def propose(self, plan, state, workload, base_config, step, context=None, guidance=None):
        return None, None


class DummyCompiler:
    def __init__(self, parameter_space):
        self.parameter_space = parameter_space


class DummyAdvice:
    def __init__(self, output):
        self.output = output
        self.call_id = "call_1"


class TestOnlineAdvisorMerge(unittest.TestCase):
    def setUp(self):
        ps = ParameterSpace.from_list(
            [
                ParameterSpec(name="NCCL_ALGO", kind="enum", choices=["RING", "TREE"], default="RING"),
            ]
        )
        cfg = AgentConfig(
            parameter_space=ps,
            budget=TuningBudget(max_steps=3, hypothesis_every=1),
            memory=MemoryConfig(),
            rag=RagConfig(),
            llm=LLMSettings(),
            warm_start=WarmStartSettings(),
            microbench=MicrobenchSettings(),
            metrics=MetricsConfig(),
            numeric_search=NumericSearchSettings(),
            safety=SafetyConfig(),
            execution=ExecutionConfig(),
            surrogate=SurrogateConfig(),
        )
        self.analyzer = TuningAnalyzer(cfg, DummyHypothesisGen(), DummyNumeric(), DummyCompiler(ps))

    def test_hypotheses_from_advice(self):
        output = {
            "hypotheses": [
                {"id": "H1", "summary": "test", "patch": {"NCCL_ALGO": "TREE"}},
            ]
        }
        hyps = self.analyzer._hypotheses_from_advice(output)
        self.assertEqual(len(hyps), 1)
        self.assertEqual(hyps[0].patch.get("NCCL_ALGO"), "TREE")

    def test_hypotheses_from_advice_rewrites_generic_summary(self):
        output = {
            "hypotheses": [
                {"id": "H1", "summary": "apply memory rule", "patch": {"NCCL_ALGO": "TREE"}},
            ]
        }
        hyps = self.analyzer._hypotheses_from_advice(output)
        self.assertEqual(len(hyps), 1)
        self.assertNotIn("apply memory rule", hyps[0].summary.lower())
        self.assertIn("NCCL_ALGO", hyps[0].summary)

    def test_tool_request_gating(self):
        advice = DummyAdvice({"tool_request": {"name": "nccltest.short", "reason": "need signal"}})
        info = self.analyzer._evaluate_tool_request(advice)
        self.assertTrue(info.get("accepted"))

    def test_convergence_from_advice(self):
        advice = DummyAdvice(
            {
                "convergence": {
                    "decision": "stop",
                    "reason": "plateau",
                    "confidence": 0.9,
                    "claims": [{"claim": "flat", "refs": ["metric:1:primary"]}],
                }
            }
        )
        info = self.analyzer._convergence_from_advice(advice)
        self.assertEqual(info.get("decision"), "stop")
        self.assertGreaterEqual(info.get("confidence", 0.0), 0.9)

    def test_action_preference_from_recommended_action(self):
        advice = DummyAdvice({"recommended_action": {"kind": "measure"}})
        pref = self.analyzer._action_preference_from_advice(advice)
        self.assertEqual(pref, "numeric")

    def test_lane_selection_hides_latency_when_advice_pending(self):
        use_hypothesis, source = self.analyzer._choose_action_lane(
            action_preference=None,
            advice=None,
            default_use_hypothesis=True,
            llm_requested=True,
        )
        self.assertFalse(use_hypothesis)
        self.assertEqual(source, "llm_pending_hide_latency")


if __name__ == "__main__":
    unittest.main()
