import unittest

from src.agent.hypothesis import HypothesisGenerator
from src.memory.schema import Rule
from src.types import ContextSignature, InitialConfigPlan, NCCLConfig, ParameterSpace, ParameterSpec


class DummyMemory:
    def retrieve_rules(self, context, top_k=5):
        _ = (context, top_k)
        return [
            Rule(
                id="rule_channels",
                context={},
                config_patch={"NCCL_MAX_NCHANNELS": 16},
                improvement=0.08,
            )
        ]


class HypothesisSummarySpecificityTest(unittest.TestCase):
    def test_memory_rule_summary_is_specific(self):
        parameter_space = ParameterSpace.from_list(
            [
                ParameterSpec(name="NCCL_MAX_NCHANNELS", kind="int", min_value=1, max_value=32, default=8),
            ]
        )
        generator = HypothesisGenerator(memory=DummyMemory(), parameter_space=parameter_space)
        context = ContextSignature(
            workload="autoccl-llama3.1-8b",
            workload_kind="training",
            topology="a40-pcie-8gpu",
            scale="32-gpu",
            nodes=4,
            gpus_per_node=8,
        )
        plan = InitialConfigPlan(
            baseline_config=NCCLConfig(params={"NCCL_MAX_NCHANNELS": 8}),
            recommended_search_params=["NCCL_MAX_NCHANNELS"],
        )
        hypotheses = generator.propose_portfolio(
            plan=plan,
            context=context,
            base_config=plan.baseline_config,
            last_metrics=None,
            max_hypotheses=1,
        )
        self.assertEqual(len(hypotheses), 1)
        summary = hypotheses[0].summary.lower()
        self.assertIn("nccl_max_nchannels", summary)
        self.assertNotIn("apply memory rule", summary)


if __name__ == "__main__":
    unittest.main()
