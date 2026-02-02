import unittest

from src.agent.analyzer import TuningAnalyzer
from src.types import AgentConfig, ExecutionConfig, MemoryConfig, MetricsConfig, MicrobenchSettings, NumericSearchSettings, RagConfig, SafetyConfig, SurrogateConfig, TuningBudget
from src.types import ParameterSpace, ParameterSpec, ContextSignature, Metrics, NCCLConfig


class DummyHypothesis:
    def propose(self, plan, context, base_config, last_metrics):
        from src.types import Hypothesis
        return Hypothesis(id="h", summary="test", patch={"NCCL_ALGO": "RING"})


class DummyNumeric:
    def propose(self, plan, state, workload, base_config, step):
        from src.types import SearchCandidate, SearchResult
        cfg = NCCLConfig(params=base_config.params)
        candidate = SearchCandidate(config=cfg, predicted_time_ms=1.0, rationale="dummy")
        return cfg, SearchResult(best=candidate, candidates=[candidate])


class DummyCompiler:
    def compile_hypothesis(self, base, patch):
        from src.types import CompiledConfig, NCCLConfig
        merged = dict(base.params)
        merged.update(patch)
        return CompiledConfig(config=NCCLConfig(params=merged), env={}, warnings=[], risk_score=0.0)


class AnalyzerSelectionTest(unittest.TestCase):
    def test_selects_hypothesis(self):
        ps = ParameterSpace.from_list([ParameterSpec(name="NCCL_ALGO", kind="enum", choices=["RING"], default="RING")])
        cfg = AgentConfig(
            parameter_space=ps,
            budget=TuningBudget(max_steps=3, hypothesis_every=1),
            memory=MemoryConfig(),
            rag=RagConfig(),
            microbench=MicrobenchSettings(),
            metrics=MetricsConfig(),
            numeric_search=NumericSearchSettings(),
            safety=SafetyConfig(),
            execution=ExecutionConfig(),
            surrogate=SurrogateConfig(),
        )
        analyzer = TuningAnalyzer(cfg, DummyHypothesis(), DummyNumeric(), DummyCompiler())
        context = ContextSignature(workload="w", workload_kind="train", topology="t", scale="s", nodes=1)
        metrics = Metrics(iteration_time_ms=1000.0, success=True)
        class DummyState:
            def __init__(self):
                self.should_stop = False
                self.best_record = None
                self.history = []
                self.plateau_count = 0
            def recent_best_window(self):
                return []
        state = DummyState()
        action = analyzer.plan_next_action(state, metrics, None, context, step=0, plan=None, workload=None, base_config=NCCLConfig())
        self.assertEqual(getattr(action, "kind", None), "hypothesis")


if __name__ == "__main__":
    unittest.main()
