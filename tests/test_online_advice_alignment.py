import unittest
from dataclasses import dataclass

from src.agent.analyzer import TuningAnalyzer
from src.types import (
    AgentConfig,
    CompiledConfig,
    ContextSignature,
    ExecutionConfig,
    LLMSettings,
    MemoryConfig,
    Metrics,
    MetricsConfig,
    MicrobenchSettings,
    NCCLConfig,
    NumericSearchSettings,
    ParameterSpace,
    ParameterSpec,
    RagConfig,
    SafetyConfig,
    SearchCandidate,
    SearchResult,
    SurrogateConfig,
    TuningBudget,
    WarmStartSettings,
)


class DummyHypothesisGen:
    def propose_portfolio(self, plan, context, base_config, last_metrics, max_hypotheses=3):
        from src.types import Hypothesis

        return [Hypothesis(id="H0", summary="test", patch={"NCCL_ALGO": "TREE"})]


class DummySurrogate:
    class Pred:
        def __init__(self):
            self.mean = 100.0
            self.std = 5.0

    def predict_one(self, config, context=None):
        return self.Pred()


class DummyNumeric:
    def __init__(self):
        self.surrogate = DummySurrogate()

    def propose(self, plan, state, workload, base_config, step, context=None, guidance=None):
        candidate = SearchCandidate(
            config=base_config or NCCLConfig(),
            predicted_time_ms=100.0,
            rationale="numeric",
        )
        return candidate.config, SearchResult(best=candidate, candidates=[candidate])


class DummyCompiler:
    def __init__(self, parameter_space):
        self.parameter_space = parameter_space

    def compile_hypothesis(self, base, patch):
        merged = dict(base.params if base else {})
        merged.update(patch)
        return CompiledConfig(config=NCCLConfig(params=merged), env={}, warnings=[], risk_score=0.0)


@dataclass
class Advice:
    step: int
    output: dict
    parse_errors: list[str]
    decision_eligible: bool
    call_id: str = "call_x"
    raw_is_valid_json: bool = True
    schema_passed: bool = True
    raw_text: str = "{}"


class DummyAdvisor:
    def __init__(self, current=None, ready=None):
        self._current = current
        self._ready = ready or []

    def try_get(self, *, step: int, timeout_s: float = 0.0):
        return self._current

    def collect_ready(self):
        return list(self._ready)

    def decide_convergence(self, **kwargs):
        return None


class DummyState:
    def __init__(self):
        self.should_stop = False
        self.best_record = None
        self.history = []
        self.plateau_count = 0
        self.last_known_good = None

    def recent_best_window(self):
        return []


class TestOnlineAdviceAlignment(unittest.TestCase):
    def setUp(self):
        ps = ParameterSpace.from_list(
            [
                ParameterSpec(name="NCCL_ALGO", kind="enum", choices=["RING", "TREE"], default="RING"),
            ]
        )
        cfg = AgentConfig(
            parameter_space=ps,
            budget=TuningBudget(max_steps=5, hypothesis_every=1),
            memory=MemoryConfig(),
            rag=RagConfig(),
            llm=LLMSettings(online_enabled=True, online_triggers=["always"], online_soft_wait_s=0.0),
            warm_start=WarmStartSettings(),
            microbench=MicrobenchSettings(),
            metrics=MetricsConfig(),
            numeric_search=NumericSearchSettings(),
            safety=SafetyConfig(),
            execution=ExecutionConfig(),
            surrogate=SurrogateConfig(),
        )
        self.ps = ps
        self.cfg = cfg
        self.context = ContextSignature(workload="w", workload_kind="train", topology="t", scale="s", nodes=1)
        self.metrics = Metrics(iteration_time_ms=1000.0, success=True, raw={})
        self.base = NCCLConfig(params={"NCCL_ALGO": "RING"})

    def _build_analyzer(self, advisor):
        return TuningAnalyzer(
            self.cfg,
            DummyHypothesisGen(),
            DummyNumeric(),
            DummyCompiler(self.ps),
            llm_advisor=advisor,
        )

    def test_cross_step_late_advice_is_not_used_for_current_step(self):
        late = Advice(
            step=0,
            output={"action_preference": "hypothesis", "numeric_guidance": {}, "tool_request": {"name": "none"}, "hypotheses": [], "convergence": {"decision": "continue", "confidence": 0.9}},
            parse_errors=[],
            decision_eligible=True,
        )
        analyzer = self._build_analyzer(DummyAdvisor(current=None, ready=[late]))
        action = analyzer.plan_next_action(
            DummyState(),
            self.metrics,
            None,
            self.context,
            step=1,
            plan=None,
            workload=None,
            base_config=self.base,
        )
        # Late advice from another step must not drive current-step lane.
        # With hypothesis_every=1, schedule defaults to hypothesis.
        self.assertEqual(getattr(action, "kind", None), "hypothesis")

    def test_ineligible_current_step_advice_is_not_used(self):
        current = Advice(
            step=1,
            output={"action_preference": "hypothesis", "numeric_guidance": {}, "tool_request": {"name": "none"}, "hypotheses": [], "convergence": {"decision": "continue", "confidence": 0.9}},
            parse_errors=["invalid_json"],
            decision_eligible=False,
            raw_is_valid_json=False,
            schema_passed=False,
        )
        analyzer = self._build_analyzer(DummyAdvisor(current=current, ready=[]))
        action = analyzer.plan_next_action(
            DummyState(),
            self.metrics,
            None,
            self.context,
            step=1,
            plan=None,
            workload=None,
            base_config=self.base,
        )
        # Ineligible advice must be ignored; schedule still applies.
        self.assertEqual(getattr(action, "kind", None), "hypothesis")


if __name__ == "__main__":
    unittest.main()
