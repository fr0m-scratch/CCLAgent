import json
import tempfile
import unittest
from pathlib import Path

from src.agent.analyzer import TuningAnalyzer
from src.types import (
    AgentConfig,
    ContextSignature,
    ExecutionConfig,
    ImportantParam,
    LLMSettings,
    MemoryConfig,
    Metrics,
    MetricsConfig,
    MicrobenchResult,
    MicrobenchSettings,
    NCCLConfig,
    NumericSearchSettings,
    ParameterSpace,
    ParameterSpec,
    RagConfig,
    RunContext,
    SafetyConfig,
    SurrogateConfig,
    TuningBudget,
    WarmStartSettings,
    WorkloadSpec,
    RAGChunk,
)


class DummyHypothesis:
    def propose(self, plan, context, base_config, last_metrics):
        from src.types import Hypothesis
        return Hypothesis(id="h", summary="test", patch={"NCCL_ALGO": "RING"})

    def propose_portfolio(self, plan, context, base_config, last_metrics, max_hypotheses=3):
        return [self.propose(plan, context, base_config, last_metrics)]


class DummyNumeric:
    def __init__(self):
        class _S:
            model_type = "rf"
            _y = [1.0]

            def predict_one(self, config, context=None):
                class _P:
                    mean = 1000.0
                    std = 10.0

                return _P()

        self.surrogate = _S()

    def propose(self, plan, state, workload, base_config, step, context=None, guidance=None):
        from src.types import SearchCandidate, SearchResult

        cfg = NCCLConfig(params=base_config.params)
        candidate = SearchCandidate(config=cfg, predicted_time_ms=1.0, rationale="dummy")
        return cfg, SearchResult(best=candidate, candidates=[candidate])


class DummyCompiler:
    def __init__(self, ps):
        self.parameter_space = ps

    def compile_hypothesis(self, base, patch):
        from src.types import CompiledConfig, NCCLConfig

        merged = dict(base.params)
        merged.update(patch)
        return CompiledConfig(config=NCCLConfig(params=merged), env={}, warnings=[], risk_score=0.0)


class DummyRag:
    def __init__(self):
        self.loaded = False
        self.search_calls = 0

    def load_documents(self, config=None):
        self.loaded = True

    def search(self, query: str, top_k: int = 5):
        self.search_calls += 1
        return [
            RAGChunk(doc_id="doc/Knowledge/nccl_algorithms.md", chunk_id="0", text="Tree vs Ring", score=0.9),
            RAGChunk(doc_id="doc/Knowledge/nccl_protocols.md", chunk_id="1", text="LL128 details", score=0.8),
        ][:top_k]


class DummyState:
    def __init__(self):
        self.should_stop = False
        self.best_record = None
        self.history = []
        self.plateau_count = 0
        self.last_known_good = None


class OnlineRagUsageTest(unittest.TestCase):
    def test_online_context_pack_contains_rag_chunks(self):
        ps = ParameterSpace.from_list(
            [ParameterSpec(name="NCCL_ALGO", kind="enum", choices=["RING"], default="RING")]
        )
        cfg = AgentConfig(
            parameter_space=ps,
            budget=TuningBudget(max_steps=3, hypothesis_every=1),
            memory=MemoryConfig(),
            rag=RagConfig(top_k=4),
            llm=LLMSettings(),
            warm_start=WarmStartSettings(),
            microbench=MicrobenchSettings(),
            metrics=MetricsConfig(),
            numeric_search=NumericSearchSettings(),
            safety=SafetyConfig(),
            execution=ExecutionConfig(),
            surrogate=SurrogateConfig(),
        )

        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            (run_dir / "steps").mkdir(parents=True, exist_ok=True)
            (run_dir / "offline").mkdir(parents=True, exist_ok=True)
            (run_dir / "online").mkdir(parents=True, exist_ok=True)
            (run_dir / "postrun").mkdir(parents=True, exist_ok=True)

            run_context = RunContext(
                run_id="test-run",
                started_at_iso="2026-02-06T00:00:00Z",
                artifacts_dir=str(run_dir),
                dry_run=True,
                seed=7,
                git_commit=None,
                host_info={},
            )

            analyzer = TuningAnalyzer(
                cfg,
                DummyHypothesis(),
                DummyNumeric(),
                DummyCompiler(ps),
                run_context=run_context,
                rag=DummyRag(),
            )

            context = ContextSignature(
                workload="autoccl-llama3.1-8b-agentic-showcase",
                workload_kind="training",
                topology="a40-pcie-8gpu",
                scale="32-gpu",
                nodes=4,
            )
            workload = WorkloadSpec(name="autoccl-llama3.1-8b-agentic-showcase", command=[])
            metrics = Metrics(iteration_time_ms=1000.0, success=True, raw={"bottleneck": "comm_bound"})
            microbench = MicrobenchResult(
                important_params=[
                    ImportantParam(param="NCCL_ALGO", importance=0.9, reason="high impact"),
                    ImportantParam(param="NCCL_PROTO", importance=0.8, reason="high impact"),
                ]
            )

            action = analyzer.plan_next_action(
                state=DummyState(),
                last_metrics=metrics,
                microbench=microbench,
                context=context,
                step=0,
                plan=None,
                workload=workload,
                base_config=NCCLConfig(),
            )
            self.assertIsNotNone(action)

            cp = json.loads((run_dir / "steps" / "step_0_context_pack.json").read_text(encoding="utf-8"))
            rag_chunks = cp.get("retrieval", {}).get("rag_chunks", [])
            self.assertGreater(len(rag_chunks), 0)
            self.assertTrue(rag_chunks[0]["ref"].startswith("rag:"))

            rr_path = run_dir / "steps" / "step_0_online_rag_retrieval.json"
            self.assertTrue(rr_path.exists())
            rr = json.loads(rr_path.read_text(encoding="utf-8"))
            self.assertGreater(len(rr.get("chunks", [])), 0)

    def test_online_rag_reuses_cached_chunks_between_steps(self):
        ps = ParameterSpace.from_list(
            [ParameterSpec(name="NCCL_ALGO", kind="enum", choices=["RING"], default="RING")]
        )
        cfg = AgentConfig(
            parameter_space=ps,
            budget=TuningBudget(max_steps=4, hypothesis_every=100),
            memory=MemoryConfig(),
            rag=RagConfig(top_k=4),
            llm=LLMSettings(),
            warm_start=WarmStartSettings(),
            microbench=MicrobenchSettings(),
            metrics=MetricsConfig(),
            numeric_search=NumericSearchSettings(),
            safety=SafetyConfig(),
            execution=ExecutionConfig(),
            surrogate=SurrogateConfig(),
        )

        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            (run_dir / "steps").mkdir(parents=True, exist_ok=True)
            (run_dir / "offline").mkdir(parents=True, exist_ok=True)
            (run_dir / "online").mkdir(parents=True, exist_ok=True)
            (run_dir / "postrun").mkdir(parents=True, exist_ok=True)

            run_context = RunContext(
                run_id="test-run",
                started_at_iso="2026-02-06T00:00:00Z",
                artifacts_dir=str(run_dir),
                dry_run=True,
                seed=7,
                git_commit=None,
                host_info={},
            )
            rag = DummyRag()
            analyzer = TuningAnalyzer(
                cfg,
                DummyHypothesis(),
                DummyNumeric(),
                DummyCompiler(ps),
                run_context=run_context,
                rag=rag,
            )
            context = ContextSignature(
                workload="autoccl-llama3.1-8b-agentic-showcase",
                workload_kind="training",
                topology="a40-pcie-8gpu",
                scale="32-gpu",
                nodes=4,
            )
            workload = WorkloadSpec(name="autoccl-llama3.1-8b-agentic-showcase", command=[])
            metrics = Metrics(iteration_time_ms=1000.0, success=True, raw={"bottleneck": "comm_bound"})
            microbench = MicrobenchResult(
                important_params=[
                    ImportantParam(param="NCCL_ALGO", importance=0.9, reason="high impact"),
                ]
            )

            state = DummyState()
            analyzer.plan_next_action(
                state=state,
                last_metrics=metrics,
                microbench=microbench,
                context=context,
                step=0,
                plan=None,
                workload=workload,
                base_config=NCCLConfig(),
            )
            analyzer.plan_next_action(
                state=state,
                last_metrics=metrics,
                microbench=microbench,
                context=context,
                step=1,
                plan=None,
                workload=workload,
                base_config=NCCLConfig(),
            )

            self.assertEqual(rag.search_calls, 1)
            rr1 = json.loads((run_dir / "steps" / "step_1_online_rag_retrieval.json").read_text(encoding="utf-8"))
            self.assertTrue(rr1.get("reused"))


if __name__ == "__main__":
    unittest.main()
