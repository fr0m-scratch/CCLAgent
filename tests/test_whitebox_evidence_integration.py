import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from src.agent.analyzer import TuningAnalyzer
from src.types import (
    AgentConfig,
    ExecutionConfig,
    LLMSettings,
    MemoryConfig,
    MetricsConfig,
    MicrobenchSettings,
    NCCLConfig,
    NumericSearchAction,
    NumericSearchSettings,
    ParameterSpace,
    ParameterSpec,
    RagConfig,
    RunContext,
    SafetyConfig,
    SearchCandidate,
    SearchResult,
    SurrogateConfig,
    TuningBudget,
    WarmStartSettings,
)
from src.whitebox import EvidenceStore


class DummyHypothesis:
    pass


class DummyNumeric:
    pass


class DummyCompiler:
    parameter_space = ParameterSpace.from_list([ParameterSpec(name="NCCL_ALGO", kind="enum", choices=["RING"], default="RING")])


class TestWhiteboxEvidenceIntegration(unittest.TestCase):
    def test_decision_record_emits_evidence_refs_and_store(self):
        ps = ParameterSpace.from_list(
            [
                ParameterSpec(name="NCCL_ALGO", kind="enum", choices=["RING", "TREE"], default="RING"),
                ParameterSpec(name="NCCL_PROTO", kind="enum", choices=["LL", "SIMPLE"], default="LL"),
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

        with tempfile.TemporaryDirectory() as tmp:
            run_context = RunContext(
                run_id="run-test",
                started_at_iso=datetime.now(timezone.utc).isoformat(),
                artifacts_dir=tmp,
                dry_run=True,
                seed=7,
                git_commit=None,
                host_info={},
            )
            # expected artifact dirs
            for sub in ("steps", "offline", "online", "postrun", "trace"):
                Path(run_context.artifacts_dir, sub).mkdir(parents=True, exist_ok=True)

            analyzer = TuningAnalyzer(
                cfg,
                DummyHypothesis(),
                DummyNumeric(),
                DummyCompiler(),
                run_context=run_context,
                evidence_store=EvidenceStore(),
            )

            selected_cfg = NCCLConfig(params={"NCCL_ALGO": "TREE", "NCCL_PROTO": "LL"})
            other_cfg = NCCLConfig(params={"NCCL_ALGO": "RING", "NCCL_PROTO": "SIMPLE"})
            selected = SearchCandidate(
                config=selected_cfg,
                predicted_time_ms=10.0,
                rationale="selected",
                candidate_id="1_0",
                uncertainty=0.2,
                risk_score=0.1,
            )
            rejected = SearchCandidate(
                config=other_cfg,
                predicted_time_ms=12.0,
                rationale="rejected",
                candidate_id="1_1",
                uncertainty=0.3,
                risk_score=0.2,
            )
            action = NumericSearchAction(
                kind="numeric",
                config=selected_cfg,
                rationale="numeric",
                search=SearchResult(best=selected, candidates=[selected, rejected]),
            )

            class DummyState:
                def __init__(self):
                    self.last_known_good = selected_cfg

            analyzer._write_decision_record(
                step=1,
                action=action,
                state=DummyState(),
                why_selected=["best predicted"],
                why_rejected=["dominated"],
                advice=None,
            )

            bundle_path = Path(tmp, "steps", "step_1_decision_bundle.json")
            self.assertTrue(bundle_path.exists())
            bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
            call_chain = bundle.get("chosen_action", {}).get("call_chain", [])
            self.assertTrue(any(str(ref).startswith("evidence:") for ref in call_chain))

            evidence_path = Path(tmp, "whitebox", "evidence.jsonl")
            self.assertTrue(evidence_path.exists())
            lines = [line for line in evidence_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertGreater(len(lines), 0)


if __name__ == "__main__":
    unittest.main()
