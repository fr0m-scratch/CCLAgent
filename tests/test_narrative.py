"""Tests for WP10a: Run narrative generation."""

import unittest

from src.agent.narrative import RunNarrative, generate_narrative
from src.agent.hypothesis_tracker import HypothesisScorecard
from src.agent.attribution import AttributionReport, ParamAttribution
from src.agent.convergence_argument import ConvergenceEvidence
from src.agent.llm_influence import LLMInfluenceSummary
from src.types import ContextSignature


def _make_context() -> ContextSignature:
    return ContextSignature(
        workload="llama3.1-8b",
        workload_kind="training",
        topology="nvlink",
        scale="8gpu",
        nodes=1,
        gpus_per_node=8,
    )


class TestRunNarrative(unittest.TestCase):
    def test_to_markdown_empty(self):
        n = RunNarrative(title="Test Report")
        md = n.to_markdown()
        self.assertIn("# Test Report", md)

    def test_to_markdown_with_sections(self):
        n = RunNarrative(
            title="Test",
            context_summary="NVLink 8 GPU",
            performance_summary="10% improvement",
            key_findings=["Finding 1", "Finding 2"],
        )
        md = n.to_markdown()
        self.assertIn("## Context", md)
        self.assertIn("## Performance", md)
        self.assertIn("## Key Findings", md)
        self.assertIn("1. Finding 1", md)


class TestGenerateNarrative(unittest.TestCase):
    def test_minimal_narrative(self):
        n = generate_narrative(baseline_ms=100.0, best_ms=90.0, total_steps=5)
        self.assertIn("CCLAgent Tuning Report", n.title)
        self.assertIn("10.0%", n.performance_summary)
        self.assertTrue(len(n.full_text) > 0)

    def test_with_context(self):
        ctx = _make_context()
        n = generate_narrative(context=ctx, baseline_ms=100.0, best_ms=85.0, total_steps=10)
        self.assertIn("llama3.1-8b", n.title)
        self.assertIn("nvlink", n.context_summary)

    def test_with_scorecard(self):
        sc = HypothesisScorecard(
            total=5, confirmed=3, refuted=1, inconclusive=1,
            confirmation_rate=0.6, avg_margin=2.5,
        )
        n = generate_narrative(baseline_ms=100.0, best_ms=90.0, total_steps=5, scorecard=sc)
        self.assertIn("5 hypotheses", n.hypothesis_summary)
        self.assertIn("3 confirmed", n.hypothesis_summary)
        self.assertIn("60%", n.hypothesis_summary)

    def test_with_attribution(self):
        attr = AttributionReport(
            baseline_ms=100.0, best_ms=85.0,
            total_improvement_ms=15.0, total_improvement_pct=15.0,
            param_attributions=[
                ParamAttribution(param="NCCL_MAX_NCHANNELS", from_value="8", to_value="16",
                                 attributed_delta_ms=10.0, attributed_delta_pct=10.0),
            ],
            consistency_check=0.95,
        )
        n = generate_narrative(baseline_ms=100.0, best_ms=85.0, total_steps=5, attribution=attr)
        self.assertIn("NCCL_MAX_NCHANNELS", n.attribution_summary)

    def test_with_convergence(self):
        conv = ConvergenceEvidence(
            converged=True,
            effect_size=0.15,
            claims=[{"claim": "CI contains zero", "refs": ["bootstrap_ci"]}],
        )
        n = generate_narrative(baseline_ms=100.0, best_ms=90.0, total_steps=5, convergence=conv)
        self.assertIn("Converged: True", n.convergence_summary)
        self.assertIn("CI contains zero", n.convergence_summary)

    def test_with_influence(self):
        inf = LLMInfluenceSummary(
            total_steps=5, advice_available_count=4,
            influence_rate=0.75, override_rate=0.25,
            llm_attributed_gain_pct=60.0,
        )
        n = generate_narrative(baseline_ms=100.0, best_ms=90.0, total_steps=5, influence=inf)
        self.assertIn("75%", n.influence_summary)
        self.assertIn("60.0%", n.influence_summary)

    def test_key_findings_populated(self):
        n = generate_narrative(baseline_ms=100.0, best_ms=80.0, total_steps=10)
        self.assertTrue(len(n.key_findings) >= 1)
        self.assertIn("20.0%", n.key_findings[0])


if __name__ == "__main__":
    unittest.main()
