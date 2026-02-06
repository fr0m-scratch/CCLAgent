"""Tests for WP4: Hypothesis lifecycle tracking with falsifiability."""

import unittest
from dataclasses import dataclass
from typing import Any, Dict, Optional

from src.agent.hypothesis_tracker import (
    HypothesisPrediction,
    HypothesisVerdict,
    HypothesisScorecard,
    make_prediction,
    compute_verdict,
    build_scorecard,
    INCONCLUSIVE_THRESHOLD,
)
from src.types import Hypothesis, Metrics


def _make_hypothesis(
    id: str = "h1",
    mechanism: str = "channel_parallelism",
    expected_effect: Optional[Dict[str, Any]] = None,
) -> Hypothesis:
    return Hypothesis(
        id=id,
        summary="Test hypothesis",
        patch={"NCCL_MAX_NCHANNELS": "16"},
        mechanism=mechanism,
        expected_effect=expected_effect or {"improvement_pct": 5.0, "direction": "decrease"},
    )


def _make_metrics(iteration_time_ms: float = 100.0, success: bool = True) -> Metrics:
    return Metrics(iteration_time_ms=iteration_time_ms, success=success)


class TestMakePrediction(unittest.TestCase):
    def test_extracts_improvement_pct_from_expected_effect(self):
        hyp = _make_hypothesis(expected_effect={"improvement_pct": 8.5, "direction": "decrease"})
        pred = make_prediction(hyp, step=1, baseline_ms=100.0)
        self.assertEqual(pred.predicted_delta_pct, 8.5)
        self.assertEqual(pred.predicted_direction, "decrease")

    def test_fallback_to_surrogate_when_no_effect(self):
        hyp = Hypothesis(id="h1", summary="Test", patch={"A": "1"}, mechanism="test")
        pred = make_prediction(hyp, step=1, baseline_ms=100.0, surrogate_mean=90.0, surrogate_std=2.0)
        self.assertAlmostEqual(pred.predicted_delta_pct, 10.0)
        self.assertEqual(pred.surrogate_mean, 90.0)
        self.assertEqual(pred.surrogate_std, 2.0)

    def test_mechanism_propagated(self):
        hyp = _make_hypothesis(mechanism="tree_latency")
        pred = make_prediction(hyp, step=2, baseline_ms=50.0)
        self.assertEqual(pred.mechanism, "tree_latency")

    def test_increase_direction(self):
        hyp = _make_hypothesis(expected_effect={"improvement_pct": 3.0, "direction": "increase"})
        pred = make_prediction(hyp, step=1, baseline_ms=100.0)
        self.assertEqual(pred.predicted_direction, "increase")


class TestComputeVerdict(unittest.TestCase):
    def test_confirmed_improvement(self):
        pred = HypothesisPrediction(
            hypothesis_id="h1", step=1, predicted_direction="decrease",
            predicted_delta_pct=5.0, baseline_ms=100.0,
        )
        metrics = _make_metrics(iteration_time_ms=92.0)  # 8% improvement
        verdict = compute_verdict(pred, metrics)
        self.assertTrue(verdict.confirmed)
        self.assertEqual(verdict.verdict, "confirmed")
        self.assertAlmostEqual(verdict.actual_delta_pct, 8.0)
        self.assertAlmostEqual(verdict.margin, 3.0)  # 8% - 5%

    def test_refuted_regression(self):
        pred = HypothesisPrediction(
            hypothesis_id="h1", step=1, predicted_direction="decrease",
            predicted_delta_pct=5.0, baseline_ms=100.0,
        )
        metrics = _make_metrics(iteration_time_ms=105.0)  # 5% regression
        verdict = compute_verdict(pred, metrics)
        self.assertFalse(verdict.confirmed)
        self.assertEqual(verdict.verdict, "refuted")

    def test_inconclusive_within_noise(self):
        pred = HypothesisPrediction(
            hypothesis_id="h1", step=1, predicted_direction="decrease",
            predicted_delta_pct=5.0, baseline_ms=100.0,
        )
        # 0.3% change — within INCONCLUSIVE_THRESHOLD (0.5%)
        metrics = _make_metrics(iteration_time_ms=99.7)
        verdict = compute_verdict(pred, metrics)
        self.assertEqual(verdict.verdict, "inconclusive")

    def test_zero_baseline_returns_inconclusive(self):
        pred = HypothesisPrediction(
            hypothesis_id="h1", step=1, predicted_direction="decrease",
            predicted_delta_pct=5.0, baseline_ms=0.0,
        )
        metrics = _make_metrics(iteration_time_ms=50.0)
        verdict = compute_verdict(pred, metrics)
        self.assertEqual(verdict.verdict, "inconclusive")

    def test_confirmed_degradation(self):
        """When we predict degradation and it actually degrades."""
        pred = HypothesisPrediction(
            hypothesis_id="h1", step=1, predicted_direction="increase",
            predicted_delta_pct=0.0, baseline_ms=100.0,
        )
        metrics = _make_metrics(iteration_time_ms=110.0)  # 10% worse
        verdict = compute_verdict(pred, metrics)
        self.assertTrue(verdict.confirmed)
        self.assertEqual(verdict.verdict, "confirmed")


class TestBuildScorecard(unittest.TestCase):
    def test_empty_verdicts(self):
        sc = build_scorecard([])
        self.assertEqual(sc.total, 0)
        self.assertEqual(sc.confirmation_rate, 0.0)

    def test_mixed_verdicts(self):
        pred = HypothesisPrediction(
            hypothesis_id="h1", step=1, predicted_direction="decrease",
            predicted_delta_pct=5.0, baseline_ms=100.0, mechanism="ring_bw",
        )
        verdicts = [
            HypothesisVerdict(
                hypothesis_id="h1", step=1, prediction=pred,
                actual_ms=92.0, actual_delta_pct=8.0, confirmed=True, margin=3.0,
                verdict="confirmed",
            ),
            HypothesisVerdict(
                hypothesis_id="h2", step=2, prediction=pred,
                actual_ms=105.0, actual_delta_pct=-5.0, confirmed=False, margin=-10.0,
                verdict="refuted",
            ),
            HypothesisVerdict(
                hypothesis_id="h3", step=3, prediction=pred,
                actual_ms=99.9, actual_delta_pct=0.1, confirmed=False, margin=-4.9,
                verdict="inconclusive",
            ),
        ]
        sc = build_scorecard(verdicts)
        self.assertEqual(sc.total, 3)
        self.assertEqual(sc.confirmed, 1)
        self.assertEqual(sc.refuted, 1)
        self.assertEqual(sc.inconclusive, 1)
        self.assertAlmostEqual(sc.confirmation_rate, 1 / 3)
        self.assertEqual(len(sc.verdicts), 3)
        self.assertIn("ring_bw", sc.per_mechanism)

    def test_to_dict(self):
        sc = build_scorecard([])
        d = sc.to_dict()
        self.assertIn("total", d)
        self.assertIn("confirmed", d)
        self.assertIn("per_mechanism", d)


if __name__ == "__main__":
    unittest.main()
