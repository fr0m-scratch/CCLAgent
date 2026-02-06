"""Tests for WP8: Statistical convergence evidence."""

import unittest

from src.agent.convergence_argument import (
    BootstrapCI,
    ConvergenceEvidence,
    _bootstrap_mean_ci,
    _cohens_d,
    compute_convergence_evidence,
)
from src.agent.state import TuningState
from src.types import (
    Metrics,
    NCCLConfig,
    TuningAction,
    TuningBudget,
    TuningRecord,
)


def _make_record(step: int, time_ms: float) -> TuningRecord:
    return TuningRecord(
        step=step,
        action=TuningAction(kind="apply", config=NCCLConfig(params={"A": str(step)}), rationale="test"),
        metrics=Metrics(iteration_time_ms=time_ms, success=True),
    )


class TestBootstrapMeanCI(unittest.TestCase):
    def test_empty_values(self):
        ci = _bootstrap_mean_ci([])
        self.assertEqual(ci.n_resamples, 0)
        self.assertAlmostEqual(ci.mean, 0.0)

    def test_constant_values(self):
        ci = _bootstrap_mean_ci([5.0] * 20, n_resamples=500)
        self.assertAlmostEqual(ci.lower, 5.0)
        self.assertAlmostEqual(ci.upper, 5.0)
        self.assertAlmostEqual(ci.mean, 5.0)

    def test_ci_contains_true_mean(self):
        values = [float(i) for i in range(100)]
        ci = _bootstrap_mean_ci(values, n_resamples=2000)
        true_mean = 49.5
        self.assertLessEqual(ci.lower, true_mean)
        self.assertGreaterEqual(ci.upper, true_mean)

    def test_deterministic_with_seed(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        ci1 = _bootstrap_mean_ci(values, rng_seed=42)
        ci2 = _bootstrap_mean_ci(values, rng_seed=42)
        self.assertEqual(ci1.lower, ci2.lower)
        self.assertEqual(ci1.upper, ci2.upper)


class TestCohensD(unittest.TestCase):
    def test_identical_groups(self):
        g = [1.0, 2.0, 3.0, 4.0, 5.0]
        self.assertAlmostEqual(_cohens_d(g, g), 0.0)

    def test_small_groups(self):
        self.assertAlmostEqual(_cohens_d([1.0], [2.0]), 0.0)

    def test_large_effect(self):
        g1 = [100.0, 100.0, 100.0, 100.0]
        g2 = [50.0, 50.0, 50.0, 50.0]
        # Should be a very large effect size (different means, zero variance)
        # But since variance is zero, it'll return 0.0 (div by zero guard)
        d = _cohens_d(g1, g2)
        # With zero variance in both groups, pooled_std is 0
        self.assertAlmostEqual(d, 0.0)

    def test_moderate_effect(self):
        g1 = [10.0, 12.0, 11.0, 13.0, 10.0]
        g2 = [8.0, 9.0, 7.0, 10.0, 8.0]
        d = _cohens_d(g1, g2)
        self.assertGreater(d, 0.5)  # positive because g1 > g2


class TestComputeConvergenceEvidence(unittest.TestCase):
    def test_insufficient_data(self):
        state = TuningState(budget=TuningBudget())
        state.record(_make_record(0, 100.0))
        state.record(_make_record(1, 95.0))
        ev = compute_convergence_evidence(state)
        self.assertFalse(ev.converged)
        self.assertEqual(len(ev.claims), 1)
        self.assertIn("Insufficient", ev.claims[0]["claim"])

    def test_converged_plateau(self):
        """Long plateau should produce convergence evidence."""
        state = TuningState(budget=TuningBudget())
        # Improvement then long plateau
        state.record(_make_record(0, 100.0))
        state.record(_make_record(1, 90.0))
        for i in range(2, 12):
            state.record(_make_record(i, 90.0 + (i % 2) * 0.1))  # tiny oscillation
        ev = compute_convergence_evidence(state, plateau_window=5)
        # Should have claims about CI and plateau
        self.assertTrue(len(ev.claims) >= 1)
        self.assertIsNotNone(ev.bootstrap_ci)

    def test_not_converged_improving(self):
        """Steadily improving should not converge."""
        state = TuningState(budget=TuningBudget())
        for i in range(10):
            state.record(_make_record(i, 100.0 - i * 5))  # 100, 95, 90, ...
        ev = compute_convergence_evidence(state, plateau_window=3)
        # Should NOT be converged since still improving significantly
        self.assertFalse(ev.converged)

    def test_to_dict(self):
        state = TuningState(budget=TuningBudget())
        for i in range(5):
            state.record(_make_record(i, 100.0))
        ev = compute_convergence_evidence(state)
        d = ev.to_dict()
        self.assertIn("converged", d)
        self.assertIn("effect_size", d)
        self.assertIn("claims", d)
        self.assertIn("statistics", d)

    def test_llm_confidence_claim(self):
        state = TuningState(budget=TuningBudget())
        for i in range(5):
            state.record(_make_record(i, 100.0))
        ev = compute_convergence_evidence(state, llm_confidence=0.9)
        llm_claims = [c for c in ev.claims if "LLM" in c["claim"]]
        self.assertEqual(len(llm_claims), 1)


if __name__ == "__main__":
    unittest.main()
