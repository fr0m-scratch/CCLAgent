"""Tests for WP9: Risk formalization — decomposition and budget."""

import unittest

from src.safety.risk import (
    RiskScore,
    RiskDecomposition,
    RiskBudgetState,
    RiskScorer,
)
from src.types import NCCLConfig, SafetyConfig


def _make_safety(**kwargs) -> SafetyConfig:
    defaults = {
        "max_risk_score": 0.7,
        "max_channels_safe": 16,
        "min_buffsize_safe": 1048576,
        "safe_envelope": {},
        "known_bad_combos_path": None,
    }
    defaults.update(kwargs)
    return SafetyConfig(**defaults)


class TestRiskDecomposition(unittest.TestCase):
    def test_total_capped_at_one(self):
        rd = RiskDecomposition(parameter_risk=0.5, combination_risk=0.4, novelty_risk=0.3)
        self.assertAlmostEqual(rd.total, 1.0)

    def test_total_additive_below_cap(self):
        rd = RiskDecomposition(parameter_risk=0.1, combination_risk=0.1, novelty_risk=0.1)
        self.assertAlmostEqual(rd.total, 0.3)


class TestRiskBudgetState(unittest.TestCase):
    def test_initial_state(self):
        rb = RiskBudgetState(total_budget=5.0)
        self.assertAlmostEqual(rb.remaining, 5.0)
        self.assertAlmostEqual(rb.utilization_pct, 0.0)
        self.assertTrue(rb.can_proceed(1.0))

    def test_record_step(self):
        rb = RiskBudgetState(total_budget=5.0)
        rb.record_step(0.3)
        rb.record_step(0.5)
        self.assertEqual(rb.steps_taken, 2)
        self.assertAlmostEqual(rb.consumed, 0.8)
        self.assertAlmostEqual(rb.remaining, 4.2)
        self.assertEqual(len(rb.per_step), 2)

    def test_can_proceed_false_when_exceeded(self):
        rb = RiskBudgetState(total_budget=1.0)
        rb.record_step(0.8)
        self.assertFalse(rb.can_proceed(0.3))
        self.assertTrue(rb.can_proceed(0.2))

    def test_utilization_pct(self):
        rb = RiskBudgetState(total_budget=10.0)
        rb.record_step(2.5)
        self.assertAlmostEqual(rb.utilization_pct, 25.0)


class TestRiskScorerDecompose(unittest.TestCase):
    def test_no_violations(self):
        safety = _make_safety()
        scorer = RiskScorer(safety)
        config = NCCLConfig(params={"NCCL_ALGO": "RING"})
        decomp = scorer.decompose(config)
        self.assertAlmostEqual(decomp.parameter_risk, 0.0)
        self.assertAlmostEqual(decomp.combination_risk, 0.0)
        # Novelty risk is 0.15 when no memory_configs
        self.assertAlmostEqual(decomp.novelty_risk, 0.15)

    def test_envelope_violation(self):
        safety = _make_safety(safe_envelope={
            "NCCL_MAX_NCHANNELS": {"min": 1, "max": 16},
        })
        scorer = RiskScorer(safety)
        config = NCCLConfig(params={"NCCL_MAX_NCHANNELS": "32"})
        decomp = scorer.decompose(config)
        self.assertGreater(decomp.parameter_risk, 0.0)
        self.assertIn("above_max:NCCL_MAX_NCHANNELS", decomp.components)

    def test_novelty_with_memory(self):
        safety = _make_safety()
        scorer = RiskScorer(safety)
        config = NCCLConfig(params={"A": "1", "B": "2"})
        memory_configs = [{"A": "1", "B": "2"}]  # exact match
        decomp = scorer.decompose(config, memory_configs=memory_configs)
        self.assertAlmostEqual(decomp.novelty_risk, 0.0)

    def test_novelty_no_overlap(self):
        safety = _make_safety()
        scorer = RiskScorer(safety)
        config = NCCLConfig(params={"A": "1", "B": "2"})
        memory_configs = [{"C": "3", "D": "4"}]  # no overlap
        decomp = scorer.decompose(config, memory_configs=memory_configs)
        self.assertAlmostEqual(decomp.novelty_risk, 0.3)  # 1.0 * 0.3

    def test_avoid_rules_add_combo_risk(self):
        safety = _make_safety()
        scorer = RiskScorer(safety)
        config = NCCLConfig(params={"NCCL_PROTO": "LL128"})
        avoid_rules = [{"NCCL_PROTO": "LL128"}]
        decomp = scorer.decompose(config, avoid_rules=avoid_rules)
        self.assertGreater(decomp.combination_risk, 0.0)


if __name__ == "__main__":
    unittest.main()
