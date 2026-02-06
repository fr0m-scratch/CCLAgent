"""Tests for WP6: Causal attribution — step-level and parameter-level."""

import unittest

from src.agent.attribution import (
    StepAttribution,
    ParamAttribution,
    LaneAttribution,
    AttributionReport,
    compute_step_attributions,
    compute_param_attributions,
    compute_lane_attributions,
    build_attribution_report,
)
from src.agent.state import TuningState
from src.types import (
    Metrics,
    NCCLConfig,
    TuningAction,
    TuningBudget,
    TuningRecord,
)


def _make_record(step: int, params: dict, time_ms: float, kind: str = "apply") -> TuningRecord:
    return TuningRecord(
        step=step,
        action=TuningAction(kind=kind, config=NCCLConfig(params=params), rationale="test"),
        metrics=Metrics(iteration_time_ms=time_ms, success=True),
    )


class TestComputeStepAttributions(unittest.TestCase):
    def test_empty_history(self):
        state = TuningState(budget=TuningBudget())
        self.assertEqual(compute_step_attributions(state), [])

    def test_single_record(self):
        state = TuningState(budget=TuningBudget())
        state.record(_make_record(0, {"A": "1"}, 100.0, kind="initial"))
        self.assertEqual(compute_step_attributions(state), [])

    def test_improvement_is_positive(self):
        state = TuningState(budget=TuningBudget())
        state.record(_make_record(0, {"A": "1"}, 100.0, kind="initial"))
        state.record(_make_record(1, {"A": "2"}, 90.0, kind="hypothesis"))
        attrs = compute_step_attributions(state)
        self.assertEqual(len(attrs), 1)
        self.assertAlmostEqual(attrs[0].delta_ms, 10.0)
        self.assertAlmostEqual(attrs[0].delta_pct, 10.0)
        self.assertEqual(attrs[0].action_lane, "hypothesis")
        self.assertIn("A", attrs[0].config_changes)

    def test_regression_is_negative(self):
        state = TuningState(budget=TuningBudget())
        state.record(_make_record(0, {"A": "1"}, 100.0, kind="initial"))
        state.record(_make_record(1, {"A": "2"}, 110.0, kind="numeric"))
        attrs = compute_step_attributions(state)
        self.assertAlmostEqual(attrs[0].delta_ms, -10.0)


class TestComputeParamAttributions(unittest.TestCase):
    def test_no_changes(self):
        base = NCCLConfig(params={"A": "1", "B": "2"})
        best = NCCLConfig(params={"A": "1", "B": "2"})
        self.assertEqual(compute_param_attributions(base, best, 100.0), [])

    def test_no_surrogate_returns_zero_deltas(self):
        base = NCCLConfig(params={"A": "1"})
        best = NCCLConfig(params={"A": "2"})
        attrs = compute_param_attributions(base, best, 100.0, surrogate=None)
        self.assertEqual(len(attrs), 1)
        self.assertEqual(attrs[0].param, "A")
        self.assertAlmostEqual(attrs[0].attributed_delta_ms, 0.0)

    def test_sorted_by_abs_delta(self):
        base = NCCLConfig(params={"A": "1", "B": "x"})
        best = NCCLConfig(params={"A": "2", "B": "y"})
        attrs = compute_param_attributions(base, best, 100.0, surrogate=None)
        self.assertEqual(len(attrs), 2)


class TestComputeLaneAttributions(unittest.TestCase):
    def test_aggregation(self):
        step_attrs = [
            StepAttribution(step=1, action_lane="hypothesis", delta_ms=5.0, delta_pct=5.0),
            StepAttribution(step=2, action_lane="numeric", delta_ms=3.0, delta_pct=3.0),
            StepAttribution(step=3, action_lane="hypothesis", delta_ms=2.0, delta_pct=2.0),
        ]
        lane_attrs = compute_lane_attributions(step_attrs)
        by_lane = {la.lane: la for la in lane_attrs}
        self.assertAlmostEqual(by_lane["hypothesis"].total_delta_ms, 7.0)
        self.assertEqual(by_lane["hypothesis"].step_count, 2)
        self.assertAlmostEqual(by_lane["numeric"].total_delta_ms, 3.0)


class TestBuildAttributionReport(unittest.TestCase):
    def test_empty_state(self):
        state = TuningState(budget=TuningBudget())
        report = build_attribution_report(state)
        self.assertAlmostEqual(report.baseline_ms, 0.0)

    def test_full_report(self):
        state = TuningState(budget=TuningBudget())
        state.record(_make_record(0, {"A": "1"}, 100.0, kind="initial"))
        state.record(_make_record(1, {"A": "2"}, 90.0, kind="hypothesis"))
        state.record(_make_record(2, {"A": "3"}, 85.0, kind="numeric"))
        report = build_attribution_report(state)
        self.assertAlmostEqual(report.baseline_ms, 100.0)
        self.assertAlmostEqual(report.best_ms, 85.0)
        self.assertAlmostEqual(report.total_improvement_ms, 15.0)
        self.assertAlmostEqual(report.total_improvement_pct, 15.0)
        self.assertEqual(len(report.step_attributions), 2)
        self.assertTrue(len(report.lane_attributions) > 0)

    def test_to_dict(self):
        state = TuningState(budget=TuningBudget())
        state.record(_make_record(0, {"A": "1"}, 100.0, kind="initial"))
        report = build_attribution_report(state)
        d = report.to_dict()
        self.assertIn("baseline_ms", d)
        self.assertIn("step_attributions", d)


if __name__ == "__main__":
    unittest.main()
