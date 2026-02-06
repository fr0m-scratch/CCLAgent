"""Tests for WP5: LLM influence measurement and attribution."""

import unittest

from src.agent.llm_influence import (
    LLMInfluenceRecord,
    LLMInfluenceSummary,
    record_influence,
    build_influence_summary,
)


class TestRecordInfluence(unittest.TestCase):
    def test_influenced_when_available_and_used(self):
        rec = record_influence(step=1, advice_available=True, advice_used=True, action_lane="hypothesis")
        self.assertTrue(rec.llm_influenced)
        self.assertFalse(rec.llm_overridden)
        self.assertTrue(rec.llm_advice_available)

    def test_overridden_when_available_but_not_used(self):
        rec = record_influence(step=1, advice_available=True, advice_used=False, action_lane="numeric")
        self.assertFalse(rec.llm_influenced)
        self.assertTrue(rec.llm_overridden)

    def test_no_advice(self):
        rec = record_influence(step=1, advice_available=False, advice_used=False, action_lane="initial")
        self.assertFalse(rec.llm_influenced)
        self.assertFalse(rec.llm_overridden)
        self.assertFalse(rec.llm_advice_available)

    def test_improvement_ms_stored(self):
        rec = record_influence(step=1, advice_available=True, advice_used=True, action_lane="hypothesis", improvement_ms=5.3)
        self.assertAlmostEqual(rec.improvement_ms, 5.3)


class TestBuildInfluenceSummary(unittest.TestCase):
    def test_empty_records(self):
        s = build_influence_summary([])
        self.assertEqual(s.total_steps, 0)
        self.assertEqual(s.influence_rate, 0.0)

    def test_all_influenced(self):
        records = [
            record_influence(step=i, advice_available=True, advice_used=True, action_lane="hypothesis", improvement_ms=2.0)
            for i in range(5)
        ]
        s = build_influence_summary(records)
        self.assertEqual(s.total_steps, 5)
        self.assertEqual(s.advice_available_count, 5)
        self.assertEqual(s.influenced_count, 5)
        self.assertAlmostEqual(s.influence_rate, 1.0)
        self.assertAlmostEqual(s.override_rate, 0.0)
        self.assertAlmostEqual(s.llm_attributed_gain_pct, 100.0)

    def test_mixed_influence(self):
        records = [
            record_influence(step=0, advice_available=False, advice_used=False, action_lane="initial"),
            record_influence(step=1, advice_available=True, advice_used=True, action_lane="hypothesis", improvement_ms=10.0),
            record_influence(step=2, advice_available=True, advice_used=False, action_lane="numeric", improvement_ms=5.0),
            record_influence(step=3, advice_available=True, advice_used=True, action_lane="hypothesis", improvement_ms=3.0),
        ]
        s = build_influence_summary(records)
        self.assertEqual(s.total_steps, 4)
        self.assertEqual(s.advice_available_count, 3)
        self.assertEqual(s.influenced_count, 2)
        self.assertEqual(s.overridden_count, 1)
        self.assertEqual(s.no_advice_count, 1)
        self.assertAlmostEqual(s.influence_rate, 2 / 3)
        self.assertAlmostEqual(s.override_rate, 1 / 3)
        # LLM-attributed gain: 10+3=13ms, heuristic: 5ms, total: 18ms
        self.assertAlmostEqual(s.llm_attributed_gain_ms, 13.0)
        self.assertAlmostEqual(s.heuristic_attributed_gain_ms, 5.0)
        self.assertAlmostEqual(s.llm_attributed_gain_pct, 13 / 18 * 100)

    def test_per_lane_breakdown(self):
        records = [
            record_influence(step=0, advice_available=True, advice_used=True, action_lane="hypothesis"),
            record_influence(step=1, advice_available=True, advice_used=False, action_lane="numeric"),
        ]
        s = build_influence_summary(records)
        self.assertIn("hypothesis", s.per_lane)
        self.assertEqual(s.per_lane["hypothesis"]["influenced"], 1)
        self.assertIn("numeric", s.per_lane)
        self.assertEqual(s.per_lane["numeric"]["overridden"], 1)

    def test_to_dict_roundtrip(self):
        records = [record_influence(step=0, advice_available=True, advice_used=True, action_lane="hypothesis")]
        s = build_influence_summary(records)
        d = s.to_dict()
        self.assertIn("influence_rate", d)
        self.assertIn("records", d)
        self.assertEqual(len(d["records"]), 1)


if __name__ == "__main__":
    unittest.main()
