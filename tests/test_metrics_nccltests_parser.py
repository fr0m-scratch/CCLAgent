import unittest

from src.tools.metrics import MetricsCollector
from src.types import MetricsConfig


class TestNcclTestsParser(unittest.TestCase):
    def test_parses_largest_size_row(self):
        raw = """
# size count type redop root time algbw busbw error
8 2 float sum -1 5.00 0.10 0.08 0
1024 256 float sum -1 12.50 22.30 18.70 0
"""
        collector = MetricsCollector(MetricsConfig(parse_mode="nccltests_v1", allow_missing_metrics=False))
        metrics = collector.parse(raw, parse_mode="nccltests_v1")

        self.assertTrue(metrics.success)
        self.assertAlmostEqual(metrics.iteration_time_ms, 0.0125)
        self.assertAlmostEqual(metrics.algbw_gbps or 0.0, 22.30)
        self.assertAlmostEqual(metrics.busbw_gbps or 0.0, 18.70)
        self.assertEqual(metrics.raw.get("rows_parsed"), 2)
        selected = metrics.raw.get("selected")
        self.assertIsInstance(selected, dict)
        self.assertEqual(selected.get("size_bytes"), 1024)

    def test_invalid_output_marks_missing_iteration(self):
        raw = "no table rows"
        collector = MetricsCollector(MetricsConfig(parse_mode="nccltests_v1", allow_missing_metrics=False))
        metrics = collector.parse(raw, parse_mode="nccltests_v1")

        self.assertFalse(metrics.success)
        self.assertEqual(metrics.failure_reason, "missing_iteration_time")


if __name__ == "__main__":
    unittest.main()
