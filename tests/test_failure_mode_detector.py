import unittest

from src.observability.failure_modes import FailureModeDetector
from src.types import Metrics


class TestFailureModeDetector(unittest.TestCase):
    def test_detect_hang_from_failure_reason(self):
        detector = FailureModeDetector()
        metrics = Metrics(iteration_time_ms=float("inf"), success=False, failure_reason="Timed out waiting for NCCL")
        signal = detector.detect(metrics=metrics)
        self.assertIsNotNone(signal)
        self.assertEqual(signal.kind, "hang")
        self.assertEqual(detector.policy_lane(signal), "debug")

    def test_detect_regression_on_success(self):
        detector = FailureModeDetector(regression_threshold=0.1)
        metrics = Metrics(iteration_time_ms=1200.0, success=True)
        signal = detector.detect(metrics=metrics, baseline_ms=1000.0)
        self.assertIsNotNone(signal)
        self.assertEqual(signal.kind, "regression")
        self.assertEqual(detector.policy_lane(signal), "cautious")


if __name__ == "__main__":
    unittest.main()
