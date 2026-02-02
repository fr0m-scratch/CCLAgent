import unittest

from src.safety.risk import RiskScorer
from src.types import NCCLConfig, SafetyConfig


class RiskScoreTest(unittest.TestCase):
    def test_risk_score_high(self):
        scorer = RiskScorer(SafetyConfig(max_channels_safe=8))
        cfg = NCCLConfig(params={"NCCL_MAX_NCHANNELS": 64})
        score = scorer.score(cfg)
        self.assertTrue(score.risk_score > 0.3)


if __name__ == "__main__":
    unittest.main()
