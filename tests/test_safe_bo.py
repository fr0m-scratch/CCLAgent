import unittest

from src.search.safe_bo import SafeBO, SafeBOConfig
from src.types import NCCLConfig, SearchCandidate


class TestSafeBO(unittest.TestCase):
    def test_risk_penalty_reorders_candidates(self):
        bo = SafeBO(SafeBOConfig(beta=1.0, risk_threshold=0.6, risk_penalty=100.0))
        low_risk = SearchCandidate(
            config=NCCLConfig(params={"A": 1}),
            predicted_time_ms=100.0,
            uncertainty=2.0,
            rationale="",
            candidate_id="c1",
            risk_score=0.2,
        )
        high_risk = SearchCandidate(
            config=NCCLConfig(params={"A": 2}),
            predicted_time_ms=90.0,
            uncertainty=1.0,
            rationale="",
            candidate_id="c2",
            risk_score=0.9,
        )
        ranked = bo.rank([high_risk, low_risk])
        self.assertEqual(ranked[0].candidate_id, "c1")


if __name__ == "__main__":
    unittest.main()
