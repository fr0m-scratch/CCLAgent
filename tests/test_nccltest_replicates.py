import unittest

from src.tools.nccltest import NcclTestConfig, NcclTestRunner
from src.types import NCCLConfig


class TestNcclTestReplicates(unittest.TestCase):
    def test_dry_run_replicates_aggregated(self):
        runner = NcclTestRunner(NcclTestConfig(dry_run=True, replicates=3))
        metrics = runner.run(NCCLConfig(params={}), env={"CCL_REPLICATES": "3", "CCL_SIMULATE_ITERS": "5"})
        self.assertTrue(metrics.success)
        self.assertIn("replicate_count", metrics.raw)
        self.assertEqual(metrics.raw["replicate_count"], 3)
        self.assertEqual(len(metrics.raw["replicate_samples_ms"]), 3)
        self.assertIn("replicate_ci95_ms", metrics.raw)


if __name__ == "__main__":
    unittest.main()
