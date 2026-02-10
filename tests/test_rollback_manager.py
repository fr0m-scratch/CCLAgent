import unittest

from src.safety.rollback import RollbackManager
from src.types import NCCLConfig


class TestRollbackManager(unittest.TestCase):
    def test_hard_rollback_uses_last_known_good(self):
        mgr = RollbackManager()
        good = NCCLConfig(params={"NCCL_ALGO": "TREE", "NCCL_PROTO": "LL"})
        mgr.update_success(good)
        decision = mgr.decide(
            failure_mode="hang",
            reason="failure_mode:hang",
            current=NCCLConfig(params={"NCCL_ALGO": "RING"}),
        )
        self.assertTrue(decision.should_rollback)
        self.assertEqual(decision.mode, "hard")
        self.assertEqual(decision.config.params["NCCL_ALGO"], "TREE")

    def test_soft_rollback_reverts_subset_keys(self):
        mgr = RollbackManager()
        mgr.update_success(NCCLConfig(params={"NCCL_ALGO": "TREE", "NCCL_PROTO": "LL", "X": 1}))
        decision = mgr.decide(
            failure_mode="regression",
            reason="failure_mode:regression",
            current=NCCLConfig(params={"NCCL_ALGO": "RING", "NCCL_PROTO": "SIMPLE", "X": 9}),
        )
        self.assertTrue(decision.should_rollback)
        self.assertEqual(decision.mode, "soft")
        self.assertEqual(decision.config.params["NCCL_ALGO"], "TREE")
        self.assertEqual(decision.config.params["X"], 9)
        self.assertIn("NCCL_ALGO", decision.changed_keys)


if __name__ == "__main__":
    unittest.main()
