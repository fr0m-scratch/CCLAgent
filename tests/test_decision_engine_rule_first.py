import unittest

from src.agent.decision_engine import DecisionEngine, TunerContext


class TestDecisionEngineRuleFirst(unittest.TestCase):
    def test_rule_priority(self):
        engine = DecisionEngine(
            rules=[
                {
                    "name": "allreduce-large",
                    "coll_type": "all_reduce",
                    "min_bytes": 1024,
                    "override": {"algo": "TREE", "proto": "LL", "channels": 8},
                }
            ]
        )
        ctx = TunerContext(
            req_id="r1",
            coll_type="all_reduce",
            bytes=4096,
            nranks=8,
            topo_sig="a100_ib",
        )
        resp = engine.decide_for_collective(ctx)
        self.assertEqual(resp.status, "ok")
        self.assertEqual(resp.source, "rule")
        self.assertEqual(resp.override.get("algo"), "TREE")
        self.assertEqual(resp.override.get("proto"), "LL")
        self.assertEqual(resp.override.get("channels"), 8)

    def test_fallback_without_rule(self):
        engine = DecisionEngine(rules=[])
        ctx = TunerContext(req_id="r2", coll_type="all_reduce", bytes=16, nranks=2)
        resp = engine.decide_for_collective(ctx)
        self.assertEqual(resp.status, "ok")
        self.assertEqual(resp.source, "fallback")
        self.assertEqual(resp.override, {})


if __name__ == "__main__":
    unittest.main()
