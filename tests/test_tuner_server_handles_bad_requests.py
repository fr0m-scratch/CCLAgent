import tempfile
import unittest

from src.agent.decision_engine import DecisionEngine
from src.plugins.tuner_server import TunerServer
from src.tools.tuner_plugin_protocol import FileTunerProtocol, ProtocolConfig


class TestTunerServerHandlesBadRequests(unittest.TestCase):
    def test_bad_request_returns_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            proto = FileTunerProtocol(tmp, ProtocolConfig(poll_interval_s=0.01, timeout_s=0.1))
            server = TunerServer(protocol=proto, decision_engine=DecisionEngine(rules=[]))

            proto.send_request({"type": "GET_TUNING_DECISION"}, req_id="req-bad")
            handled = server.process_once(timeout_s=0.1)
            self.assertTrue(handled)
            response = proto.read_response(req_id="req-bad", timeout_s=0.1)
            self.assertEqual(response.get("status"), "error")
            reasons = response.get("reasons") or []
            self.assertTrue(any("bad_request" in str(item) for item in reasons))

    def test_good_request_uses_rule_engine(self):
        with tempfile.TemporaryDirectory() as tmp:
            proto = FileTunerProtocol(tmp, ProtocolConfig(poll_interval_s=0.01, timeout_s=0.1))
            engine = DecisionEngine(
                rules=[
                    {
                        "name": "allreduce-medium",
                        "coll_type": "all_reduce",
                        "min_bytes": 128,
                        "max_bytes": 8192,
                        "override": {"algo": "RING", "proto": "LL128"},
                    }
                ]
            )
            server = TunerServer(protocol=proto, decision_engine=engine)

            proto.send_request(
                {
                    "type": "GET_TUNING_DECISION",
                    "coll_type": "all_reduce",
                    "bytes": 4096,
                    "nranks": 8,
                    "topo_sig": "a100_ib",
                },
                req_id="req-good",
            )
            handled = server.process_once(timeout_s=0.1)
            self.assertTrue(handled)
            response = proto.read_response(req_id="req-good", timeout_s=0.1)
            self.assertEqual(response.get("status"), "ok")
            self.assertEqual(response.get("source"), "rule")
            override = response.get("override") or {}
            self.assertEqual(override.get("algo"), "RING")
            self.assertEqual(override.get("proto"), "LL128")


if __name__ == "__main__":
    unittest.main()
