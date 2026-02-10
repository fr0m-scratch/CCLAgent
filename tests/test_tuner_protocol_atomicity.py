import tempfile
import unittest
from pathlib import Path

from src.tools.tuner_plugin_protocol import FileTunerProtocol, ProtocolConfig, ProtocolError


class TestTunerProtocolAtomicity(unittest.TestCase):
    def test_versioned_request_response_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            proto = FileTunerProtocol(tmp, ProtocolConfig(poll_interval_s=0.01, timeout_s=0.2))
            req_id = proto.send_request({"type": "CONFIG", "x": 1}, req_id="req-1")
            self.assertEqual(req_id, "req-1")

            request = proto.wait_for_request()
            self.assertEqual(request.get("req_id"), "req-1")
            self.assertEqual(request.get("protocol_version"), "2.0")

            proto.send_response({"type": "ACK"}, req_id=request.get("req_id"))
            response = proto.read_response(req_id="req-1")
            self.assertEqual(response.get("type"), "ACK")
            self.assertEqual(response.get("req_id"), "req-1")
            self.assertEqual(response.get("protocol_version"), "2.0")

    def test_invalid_json_raises_protocol_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            proto = FileTunerProtocol(tmp, ProtocolConfig(poll_interval_s=0.01, timeout_s=0.05))
            request_path = Path(tmp) / "request.json"
            request_path.write_text("{not-json", encoding="utf-8")
            with self.assertRaises(ProtocolError) as ctx:
                proto.wait_for_request(timeout_s=0.05)
            self.assertEqual(ctx.exception.code, "bad_json")


if __name__ == "__main__":
    unittest.main()
