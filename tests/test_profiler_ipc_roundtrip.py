import json
import tempfile
import unittest
from pathlib import Path

from src.plugins.profiler_ipc import ProfilerIPC, ProfilerIPCConfig, ProfilerProtocolError


class TestProfilerIPCRoundtrip(unittest.TestCase):
    def test_event_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            ipc = ProfilerIPC(tmp, ProfilerIPCConfig(poll_interval_s=0.01, timeout_s=0.1))
            event_id = ipc.send_event({"op": "all_reduce", "dur_us": 10.0}, event_id="evt-1")
            self.assertEqual(event_id, "evt-1")
            events = ipc.read_events()
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0].get("event_id"), "evt-1")
            self.assertEqual(events[0].get("op"), "all_reduce")

    def test_summary_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            ipc = ProfilerIPC(tmp, ProfilerIPCConfig(poll_interval_s=0.01, timeout_s=0.1))
            req_id = ipc.send_summary({"summary": {"event_count": 3}}, req_id="sum-1")
            self.assertEqual(req_id, "sum-1")
            payload = ipc.read_summary("sum-1", timeout_s=0.1)
            self.assertEqual(payload.get("req_id"), "sum-1")
            self.assertEqual(payload.get("summary", {}).get("event_count"), 3)

    def test_bad_json_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            ipc = ProfilerIPC(tmp, ProfilerIPCConfig(poll_interval_s=0.01, timeout_s=0.1))
            events_dir = Path(tmp) / "events"
            events_dir.mkdir(parents=True, exist_ok=True)
            (events_dir / "bad.json").write_text("{bad", encoding="utf-8")
            with self.assertRaises(ProfilerProtocolError) as ctx:
                ipc.read_events()
            self.assertEqual(ctx.exception.code, "bad_json")


if __name__ == "__main__":
    unittest.main()
