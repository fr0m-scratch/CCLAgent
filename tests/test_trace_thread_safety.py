import json
import tempfile
import threading
import unittest

from src.trace import TraceWriter, TraceEvent


class TestTraceThreadSafety(unittest.TestCase):
    def test_concurrent_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            writer = TraceWriter(run_id="test", artifacts_dir=tmp)
            total = 100

            def _write(n):
                for _ in range(n):
                    writer.write_event(
                        TraceEvent.now(
                            run_id="test",
                            phase="online",
                            step=1,
                            actor="agent",
                            type="test",
                            payload={"ok": True},
                        )
                    )

            threads = [threading.Thread(target=_write, args=(total,)) for _ in range(5)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            writer.close()

            # Ensure line count matches expected total events.
            with open(writer.events_path, "r", encoding="utf-8") as handle:
                lines = [line for line in handle if line.strip()]
            self.assertEqual(len(lines), total * 5)
            rows = [json.loads(line) for line in lines]
            event_ids = [row.get("event_id") for row in rows]
            self.assertEqual(len(set(event_ids)), total * 5)
            for row in rows:
                self.assertIsInstance(row.get("event_id"), str)
                self.assertIsInstance(row.get("refs"), list)
                self.assertIsInstance(row.get("causal_refs"), list)
                self.assertIsInstance(row.get("quality_flags"), list)


if __name__ == "__main__":
    unittest.main()
