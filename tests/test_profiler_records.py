import unittest

from src.observability.profiler_records import parse_profiler_records, summarize_profiler_events


class TestProfilerRecords(unittest.TestCase):
    def test_parse_and_summarize(self):
        text = """
{"ts_us":1,"rank":0,"op":"all_reduce","bytes":1024,"dur_us":10.0}
2,1,all_gather,2048,20.0
"""
        events = parse_profiler_records(text)
        self.assertEqual(len(events), 2)
        summary = summarize_profiler_events(events)
        payload = summary.to_dict()
        self.assertEqual(payload["event_count"], 2)
        self.assertEqual(payload["bytes_total"], 3072)
        self.assertGreater(payload["p95_dur_us"], 0)


if __name__ == "__main__":
    unittest.main()
