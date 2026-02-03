from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from .events import TraceEvent


class TraceWriter:
    def __init__(self, run_id: str, artifacts_dir: str) -> None:
        self.run_id = run_id
        self.artifacts_dir = artifacts_dir
        self.trace_dir = os.path.join(artifacts_dir, "trace")
        os.makedirs(self.trace_dir, exist_ok=True)
        self.events_path = os.path.join(self.trace_dir, "events.jsonl")
        self._handle = open(self.events_path, "a", encoding="utf-8")

    def write_event(self, event: TraceEvent) -> None:
        payload = event.to_dict()
        self._handle.write(json.dumps(payload) + "\n")
        self._handle.flush()

    def close(self) -> None:
        try:
            self._handle.close()
        except Exception:
            pass


class JsonlTraceReader:
    def __init__(self, path: str) -> None:
        self.path = path

    def read(self):
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue
        except FileNotFoundError:
            return
