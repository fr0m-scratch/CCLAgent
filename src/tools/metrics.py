from __future__ import annotations

import json
from typing import Optional

from ..types import Metrics


class MetricsCollector:
    def parse(self, raw_output: str) -> Metrics:
        try:
            payload = json.loads(raw_output)
        except json.JSONDecodeError:
            payload = {}
        return Metrics(
            iteration_time=payload.get("iteration_time", float("inf")),
            comm_time=payload.get("comm_time"),
            bandwidth=payload.get("bandwidth"),
            errors=payload.get("errors", 0),
            extras=payload,
        )

    def from_file(self, path: str) -> Optional[Metrics]:
        try:
            with open(path, "r", encoding="utf-8") as handle:
                raw = handle.read()
        except FileNotFoundError:
            return None
        return self.parse(raw)
