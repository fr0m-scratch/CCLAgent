from __future__ import annotations

import json
import re
from typing import Optional

from ..types import METRICS_SCHEMA_VERSION, Metrics, MetricsConfig, RunContext
from ..utils import artifact_path, write_json


_NCCLTESTS_RE = re.compile(r"\s+([0-9.]+)\s+([0-9.]+)\s*$")


class MetricsCollector:
    def __init__(self, config: MetricsConfig, run_context: RunContext | None = None) -> None:
        self.config = config
        self.run_context = run_context

    def parse(self, raw_output: str, parse_mode: Optional[str] = None) -> Metrics:
        mode = parse_mode or self.config.parse_mode
        if mode in ("json_stdout_v1", "autoccl_demo_v1"):
            return self._parse_json(raw_output)
        if mode == "nccltests_v1":
            return self._parse_nccltests(raw_output)
        return self._parse_json(raw_output)

    def _parse_json(self, raw_output: str) -> Metrics:
        invalid_json = False
        try:
            payload = json.loads(raw_output)
        except json.JSONDecodeError:
            invalid_json = True
            payload = {}

        iteration_time_ms = payload.get("iteration_time_ms")
        if iteration_time_ms is None and "iteration_time" in payload:
            iteration_time_ms = float(payload.get("iteration_time", 0.0)) * 1000.0
        if iteration_time_ms is None and "iter_time_ms" in payload:
            iteration_time_ms = payload.get("iter_time_ms")

        metrics = Metrics(
            iteration_time_ms=float(iteration_time_ms or 0.0),
            throughput=payload.get("throughput"),
            comm_time_ms=payload.get("comm_time_ms"),
            busbw_gbps=payload.get("busbw_gbps") or payload.get("busbw"),
            algbw_gbps=payload.get("algbw_gbps") or payload.get("algbw"),
            loss=payload.get("loss"),
            error_budget=payload.get("error_budget"),
            success=True,
            failure_reason=None,
            raw=payload,
            schema_version=payload.get("schema_version", METRICS_SCHEMA_VERSION),
        )

        if invalid_json and not self.config.allow_missing_metrics:
            metrics.success = False
            metrics.failure_reason = "invalid_json"
        elif metrics.iteration_time_ms <= 0.0 and not self.config.allow_missing_metrics:
            metrics.success = False
            metrics.failure_reason = "missing_iteration_time"
        return metrics

    def _parse_nccltests(self, raw_output: str) -> Metrics:
        lines = [line.strip() for line in raw_output.splitlines() if line.strip()]
        algbw = None
        busbw = None
        for line in reversed(lines):
            match = _NCCLTESTS_RE.search(line)
            if not match:
                continue
            try:
                algbw = float(match.group(1))
                busbw = float(match.group(2))
                break
            except ValueError:
                continue
        iteration_time_ms = 0.0
        metrics = Metrics(
            iteration_time_ms=iteration_time_ms,
            algbw_gbps=algbw,
            busbw_gbps=busbw,
            success=True,
            failure_reason=None,
            raw={"raw": raw_output, "parse": "nccltests_v1"},
        )
        if metrics.iteration_time_ms <= 0.0 and not self.config.allow_missing_metrics:
            metrics.success = False
            metrics.failure_reason = "missing_iteration_time"
        return metrics

    def save_metrics(self, metrics: Metrics, step: int) -> None:
        if not self.run_context:
            return
        path = artifact_path(self.run_context, "steps", f"step_{step}_metrics.json")
        write_json(path, metrics.__dict__)

    def from_file(self, path: str) -> Optional[Metrics]:
        try:
            with open(path, "r", encoding="utf-8") as handle:
                raw = handle.read()
        except FileNotFoundError:
            return None
        return self.parse(raw)
