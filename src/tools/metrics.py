from __future__ import annotations

import json
import re
from typing import Optional

from ..types import METRICS_SCHEMA_VERSION, Metrics, MetricsConfig, RunContext
from ..utils import artifact_path, write_json


_NCCLTESTS_DATA_PREFIX = re.compile(r"^\s*\d+")


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
        rows = self._parse_nccltests_rows(raw_output)
        best = None
        if rows:
            rows.sort(key=lambda item: item["size_bytes"])
            best = rows[-1]
        iteration_time_ms = 0.0
        algbw = None
        busbw = None
        if best is not None:
            time_us = best.get("time_us")
            if isinstance(time_us, float):
                iteration_time_ms = time_us / 1000.0
            algbw = best.get("algbw_gbps")
            busbw = best.get("busbw_gbps")
        metrics = Metrics(
            iteration_time_ms=iteration_time_ms,
            algbw_gbps=algbw,
            busbw_gbps=busbw,
            success=True,
            failure_reason=None,
            raw={
                "raw": raw_output,
                "parse": "nccltests_v1",
                "rows_parsed": len(rows),
                "selected": best,
            },
        )
        if metrics.iteration_time_ms <= 0.0 and not self.config.allow_missing_metrics:
            metrics.success = False
            metrics.failure_reason = "missing_iteration_time"
        return metrics

    def _parse_nccltests_rows(self, raw_output: str) -> list[dict]:
        rows: list[dict] = []
        for line in raw_output.splitlines():
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            if not _NCCLTESTS_DATA_PREFIX.match(text):
                continue
            parts = text.split()
            # Typical nccl-tests row:
            # size count type redop root time algbw busbw [error]
            if len(parts) < 8:
                continue
            try:
                size_bytes = int(parts[0])
            except ValueError:
                continue
            # Parse from the tail to tolerate varying middle columns.
            busbw = _try_float(parts[-2]) if len(parts) >= 2 else None
            algbw = _try_float(parts[-3]) if len(parts) >= 3 else None
            time_us = _try_float(parts[-4]) if len(parts) >= 4 else None
            if time_us is None and len(parts) >= 6:
                # Fallback for atypical formatting.
                time_us = _try_float(parts[5])
            if algbw is None and len(parts) >= 7:
                algbw = _try_float(parts[6])
            if busbw is None and len(parts) >= 8:
                busbw = _try_float(parts[7])
            rows.append(
                {
                    "size_bytes": size_bytes,
                    "time_us": time_us,
                    "algbw_gbps": algbw,
                    "busbw_gbps": busbw,
                    "line": text,
                }
            )
        return rows

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


def _try_float(text: str) -> float | None:
    try:
        return float(text)
    except (TypeError, ValueError):
        return None
