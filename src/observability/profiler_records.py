from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
import json


@dataclass
class ProfilerEvent:
    ts_us: int
    rank: int
    op: str
    bytes: int
    dur_us: float
    stream: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "ts_us": int(self.ts_us),
            "rank": int(self.rank),
            "op": str(self.op),
            "bytes": int(self.bytes),
            "dur_us": float(self.dur_us),
        }
        if self.stream:
            payload["stream"] = str(self.stream)
        if self.extra:
            payload["extra"] = dict(self.extra)
        return payload


@dataclass
class ProfilerSummary:
    event_count: int
    total_dur_us: float
    p50_dur_us: float
    p95_dur_us: float
    p99_dur_us: float
    bytes_total: int
    by_op: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": "1.0",
            "event_count": int(self.event_count),
            "total_dur_us": float(self.total_dur_us),
            "p50_dur_us": float(self.p50_dur_us),
            "p95_dur_us": float(self.p95_dur_us),
            "p99_dur_us": float(self.p99_dur_us),
            "bytes_total": int(self.bytes_total),
            "by_op": dict(self.by_op),
        }


def parse_profiler_line(text: str) -> Optional[ProfilerEvent]:
    line = (text or "").strip()
    if not line:
        return None

    # JSON-lines is the primary format.
    try:
        payload = json.loads(line)
    except json.JSONDecodeError:
        payload = None

    if isinstance(payload, dict):
        return _event_from_payload(payload)

    # CSV fallback: ts_us,rank,op,bytes,dur_us[,stream]
    parts = [item.strip() for item in line.split(",")]
    if len(parts) < 5:
        return None
    try:
        ts_us = int(parts[0])
        rank = int(parts[1])
        op = str(parts[2])
        bytes_value = int(parts[3])
        dur_us = float(parts[4])
    except (TypeError, ValueError):
        return None
    stream = parts[5] if len(parts) >= 6 and parts[5] else None
    return ProfilerEvent(
        ts_us=ts_us,
        rank=rank,
        op=op,
        bytes=bytes_value,
        dur_us=dur_us,
        stream=stream,
    )


def parse_profiler_records(text: str) -> List[ProfilerEvent]:
    events: List[ProfilerEvent] = []
    for line in (text or "").splitlines():
        event = parse_profiler_line(line)
        if event is None:
            continue
        events.append(event)
    return events


def parse_profiler_file(path: str | Path) -> List[ProfilerEvent]:
    path = Path(path)
    if not path.exists():
        return []
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return []
    return parse_profiler_records(text)


def summarize_profiler_events(events: Iterable[ProfilerEvent]) -> ProfilerSummary:
    event_list = list(events)
    if not event_list:
        return ProfilerSummary(
            event_count=0,
            total_dur_us=0.0,
            p50_dur_us=0.0,
            p95_dur_us=0.0,
            p99_dur_us=0.0,
            bytes_total=0,
            by_op={},
        )

    durations = sorted(float(item.dur_us) for item in event_list)
    p50 = _percentile(durations, 50)
    p95 = _percentile(durations, 95)
    p99 = _percentile(durations, 99)
    total = sum(durations)
    bytes_total = sum(int(item.bytes) for item in event_list)

    by_op: Dict[str, Dict[str, float]] = {}
    for event in event_list:
        key = str(event.op or "unknown")
        bucket = by_op.setdefault(key, {"count": 0.0, "dur_us": 0.0, "bytes": 0.0})
        bucket["count"] += 1.0
        bucket["dur_us"] += float(event.dur_us)
        bucket["bytes"] += float(event.bytes)

    for op_name, bucket in by_op.items():
        count = max(1.0, bucket["count"])
        bucket["mean_dur_us"] = bucket["dur_us"] / count
        bucket["mean_bytes"] = bucket["bytes"] / count

    return ProfilerSummary(
        event_count=len(event_list),
        total_dur_us=total,
        p50_dur_us=p50,
        p95_dur_us=p95,
        p99_dur_us=p99,
        bytes_total=bytes_total,
        by_op=by_op,
    )


def _event_from_payload(payload: Dict[str, Any]) -> Optional[ProfilerEvent]:
    try:
        ts_us = int(payload.get("ts_us") or payload.get("timestamp_us") or 0)
        rank = int(payload.get("rank") or 0)
        op = str(payload.get("op") or payload.get("collective") or "unknown")
        bytes_value = int(payload.get("bytes") or payload.get("nbytes") or 0)
        dur_us = float(payload.get("dur_us") or payload.get("duration_us") or 0.0)
    except (TypeError, ValueError):
        return None
    stream = payload.get("stream")
    extra = payload.get("extra") if isinstance(payload.get("extra"), dict) else {}
    return ProfilerEvent(
        ts_us=ts_us,
        rank=rank,
        op=op,
        bytes=bytes_value,
        dur_us=dur_us,
        stream=str(stream) if stream else None,
        extra=extra,
    )


def _percentile(values: List[float], q: int) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    # Inclusive percentile index to keep behavior deterministic without numpy.
    idx = (len(values) - 1) * (float(q) / 100.0)
    lo = int(idx)
    hi = min(len(values) - 1, lo + 1)
    if lo == hi:
        return float(values[lo])
    frac = idx - lo
    return float(values[lo] * (1.0 - frac) + values[hi] * frac)
