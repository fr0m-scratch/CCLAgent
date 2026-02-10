from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..observability.profiler_records import (
    ProfilerEvent,
    parse_profiler_file,
    parse_profiler_records,
    summarize_profiler_events,
)
from ..plugins.profiler_ipc import ProfilerIPC, ProfilerIPCConfig
from ..types import RunContext
from ..utils import artifact_path, write_json


@dataclass
class ProfilerConfig:
    enabled: bool = False
    session_dir: str = "plugins/profiler"
    dry_run: bool = True
    poll_interval_s: float = 0.1
    timeout_s: float = 0.5


class ProfilerCollector:
    def __init__(self, config: ProfilerConfig, run_context: Optional[RunContext] = None) -> None:
        self.config = config
        self.run_context = run_context
        self._ipc: Optional[ProfilerIPC] = None
        if self.config.enabled:
            self._ipc = ProfilerIPC(
                self.config.session_dir,
                ProfilerIPCConfig(
                    poll_interval_s=self.config.poll_interval_s,
                    timeout_s=self.config.timeout_s,
                ),
            )

    def env_overrides(self) -> Dict[str, str]:
        if not self.config.enabled:
            return {}
        return {
            "CCL_PROFILER_ENABLED": "1",
            "CCL_PROFILER_SESSION_DIR": self.config.session_dir,
        }

    def collect_step(self, step: int, *, artifact_subdir: str = "steps") -> Dict[str, Any]:
        events: List[ProfilerEvent] = []
        source = "none"

        if self.config.enabled and self._ipc is not None:
            payloads = self._ipc.read_events()
            for item in payloads:
                event = _event_from_payload(item)
                if event is not None:
                    events.append(event)
            if events:
                source = "plugin_ipc"

        if not events and self.config.dry_run:
            source = "simulated"
            events = _simulated_events(step)

        summary = summarize_profiler_events(events)
        payload = {
            "schema_version": "1.0",
            "step": int(step),
            "source": source,
            "summary": summary.to_dict(),
            "events": [item.to_dict() for item in events],
        }
        self._persist(step, payload, artifact_subdir=artifact_subdir)
        return payload

    def collect_from_file(self, path: str | Path, *, step: int, artifact_subdir: str = "steps") -> Dict[str, Any]:
        events = parse_profiler_file(path)
        summary = summarize_profiler_events(events)
        payload = {
            "schema_version": "1.0",
            "step": int(step),
            "source": "file",
            "source_path": str(path),
            "summary": summary.to_dict(),
            "events": [item.to_dict() for item in events],
        }
        self._persist(step, payload, artifact_subdir=artifact_subdir)
        return payload

    def collect_from_text(self, text: str, *, step: int, artifact_subdir: str = "steps") -> Dict[str, Any]:
        events = parse_profiler_records(text)
        summary = summarize_profiler_events(events)
        payload = {
            "schema_version": "1.0",
            "step": int(step),
            "source": "text",
            "summary": summary.to_dict(),
            "events": [item.to_dict() for item in events],
        }
        self._persist(step, payload, artifact_subdir=artifact_subdir)
        return payload

    def _persist(self, step: int, payload: Dict[str, Any], *, artifact_subdir: str) -> None:
        if self.run_context is None:
            return
        base = artifact_path(self.run_context, artifact_subdir)
        Path(base).mkdir(parents=True, exist_ok=True)
        write_json(artifact_path(self.run_context, artifact_subdir, f"step_{step}_profiler_summary.json"), payload)

        events_path = Path(artifact_path(self.run_context, artifact_subdir, f"step_{step}_profiler_events.jsonl"))
        with events_path.open("w", encoding="utf-8") as handle:
            for item in payload.get("events", []):
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def _simulated_events(step: int) -> List[ProfilerEvent]:
    base_ts = int(time.time() * 1_000_000)
    ops = ["all_reduce", "all_reduce", "all_gather", "reduce_scatter"]
    events: List[ProfilerEvent] = []
    for idx, op_name in enumerate(ops):
        events.append(
            ProfilerEvent(
                ts_us=base_ts + idx * 1000,
                rank=idx % 8,
                op=op_name,
                bytes=1 << (20 + idx),
                dur_us=float(80 + idx * 15 + step),
                stream="sim",
                extra={"simulated": True},
            )
        )
    return events


def _event_from_payload(payload: Dict[str, Any]) -> Optional[ProfilerEvent]:
    try:
        return ProfilerEvent(
            ts_us=int(payload.get("ts_us") or payload.get("timestamp_us") or 0),
            rank=int(payload.get("rank") or 0),
            op=str(payload.get("op") or payload.get("collective") or "unknown"),
            bytes=int(payload.get("bytes") or payload.get("nbytes") or 0),
            dur_us=float(payload.get("dur_us") or payload.get("duration_us") or 0.0),
            stream=str(payload.get("stream")) if payload.get("stream") else None,
            extra=payload.get("extra") if isinstance(payload.get("extra"), dict) else {},
        )
    except (TypeError, ValueError):
        return None
