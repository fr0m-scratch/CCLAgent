from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


class ProfilerProtocolError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass
class ProfilerIPCConfig:
    poll_interval_s: float = 0.1
    timeout_s: float = 1.0


class ProfilerIPC:
    """Atomic file IPC for profiler plugin records and summaries."""

    def __init__(self, session_dir: str, config: Optional[ProfilerIPCConfig] = None) -> None:
        self.session_dir = Path(session_dir)
        self.config = config or ProfilerIPCConfig()
        self.events_dir = self.session_dir / "events"
        self.summaries_dir = self.session_dir / "summaries"
        self.events_dir.mkdir(parents=True, exist_ok=True)
        self.summaries_dir.mkdir(parents=True, exist_ok=True)

    def send_event(self, payload: Dict[str, Any], *, event_id: str | None = None) -> str:
        event_id = str(event_id or uuid.uuid4())
        out = dict(payload or {})
        out.setdefault("event_id", event_id)
        out.setdefault("schema_version", "1.0")
        self._atomic_write_json(self.events_dir / f"{event_id}.json", out)
        return event_id

    def read_events(self, *, max_events: int | None = None) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        limit = max_events if max_events is not None and max_events > 0 else None
        for path in self._sorted_json(self.events_dir):
            payload = self._read_and_delete(path)
            if payload is None:
                continue
            records.append(payload)
            if limit is not None and len(records) >= limit:
                break
        return records

    def send_summary(self, payload: Dict[str, Any], *, req_id: str | None = None) -> str:
        req_id = str(req_id or uuid.uuid4())
        out = dict(payload or {})
        out.setdefault("req_id", req_id)
        out.setdefault("schema_version", "1.0")
        self._atomic_write_json(self.summaries_dir / f"{req_id}.json", out)
        return req_id

    def read_summary(self, req_id: str, *, timeout_s: float | None = None) -> Dict[str, Any]:
        deadline = time.time() + (timeout_s if timeout_s is not None else self.config.timeout_s)
        target = self.summaries_dir / f"{str(req_id)}.json"
        while True:
            payload = self._read_and_delete(target)
            if payload is not None:
                return payload
            if time.time() > deadline:
                raise ProfilerProtocolError("timeout", "timed out waiting for profiler summary")
            time.sleep(self.config.poll_interval_s)

    def _atomic_write_json(self, path: Path, payload: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        os.replace(str(tmp), str(path))

    def _read_and_delete(self, path: Path) -> Dict[str, Any] | None:
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            try:
                path.unlink()
            except OSError:
                pass
            raise ProfilerProtocolError("bad_json", f"invalid profiler payload: {exc}") from exc
        try:
            path.unlink()
        except OSError:
            pass
        if not isinstance(payload, dict):
            raise ProfilerProtocolError("bad_json", "profiler payload must be object")
        return payload

    def _sorted_json(self, folder: Path) -> List[Path]:
        try:
            return sorted(folder.glob("*.json"), key=lambda p: p.stat().st_mtime)
        except OSError:
            return []
