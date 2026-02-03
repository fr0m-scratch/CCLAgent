from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import time


SCHEMA_VERSION = "1.0"


@dataclass
class TraceEvent:
    ts: float
    run_id: str
    phase: str
    step: Optional[int]
    actor: str
    type: str
    payload: Dict[str, Any]
    refs: List[str] = field(default_factory=list)
    status: str = "ok"
    duration_ms: Optional[float] = None
    error: Optional[str] = None
    tags: Optional[List[str]] = None
    schema_version: str = SCHEMA_VERSION

    @classmethod
    def now(
        cls,
        *,
        run_id: str,
        phase: str,
        step: Optional[int],
        actor: str,
        type: str,
        payload: Dict[str, Any],
        refs: Optional[List[str]] = None,
        status: str = "ok",
        duration_ms: Optional[float] = None,
        error: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> "TraceEvent":
        return cls(
            ts=time.time(),
            run_id=run_id,
            phase=phase,
            step=step,
            actor=actor,
            type=type,
            payload=payload,
            refs=refs or [],
            status=status,
            duration_ms=duration_ms,
            error=error,
            tags=tags,
        )

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "schema_version": self.schema_version,
            "ts": self.ts,
            "run_id": self.run_id,
            "phase": self.phase,
            "step": self.step,
            "actor": self.actor,
            "type": self.type,
            "payload": self.payload,
            "refs": self.refs,
            "status": self.status,
            "duration_ms": self.duration_ms,
        }
        if self.error:
            data["error"] = self.error
        if self.tags:
            data["tags"] = self.tags
        return data
