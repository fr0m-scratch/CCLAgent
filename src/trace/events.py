from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import time
import uuid


SCHEMA_VERSION = "2.0"


@dataclass
class TraceEvent:
    event_id: str
    ts: float
    run_id: str
    phase: str
    step: Optional[int]
    actor: str
    type: str
    payload: Dict[str, Any]
    parent_event_id: Optional[str] = None
    span_id: Optional[str] = None
    refs: List[str] = field(default_factory=list)
    causal_refs: List[str] = field(default_factory=list)
    quality_flags: List[str] = field(default_factory=list)
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
        parent_event_id: Optional[str] = None,
        span_id: Optional[str] = None,
        refs: Optional[List[str]] = None,
        causal_refs: Optional[List[str]] = None,
        quality_flags: Optional[List[str]] = None,
        status: str = "ok",
        duration_ms: Optional[float] = None,
        error: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> "TraceEvent":
        return cls(
            event_id=str(uuid.uuid4()),
            ts=time.time(),
            run_id=run_id,
            phase=phase,
            step=step,
            actor=actor,
            type=type,
            payload=payload,
            parent_event_id=parent_event_id,
            span_id=span_id,
            refs=refs or [],
            causal_refs=causal_refs or [],
            quality_flags=quality_flags or [],
            status=status,
            duration_ms=duration_ms,
            error=error,
            tags=tags,
        )

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "schema_version": self.schema_version,
            "event_id": self.event_id,
            "ts": self.ts,
            "run_id": self.run_id,
            "phase": self.phase,
            "step": self.step,
            "actor": self.actor,
            "type": self.type,
            "payload": self.payload,
            "parent_event_id": self.parent_event_id,
            "span_id": self.span_id,
            "refs": self.refs,
            "causal_refs": self.causal_refs,
            "quality_flags": self.quality_flags,
            "status": self.status,
            "duration_ms": self.duration_ms,
        }
        if self.error:
            data["error"] = self.error
        if self.tags:
            data["tags"] = self.tags
        return data
