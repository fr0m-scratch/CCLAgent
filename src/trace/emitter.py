from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from .events import TraceEvent
from .writer import TraceWriter


class TraceEmitter:
    def emit(self, event: TraceEvent) -> None:
        raise NotImplementedError

    def event(
        self,
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
    ) -> None:
        event = TraceEvent.now(
            run_id=run_id,
            phase=phase,
            step=step,
            actor=actor,
            type=type,
            payload=payload,
            refs=refs,
            status=status,
            duration_ms=duration_ms,
            error=error,
            tags=tags,
        )
        self.emit(event)


class NullTraceEmitter(TraceEmitter):
    def emit(self, event: TraceEvent) -> None:
        return


class TraceEmitterWriter(TraceEmitter):
    def __init__(self, writer: TraceWriter) -> None:
        self.writer = writer

    def emit(self, event: TraceEvent) -> None:
        self.writer.write_event(event)
