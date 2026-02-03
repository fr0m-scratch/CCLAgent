from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from .emitter import TraceEmitter


class TraceSpan:
    def __init__(
        self,
        emitter: TraceEmitter,
        *,
        run_id: str,
        phase: str,
        step: Optional[int],
        actor: str,
        type: str,
        payload: Dict[str, Any],
        refs: Optional[List[str]] = None,
    ) -> None:
        self.emitter = emitter
        self.run_id = run_id
        self.phase = phase
        self.step = step
        self.actor = actor
        self.type = type
        self.payload = payload
        self.refs = refs or []
        self._start = None

    def __enter__(self):
        self._start = time.time()
        return self

    def __exit__(self, exc_type, exc, tb):
        duration_ms = None
        if self._start is not None:
            duration_ms = (time.time() - self._start) * 1000.0
        status = "ok"
        error = None
        if exc is not None:
            status = "error"
            error = str(exc)
        self.emitter.event(
            run_id=self.run_id,
            phase=self.phase,
            step=self.step,
            actor=self.actor,
            type=self.type,
            payload=self.payload,
            refs=self.refs,
            status=status,
            duration_ms=duration_ms,
            error=error,
        )
        return False
