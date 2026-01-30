from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..types import Metrics


@dataclass
class SLAResult:
    ok: bool
    reason: Optional[str] = None


class SLAEnforcer:
    def __init__(self, max_iteration_time: Optional[float] = None, max_errors: int = 0):
        self.max_iteration_time = max_iteration_time
        self.max_errors = max_errors

    def check(self, metrics: Metrics) -> SLAResult:
        if self.max_iteration_time is not None and metrics.iteration_time > self.max_iteration_time:
            return SLAResult(False, "iteration_time_exceeded")
        if metrics.errors > self.max_errors:
            return SLAResult(False, "error_budget_exceeded")
        return SLAResult(True)
