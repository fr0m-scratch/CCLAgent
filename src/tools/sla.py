from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from ..types import Metrics


@dataclass
class SLAResult:
    ok: bool
    violations: List[str] = field(default_factory=list)
    severity: str = "soft"
    rollback_recommended: bool = False


class SLAEnforcer:
    def __init__(self, max_iteration_time: Optional[float] = None, max_errors: int = 0):
        self.max_iteration_time = max_iteration_time
        self.max_errors = max_errors

    def check(self, metrics: Metrics) -> SLAResult:
        violations: List[str] = []
        severity = "soft"
        rollback = False
        if not metrics.success:
            violations.append("metrics_failed")
            severity = "hard"
            rollback = True
        if self.max_iteration_time is not None and metrics.iteration_time_ms > self.max_iteration_time:
            violations.append("iteration_time_exceeded")
            if metrics.iteration_time_ms > self.max_iteration_time * 1.2:
                severity = "hard"
                rollback = True
        if metrics.error_budget is not None and metrics.error_budget > self.max_errors:
            violations.append("error_budget_exceeded")
            severity = "hard"
            rollback = True
        return SLAResult(ok=not violations, violations=violations, severity=severity, rollback_recommended=rollback)
