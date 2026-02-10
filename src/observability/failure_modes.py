from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..types import Metrics


HANG_HINTS = (
    "watchdog",
    "timed out",
    "timeout waiting",
    "stuck",
    "deadlock",
)

CRASH_HINTS = (
    "segmentation fault",
    "sigsegv",
    "abort",
    "core dumped",
    "unhandled system error",
)

MISMATCH_HINTS = (
    "mismatch",
    "invalid rank",
    "wrong rank",
    "size mismatch",
)


@dataclass
class FailureSignal:
    kind: str
    severity: str
    confidence: float
    reasons: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": "1.0",
            "kind": self.kind,
            "severity": self.severity,
            "confidence": float(self.confidence),
            "reasons": list(self.reasons),
            "metadata": dict(self.metadata),
        }


class FailureModeDetector:
    def __init__(self, regression_threshold: float = 0.1) -> None:
        self.regression_threshold = max(0.0, float(regression_threshold))

    def detect(
        self,
        *,
        metrics: Metrics,
        stdout_text: str = "",
        stderr_text: str = "",
        baseline_ms: float | None = None,
    ) -> Optional[FailureSignal]:
        reasons: List[str] = []
        logs = "\n".join([stdout_text or "", stderr_text or ""]).lower()

        if not metrics.success:
            failure_reason = str(metrics.failure_reason or metrics.raw.get("error") or "").lower()
            text = (failure_reason + "\n" + logs).strip()
            if _has_any(text, HANG_HINTS):
                reasons.append("timeout_or_watchdog")
                return FailureSignal(
                    kind="hang",
                    severity="high",
                    confidence=0.9,
                    reasons=reasons,
                    metadata={"failure_reason": metrics.failure_reason},
                )
            if _has_any(text, CRASH_HINTS):
                reasons.append("crash_signature")
                return FailureSignal(
                    kind="crash",
                    severity="high",
                    confidence=0.9,
                    reasons=reasons,
                    metadata={"failure_reason": metrics.failure_reason},
                )
            if _has_any(text, MISMATCH_HINTS):
                reasons.append("rank_or_shape_mismatch")
                return FailureSignal(
                    kind="rank_mismatch",
                    severity="high",
                    confidence=0.85,
                    reasons=reasons,
                    metadata={"failure_reason": metrics.failure_reason},
                )
            return FailureSignal(
                kind="failure",
                severity="high",
                confidence=0.6,
                reasons=["generic_failure"],
                metadata={"failure_reason": metrics.failure_reason},
            )

        # Successful step but high-risk signals in logs.
        if _has_any(logs, MISMATCH_HINTS):
            return FailureSignal(
                kind="rank_mismatch",
                severity="med",
                confidence=0.7,
                reasons=["mismatch_signature_in_logs"],
            )

        iter_ms = float(metrics.iteration_time_ms or 0.0)
        if baseline_ms and baseline_ms > 0 and iter_ms > 0:
            ratio = (iter_ms - baseline_ms) / baseline_ms
            if ratio >= self.regression_threshold:
                return FailureSignal(
                    kind="regression",
                    severity="med",
                    confidence=0.8,
                    reasons=[f"iteration_time_regressed:{ratio:.3f}"],
                    metadata={"baseline_ms": baseline_ms, "iteration_time_ms": iter_ms, "ratio": ratio},
                )

        if _has_any(logs, HANG_HINTS):
            return FailureSignal(
                kind="hang",
                severity="med",
                confidence=0.6,
                reasons=["hang_signature_in_logs"],
            )

        return None

    def policy_lane(self, signal: Optional[FailureSignal]) -> str:
        if signal is None:
            return "numeric"
        if signal.kind in ("hang", "crash", "rank_mismatch"):
            return "debug"
        if signal.kind in ("regression", "failure"):
            return "cautious"
        return "numeric"


def _has_any(text: str, terms: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in terms)
