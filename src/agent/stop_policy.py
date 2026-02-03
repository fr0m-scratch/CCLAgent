from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class StopDecision:
    reason: str
    details: dict


class StopPolicy:
    def __init__(self, config) -> None:
        self.config = config

    def evaluate(self, state, step: int) -> Optional[StopDecision]:
        if step >= self.config.budget.max_steps - 1:
            return StopDecision(reason="budget_exhausted", details={})
        if self.config.budget.target_gain and state.best_record and state.history:
            baseline = state.history[0].metrics.iteration_time_ms
            best = state.best_record.metrics.iteration_time_ms
            gain = (baseline - best) / max(1e-9, baseline)
            if gain >= self.config.budget.target_gain and state.plateau_count >= self.config.budget.stable_steps:
                return StopDecision(reason="target_gain", details={"gain": gain})
        if state.should_stop:
            return StopDecision(reason="plateau", details={"plateau_count": state.plateau_count})
        return None
