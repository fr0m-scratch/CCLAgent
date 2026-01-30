from __future__ import annotations

from typing import Any, Optional

from ..types import NCCLConfig, TuningAction, TuningRecord, WorkloadSpec
from .state import TuningState


class ExtTunerSession:
    """Session adapter for NCCL ext-tuner style integrations."""
    def __init__(self, agent: Any, workload: WorkloadSpec) -> None:
        self.agent = agent
        self.workload = workload
        self.context = agent.planner.build_context(workload)
        self.microbench = agent.planner.offline_plan(workload)
        self.state = TuningState(budget=agent.config.budget)
        self.current_config = agent.planner.propose_initial_config(
            workload, self.microbench, self.context
        )

    def initial_config(self) -> NCCLConfig:
        return self.current_config

    def get_candidate(self) -> NCCLConfig:
        return self.current_config

    def update(self, metrics: Any) -> Optional[NCCLConfig]:
        action = TuningAction(
            kind="external", config=self.current_config, rationale="external metrics"
        )
        self.state.record(TuningRecord(action=action, metrics=metrics))
        if metrics.extras.get("sla_ok") is False or self.state.should_stop:
            return None
        next_action = self.agent.policy.decide_next_action(
            self.state, self.microbench, self.context, len(self.state.history)
        )
        if next_action is None:
            return None
        self.current_config = next_action.config
        return self.current_config

    def report_metrics(self, metrics: Any) -> Optional[NCCLConfig]:
        return self.update(metrics)
