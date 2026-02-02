from __future__ import annotations

from typing import Any, Optional

from ..types import Metrics, NCCLConfig, TuningAction, TuningRecord, WorkloadSpec
from ..tools.tuner_plugin_protocol import FileTunerProtocol
from .state import TuningState


class ExtTunerSession:
    """Session adapter for NCCL ext-tuner style integrations."""

    def __init__(self, agent: Any, workload: WorkloadSpec) -> None:
        self.agent = agent
        self.workload = workload
        self.context = agent.planner.build_context(workload)
        self.microbench = agent.planner.offline_plan(workload)
        self.plan = agent.planner.build_initial_plan(workload, self.microbench, self.context)
        self.state = TuningState(budget=agent.config.budget)
        self.current_config = self.plan.baseline_config

    def initial_config(self) -> NCCLConfig:
        return self.current_config

    def get_candidate(self) -> NCCLConfig:
        return self.current_config

    def update(self, metrics: Metrics) -> Optional[NCCLConfig]:
        action = TuningAction(
            kind="external", config=self.current_config, rationale="external metrics"
        )
        record = TuningRecord(step=len(self.state.history), action=action, metrics=metrics)
        self.state.record(record)
        if metrics.raw.get("sla_ok") is False or self.state.should_stop:
            return None
        decision = self.agent.analyzer.plan_next_action(
            state=self.state,
            last_metrics=metrics,
            microbench=self.microbench,
            context=self.context,
            step=len(self.state.history),
            plan=self.plan,
            workload=self.workload,
            base_config=self.current_config,
        )
        if decision is None or getattr(decision, "kind", "") == "stop":
            return None
        if getattr(decision, "kind", "") == "rollback":
            self.current_config = decision.config
            return self.current_config
        self.current_config = decision.config
        return self.current_config

    def report_metrics(self, metrics: Metrics) -> Optional[NCCLConfig]:
        return self.update(metrics)


class ExtTunerServer:
    def __init__(self, agent: Any, workload: WorkloadSpec, session_dir: str) -> None:
        self.session = ExtTunerSession(agent, workload)
        self.protocol = FileTunerProtocol(session_dir)

    def serve(self) -> None:
        current = self.session.initial_config()
        while True:
            request = self.protocol.wait_for_request()
            if request.get("type") == "GET_CONFIG":
                self.protocol.send_response({"type": "CONFIG", "config": current.params})
                continue
            if request.get("type") == "REPORT_METRICS":
                metrics = Metrics(**request.get("metrics", {}))
                next_cfg = self.session.report_metrics(metrics)
                if next_cfg is None:
                    self.protocol.send_response({"type": "STOP"})
                    break
                current = next_cfg
                self.protocol.send_response({"type": "CONFIG", "config": current.params})
                continue
