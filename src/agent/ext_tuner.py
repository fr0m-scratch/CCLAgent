from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from ..agent.decision_engine import DecisionEngine
from ..plugins.tuner_server import TunerServer
from ..tools.tuner_plugin_protocol import FileTunerProtocol, ProtocolConfig
from ..types import Metrics, NCCLConfig, TuningAction, TuningRecord, WorkloadSpec
from ..utils import artifact_path, write_json
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
        self.agent = agent
        self.session = ExtTunerSession(agent, workload)
        timeout_s = getattr(getattr(agent.config, "plugins", None), "protocol_timeout_s", None)
        self.protocol = FileTunerProtocol(
            session_dir,
            ProtocolConfig(timeout_s=timeout_s if isinstance(timeout_s, (int, float)) and timeout_s > 0 else None),
        )
        tuner_rules = self._build_rule_table()
        self.tuner_server = TunerServer(protocol=self.protocol, decision_engine=DecisionEngine(rules=tuner_rules))
        self.session_dir = session_dir
        self._event_seq = 0
        self.session_artifact_dir = None
        if getattr(agent, "run_context", None) is not None:
            self.session_artifact_dir = artifact_path(agent.run_context, "plugins", "session")
            Path(self.session_artifact_dir).mkdir(parents=True, exist_ok=True)

    def serve(self) -> None:
        current = self.session.initial_config()
        while True:
            request = self.protocol.wait_for_request()
            req_type = str(request.get("type") or "GET_TUNING_DECISION")
            req_id = str(request.get("req_id") or "")
            if req_type == "GET_CONFIG":
                self.protocol.send_response(
                    {
                        "type": "CONFIG",
                        "status": "ok",
                        "config": current.params,
                        "fallback_used": True,
                    },
                    req_id=req_id or None,
                )
                self._persist_session_event("get_config", request=request, response={"config": current.params})
                continue
            if req_type == "REPORT_METRICS":
                metrics_payload = request.get("metrics") if isinstance(request.get("metrics"), dict) else {}
                metrics = Metrics(**metrics_payload)
                next_cfg = self.session.report_metrics(metrics)
                if next_cfg is None:
                    self.protocol.send_response({"type": "STOP", "status": "ok"}, req_id=req_id or None)
                    self._persist_session_event("stop", request=request, response={"type": "STOP"})
                    break
                current = next_cfg
                self.protocol.send_response(
                    {
                        "type": "CONFIG",
                        "status": "ok",
                        "config": current.params,
                        "fallback_used": True,
                    },
                    req_id=req_id or None,
                )
                self._persist_session_event("report_metrics", request=request, response={"config": current.params})
                continue
            if req_type == "GET_TUNING_DECISION":
                # Delegate structured decision responses to the new tuner server.
                response = self.tuner_server._handle_request(request)
                if response.status == "ok" and response.override:
                    params = dict(current.params)
                    params.update(response.override)
                    current = NCCLConfig(params=params, metadata={"source": "tuner_override"})
                payload = response.to_dict()
                payload["type"] = "TUNING_DECISION"
                payload["config"] = current.params
                self.protocol.send_response(payload, req_id=req_id or response.req_id)
                self._persist_session_event("get_tuning_decision", request=request, response=payload)
                continue

            # Unknown request type: protocol-safe error response.
            err = {
                "type": "ERROR",
                "status": "error",
                "reasons": [f"bad_request_type:{req_type}"],
                "fallback_used": True,
                "config": current.params,
            }
            self.protocol.send_response(err, req_id=req_id or None)
            self._persist_session_event("bad_request", request=request, response=err)

    def _persist_session_event(self, kind: str, *, request: dict, response: dict) -> None:
        if self.session_artifact_dir is None:
            return
        self._event_seq += 1
        idx = int(self.session.state.history[-1].step) if self.session.state.history else 0
        path = Path(self.session_artifact_dir) / f"event_{self._event_seq:06d}_{kind}_step{idx}.json"
        payload = {
            "schema_version": "1.0",
            "kind": kind,
            "sequence": self._event_seq,
            "request": request,
            "response": response,
            "current_config": self.session.current_config.params,
            "history_len": len(self.session.state.history),
        }
        try:
            write_json(str(path), payload)
        except Exception:
            pass

    def _build_rule_table(self) -> list[dict]:
        table: list[dict] = []
        memory = getattr(self.agent, "memory", None)
        if memory is None:
            return table
        try:
            rules = memory.retrieve_rules(self.session.context, top_k=8)
        except Exception:
            rules = []
        for rule in rules:
            patch = rule.config_patch if isinstance(getattr(rule, "config_patch", None), dict) else {}
            override = {}
            if "NCCL_ALGO" in patch:
                override["algo"] = patch["NCCL_ALGO"]
            if "NCCL_PROTO" in patch:
                override["proto"] = patch["NCCL_PROTO"]
            if "NCCL_MAX_NCHANNELS" in patch:
                override["channels"] = patch["NCCL_MAX_NCHANNELS"]
            if not override:
                continue
            table.append(
                {
                    "name": f"memory_rule_{rule.id}",
                    "override": override,
                    "min_ranks": 2,
                }
            )
        return table
