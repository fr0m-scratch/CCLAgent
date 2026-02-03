from __future__ import annotations

from dataclasses import asdict
from typing import Any, Optional

from ..types import (
    HypothesisAction,
    NumericSearchAction,
    RollbackAction,
    StopAction,
    TuningAction,
    NCCLConfig,
)
from ..utils import artifact_path, setup_logger, write_json
from ..trace import TraceEmitter, NullTraceEmitter
from .context_pack import build_context_pack, write_context_pack
from .stop_policy import StopPolicy


logger = setup_logger("cclagent.analyzer")


class TuningAnalyzer:
    def __init__(
        self,
        config: Any,
        hypothesis_generator: Any,
        numeric_manager: Any,
        compiler: Any,
        run_context=None,
        trace: TraceEmitter | None = None,
        memory: Any | None = None,
    ):
        self.config = config
        self.hypothesis_generator = hypothesis_generator
        self.numeric_manager = numeric_manager
        self.compiler = compiler
        self.run_context = run_context
        self.trace = trace or NullTraceEmitter()
        self.memory = memory
        self.stop_policy = StopPolicy(config)

    def plan_next_action(
        self,
        state,
        last_metrics,
        microbench,
        context,
        step: int,
        plan=None,
        workload=None,
        base_config=None,
    ):
        if self.run_context and workload is not None:
            memory_rules = []
            if self.memory is not None:
                rules_with_scores = self.memory.retrieve_rules_with_scores(context, top_k=3)
                memory_rules = [
                    {"rule_id": item["rule"].id, "score": item["score"]}
                    for item in rules_with_scores
                ]
            surrogate = {
                "model_type": getattr(self.numeric_manager.surrogate, "model_type", "unknown"),
                "n_train": len(getattr(self.numeric_manager.surrogate, "_y", [])),
            }
            observations = {
                "last_metrics_ref": f"metric:{step-1}:primary" if step > 0 else None,
                "best_ms": state.best_record.metrics.iteration_time_ms if state.best_record else None,
                "baseline_ms": state.history[0].metrics.iteration_time_ms if state.history else None,
                "plateau_count": state.plateau_count,
            }
            constraints = {
                "sla": {"max_iteration_time": self.config.sla_max_iteration_time},
                "risk_budget": {"max_risk_score": self.config.safety.max_risk_score},
            }
            ctx_pack = build_context_pack(
                phase="online",
                step=step,
                workload=workload,
                context=context,
                observations=observations,
                memory_rules=memory_rules,
                rag_chunks=[],
                surrogate=surrogate,
                constraints=constraints,
            )
            write_context_pack(artifact_path(self.run_context, "steps", f"step_{step}_context_pack.json"), ctx_pack)
        decision = {
            "step": step,
            "plateau": state.should_stop,
            "last_success": last_metrics.success if last_metrics else True,
            "sla_rollback": last_metrics.raw.get("sla_rollback") if last_metrics else False,
        }

        if last_metrics and (not last_metrics.success or last_metrics.raw.get("sla_rollback")):
            if state.last_known_good is not None:
                action = RollbackAction(kind="rollback", reason="sla_or_failure", config=state.last_known_good)
                decision["action"] = "rollback"
                self._persist(decision, step)
                if self.run_context:
                    write_json(
                        artifact_path(self.run_context, "steps", f"step_{step}_rollback_decision.json"),
                        {
                            "schema_version": "1.0",
                            "reason": "sla_or_failure",
                            "config": state.last_known_good.params,
                        },
                    )
                    self.trace.event(
                        run_id=self.run_context.run_id,
                        phase="online",
                        step=step,
                        actor="agent",
                        type="safety.rollback",
                        payload={"reason": "sla_or_failure"},
                    )
                self._write_decision_record(step, action, state, why_selected=["sla_or_failure"], why_rejected=[])
                return action
            action = StopAction(kind="stop", reason="failure_without_rollback")
            decision["action"] = "stop"
            self._persist(decision, step)
            self._write_decision_record(step, action, state, why_selected=["failure_without_rollback"], why_rejected=[])
            return action

        stop_decision = self.stop_policy.evaluate(state, step)
        if stop_decision is not None:
            action = StopAction(kind="stop", reason=stop_decision.reason)
            decision["action"] = "stop"
            decision.update(stop_decision.details)
            self._persist(decision, step)
            if self.run_context:
                write_json(
                    artifact_path(self.run_context, "steps", f"step_{step}_stop_decision.json"),
                    {
                        "schema_version": "1.0",
                        "reason": stop_decision.reason,
                        "claims": [{"claim": stop_decision.reason, "refs": []}],
                    },
                )
                self.trace.event(
                    run_id=self.run_context.run_id,
                    phase="online",
                    step=step,
                    actor="agent",
                    type="stop.decision",
                    payload={"reason": stop_decision.reason},
                )
            self._write_decision_record(step, action, state, why_selected=[stop_decision.reason], why_rejected=[])
            return action

        use_hypothesis = (step % self.config.budget.hypothesis_every) == 0

        base_config = base_config or (state.best_record.action.config if state.best_record else None)

        if use_hypothesis:
            portfolio = self.hypothesis_generator.propose_portfolio(
                plan, context, base_config, last_metrics, max_hypotheses=3
            )
            scored = []
            for hyp in portfolio:
                merged = dict(base_config.params if base_config else {})
                merged.update(hyp.patch)
                predicted = None
                uncertainty = None
                try:
                    pred = self.numeric_manager.surrogate.predict_one(
                        NCCLConfig(params=merged), context=context
                    )
                    predicted = pred.mean
                    uncertainty = pred.std
                except Exception:
                    predicted = None
                scored.append(
                    {
                        "hypothesis": asdict(hyp),
                        "predicted_time_ms": predicted,
                        "uncertainty": uncertainty,
                    }
                )
            ranked = sorted(
                scored,
                key=lambda item: item["predicted_time_ms"] if item["predicted_time_ms"] is not None else float("inf"),
            )
            if not ranked:
                hypothesis = portfolio[0]
            else:
                by_id = {hyp.id: hyp for hyp in portfolio}
                best_id = ranked[0]["hypothesis"]["id"]
                hypothesis = by_id.get(best_id, portfolio[0])
            if self.run_context:
                write_json(
                    artifact_path(self.run_context, "steps", f"step_{step}_hypothesis_portfolio.json"),
                    scored,
                )
                write_json(
                    artifact_path(self.run_context, "steps", f"step_{step}_hypothesis_ranked.json"),
                    ranked,
                )
                self.trace.event(
                    run_id=self.run_context.run_id,
                    phase="online",
                    step=step,
                    actor="agent",
                    type="model.surrogate.predict",
                    payload={"count": len(scored)},
                )
                self.trace.event(
                    run_id=self.run_context.run_id,
                    phase="online",
                    step=step,
                    actor="agent",
                    type="proposal.hypothesis",
                    payload={"count": len(portfolio)},
                    refs=[f"rule:{h.id}" for h in portfolio if h.id],
                )
            compiled = self.compiler.compile_hypothesis(base_config, hypothesis.patch)
            max_risk = self.config.safety.max_risk_score
            if max_risk is None:
                max_risk = self.config.safety.risk_threshold
            if self.run_context:
                write_json(
                    artifact_path(self.run_context, "steps", f"step_{step}_hypothesis.json"),
                    asdict(hypothesis),
                )
                write_json(
                    artifact_path(self.run_context, "steps", f"step_{step}_compiled_config.json"),
                    {
                        "config": compiled.config.params,
                        "env": compiled.env,
                        "warnings": compiled.warnings,
                        "risk_score": compiled.risk_score,
                        "risk_reasons": compiled.risk_reasons,
                    },
                )
                write_json(
                    artifact_path(self.run_context, "steps", f"step_{step}_risk_report.json"),
                    {"schema_version": "1.0", "risk_score": compiled.risk_score, "reasons": compiled.risk_reasons},
                )
                self.trace.event(
                    run_id=self.run_context.run_id,
                    phase="online",
                    step=step,
                    actor="agent",
                    type="safety.risk_score",
                    payload={"risk_score": compiled.risk_score, "reasons": compiled.risk_reasons},
                )
            if compiled.risk_score > max_risk:
                action = NumericSearchAction(
                    kind="numeric",
                    config=compiled.config,
                    rationale="risk_too_high_fallback",
                )
                decision["action"] = "numeric_fallback"
                decision["risk_score"] = compiled.risk_score
                self._persist(decision, step)
                self._write_decision_record(
                    step,
                    action,
                    state,
                    why_selected=["risk_too_high_fallback"],
                    why_rejected=["hypothesis_risk_too_high"],
                )
                return action
            action = HypothesisAction(
                kind="hypothesis",
                config=compiled.config,
                rationale=hypothesis.summary,
                hypothesis=hypothesis,
                compiled=compiled,
                metadata={"risk_score": compiled.risk_score},
            )
            decision["action"] = "hypothesis"
            decision["hypothesis"] = asdict(hypothesis)
            decision["compiled"] = {"warnings": compiled.warnings, "risk_score": compiled.risk_score}
            self._persist(decision, step)
            self._write_decision_record(step, action, state, why_selected=[hypothesis.summary], why_rejected=[])
            return action

        config, search_result = self.numeric_manager.propose(
            plan, state, workload, base_config, step, context=context
        )
        candidate_payload = [
            {
                "config": c.config.params,
                "predicted_time_ms": c.predicted_time_ms,
                "uncertainty": c.uncertainty,
                "evaluation_mode": c.evaluation_mode,
            }
            for c in search_result.candidates
        ]
        action = NumericSearchAction(
            kind="numeric",
            config=config,
            rationale="numeric search",
            metadata={"candidates": candidate_payload},
            search=search_result,
        )
        decision["action"] = "numeric"
        decision["candidates"] = candidate_payload
        self._persist(decision, step)
        self._write_decision_record(step, action, state, why_selected=["numeric_search"], why_rejected=[])
        return action

    def _persist(self, decision: dict, step: int) -> None:
        if not self.run_context:
            return
        write_json(artifact_path(self.run_context, "steps", f"step_{step}_decision.json"), decision)
    def _plateau(self, state) -> bool:
        window = state.recent_best_window()
        if len(window) < self.config.budget.plateau_window:
            return False
        best = min(window)
        worst = max(window)
        if worst <= 0:
            return False
        improvement = (worst - best) / worst
        return improvement < self.config.budget.plateau_eps

    def _write_decision_record(self, step: int, action: Any, state: Any, why_selected: list, why_rejected: list) -> None:
        if not self.run_context:
            return
        record = {
            "schema_version": "1.0",
            "step": step,
            "chosen_action": {
                "kind": getattr(action, "kind", None),
                "rationale": getattr(action, "rationale", None),
            },
            "candidates_considered": [],
            "why_selected": [{"claim": item, "refs": []} for item in why_selected],
            "why_rejected": [{"claim": item, "refs": []} for item in why_rejected],
            "expected_outcome": {},
            "success_criteria": {"metric": "iteration_time_ms", "direction": "decrease"},
            "rollback_plan": {
                "last_known_good": state.last_known_good.params if state.last_known_good else None
            },
        }
        write_json(artifact_path(self.run_context, "steps", f"step_{step}_decision_record.json"), record)
        self.trace.event(
            run_id=self.run_context.run_id,
            phase="online",
            step=step,
            actor="agent",
            type="decision.select_action",
            payload={"action": record["chosen_action"]},
        )
