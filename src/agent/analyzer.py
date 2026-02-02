from __future__ import annotations

from dataclasses import asdict
from typing import Any, Optional

from ..types import (
    HypothesisAction,
    NumericSearchAction,
    RollbackAction,
    StopAction,
    TuningAction,
)
from ..utils import artifact_path, setup_logger, write_json


logger = setup_logger("cclagent.analyzer")


class TuningAnalyzer:
    def __init__(
        self,
        config: Any,
        hypothesis_generator: Any,
        numeric_manager: Any,
        compiler: Any,
        run_context=None,
    ):
        self.config = config
        self.hypothesis_generator = hypothesis_generator
        self.numeric_manager = numeric_manager
        self.compiler = compiler
        self.run_context = run_context

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
                return action
            action = StopAction(kind="stop", reason="failure_without_rollback")
            decision["action"] = "stop"
            self._persist(decision, step)
            return action

        if step >= self.config.budget.max_steps - 1:
            action = StopAction(kind="stop", reason="budget_exhausted")
            decision["action"] = "stop"
            self._persist(decision, step)
            return action

        if self.config.budget.target_gain and state.best_record and state.history:
            baseline = state.history[0].metrics.iteration_time_ms
            best = state.best_record.metrics.iteration_time_ms
            gain = (baseline - best) / max(1e-9, baseline)
            if gain >= self.config.budget.target_gain and state.plateau_count >= self.config.budget.stable_steps:
                action = StopAction(kind="stop", reason="target_gain")
                decision["action"] = "stop"
                decision["gain"] = gain
                self._persist(decision, step)
                return action

        use_hypothesis = (step % self.config.budget.hypothesis_every) == 0
        if state.should_stop or self._plateau(state):
            action = StopAction(kind="stop", reason="plateau")
            decision["action"] = "stop"
            self._persist(decision, step)
            return action

        base_config = base_config or (state.best_record.action.config if state.best_record else None)

        if use_hypothesis:
            hypothesis = self.hypothesis_generator.propose(plan, context, base_config, last_metrics)
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
                    },
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
