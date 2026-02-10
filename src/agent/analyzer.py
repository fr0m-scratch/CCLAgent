from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Optional

from ..types import (
    HypothesisAction,
    NumericSearchAction,
    RollbackAction,
    StopAction,
    TuningAction,
    NCCLConfig,
    Hypothesis,
)
from ..utils import artifact_path, setup_logger, write_json
from ..trace import TraceEmitter, NullTraceEmitter
from ..whitebox import Evidence, EvidenceStore
from ..safety.rollback import RollbackManager
from .decision_bundle import build_decision_bundle, validate_decision_bundle
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
        llm_advisor: Any | None = None,
        rag: Any | None = None,
        evidence_store: EvidenceStore | None = None,
        rollback_manager: RollbackManager | None = None,
    ):
        self.config = config
        self.hypothesis_generator = hypothesis_generator
        self.numeric_manager = numeric_manager
        self.compiler = compiler
        self.run_context = run_context
        self.trace = trace or NullTraceEmitter()
        self.memory = memory
        self.stop_policy = StopPolicy(config)
        self.llm_advisor = llm_advisor
        self.rag = rag
        self.evidence_store = evidence_store
        self.rollback_manager = rollback_manager or RollbackManager()
        self._online_rag_loaded = False
        self._online_rag_cache: List[Dict[str, Any]] = []
        self._online_rag_cache_meta: Dict[str, Any] = {}

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
        ctx_pack = None
        recent_history = None
        llm_requested = bool(self.llm_advisor and self._should_request_llm_advice(step, state, last_metrics))
        if self.run_context and workload is not None:
            rag_chunks = self._retrieve_online_rag_chunks(
                step=step,
                workload=workload,
                context=context,
                microbench=microbench,
                last_metrics=last_metrics,
            )
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
                rag_chunks=rag_chunks,
                surrogate=surrogate,
                constraints=constraints,
            )
            write_context_pack(artifact_path(self.run_context, "steps", f"step_{step}_context_pack.json"), ctx_pack)

            # Schedule online LLM advice (async) if enabled.
            if self.llm_advisor and llm_requested:
                recent_history = {
                    "steps": [
                        {
                            "step": rec.step,
                            "iteration_time_ms": rec.metrics.iteration_time_ms,
                            "success": rec.metrics.success,
                            "config": rec.action.config.params,
                        }
                        for rec in state.history[-5:]
                    ],
                    "best_ms": state.best_record.metrics.iteration_time_ms if state.best_record else None,
                    "plateau_count": state.plateau_count,
                }
                param_specs = {name: asdict(spec) for name, spec in self.compiler.parameter_space.specs.items()}
                recent_pruning = {"pruning_guidance": plan.pruning_guidance if plan else []}
                playbook = plan.hypothesis_playbook if plan else []
                self.llm_advisor.request(
                    step=step,
                    context_pack=ctx_pack,
                    playbook=playbook,
                    param_specs=param_specs,
                    recent_history=recent_history,
                    recent_pruning=recent_pruning,
                )
        decision = {
            "step": step,
            "plateau": state.should_stop,
            "last_success": last_metrics.success if last_metrics else True,
            "sla_rollback": last_metrics.raw.get("sla_rollback") if last_metrics else False,
        }
        self._last_advice_available = False
        self._last_advice_used = False

        advice = None
        decision_advice = None
        advice_used = False
        def _stamp_advice_state() -> None:
            self._last_advice_available = decision_advice is not None
            self._last_advice_used = bool(advice_used)
        if self.llm_advisor and llm_requested:
            timeout_s = getattr(self.config.llm, "online_soft_wait_s", 0.0) if getattr(self.config, "llm", None) else 0.0
            advice = self.llm_advisor.try_get(step=step, timeout_s=timeout_s)
            # Collect any late advice from previous steps and persist it.
            late_ready = self.llm_advisor.collect_ready()
            for late in late_ready:
                if late.step == step and advice is None:
                    advice = late
                elif late.step != step:
                    self._persist_late_llm_advice(late)
        if self._is_decision_eligible_advice(advice):
            decision_advice = advice

        tool_request_info = self._evaluate_tool_request(decision_advice)
        if tool_request_info:
            decision["tool_request"] = tool_request_info
            if tool_request_info.get("accepted"):
                advice_used = True
        action_preference = self._action_preference_from_advice(decision_advice)
        if action_preference:
            decision["action_preference"] = action_preference
            if action_preference != "auto":
                advice_used = True
        llm_convergence = self._convergence_from_advice(decision_advice)
        explicit_convergence_advice = None
        if llm_convergence is None and self.llm_advisor is not None and self._llm_convergence_enabled():
            explicit_convergence_advice = self._request_explicit_convergence(
                step=step,
                state=state,
                last_metrics=last_metrics,
                context_pack=ctx_pack,
                recent_history=recent_history,
            )
            llm_convergence = self._convergence_from_advice(explicit_convergence_advice)
            if llm_convergence:
                advice_used = True
                decision["llm_convergence_call_id"] = getattr(explicit_convergence_advice, "call_id", None)
        if llm_convergence:
            decision["llm_convergence"] = llm_convergence
            advice_used = True

        failure_mode = None
        if last_metrics is not None and isinstance(getattr(last_metrics, "raw", None), dict):
            failure_mode = str(last_metrics.raw.get("failure_mode") or "").strip().lower() or None
        if failure_mode in ("hang", "crash", "rank_mismatch", "regression"):
            reason = f"failure_mode:{failure_mode}"
            rb_decision = self.rollback_manager.decide(
                failure_mode=failure_mode,
                reason=reason,
                current=base_config or (state.last_known_good or NCCLConfig(params={})),
            )
            if rb_decision.should_rollback and rb_decision.config is not None:
                action = RollbackAction(kind="rollback", reason=reason, config=rb_decision.config)
                decision["action"] = "rollback"
                decision["failure_mode"] = failure_mode
                decision["rollback_mode"] = rb_decision.mode
                if rb_decision.changed_keys:
                    decision["rollback_changed_keys"] = list(rb_decision.changed_keys)
                self._persist(decision, step)
                if self.run_context:
                    write_json(
                        artifact_path(self.run_context, "steps", f"step_{step}_rollback_decision.json"),
                        {
                            "schema_version": "1.0",
                            "reason": reason,
                            "mode": rb_decision.mode,
                            "changed_keys": list(rb_decision.changed_keys),
                            "config": rb_decision.config.params,
                        },
                    )
                    self.trace.event(
                        run_id=self.run_context.run_id,
                        phase="online",
                        step=step,
                        actor="agent",
                        type="safety.rollback",
                        payload={
                            "reason": reason,
                            "mode": rb_decision.mode,
                            "changed_keys": list(rb_decision.changed_keys),
                        },
                    )
                self._write_decision_record(
                    step,
                    action,
                    state,
                    why_selected=[reason],
                    why_rejected=[],
                    advice=decision_advice,
                )
                _stamp_advice_state()
                self._persist_llm_advice(step, advice, advice_used)
                self._persist_llm_convergence(step, explicit_convergence_advice, used=False)
                return action
            action = StopAction(kind="stop", reason=f"{reason}_without_rollback")
            decision["action"] = "stop"
            decision["failure_mode"] = failure_mode
            self._persist(decision, step)
            self._write_decision_record(
                step,
                action,
                state,
                why_selected=[action.reason],
                why_rejected=[],
                advice=decision_advice,
            )
            _stamp_advice_state()
            self._persist_llm_advice(step, advice, advice_used)
            self._persist_llm_convergence(step, explicit_convergence_advice, used=False)
            return action

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
                self._write_decision_record(
                    step,
                    action,
                    state,
                    why_selected=["sla_or_failure"],
                    why_rejected=[],
                    advice=decision_advice,
                )
                _stamp_advice_state()
                self._persist_llm_advice(step, advice, advice_used)
                self._persist_llm_convergence(step, explicit_convergence_advice, used=False)
                return action
            action = StopAction(kind="stop", reason="failure_without_rollback")
            decision["action"] = "stop"
            self._persist(decision, step)
            self._write_decision_record(
                step,
                action,
                state,
                why_selected=["failure_without_rollback"],
                why_rejected=[],
                advice=decision_advice,
            )
            _stamp_advice_state()
            self._persist_llm_advice(step, advice, advice_used)
            self._persist_llm_convergence(step, explicit_convergence_advice, used=False)
            return action

        if llm_convergence and llm_convergence.get("decision") == "stop":
            reason = llm_convergence.get("reason") or "llm_convergence"
            action = StopAction(kind="stop", reason=f"llm_convergence:{reason}")
            decision["action"] = "stop"
            decision["reason"] = action.reason
            self._persist(decision, step)
            if self.run_context:
                claims_raw = llm_convergence.get("claims") if isinstance(llm_convergence.get("claims"), list) else []
                claims = self._claims_with_refs(
                    claims_raw if claims_raw else [reason],
                    self._default_reason_refs(step, advice=explicit_convergence_advice or decision_advice),
                )
                write_json(
                    artifact_path(self.run_context, "steps", f"step_{step}_stop_decision.json"),
                    {
                        "schema_version": "1.0",
                        "reason": action.reason,
                        "source": "llm",
                        "confidence": llm_convergence.get("confidence"),
                        "claims": claims,
                    },
                )
                stop_refs = self._extract_refs_from_claims(claims) or self._default_reason_refs(
                    step,
                    advice=explicit_convergence_advice or decision_advice,
                )
                self.trace.event(
                    run_id=self.run_context.run_id,
                    phase="online",
                    step=step,
                    actor="llm",
                    type="stop.decision",
                    payload={
                        "reason": action.reason,
                        "source": "llm",
                        "confidence": llm_convergence.get("confidence"),
                    },
                    refs=stop_refs,
                )
            self._write_decision_record(
                step,
                action,
                state,
                why_selected=[action.reason],
                why_rejected=[],
                advice=explicit_convergence_advice or decision_advice,
            )
            _stamp_advice_state()
            self._persist_llm_advice(step, advice, advice_used)
            self._persist_llm_convergence(step, explicit_convergence_advice, used=True)
            return action

        require_llm_convergence = False
        llm_cfg = getattr(self.config, "llm", None)
        if llm_cfg is not None:
            require_llm_convergence = bool(getattr(llm_cfg, "convergence_require_llm", False))
        # If LLM convergence is required, heuristic stop policy is fully disabled.
        # If LLM convergence is optional, only use heuristic policy when LLM gave no convergence decision.
        allow_policy_stop = (llm_convergence is None) and (not require_llm_convergence)
        if allow_policy_stop:
            stop_decision = self.stop_policy.evaluate(state, step)
        else:
            stop_decision = None
        if stop_decision is not None:
            action = StopAction(kind="stop", reason=stop_decision.reason)
            decision["action"] = "stop"
            decision.update(stop_decision.details)
            self._persist(decision, step)
            if self.run_context:
                stop_claims = self._claims_with_refs(
                    [stop_decision.reason],
                    self._default_reason_refs(step, advice=decision_advice),
                )
                write_json(
                    artifact_path(self.run_context, "steps", f"step_{step}_stop_decision.json"),
                    {
                        "schema_version": "1.0",
                        "reason": stop_decision.reason,
                        "claims": stop_claims,
                    },
                )
                stop_refs = self._default_reason_refs(step, advice=decision_advice)
                self.trace.event(
                    run_id=self.run_context.run_id,
                    phase="online",
                    step=step,
                    actor="agent",
                    type="stop.decision",
                    payload={"reason": stop_decision.reason},
                    refs=stop_refs,
                )
            self._write_decision_record(
                step,
                action,
                state,
                why_selected=[stop_decision.reason],
                why_rejected=[],
                advice=decision_advice,
            )
            _stamp_advice_state()
            self._persist_llm_advice(step, advice, advice_used)
            self._persist_llm_convergence(step, explicit_convergence_advice, used=False)
            return action

        default_use_hypothesis, default_lane_source = self._agentic_default_lane(
            step=step,
            state=state,
            last_metrics=last_metrics,
        )
        use_hypothesis, lane_source = self._choose_action_lane(
            action_preference=action_preference,
            advice=decision_advice,
            default_use_hypothesis=default_use_hypothesis,
            default_lane_source=default_lane_source,
            llm_requested=llm_requested,
        )
        decision["lane_source"] = lane_source
        if lane_source.startswith("llm_") and lane_source != "llm_pending_hide_latency" and decision_advice is not None:
            advice_used = True

        base_config = base_config or (state.best_record.action.config if state.best_record else None)

        if use_hypothesis:
            portfolio = self.hypothesis_generator.propose_portfolio(
                plan, context, base_config, last_metrics, max_hypotheses=3
            )
            if decision_advice:
                advice_hypotheses = self._hypotheses_from_advice(decision_advice.output)
                if advice_hypotheses:
                    portfolio.extend(advice_hypotheses)
                    advice_used = True
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
            selected_pred = None
            selected_uncertainty = None
            if ranked:
                top = ranked[0]
                selected_pred = top.get("predicted_time_ms")
                selected_uncertainty = top.get("uncertainty")
            why_selected = [self._selected_hypothesis_reason(hypothesis, selected_pred, selected_uncertainty)]
            why_rejected = self._rejected_hypothesis_reasons(ranked, chosen_id=hypothesis.id)
            if self.run_context:
                series_payload = {
                    "schema_version": "1.0",
                    "step": step,
                    "chosen_id": hypothesis.id,
                    "chosen_reason": why_selected[0],
                    "series": [
                        {
                            "rank": rank,
                            "id": item.get("hypothesis", {}).get("id"),
                            "summary": item.get("hypothesis", {}).get("summary"),
                            "mechanism": item.get("hypothesis", {}).get("mechanism"),
                            "patch": item.get("hypothesis", {}).get("patch"),
                            "predicted_time_ms": item.get("predicted_time_ms"),
                            "uncertainty": item.get("uncertainty"),
                        }
                        for rank, item in enumerate(ranked, start=1)
                    ],
                }
                write_json(
                    artifact_path(self.run_context, "steps", f"step_{step}_hypothesis_series.json"),
                    series_payload,
                )
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
                    advice=decision_advice,
                )
                _stamp_advice_state()
                self._persist_llm_advice(step, advice, advice_used)
                self._persist_llm_convergence(step, explicit_convergence_advice, used=False)
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
            self._write_decision_record(
                step,
                action,
                state,
                why_selected=why_selected,
                why_rejected=why_rejected,
                advice=decision_advice,
            )
            _stamp_advice_state()
            self._persist_llm_advice(step, advice, advice_used)
            self._persist_llm_convergence(step, explicit_convergence_advice, used=False)
            return action

        guidance = None
        if decision_advice and isinstance(decision_advice.output, dict) and decision_advice.output.get("numeric_guidance"):
            guidance = decision_advice.output.get("numeric_guidance")
            advice_used = True
        config, search_result = self.numeric_manager.propose(
            plan, state, workload, base_config, step, context=context, guidance=guidance
        )
        selected_idx = 0
        selected_candidate = None
        for idx, candidate in enumerate(search_result.candidates):
            cand_cfg = getattr(getattr(candidate, "config", None), "params", None)
            if isinstance(cand_cfg, dict) and cand_cfg == config.params:
                selected_idx = idx
                selected_candidate = candidate
                break
        if selected_candidate is None and search_result.candidates:
            selected_candidate = search_result.candidates[0]
        candidate_payload = [
            {
                "candidate_id": str(getattr(c, "candidate_id", f"{step}_{idx}")),
                "candidate_ref": f"candidate:{step}:{str(getattr(c, 'candidate_id', f'{step}_{idx}'))}",
                "config": c.config.params,
                "predicted_time_ms": c.predicted_time_ms,
                "uncertainty": c.uncertainty,
                "evaluation_mode": c.evaluation_mode,
                "risk_score": self._safe_float(getattr(c, "risk_score", None)),
            }
            for idx, c in enumerate(search_result.candidates)
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
        selected_pred = self._safe_float(getattr(selected_candidate, "predicted_time_ms", None)) if selected_candidate else None
        selected_unc = self._safe_float(getattr(selected_candidate, "uncertainty", None)) if selected_candidate else None
        selected_candidate_id = str(getattr(selected_candidate, "candidate_id", f"{step}_{selected_idx}"))
        selected_candidate_ref = f"candidate:{step}:{selected_candidate_id}"
        selected_score_bits = [selected_candidate_ref]
        if selected_pred is not None:
            selected_score_bits.append(f"pred_ms={selected_pred:.3f}")
        if selected_unc is not None:
            selected_score_bits.append(f"unc={selected_unc:.3f}")
        why_selected = [
            {
                "claim": "numeric search selected " + ", ".join(selected_score_bits),
                "refs": [
                    selected_candidate_ref,
                    f"steps/step_{step}_candidates_trace.json",
                ],
                "confidence": 0.75,
            }
        ]
        why_rejected = []
        if len(search_result.candidates) > 1:
            why_rejected.append(
                {
                    "claim": f"{len(search_result.candidates) - 1} candidates ranked lower by surrogate objective",
                    "refs": [f"steps/step_{step}_candidates_trace.json"],
                    "confidence": 0.7,
                }
            )
        self._write_decision_record(
            step,
            action,
            state,
            why_selected=why_selected,
            why_rejected=why_rejected,
            advice=decision_advice,
        )
        _stamp_advice_state()
        self._persist_llm_advice(step, advice, advice_used)
        self._persist_llm_convergence(step, explicit_convergence_advice, used=False)
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

    def _write_decision_record(
        self,
        step: int,
        action: Any,
        state: Any,
        why_selected: list,
        why_rejected: list,
        advice: Any | None = None,
    ) -> None:
        if not self.run_context:
            return
        reason_refs = self._default_reason_refs(step, advice=advice)
        chosen_action = {
            "kind": getattr(action, "kind", None),
            "rationale": getattr(action, "rationale", None),
        }
        candidates_considered = self._candidate_records_for_action(step, action, reason_refs)
        counterfactuals = self._build_counterfactuals(candidates_considered)
        budget_remaining = None
        try:
            budget_remaining = int(self.config.budget.max_steps) - int(step) - 1
        except Exception:
            budget_remaining = None
        constraints_snapshot = {
            "risk_max": getattr(self.config.safety, "max_risk_score", None),
            "sla_max_iteration_time": getattr(self.config, "sla_max_iteration_time", None),
            "budget_remaining_steps": budget_remaining,
        }
        context_ref = f"steps/step_{step}_context_pack.json"
        rollback_plan = {
            "last_known_good": state.last_known_good.params if state.last_known_good else None,
            "last_known_good_ref": f"metric:{max(step - 1, 0)}:primary",
        }
        evidence_refs = self._register_decision_evidence(
            step=step,
            chosen_action=chosen_action,
            context_ref=context_ref,
            constraints_snapshot=constraints_snapshot,
            rollback_plan=rollback_plan,
            candidates_considered=candidates_considered,
        )
        reason_refs = self._dedupe_refs(reason_refs + evidence_refs)
        selected_claims = self._claims_with_refs(why_selected, reason_refs)
        rejected_claims = self._claims_with_refs(why_rejected, reason_refs)
        record = {
            "schema_version": "1.0",
            "step": step,
            "context_ref": context_ref,
            "chosen_action": chosen_action,
            "candidates_considered": candidates_considered,
            "why_selected": selected_claims,
            "why_rejected": rejected_claims,
            "expected_outcome": {},
            "success_criteria": {"metric": "iteration_time_ms", "direction": "decrease"},
            "rollback_plan": rollback_plan,
        }
        write_json(artifact_path(self.run_context, "steps", f"step_{step}_decision_record.json"), record)

        call_chain = list(reason_refs)
        selected_ref = next(
            (
                item.get("candidate_ref")
                for item in candidates_considered
                if isinstance(item, dict) and item.get("status") == "selected"
            ),
            None,
        )
        if selected_ref:
            call_chain.append(str(selected_ref))
        bundle = build_decision_bundle(
            step=step,
            chosen_action=chosen_action,
            why_selected=selected_claims,
            why_rejected=rejected_claims,
            context_ref=context_ref,
            constraints_snapshot=constraints_snapshot,
            rollback_plan=rollback_plan,
            refs_fallback=reason_refs,
            candidates_considered=candidates_considered,
            counterfactuals=counterfactuals,
            call_chain=call_chain,
        )
        bundle_errors = validate_decision_bundle(bundle)
        if bundle_errors:
            bundle["quality_flags"] = sorted(set(bundle.get("quality_flags", []) + bundle_errors))
        if candidates_considered and not selected_ref:
            bundle["quality_flags"] = sorted(set(bundle.get("quality_flags", []) + ["missing_selected_candidate_ref"]))
        bundle_rel_path = f"steps/step_{step}_decision_bundle.json"
        write_json(
            artifact_path(self.run_context, "steps", f"step_{step}_decision_bundle.json"),
            bundle,
        )
        bundle_evidence_ref = self._add_evidence(
            kind="experiment",
            source="analyzer.decision_bundle",
            payload={
                "step": step,
                "path": bundle_rel_path,
                "chosen_candidate_ref": selected_ref,
                "quality_flags": bundle.get("quality_flags", []),
            },
        )
        trace_refs = list(reason_refs)
        if selected_ref:
            trace_refs.append(str(selected_ref))
        if bundle_evidence_ref:
            trace_refs.append(bundle_evidence_ref)
        if trace_refs:
            trace_refs = list(dict.fromkeys(trace_refs))
        self.trace.event(
            run_id=self.run_context.run_id,
            phase="online",
            step=step,
            actor="agent",
            type="decision.select_action",
            payload={
                "action": record["chosen_action"],
                "decision_bundle_path": bundle_rel_path,
                "chosen_candidate_ref": selected_ref,
            },
            refs=trace_refs,
            quality_flags=bundle.get("quality_flags", []),
        )
        top_claim = selected_claims[0]["claim"] if selected_claims else "no_explicit_claim"
        top_refs = selected_claims[0].get("refs", []) if selected_claims else []
        logger.info(
            "Step %d decision: action=%s claim=%s refs=%s bundle=%s",
            step,
            chosen_action.get("kind"),
            top_claim,
            len(top_refs),
            bundle_rel_path,
        )
        self._flush_evidence_store()

    def _is_decision_eligible_advice(self, advice: Any) -> bool:
        if advice is None:
            return False
        explicit = getattr(advice, "decision_eligible", None)
        if explicit is not None:
            return bool(explicit)
        output = getattr(advice, "output", None)
        parse_errors = getattr(advice, "parse_errors", [])
        return isinstance(output, dict) and bool(output) and not parse_errors

    def _default_reason_refs(self, step: int, advice: Any | None = None) -> list[str]:
        refs = [f"metric:{max(step - 1, 0)}:primary", f"steps/step_{step}_context_pack.json"]
        call_id = getattr(advice, "call_id", None) if advice is not None else None
        if call_id:
            refs.append(f"llm:call_{call_id}")
        return refs

    def _claims_with_refs(self, claims: list[Any], default_refs: list[str]) -> list[dict]:
        out: list[dict] = []
        for item in claims:
            if isinstance(item, dict):
                claim_text = str(item.get("claim", "")).strip()
                refs = item.get("refs") if isinstance(item.get("refs"), list) else []
                confidence = item.get("confidence", 0.7)
            else:
                claim_text = str(item).strip()
                refs = []
                confidence = 0.7
            if not claim_text:
                continue
            out.append(
                {
                    "claim": claim_text,
                    "refs": refs or list(default_refs),
                    "confidence": self._safe_float(confidence) if self._safe_float(confidence) is not None else 0.7,
                }
            )
        if out:
            return out
        return [{"claim": "no_explicit_claim", "refs": list(default_refs), "confidence": 0.5}]

    def _candidate_records_for_action(self, step: int, action: Any, default_refs: list[str]) -> list[dict]:
        kind = str(getattr(action, "kind", "") or "")
        records: list[dict] = []
        if kind == "numeric":
            search = getattr(action, "search", None)
            candidates = getattr(search, "candidates", None) if search is not None else None
            if isinstance(candidates, list):
                selected_idx = 0
                action_config = getattr(getattr(action, "config", None), "params", None)
                if isinstance(action_config, dict):
                    for idx, candidate in enumerate(candidates):
                        cand_cfg = getattr(getattr(candidate, "config", None), "params", None)
                        if isinstance(cand_cfg, dict) and cand_cfg == action_config:
                            selected_idx = idx
                            break
                for idx, candidate in enumerate(candidates):
                    candidate_id = str(getattr(candidate, "candidate_id", f"{step}_{idx}"))
                    candidate_ref = f"candidate:{step}:{candidate_id}"
                    pred = getattr(candidate, "predicted_time_ms", None)
                    unc = getattr(candidate, "uncertainty", None)
                    risk_score = getattr(candidate, "risk_score", None)
                    records.append(
                        {
                            "candidate_id": candidate_id,
                            "candidate_ref": candidate_ref,
                            "rank": idx + 1,
                            "predicted_iteration_time_ms": self._safe_float(pred),
                            "score_breakdown": {
                                "pred_time_ms": self._safe_float(pred),
                                "uncertainty": self._safe_float(unc),
                                "risk_score": self._safe_float(risk_score),
                                "feasibility": 1.0,
                                "final_rank_score": None,
                            },
                            "status": "selected" if idx == selected_idx else "rejected",
                            "reject_reason": "" if idx == selected_idx else "dominated",
                            "refs": list(default_refs),
                        }
                    )
        elif kind == "hypothesis":
            compiled = getattr(action, "compiled", None)
            risk_score = getattr(compiled, "risk_score", None) if compiled is not None else None
            records.append(
                {
                    "candidate_id": "hypothesis",
                    "candidate_ref": f"candidate:{step}:hypothesis",
                    "rank": 1,
                    "predicted_iteration_time_ms": None,
                    "score_breakdown": {
                        "pred_time_ms": None,
                        "uncertainty": None,
                        "risk_score": self._safe_float(risk_score),
                        "feasibility": 1.0,
                        "final_rank_score": None,
                    },
                    "status": "selected",
                    "reject_reason": "",
                    "refs": list(default_refs),
                }
            )
        elif kind in ("stop", "rollback"):
            records.append(
                {
                    "candidate_id": kind,
                    "candidate_ref": f"candidate:{step}:{kind}",
                    "rank": 1,
                    "predicted_iteration_time_ms": None,
                    "score_breakdown": {
                        "pred_time_ms": None,
                        "uncertainty": None,
                        "risk_score": None,
                        "feasibility": 1.0,
                        "final_rank_score": None,
                    },
                    "status": "selected",
                    "reject_reason": "",
                    "refs": list(default_refs),
                }
            )
        return records

    def _build_counterfactuals(self, candidates_considered: list[dict]) -> list[dict]:
        selected_pred = None
        selected_risk = None
        rejected: list[dict] = []
        for item in candidates_considered:
            score = item.get("score_breakdown", {}) if isinstance(item.get("score_breakdown"), dict) else {}
            if item.get("status") == "selected":
                selected_pred = self._safe_float(score.get("pred_time_ms"))
                selected_risk = self._safe_float(score.get("risk_score"),) or 0.0
            else:
                rejected.append(item)
        out: list[dict] = []
        for item in rejected[:2]:
            score = item.get("score_breakdown", {}) if isinstance(item.get("score_breakdown"), dict) else {}
            pred = self._safe_float(score.get("pred_time_ms"))
            risk = self._safe_float(score.get("risk_score")) or 0.0
            expected_delta_ms = None
            if pred is not None and selected_pred is not None:
                expected_delta_ms = pred - selected_pred
            risk_delta = risk - selected_risk if selected_risk is not None else None
            out.append(
                {
                    "candidate_ref": item.get("candidate_ref"),
                    "expected_delta_ms": expected_delta_ms,
                    "risk_delta": risk_delta,
                    "why_not": item.get("reject_reason") or "dominated",
                }
            )
        return out

    def _add_evidence(self, *, kind: str, source: str, payload: dict[str, Any]) -> str:
        if self.evidence_store is None:
            return ""
        evidence = Evidence(id="", kind=kind, source=source, payload=dict(payload))
        return self.evidence_store.add_evidence(evidence)

    def _flush_evidence_store(self) -> None:
        if self.evidence_store is None or self.run_context is None:
            return
        try:
            self.evidence_store.flush(self.run_context.artifacts_dir)
        except Exception as exc:
            logger.warning("Failed to flush analyzer evidence: %s", exc)

    def _dedupe_refs(self, refs: list[str]) -> list[str]:
        out: list[str] = []
        for ref in refs:
            text = str(ref or "").strip()
            if not text or text in out:
                continue
            out.append(text)
        return out

    def _register_decision_evidence(
        self,
        *,
        step: int,
        chosen_action: dict[str, Any],
        context_ref: str,
        constraints_snapshot: dict[str, Any],
        rollback_plan: dict[str, Any],
        candidates_considered: list[dict],
    ) -> list[str]:
        refs: list[str] = []
        decision_ref = self._add_evidence(
            kind="experiment",
            source="analyzer.decision",
            payload={
                "step": step,
                "chosen_action": chosen_action,
                "context_ref": context_ref,
                "constraints_snapshot": constraints_snapshot,
                "rollback_plan": rollback_plan,
            },
        )
        if decision_ref:
            refs.append(decision_ref)
        candidates_ref = self._add_evidence(
            kind="model",
            source="analyzer.candidates",
            payload={
                "step": step,
                "count": len(candidates_considered),
                "candidates": candidates_considered,
            },
        )
        if candidates_ref:
            refs.append(candidates_ref)
        selected = next(
            (
                item
                for item in candidates_considered
                if isinstance(item, dict) and item.get("status") == "selected"
            ),
            None,
        )
        if isinstance(selected, dict):
            selected_ref = self._add_evidence(
                kind="metric",
                source="analyzer.selected_candidate",
                payload={
                    "step": step,
                    "candidate_id": selected.get("candidate_id"),
                    "candidate_ref": selected.get("candidate_ref"),
                    "score_breakdown": selected.get("score_breakdown", {}),
                },
            )
            if selected_ref:
                refs.append(selected_ref)
        return self._dedupe_refs(refs)

    def _extract_refs_from_claims(self, claims: Any) -> list[str]:
        refs: list[str] = []
        if not isinstance(claims, list):
            return refs
        for item in claims:
            if not isinstance(item, dict):
                continue
            claim_refs = item.get("refs")
            if not isinstance(claim_refs, list):
                continue
            for ref in claim_refs:
                text = str(ref)
                if text and text not in refs:
                    refs.append(text)
        return refs

    def _should_request_llm_advice(self, step: int, state: Any, last_metrics: Any) -> bool:
        llm_cfg = getattr(self.config, "llm", None)
        if llm_cfg is None or not getattr(llm_cfg, "online_enabled", False):
            return False
        triggers = set(getattr(llm_cfg, "online_triggers", ["always"]) or ["always"])
        if "always" in triggers:
            return True
        if "plateau" in triggers and state.should_stop:
            return True
        if "failure" in triggers and last_metrics and not last_metrics.success:
            return True
        return False

    def _hypotheses_from_advice(self, advice: Dict[str, Any]) -> list[Hypothesis]:
        if not isinstance(advice, dict):
            return []
        hypotheses = []
        for item in advice.get("hypotheses", []):
            if not isinstance(item, dict):
                continue
            patch = item.get("patch", {}) if isinstance(item.get("patch"), dict) else {}
            if not patch:
                continue
            summary = str(item.get("summary", "")).strip()
            reason_claims = item.get("reason_claims") if isinstance(item.get("reason_claims"), list) else []
            if self._is_generic_summary(summary):
                summary = self._build_specific_summary_from_patch(
                    patch=patch,
                    mechanism=item.get("mechanism"),
                    reason_claims=reason_claims,
                )
            hypotheses.append(
                Hypothesis(
                    id=item.get("id", "llm_hypothesis"),
                    summary=summary or self._build_specific_summary_from_patch(patch=patch, mechanism=item.get("mechanism"), reason_claims=reason_claims),
                    patch=patch,
                    mechanism=item.get("mechanism"),
                    expected_effect=item.get("expected_effect", {}),
                    risk=str(item.get("risk", {}).get("level", "low")),
                    evidence={"refs": item.get("evidence_refs", []), "reason_claims": reason_claims},
                    test_plan={"success_criteria": item.get("success_criteria", "")},
                )
            )
        return hypotheses

    def _persist_llm_advice(self, step: int, advice: Any, used: bool) -> None:
        if not self.run_context or advice is None:
            return
        parse_errors = getattr(advice, "parse_errors", [])
        raw_is_valid_json = bool(getattr(advice, "raw_is_valid_json", False))
        schema_passed = bool(getattr(advice, "schema_passed", False))
        decision_eligible = self._is_decision_eligible_advice(advice)
        payload = {
            "schema_version": "1.0",
            "step": step,
            "advice_step": getattr(advice, "step", step),
            "call_id": getattr(advice, "call_id", None),
            "used_in_decision": bool(used),
            "output": getattr(advice, "output", {}),
            "parse_errors": parse_errors,
            "raw_is_valid_json": raw_is_valid_json,
            "schema_passed": schema_passed,
            "decision_eligible": decision_eligible,
            "raw_text": getattr(advice, "raw_text", ""),
        }
        write_json(artifact_path(self.run_context, "steps", f"step_{step}_llm_decision_support.json"), payload)
        self.trace.event(
            run_id=self.run_context.run_id,
            phase="online",
            step=step,
            actor="llm",
            type="llm.advice",
            payload={
                "call_id": payload["call_id"],
                "used": payload["used_in_decision"],
                "decision_eligible": payload["decision_eligible"],
                "raw_is_valid_json": payload["raw_is_valid_json"],
                "schema_passed": payload["schema_passed"],
            },
            refs=[f"llm:call_{payload['call_id']}"] if payload.get("call_id") else [],
        )

    def _persist_late_llm_advice(self, advice: Any) -> None:
        if not self.run_context or advice is None:
            return
        parse_errors = getattr(advice, "parse_errors", [])
        raw_is_valid_json = bool(getattr(advice, "raw_is_valid_json", False))
        schema_passed = bool(getattr(advice, "schema_passed", False))
        decision_eligible = self._is_decision_eligible_advice(advice)
        payload = {
            "schema_version": "1.0",
            "step": getattr(advice, "step", None),
            "call_id": getattr(advice, "call_id", None),
            "output": getattr(advice, "output", {}),
            "parse_errors": parse_errors,
            "raw_is_valid_json": raw_is_valid_json,
            "schema_passed": schema_passed,
            "decision_eligible": decision_eligible,
            "raw_text": getattr(advice, "raw_text", ""),
        }
        write_json(
            artifact_path(self.run_context, "online", f"llm_advice_step_{payload['step']}.json"),
            payload,
        )

    def _evaluate_tool_request(self, advice: Any) -> dict | None:
        if not self._is_decision_eligible_advice(advice):
            return None
        if advice is None or not isinstance(advice.output, dict):
            return None
        tool_request = advice.output.get("tool_request") or {}
        name = tool_request.get("name", "none")
        allowed = {"none", "nccltest.short", "workload.short", "microbench.reduced"}
        accepted = name in allowed and name != "none"
        return {
            "name": name,
            "accepted": accepted,
            "reason": tool_request.get("reason", ""),
        }

    def _action_preference_from_advice(self, advice: Any) -> str | None:
        if not self._is_decision_eligible_advice(advice):
            return None
        if advice is None or not isinstance(advice.output, dict):
            return None
        pref = advice.output.get("action_preference")
        if pref in ("auto", "hypothesis", "numeric"):
            return str(pref)
        recommended = advice.output.get("recommended_action")
        if isinstance(recommended, dict):
            kind = str(recommended.get("kind", "")).strip().lower()
            if kind == "hypothesis":
                return "hypothesis"
            if kind in ("numeric", "measure"):
                return "numeric"
        return None

    def _choose_action_lane(
        self,
        *,
        action_preference: str | None,
        advice: Any,
        default_use_hypothesis: bool,
        default_lane_source: str,
        llm_requested: bool,
    ) -> tuple[bool, str]:
        if action_preference == "hypothesis":
            return True, "llm_action_preference"
        if action_preference == "numeric":
            return False, "llm_action_preference"

        output = advice.output if advice is not None and isinstance(getattr(advice, "output", None), dict) else {}
        if output:
            recommended = output.get("recommended_action")
            if isinstance(recommended, dict):
                kind = str(recommended.get("kind", "")).strip().lower()
                if kind == "hypothesis":
                    return True, "llm_recommended_action"
                if kind in ("numeric", "measure"):
                    return False, "llm_recommended_action"

            hypotheses = output.get("hypotheses")
            if isinstance(hypotheses, list) and hypotheses:
                return True, "llm_hypothesis_portfolio"

            numeric_guidance = output.get("numeric_guidance")
            if isinstance(numeric_guidance, dict) and numeric_guidance:
                return False, "llm_numeric_guidance"

        if llm_requested and advice is None:
            if default_use_hypothesis:
                return True, "llm_pending_follow_schedule"
            return False, "llm_pending_hide_latency"
        return default_use_hypothesis, default_lane_source

    def _agentic_default_lane(
        self,
        *,
        step: int,
        state: Any,
        last_metrics: Any,
    ) -> tuple[bool, str]:
        configured_every = max(1, int(getattr(self.config.budget, "hypothesis_every", 2)))
        llm_cfg = getattr(self.config, "llm", None)
        llm_enabled = bool(llm_cfg and getattr(llm_cfg, "online_enabled", False))
        effective_every = configured_every
        if llm_enabled:
            # Keep cadence agentic even when config uses a large numeric-heavy interval.
            effective_every = min(configured_every, 3)

        if llm_enabled and step == 1:
            return True, "agentic_bootstrap_hypothesis"
        if step % effective_every == 0:
            return True, "agentic_schedule"

        if llm_enabled:
            action_kinds = [
                getattr(getattr(rec, "action", None), "kind", None)
                for rec in getattr(state, "history", [])
            ]
            explored = [k for k in action_kinds if k in ("numeric", "hypothesis")]
            hyp_count = sum(1 for k in explored if k == "hypothesis")
            total = max(1, len(explored))
            hyp_ratio = hyp_count / total
            if step >= 2 and hyp_ratio < 0.25 and (step % 2 == 0):
                return True, "agentic_balance_hypothesis_ratio"
            if getattr(state, "plateau_count", 0) >= 2:
                return True, "agentic_plateau_probe"

        return False, "schedule_fallback"

    def _is_generic_summary(self, summary: str) -> bool:
        text = " ".join(str(summary or "").lower().split())
        if not text:
            return True
        generic_markers = (
            "apply memory rule",
            "memory rule",
            "llm hypothesis",
            "generic optimization",
            "tune params",
            "tune parameters",
            "adjust config",
        )
        if text in generic_markers:
            return True
        if "memory rule" in text and len(text) < 80:
            return True
        return False

    def _build_specific_summary_from_patch(
        self,
        *,
        patch: Dict[str, Any],
        mechanism: Any,
        reason_claims: list[Any],
    ) -> str:
        if isinstance(reason_claims, list):
            for item in reason_claims:
                if isinstance(item, dict):
                    claim = str(item.get("claim", "")).strip()
                    if claim:
                        return claim
                elif isinstance(item, str) and item.strip():
                    return item.strip()
        parts = [f"{k}={v}" for k, v in list((patch or {}).items())[:4]]
        if not parts:
            return "targeted hypothesis derived from online evidence"
        mech = str(mechanism or "").strip()
        mech_text = f"{mech}: " if mech else ""
        return f"{mech_text}adjust {'; '.join(parts)} to test measured communication bottleneck"

    def _selected_hypothesis_reason(
        self,
        hypothesis: Hypothesis,
        predicted_time_ms: Any,
        uncertainty: Any,
    ) -> str:
        summary = str(getattr(hypothesis, "summary", "") or "selected hypothesis").strip()
        pred = self._safe_float(predicted_time_ms)
        unc = self._safe_float(uncertainty)
        if pred is None:
            return summary
        if unc is None:
            return f"{summary} | predicted_time_ms={pred:.2f}"
        return f"{summary} | predicted_time_ms={pred:.2f} (uncertainty={unc:.3f})"

    def _rejected_hypothesis_reasons(self, ranked: list[Dict[str, Any]], *, chosen_id: str) -> list[str]:
        reasons: list[str] = []
        for item in ranked:
            hyp = item.get("hypothesis", {}) if isinstance(item.get("hypothesis"), dict) else {}
            hyp_id = str(hyp.get("id", ""))
            if not hyp_id or hyp_id == chosen_id:
                continue
            summary = str(hyp.get("summary", "alternative hypothesis")).strip()
            pred = self._safe_float(item.get("predicted_time_ms"))
            if pred is None:
                reasons.append(f"{hyp_id} not selected: {summary}")
            else:
                reasons.append(f"{hyp_id} not selected: {summary} (predicted_time_ms={pred:.2f})")
            if len(reasons) >= 3:
                break
        return reasons

    def _safe_float(self, value: Any) -> Optional[float]:
        try:
            if value is None:
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    def _convergence_from_advice(self, advice: Any) -> dict | None:
        if not self._is_decision_eligible_advice(advice):
            return None
        if advice is None or not isinstance(advice.output, dict):
            return None
        if not self._llm_convergence_enabled():
            return None
        llm_cfg = getattr(self.config, "llm", None)
        convergence = advice.output.get("convergence")
        if not isinstance(convergence, dict):
            return None
        decision = convergence.get("decision")
        if decision not in ("continue", "stop"):
            return None
        confidence_raw = convergence.get("confidence", 0.0)
        try:
            confidence = float(confidence_raw)
        except (TypeError, ValueError):
            confidence = 0.0
        min_conf = float(getattr(llm_cfg, "convergence_min_confidence", 0.55)) if llm_cfg is not None else 0.55
        if confidence < min_conf:
            return None
        claims = convergence.get("claims") if isinstance(convergence.get("claims"), list) else []
        return {
            "decision": decision,
            "reason": convergence.get("reason", ""),
            "confidence": confidence,
            "claims": claims,
        }

    def _llm_convergence_enabled(self) -> bool:
        llm_cfg = getattr(self.config, "llm", None)
        if llm_cfg is None:
            return True
        return bool(getattr(llm_cfg, "convergence_enabled", True))

    def _request_explicit_convergence(
        self,
        *,
        step: int,
        state: Any,
        last_metrics: Any,
        context_pack: Dict[str, Any] | None,
        recent_history: Dict[str, Any] | None,
    ) -> Any | None:
        if self.llm_advisor is None:
            return None
        if context_pack is None:
            context_pack = {
                "schema_version": "1.0",
                "phase": "online",
                "step": step,
                "observations": {
                    "last_success": getattr(last_metrics, "success", True) if last_metrics is not None else True,
                    "plateau_count": getattr(state, "plateau_count", 0),
                },
            }
        if recent_history is None:
            recent_history = {
                "steps": [
                    {
                        "step": rec.step,
                        "iteration_time_ms": rec.metrics.iteration_time_ms,
                        "success": rec.metrics.success,
                    }
                    for rec in state.history[-5:]
                ],
                "best_ms": state.best_record.metrics.iteration_time_ms if state.best_record else None,
                "plateau_count": state.plateau_count,
            }
        policy_hints = {
            "plateau": bool(getattr(state, "should_stop", False)),
            "patience": self.config.budget.patience,
            "max_steps": self.config.budget.max_steps,
            "current_step": step,
            "last_success": getattr(last_metrics, "success", True) if last_metrics is not None else True,
        }
        try:
            return self.llm_advisor.decide_convergence(
                step=step,
                context_pack=context_pack,
                recent_history=recent_history,
                policy_hints=policy_hints,
            )
        except Exception:
            return None

    def _retrieve_online_rag_chunks(
        self,
        *,
        step: int,
        workload: Any,
        context: Any,
        microbench: Any,
        last_metrics: Any,
    ) -> List[Dict[str, Any]]:
        if self.rag is None:
            return []
        try:
            if not self._online_rag_loaded:
                self.rag.load_documents(self.config.rag)
                self._online_rag_loaded = True

            query_parts = [
                str(getattr(workload, "name", "") or ""),
                str(getattr(workload, "topology", "") or ""),
                str(getattr(workload, "scale", "") or ""),
                str(getattr(context, "workload_kind", "") or ""),
            ]
            if last_metrics is not None and isinstance(getattr(last_metrics, "raw", None), dict):
                bottleneck = last_metrics.raw.get("bottleneck")
                if bottleneck:
                    query_parts.append(str(bottleneck))
            important = getattr(microbench, "important_params", []) if microbench is not None else []
            important_names: list[str] = []
            for item in (important or [])[:3]:
                param = getattr(item, "param", None)
                if param:
                    name = str(param)
                    important_names.append(name)
                    query_parts.append(name)

            query = " ".join(p for p in query_parts if p).strip()
            if not query:
                return []

            raw = getattr(last_metrics, "raw", {}) if last_metrics is not None else {}
            bottleneck = raw.get("bottleneck") if isinstance(raw, dict) else None
            fingerprint = "|".join(
                [
                    str(getattr(workload, "name", "") or ""),
                    str(getattr(context, "topology", "") or ""),
                    str(getattr(context, "scale", "") or ""),
                    str(getattr(context, "workload_kind", "") or ""),
                    str(bottleneck or ""),
                    ",".join(important_names),
                ]
            )
            refresh_every = max(1, int(getattr(self.config.rag, "online_refresh_every", 3)))
            cache_step = int(self._online_rag_cache_meta.get("step", -10_000))
            cache_fp = self._online_rag_cache_meta.get("fingerprint")
            can_reuse = bool(self._online_rag_cache) and cache_fp == fingerprint and (step - cache_step) < refresh_every
            if can_reuse:
                chunks = [dict(item) for item in self._online_rag_cache]
                if self.run_context is not None:
                    write_json(
                        artifact_path(self.run_context, "steps", f"step_{step}_online_rag_retrieval.json"),
                        {
                            "schema_version": "1.0",
                            "query": query,
                            "top_k": int(self._online_rag_cache_meta.get("top_k", 0) or 0),
                            "reused": True,
                            "source_step": cache_step,
                            "chunks": chunks,
                        },
                    )
                return chunks

            top_k = max(1, int(getattr(self.config.rag, "top_k", 5)))
            docs = self.rag.search(query, top_k=min(6, top_k))
            chunks = [
                {
                    "ref": f"rag:{doc.doc_id}:{doc.chunk_id}",
                    "doc_id": doc.doc_id,
                    "chunk_id": doc.chunk_id,
                    "score": doc.score,
                    "text": doc.text,
                }
                for doc in docs
            ]
            self._online_rag_cache = [dict(item) for item in chunks]
            self._online_rag_cache_meta = {
                "step": step,
                "fingerprint": fingerprint,
                "top_k": min(6, top_k),
                "query": query,
            }

            if self.run_context is not None:
                write_json(
                    artifact_path(self.run_context, "steps", f"step_{step}_online_rag_retrieval.json"),
                    {
                        "schema_version": "1.0",
                        "query": query,
                        "top_k": min(6, top_k),
                        "reused": False,
                        "chunks": chunks,
                    },
                )
                if chunks:
                    self.trace.event(
                        run_id=self.run_context.run_id,
                        phase="online",
                        step=step,
                        actor="agent",
                        type="retrieval.rag",
                        payload={
                            "query": query,
                            "count": len(chunks),
                            "chunks": [
                                {
                                    "doc_id": chunk["doc_id"],
                                    "chunk_id": chunk["chunk_id"],
                                    "score": chunk["score"],
                                }
                                for chunk in chunks
                            ],
                        },
                        refs=[chunk["ref"] for chunk in chunks if chunk.get("ref")],
                    )
            return chunks
        except Exception as exc:
            logger.warning("Online RAG retrieval failed at step %d: %s", step, exc)
            return []

    def _persist_llm_convergence(self, step: int, advice: Any, used: bool) -> None:
        if not self.run_context or advice is None:
            return
        parse_errors = getattr(advice, "parse_errors", [])
        raw_is_valid_json = bool(getattr(advice, "raw_is_valid_json", False))
        schema_passed = bool(getattr(advice, "schema_passed", False))
        decision_eligible = self._is_decision_eligible_advice(advice)
        payload = {
            "schema_version": "1.0",
            "step": step,
            "advice_step": getattr(advice, "step", step),
            "call_id": getattr(advice, "call_id", None),
            "used_in_decision": bool(used),
            "output": getattr(advice, "output", {}),
            "parse_errors": parse_errors,
            "raw_is_valid_json": raw_is_valid_json,
            "schema_passed": schema_passed,
            "decision_eligible": decision_eligible,
            "raw_text": getattr(advice, "raw_text", ""),
        }
        write_json(artifact_path(self.run_context, "steps", f"step_{step}_llm_convergence.json"), payload)
