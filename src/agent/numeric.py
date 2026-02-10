from __future__ import annotations

from typing import Any, List, Dict

from ..models.digital_twin import DigitalTwinModel
from ..search.safe_bo import SafeBO, SafeBOConfig
from ..search.coordinate_descent import CoordinateDescentSearch, SearchState
from ..safety.risk import RiskScorer
from ..types import InitialConfigPlan, NCCLConfig, SearchCandidate, SearchResult
from ..utils import artifact_path, setup_logger, write_json


logger = setup_logger("cclagent.numeric")


def _find_record(trace_records: List[Dict[str, Any]], params: Dict[str, Any]) -> Dict[str, Any] | None:
    for record in trace_records:
        if record.get("config") == params:
            return record
    return None


def _candidate_ref(step: int, candidate_id: str) -> str:
    return f"candidate:{step}:{candidate_id}"


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


class NumericSearchManager:
    def __init__(
        self,
        config: Any,
        parameter_space: Any,
        surrogate: Any,
        executor: Any,
        run_context=None,
        trace=None,
    ) -> None:
        self.config = config
        self.parameter_space = parameter_space
        self.surrogate = surrogate
        self.executor = executor
        self.risk_scorer = RiskScorer(config.safety)
        self.cd = CoordinateDescentSearch(parameter_space)
        self.safe_bo = SafeBO(
            SafeBOConfig(
                beta=1.5,
                risk_threshold=float(config.safety.max_risk_score if config.safety.max_risk_score is not None else config.safety.risk_threshold),
            )
        )
        self.digital_twin = DigitalTwinModel()
        self.run_context = run_context
        self.trace = trace

    def propose(
        self,
        plan: InitialConfigPlan,
        state,
        workload,
        base_config: NCCLConfig,
        step: int,
        context=None,
        guidance: Dict[str, Any] | None = None,
    ) -> tuple[NCCLConfig, SearchResult]:
        if state.search_state is None:
            state.search_state = SearchState()
        base_config = base_config or plan.baseline_config
        if guidance is None and getattr(plan, "subspace_priors", None):
            guidance = {"subspace_bias": plan.subspace_priors}
        plan_subspaces = self._apply_guidance(plan.candidate_subspaces, guidance or {})
        raw_candidates = self.cd.propose_candidates(
            plan_subspaces=plan_subspaces,
            state=state.search_state,
            base_config=base_config,
            max_candidates=self.config.numeric_search.max_candidates,
        )
        trace_records = []
        for idx, cand in enumerate(raw_candidates):
            trace_records.append(
                {
                    "candidate_id": f"{step}_{idx}",
                    "config": cand.params,
                    "stages": {
                        "generated": {"status": "kept", "reason": "neighbor"},
                    },
                }
            )
        if self.run_context and self.trace:
            self.trace.event(
                run_id=self.run_context.run_id,
                phase="online",
                step=step,
                actor="agent",
                type="proposal.numeric_candidates",
                payload={"count": len(trace_records)},
                refs=[_candidate_ref(step, str(rec["candidate_id"])) for rec in trace_records],
            )
        candidates, dedup_dropped = self._dedup_candidates(state.search_state, raw_candidates, trace_records)
        filtered = []
        validated: List[NCCLConfig] = []
        for cand in candidates:
            errors = self.parameter_space.validate(cand.params)
            record = _find_record(trace_records, cand.params)
            if errors:
                if record is not None:
                    record["stages"]["validated"] = {"status": "dropped", "reason": "invalid_config", "errors": errors}
                continue
            if record is not None:
                record["stages"]["validated"] = {"status": "kept"}
            validated.append(cand)
        candidates = validated
        max_risk = self.config.safety.max_risk_score
        if max_risk is None:
            max_risk = self.config.safety.risk_threshold
        for cand in candidates:
            risk = self.risk_scorer.score(cand)
            record = _find_record(trace_records, cand.params)
            if risk.risk_score <= max_risk:
                if record is not None:
                    record["stages"]["risk_scored"] = {"status": "kept", "risk_score": risk.risk_score}
                filtered.append(cand)
            else:
                if record is not None:
                    record["stages"]["risk_scored"] = {
                        "status": "dropped",
                        "risk_score": risk.risk_score,
                        "reason": "risk_too_high",
                    }
        if filtered:
            candidates = filtered

        search_candidates: List[SearchCandidate] = []
        if self.config.numeric_search.mode == "real_eval":
            results = self.executor.run_batch(
                workload=workload,
                candidates=candidates,
                step=step,
                eval_mode="short",
                concurrency=self.config.numeric_search.concurrency,
            )
            for candidate, metrics in results:
                predicted = metrics.iteration_time_ms if metrics.success else float("inf")
                record = _find_record(trace_records, candidate.params)
                candidate_id = (
                    str(record.get("candidate_id"))
                    if isinstance(record, dict) and record.get("candidate_id") is not None
                    else f"{step}_unknown_{len(search_candidates)}"
                )
                risk_score = None
                if isinstance(record, dict):
                    risk_scored = record.get("stages", {}).get("risk_scored", {})
                    if isinstance(risk_scored, dict):
                        risk_score = _safe_float(risk_scored.get("risk_score"))
                search_candidates.append(
                    SearchCandidate(
                        config=candidate,
                        predicted_time_ms=predicted,
                        rationale="real_eval",
                        candidate_id=candidate_id,
                        evaluation_mode="real_eval",
                        risk_score=risk_score,
                    )
                )
                if record is not None:
                    record["stages"]["evaluated"] = {"status": "kept", "mode": "real_eval"}
            self._persist_batch_results(step, search_candidates)
        else:
            predictions = self.surrogate.predict(candidates, context=context)
            for pred in predictions:
                record = _find_record(trace_records, pred.config.params)
                candidate_id = (
                    str(record.get("candidate_id"))
                    if isinstance(record, dict) and record.get("candidate_id") is not None
                    else f"{step}_unknown_{len(search_candidates)}"
                )
                risk_score = None
                if isinstance(record, dict):
                    risk_scored = record.get("stages", {}).get("risk_scored", {})
                    if isinstance(risk_scored, dict):
                        risk_score = _safe_float(risk_scored.get("risk_score"))
                twin_pred = self.digital_twin.estimate(
                    config=pred.config,
                    surrogate_mean_ms=pred.mean,
                    surrogate_std_ms=pred.std,
                    context=context,
                    profiler_summary=_profiler_summary_from_guidance(guidance),
                    topology_signature=_topology_signature_from_context(context),
                )
                search_candidates.append(
                    SearchCandidate(
                        config=pred.config,
                        predicted_time_ms=twin_pred.mean_ms,
                        rationale="predict_only+twin",
                        candidate_id=candidate_id,
                        uncertainty=twin_pred.std_ms,
                        evaluation_mode="predict_only",
                        risk_score=risk_score,
                    )
                )
                if record is not None:
                    record["stages"]["predicted"] = {
                        "status": "kept",
                        "predicted_time_ms": twin_pred.mean_ms,
                        "uncertainty": twin_pred.std_ms,
                        "surrogate_predicted_time_ms": pred.mean,
                        "twin_prior_delta_ms": twin_pred.prior_delta_ms,
                        "twin_calibration_bias_ms": twin_pred.calibration_bias_ms,
                        "twin_calibration_scale": twin_pred.calibration_scale,
                    }
            self._persist_surrogate_predictions(step, search_candidates)

        if self.config.numeric_search.mode == "safe_bo":
            search_candidates = self.safe_bo.rank(search_candidates)
            if self.run_context:
                write_json(
                    artifact_path(self.run_context, "steps", f"step_{step}_safe_bo_diagnostics.json"),
                    {
                        "schema_version": "1.0",
                        "step": step,
                        "diagnostics": self.safe_bo.diagnostics(search_candidates),
                    },
                )
        else:
            search_candidates.sort(key=lambda item: item.predicted_time_ms)
        for rank, item in enumerate(search_candidates, start=1):
            record = _find_record(trace_records, item.config.params)
            if record is not None:
                record["stages"]["ranked"] = {"status": "kept", "rank": rank}
        best = search_candidates[0]
        # explore uncertain candidate if reasonable
        if self.config.numeric_search.mode == "predict_only":
            uncertain = max(search_candidates, key=lambda item: item.uncertainty) if search_candidates else None
            if uncertain and uncertain.predicted_time_ms <= best.predicted_time_ms * 1.2:
                best = uncertain
        record = _find_record(trace_records, best.config.params)
        if record is not None:
            record["stages"]["selected"] = {"status": "kept"}
        self.cd.update_state(plan.candidate_subspaces, state.search_state, best)
        self._record_evaluated(state.search_state, search_candidates)
        self._persist_state(state.search_state)
        self._persist_candidates(step, search_candidates)
        self._persist_candidate_trace(step, trace_records, guidance=guidance)
        self._persist_pruning_summary(step, trace_records)
        if self.run_context and self.trace:
            self.trace.event(
                run_id=self.run_context.run_id,
                phase="online",
                step=step,
                actor="agent",
                type="search.prune",
                payload={"dropped": self._prune_counts(trace_records)},
                refs=[
                    f"steps/step_{step}_pruning_summary.json",
                    f"steps/step_{step}_candidates_trace.json",
                ],
            )
        return best.config, SearchResult(best=best, candidates=search_candidates)

    def observe_outcome(self, *, config: NCCLConfig, observed_ms: float, predicted_ms: float | None = None) -> None:
        if predicted_ms is None:
            predicted_ms = observed_ms
        self.digital_twin.update(observed_ms=observed_ms, predicted_ms=predicted_ms)
        if self.run_context:
            write_json(
                artifact_path(self.run_context, "models", "twin.json"),
                self.digital_twin.snapshot(),
            )

    def _config_hash(self, config: NCCLConfig) -> str:
        payload = "|".join(f"{k}={v}" for k, v in sorted(config.params.items()))
        return payload

    def _dedup_candidates(
        self, state: SearchState, candidates: List[NCCLConfig], trace_records: List[Dict[str, Any]]
    ) -> tuple[List[NCCLConfig], List[NCCLConfig]]:
        seen = set()
        unique: List[NCCLConfig] = []
        dropped: List[NCCLConfig] = []
        for cand in candidates:
            key = self._config_hash(cand)
            if key in state.evaluated_hashes or key in seen:
                record = _find_record(trace_records, cand.params)
                if record is not None:
                    record["stages"]["deduped"] = {"status": "dropped", "reason": "duplicate"}
                dropped.append(cand)
                continue
            seen.add(key)
            record = _find_record(trace_records, cand.params)
            if record is not None:
                record["stages"]["deduped"] = {"status": "kept"}
            unique.append(cand)
        return unique or candidates, dropped

    def _record_evaluated(self, state: SearchState, candidates: List[SearchCandidate]) -> None:
        for cand in candidates:
            state.evaluated_hashes.add(self._config_hash(cand.config))

    def _persist_candidates(self, step: int, candidates: List[SearchCandidate]) -> None:
        if not self.run_context:
            return
        payload = [
            {
                "candidate_id": cand.candidate_id,
                "candidate_ref": _candidate_ref(step, cand.candidate_id),
                "config": cand.config.params,
                "predicted_time_ms": cand.predicted_time_ms,
                "uncertainty": cand.uncertainty,
                "evaluation_mode": cand.evaluation_mode,
                "rationale": cand.rationale,
                "risk_score": cand.risk_score,
            }
            for cand in candidates
        ]
        write_json(artifact_path(self.run_context, "steps", f"step_{step}_candidates.json"), payload)

    def _persist_candidate_trace(
        self, step: int, trace_records: List[Dict[str, Any]], guidance: Dict[str, Any] | None = None
    ) -> None:
        if not self.run_context:
            return
        payload = {
            "schema_version": "1.0",
            "step": step,
            "guidance": guidance or {},
            "candidates": trace_records,
        }
        write_json(artifact_path(self.run_context, "steps", f"step_{step}_candidates_trace.json"), payload)

    def _persist_pruning_summary(self, step: int, trace_records: List[Dict[str, Any]]) -> None:
        if not self.run_context:
            return
        summary = self._prune_counts(trace_records)
        payload = {
            "schema_version": "1.0",
            "step": step,
            "dropped": summary,
        }
        write_json(artifact_path(self.run_context, "steps", f"step_{step}_pruning_summary.json"), payload)

    def _prune_counts(self, trace_records: List[Dict[str, Any]]) -> Dict[str, int]:
        summary: Dict[str, int] = {}
        for record in trace_records:
            for stage, info in record.get("stages", {}).items():
                if info.get("status") == "dropped":
                    reason = info.get("reason", stage)
                    summary[reason] = summary.get(reason, 0) + 1
        return summary

    def _apply_guidance(self, subspaces: List[Any], guidance: Dict[str, Any]) -> List[Any]:
        if not guidance:
            return subspaces
        focus_params = guidance.get("focus_params") if isinstance(guidance, dict) else None
        freeze_params = guidance.get("freeze_params") if isinstance(guidance, dict) else None
        if focus_params and not isinstance(focus_params, list):
            focus_params = None
        if freeze_params and not isinstance(freeze_params, list):
            freeze_params = None
        adjusted: List[Any] = []
        for subspace in subspaces:
            free = list(subspace.free or [])
            if focus_params:
                filtered = [p for p in free if p in focus_params]
                if filtered:
                    free = filtered
            if freeze_params:
                free = [p for p in free if p not in freeze_params]
            adjusted.append(
                type(subspace)(
                    name=subspace.name,
                    fixed=dict(subspace.fixed),
                    free=free or list(subspace.free or []),
                )
            )
        bias = guidance.get("subspace_bias") if isinstance(guidance, dict) else None
        if isinstance(bias, list):
            weight_map = {item.get("name"): item.get("weight", 1.0) for item in bias if isinstance(item, dict)}
            adjusted.sort(key=lambda s: weight_map.get(s.name, 1.0), reverse=True)
        return adjusted

    def _persist_state(self, state: SearchState) -> None:
        if not self.run_context:
            return
        payload = {
            "current_subspace_idx": state.current_subspace_idx,
            "current_dim_idx": state.current_dim_idx,
            "step_size": state.step_size,
            "best_in_subspace": state.best_in_subspace.params if state.best_in_subspace else None,
            "evaluated_hashes": sorted(state.evaluated_hashes),
            "history": [
                {
                    "candidate_id": item.candidate_id,
                    "config": item.config.params,
                    "predicted_time_ms": item.predicted_time_ms,
                    "evaluation_mode": item.evaluation_mode,
                    "uncertainty": item.uncertainty,
                    "risk_score": item.risk_score,
                }
                for item in state.history
            ],
        }
        write_json(artifact_path(self.run_context, "online", "search_state.json"), payload)

    def _persist_batch_results(self, step: int, candidates: List[SearchCandidate]) -> None:
        if not self.run_context:
            return
        ranked = sorted(candidates, key=lambda item: item.predicted_time_ms)
        payload = {
            "step": step,
            "ranked": [
                {
                    "candidate_id": cand.candidate_id,
                    "candidate_ref": _candidate_ref(step, cand.candidate_id),
                    "config": cand.config.params,
                    "predicted_time_ms": cand.predicted_time_ms,
                    "evaluation_mode": cand.evaluation_mode,
                    "risk_score": cand.risk_score,
                }
                for cand in ranked
            ],
        }
        write_json(artifact_path(self.run_context, "steps", f"step_{step}_batch_results.json"), payload)

    def _persist_surrogate_predictions(self, step: int, candidates: List[SearchCandidate]) -> None:
        if not self.run_context:
            return
        payload = [
            {
                "candidate_id": cand.candidate_id,
                "candidate_ref": _candidate_ref(step, cand.candidate_id),
                "config": cand.config.params,
                "predicted_time_ms": cand.predicted_time_ms,
                "uncertainty": cand.uncertainty,
                "risk_score": cand.risk_score,
            }
            for cand in candidates
        ]
        write_json(
            artifact_path(self.run_context, "online", f"surrogate_predictions_step_{step}.json"),
            payload,
        )


def _profiler_summary_from_guidance(guidance: Dict[str, Any] | None) -> Dict[str, Any] | None:
    if not isinstance(guidance, dict):
        return None
    value = guidance.get("profiler_summary")
    return value if isinstance(value, dict) else None


def _topology_signature_from_context(context: Any) -> Dict[str, Any] | None:
    if context is None:
        return None
    extra = getattr(context, "extra", None)
    if not isinstance(extra, dict):
        return None
    probe = extra.get("system_probe")
    if not isinstance(probe, dict):
        return None
    topo = probe.get("topology_signature")
    return topo if isinstance(topo, dict) else None
