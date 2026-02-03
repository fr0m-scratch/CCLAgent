from __future__ import annotations

from typing import Any, List, Dict

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
    ) -> tuple[NCCLConfig, SearchResult]:
        if state.search_state is None:
            state.search_state = SearchState()
        base_config = base_config or plan.baseline_config
        raw_candidates = self.cd.propose_candidates(
            plan_subspaces=plan.candidate_subspaces,
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
                refs=[f"candidate:{step}:{rec['candidate_id']}" for rec in trace_records],
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
                search_candidates.append(
                    SearchCandidate(
                        config=candidate,
                        predicted_time_ms=predicted,
                        rationale="real_eval",
                        evaluation_mode="real_eval",
                    )
                )
                record = _find_record(trace_records, candidate.params)
                if record is not None:
                    record["stages"]["evaluated"] = {"status": "kept", "mode": "real_eval"}
            self._persist_batch_results(step, search_candidates)
        else:
            predictions = self.surrogate.predict(candidates, context=context)
            for pred in predictions:
                search_candidates.append(
                    SearchCandidate(
                        config=pred.config,
                        predicted_time_ms=pred.mean,
                        rationale="predict_only",
                        uncertainty=pred.std,
                        evaluation_mode="predict_only",
                    )
                )
                record = _find_record(trace_records, pred.config.params)
                if record is not None:
                    record["stages"]["predicted"] = {
                        "status": "kept",
                        "predicted_time_ms": pred.mean,
                        "uncertainty": pred.std,
                    }
            self._persist_surrogate_predictions(step, search_candidates)

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
        self._persist_candidate_trace(step, trace_records)
        self._persist_pruning_summary(step, trace_records)
        if self.run_context and self.trace:
            self.trace.event(
                run_id=self.run_context.run_id,
                phase="online",
                step=step,
                actor="agent",
                type="search.prune",
                payload={"dropped": self._prune_counts(trace_records)},
            )
        return best.config, SearchResult(best=best, candidates=search_candidates)

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
                "config": cand.config.params,
                "predicted_time_ms": cand.predicted_time_ms,
                "uncertainty": cand.uncertainty,
                "evaluation_mode": cand.evaluation_mode,
                "rationale": cand.rationale,
            }
            for cand in candidates
        ]
        write_json(artifact_path(self.run_context, "steps", f"step_{step}_candidates.json"), payload)

    def _persist_candidate_trace(self, step: int, trace_records: List[Dict[str, Any]]) -> None:
        if not self.run_context:
            return
        payload = {
            "schema_version": "1.0",
            "step": step,
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
                    "config": item.config.params,
                    "predicted_time_ms": item.predicted_time_ms,
                    "evaluation_mode": item.evaluation_mode,
                    "uncertainty": item.uncertainty,
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
                    "config": cand.config.params,
                    "predicted_time_ms": cand.predicted_time_ms,
                    "evaluation_mode": cand.evaluation_mode,
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
                "config": cand.config.params,
                "predicted_time_ms": cand.predicted_time_ms,
                "uncertainty": cand.uncertainty,
            }
            for cand in candidates
        ]
        write_json(
            artifact_path(self.run_context, "online", f"surrogate_predictions_step_{step}.json"),
            payload,
        )
