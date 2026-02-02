from __future__ import annotations

from typing import Any, List

from ..search.coordinate_descent import CoordinateDescentSearch, SearchState
from ..safety.risk import RiskScorer
from ..types import InitialConfigPlan, NCCLConfig, SearchCandidate, SearchResult
from ..utils import artifact_path, setup_logger, write_json


logger = setup_logger("cclagent.numeric")


class NumericSearchManager:
    def __init__(
        self,
        config: Any,
        parameter_space: Any,
        surrogate: Any,
        executor: Any,
        run_context=None,
    ) -> None:
        self.config = config
        self.parameter_space = parameter_space
        self.surrogate = surrogate
        self.executor = executor
        self.risk_scorer = RiskScorer(config.safety)
        self.cd = CoordinateDescentSearch(parameter_space)
        self.run_context = run_context

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
        candidates = self.cd.propose_candidates(
            plan_subspaces=plan.candidate_subspaces,
            state=state.search_state,
            base_config=base_config,
            max_candidates=self.config.numeric_search.max_candidates,
        )
        candidates = self._dedup_candidates(state.search_state, candidates)
        filtered = []
        max_risk = self.config.safety.max_risk_score
        if max_risk is None:
            max_risk = self.config.safety.risk_threshold
        for cand in candidates:
            risk = self.risk_scorer.score(cand)
            if risk.risk_score <= max_risk:
                filtered.append(cand)
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
            self._persist_surrogate_predictions(step, search_candidates)

        search_candidates.sort(key=lambda item: item.predicted_time_ms)
        best = search_candidates[0]
        # explore uncertain candidate if reasonable
        if self.config.numeric_search.mode == "predict_only":
            uncertain = max(search_candidates, key=lambda item: item.uncertainty) if search_candidates else None
            if uncertain and uncertain.predicted_time_ms <= best.predicted_time_ms * 1.2:
                best = uncertain
        self.cd.update_state(plan.candidate_subspaces, state.search_state, best)
        self._record_evaluated(state.search_state, search_candidates)
        self._persist_state(state.search_state)
        self._persist_candidates(step, search_candidates)
        return best.config, SearchResult(best=best, candidates=search_candidates)

    def _config_hash(self, config: NCCLConfig) -> str:
        payload = "|".join(f"{k}={v}" for k, v in sorted(config.params.items()))
        return payload

    def _dedup_candidates(self, state: SearchState, candidates: List[NCCLConfig]) -> List[NCCLConfig]:
        seen = set()
        unique: List[NCCLConfig] = []
        for cand in candidates:
            key = self._config_hash(cand)
            if key in state.evaluated_hashes or key in seen:
                continue
            seen.add(key)
            unique.append(cand)
        return unique or candidates

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
