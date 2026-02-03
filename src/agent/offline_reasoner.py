from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..types import ContextSignature, MicrobenchResult, NCCLConfig, ParameterSpace
from ..safety.risk import RiskScorer
from ..types import SafetyConfig


@dataclass
class WarmStartCandidate:
    candidate_id: str
    config: Dict[str, Any]
    source: str
    risk_score: float
    reason: str


@dataclass
class WarmStartDecision:
    selected_id: str
    reason: str


@dataclass
class SearchSpacePrune:
    param: str
    action: str
    reason: str


@dataclass
class OfflinePlanArtifacts:
    microbench_plan: Dict[str, Any]
    warm_start_candidates: List[WarmStartCandidate]
    warm_start_decision: WarmStartDecision
    pruning: List[SearchSpacePrune]


class OfflineReasoner:
    def __init__(self, safety: SafetyConfig) -> None:
        self.risk = RiskScorer(safety)

    def microbench_plan(self, mode: str) -> Dict[str, Any]:
        return {
            "mode": mode,
            "run": True,
            "reason": "default_plan",
        }

    def warm_start_candidates(
        self,
        parameter_space: ParameterSpace,
        rule_patches: List[Dict[str, Any]],
    ) -> List[WarmStartCandidate]:
        candidates: List[WarmStartCandidate] = []
        base = parameter_space.default_config()
        candidates.append(
            WarmStartCandidate(
                candidate_id="C0",
                config=base,
                source="defaults",
                risk_score=self.risk.score(NCCLConfig(params=base)).risk_score,
                reason="baseline defaults",
            )
        )
        for idx, patch in enumerate(rule_patches):
            cfg = dict(base)
            cfg.update(patch)
            candidates.append(
                WarmStartCandidate(
                    candidate_id=f"C{idx+1}",
                    config=cfg,
                    source="memory_rule",
                    risk_score=self.risk.score(NCCLConfig(params=cfg)).risk_score,
                    reason="memory rule patch",
                )
            )
        return candidates

    def select_warm_start(self, candidates: List[WarmStartCandidate]) -> WarmStartDecision:
        if not candidates:
            return WarmStartDecision(selected_id="C0", reason="fallback")
        ranked = sorted(candidates, key=lambda c: c.risk_score)
        chosen = ranked[0]
        return WarmStartDecision(selected_id=chosen.candidate_id, reason="min_risk")

    def prune_search_space(
        self,
        parameter_space: ParameterSpace,
        microbench: MicrobenchResult,
    ) -> List[SearchSpacePrune]:
        important = {ip.param for ip in microbench.important_params}
        prunes: List[SearchSpacePrune] = []
        for name in parameter_space.specs.keys():
            if important and name not in important:
                prunes.append(SearchSpacePrune(param=name, action="fix_default", reason="low_importance"))
        return prunes

    def build_offline_artifacts(
        self,
        parameter_space: ParameterSpace,
        microbench: MicrobenchResult,
        rule_patches: List[Dict[str, Any]],
        mode: str,
    ) -> OfflinePlanArtifacts:
        plan = self.microbench_plan(mode)
        candidates = self.warm_start_candidates(parameter_space, rule_patches)
        decision = self.select_warm_start(candidates)
        pruning = self.prune_search_space(parameter_space, microbench)
        return OfflinePlanArtifacts(
            microbench_plan=plan,
            warm_start_candidates=candidates,
            warm_start_decision=decision,
            pruning=pruning,
        )
