"""WP5: LLM influence measurement and attribution.

Tracks whether the LLM actually influenced each decision, what the
heuristic-only default would have been, and attributes performance
gains to LLM-guided vs heuristic steps.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class LLMInfluenceRecord:
    """Per-step record of LLM influence on the decision."""

    step: int
    llm_advice_available: bool
    llm_influenced: bool  # agent actually used LLM advice
    llm_overridden: bool  # LLM advice was available but agent ignored it
    llm_recommended_action: Optional[str] = None  # what LLM suggested
    agent_chosen_action: Optional[str] = None  # what agent actually did
    heuristic_default: Optional[str] = None  # what heuristic would have done
    improvement_ms: float = 0.0  # delta from previous step
    action_lane: Optional[str] = None  # "hypothesis", "numeric", "rollback", "stop"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LLMInfluenceSummary:
    """Run-level aggregation of LLM influence metrics."""

    total_steps: int = 0
    advice_available_count: int = 0
    influenced_count: int = 0
    overridden_count: int = 0
    no_advice_count: int = 0
    influence_rate: float = 0.0  # influenced / advice_available
    override_rate: float = 0.0  # overridden / advice_available
    llm_attributed_gain_ms: float = 0.0
    heuristic_attributed_gain_ms: float = 0.0
    llm_attributed_gain_pct: float = 0.0
    per_lane: Dict[str, Dict[str, int]] = field(default_factory=dict)
    records: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def record_influence(
    step: int,
    advice_available: bool,
    advice_used: bool,
    action_lane: str,
    recommended_action: Optional[str] = None,
    chosen_action: Optional[str] = None,
    heuristic_default: Optional[str] = None,
    improvement_ms: float = 0.0,
) -> LLMInfluenceRecord:
    """Create a per-step influence record."""
    influenced = advice_available and advice_used
    overridden = advice_available and not advice_used
    return LLMInfluenceRecord(
        step=step,
        llm_advice_available=advice_available,
        llm_influenced=influenced,
        llm_overridden=overridden,
        llm_recommended_action=recommended_action,
        agent_chosen_action=chosen_action,
        heuristic_default=heuristic_default,
        improvement_ms=improvement_ms,
        action_lane=action_lane,
    )


def build_influence_summary(records: List[LLMInfluenceRecord]) -> LLMInfluenceSummary:
    """Aggregate per-step influence records into a run-level summary."""
    summary = LLMInfluenceSummary()
    if not records:
        return summary
    summary.total_steps = len(records)
    llm_gain = 0.0
    heuristic_gain = 0.0
    for rec in records:
        summary.records.append(rec.to_dict())
        if rec.llm_advice_available:
            summary.advice_available_count += 1
            if rec.llm_influenced:
                summary.influenced_count += 1
                llm_gain += max(0.0, rec.improvement_ms)
            else:
                summary.overridden_count += 1
                heuristic_gain += max(0.0, rec.improvement_ms)
        else:
            summary.no_advice_count += 1
            heuristic_gain += max(0.0, rec.improvement_ms)
        lane = rec.action_lane or "unknown"
        bucket = summary.per_lane.setdefault(lane, {"influenced": 0, "overridden": 0, "no_advice": 0})
        if rec.llm_influenced:
            bucket["influenced"] += 1
        elif rec.llm_overridden:
            bucket["overridden"] += 1
        else:
            bucket["no_advice"] += 1

    avail = summary.advice_available_count
    summary.influence_rate = summary.influenced_count / avail if avail else 0.0
    summary.override_rate = summary.overridden_count / avail if avail else 0.0
    total_gain = llm_gain + heuristic_gain
    summary.llm_attributed_gain_ms = llm_gain
    summary.heuristic_attributed_gain_ms = heuristic_gain
    summary.llm_attributed_gain_pct = (llm_gain / total_gain * 100.0) if total_gain > 0 else 0.0
    return summary
