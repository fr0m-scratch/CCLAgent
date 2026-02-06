"""WP6: Causal attribution — step-level and parameter-level.

Decomposes total performance gain into per-step contributions and
per-parameter contributions using leave-one-out (LOO) analysis on
the surrogate model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .state import TuningState
    from ..models.surrogate import SurrogateModel
    from ..types import ContextSignature, NCCLConfig


@dataclass
class StepAttribution:
    """How much a single step contributed to total improvement."""

    step: int
    action_lane: str  # "hypothesis", "numeric", "rollback", "baseline"
    delta_ms: float  # positive = improved (lower time)
    delta_pct: float
    config_changes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ParamAttribution:
    """Leave-one-out attribution for a single parameter change."""

    param: str
    from_value: Any
    to_value: Any
    attributed_delta_ms: float
    attributed_delta_pct: float


@dataclass
class LaneAttribution:
    """Aggregate attribution per decision lane."""

    lane: str
    total_delta_ms: float = 0.0
    total_delta_pct: float = 0.0
    step_count: int = 0


@dataclass
class AttributionReport:
    """Complete attribution analysis for a tuning run."""

    baseline_ms: float
    best_ms: float
    total_improvement_ms: float
    total_improvement_pct: float
    step_attributions: List[StepAttribution] = field(default_factory=list)
    param_attributions: List[ParamAttribution] = field(default_factory=list)
    lane_attributions: List[LaneAttribution] = field(default_factory=list)
    consistency_check: float = 0.0  # ratio of sum(param) / total
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_step_attributions(state: "TuningState") -> List[StepAttribution]:
    """Compute per-step improvement deltas from the tuning history."""
    if not state.history or len(state.history) < 2:
        return []
    attrs: List[StepAttribution] = []
    for i in range(1, len(state.history)):
        prev = state.history[i - 1]
        curr = state.history[i]
        prev_ms = prev.metrics.iteration_time_ms
        curr_ms = curr.metrics.iteration_time_ms
        delta_ms = prev_ms - curr_ms  # positive = improvement
        delta_pct = (delta_ms / prev_ms * 100.0) if prev_ms > 0 else 0.0
        changes = {}
        for k, v in curr.action.config.params.items():
            old = prev.action.config.params.get(k)
            if old != v:
                changes[k] = {"from": old, "to": v}
        attrs.append(StepAttribution(
            step=curr.step,
            action_lane=curr.action.kind,
            delta_ms=delta_ms,
            delta_pct=delta_pct,
            config_changes=changes,
        ))
    return attrs


def compute_param_attributions(
    baseline_config: "NCCLConfig",
    best_config: "NCCLConfig",
    baseline_ms: float,
    surrogate: Optional["SurrogateModel"] = None,
    context: Optional["ContextSignature"] = None,
) -> List[ParamAttribution]:
    """LOO on surrogate: flip each changed param back to baseline, re-predict."""
    from ..types import NCCLConfig

    changed = {}
    for k, v in best_config.params.items():
        base_val = baseline_config.params.get(k)
        if base_val != v:
            changed[k] = (base_val, v)
    if not changed:
        return []
    if surrogate is None:
        return [
            ParamAttribution(param=k, from_value=fv, to_value=tv, attributed_delta_ms=0.0, attributed_delta_pct=0.0)
            for k, (fv, tv) in changed.items()
        ]
    best_pred = surrogate.predict_one(best_config, context).mean
    attrs: List[ParamAttribution] = []
    for param, (base_val, best_val) in changed.items():
        loo_params = dict(best_config.params)
        loo_params[param] = base_val
        loo_config = NCCLConfig(params=loo_params)
        loo_pred = surrogate.predict_one(loo_config, context).mean
        delta_ms = loo_pred - best_pred  # positive = this param helped
        delta_pct = (delta_ms / loo_pred * 100.0) if loo_pred > 0 else 0.0
        attrs.append(ParamAttribution(
            param=param,
            from_value=base_val,
            to_value=best_val,
            attributed_delta_ms=delta_ms,
            attributed_delta_pct=delta_pct,
        ))
    attrs.sort(key=lambda a: abs(a.attributed_delta_ms), reverse=True)
    return attrs


def compute_lane_attributions(step_attrs: List[StepAttribution]) -> List[LaneAttribution]:
    """Aggregate step attributions by decision lane."""
    lanes: Dict[str, LaneAttribution] = {}
    for sa in step_attrs:
        lane = sa.action_lane
        if lane not in lanes:
            lanes[lane] = LaneAttribution(lane=lane)
        la = lanes[lane]
        la.total_delta_ms += sa.delta_ms
        la.step_count += 1
    for la in lanes.values():
        la.total_delta_pct = sum(
            sa.delta_pct for sa in step_attrs if sa.action_lane == la.lane
        )
    return sorted(lanes.values(), key=lambda la: la.total_delta_ms, reverse=True)


def build_attribution_report(
    state: "TuningState",
    surrogate: Optional["SurrogateModel"] = None,
    context: Optional["ContextSignature"] = None,
) -> AttributionReport:
    """Build the full attribution report for a completed run."""
    if not state.history:
        return AttributionReport(baseline_ms=0, best_ms=0, total_improvement_ms=0, total_improvement_pct=0)
    baseline_ms = state.history[0].metrics.iteration_time_ms
    best_rec = state.best_record or state.history[-1]
    best_ms = best_rec.metrics.iteration_time_ms
    total_ms = baseline_ms - best_ms
    total_pct = (total_ms / baseline_ms * 100.0) if baseline_ms > 0 else 0.0

    step_attrs = compute_step_attributions(state)
    baseline_cfg = state.history[0].action.config
    best_cfg = best_rec.action.config
    param_attrs = compute_param_attributions(baseline_cfg, best_cfg, baseline_ms, surrogate, context)
    lane_attrs = compute_lane_attributions(step_attrs)

    param_sum = sum(a.attributed_delta_ms for a in param_attrs)
    consistency = (param_sum / total_ms) if abs(total_ms) > 1e-9 else 1.0

    notes = []
    if abs(consistency - 1.0) > 0.3:
        notes.append(
            f"Param attribution sum ({param_sum:.2f}ms) differs from total "
            f"improvement ({total_ms:.2f}ms) by {abs(consistency-1.0)*100:.0f}%. "
            "Interaction effects or surrogate inaccuracy may be responsible."
        )

    return AttributionReport(
        baseline_ms=baseline_ms,
        best_ms=best_ms,
        total_improvement_ms=total_ms,
        total_improvement_pct=total_pct,
        step_attributions=step_attrs,
        param_attributions=param_attrs,
        lane_attributions=lane_attrs,
        consistency_check=consistency,
        notes=notes,
    )
