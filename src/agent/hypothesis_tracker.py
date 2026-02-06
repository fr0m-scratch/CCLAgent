"""WP4: Hypothesis lifecycle tracking with falsifiability.

Every hypothesis carries a quantitative prediction.  After execution
the tracker computes a verdict (confirmed / refuted / inconclusive)
with a numeric margin so that the tuning run becomes a scientific
experiment with testable claims.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from ..types import Hypothesis, Metrics


@dataclass
class HypothesisPrediction:
    """Quantitative prediction attached to a hypothesis before execution."""

    hypothesis_id: str
    step: int
    predicted_direction: str  # "decrease" or "increase" for iteration_time_ms
    predicted_delta_pct: float  # expected magnitude (positive = improvement)
    baseline_ms: float
    surrogate_mean: Optional[float] = None
    surrogate_std: Optional[float] = None
    mechanism: Optional[str] = None


@dataclass
class HypothesisVerdict:
    """Post-execution evaluation of a hypothesis prediction."""

    hypothesis_id: str
    step: int
    prediction: HypothesisPrediction
    actual_ms: float
    actual_delta_pct: float
    confirmed: bool
    margin: float  # positive = better than predicted, negative = worse
    verdict: str  # "confirmed", "refuted", "inconclusive"
    explanation: str = ""


@dataclass
class HypothesisScorecard:
    """Accumulated verdicts across an entire run."""

    total: int = 0
    confirmed: int = 0
    refuted: int = 0
    inconclusive: int = 0
    avg_margin: float = 0.0
    confirmation_rate: float = 0.0
    verdicts: List[Dict[str, Any]] = field(default_factory=list)
    per_mechanism: Dict[str, Dict[str, int]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

INCONCLUSIVE_THRESHOLD = 0.005  # 0.5 % — changes smaller than this are noise


def make_prediction(
    hypothesis: Hypothesis,
    step: int,
    baseline_ms: float,
    surrogate_mean: Optional[float] = None,
    surrogate_std: Optional[float] = None,
) -> HypothesisPrediction:
    """Create a quantitative prediction from a hypothesis."""
    direction = "decrease"
    delta_pct = 0.0
    effect = hypothesis.expected_effect
    if isinstance(effect, dict):
        raw = effect.get("improvement_pct") or effect.get("improvement") or effect.get("delta")
        if raw is not None:
            try:
                delta_pct = float(raw)
            except (TypeError, ValueError):
                delta_pct = 0.0
        dir_str = effect.get("direction", "decrease")
        if dir_str in ("increase", "worse"):
            direction = "increase"

    if delta_pct == 0.0 and surrogate_mean is not None and baseline_ms > 0:
        delta_pct = max(0.0, (baseline_ms - surrogate_mean) / baseline_ms) * 100.0

    return HypothesisPrediction(
        hypothesis_id=hypothesis.id,
        step=step,
        predicted_direction=direction,
        predicted_delta_pct=delta_pct,
        baseline_ms=baseline_ms,
        surrogate_mean=surrogate_mean,
        surrogate_std=surrogate_std,
        mechanism=hypothesis.mechanism,
    )


def compute_verdict(
    prediction: HypothesisPrediction,
    actual_metrics: Metrics,
) -> HypothesisVerdict:
    """Compare prediction against actual outcome to produce a verdict."""
    actual_ms = actual_metrics.iteration_time_ms
    baseline = prediction.baseline_ms

    if baseline <= 0:
        return HypothesisVerdict(
            hypothesis_id=prediction.hypothesis_id,
            step=prediction.step,
            prediction=prediction,
            actual_ms=actual_ms,
            actual_delta_pct=0.0,
            confirmed=False,
            margin=0.0,
            verdict="inconclusive",
            explanation="Baseline is zero or negative; cannot evaluate.",
        )

    actual_delta_pct = (baseline - actual_ms) / baseline * 100.0
    margin = actual_delta_pct - prediction.predicted_delta_pct

    if abs(actual_delta_pct) < INCONCLUSIVE_THRESHOLD * 100:
        verdict = "inconclusive"
        confirmed = False
        explanation = (
            f"Change of {actual_delta_pct:+.2f}% is within noise threshold "
            f"({INCONCLUSIVE_THRESHOLD*100:.1f}%)."
        )
    elif prediction.predicted_direction == "decrease":
        if actual_delta_pct > 0:
            verdict = "confirmed"
            confirmed = True
            explanation = (
                f"Predicted improvement; actual {actual_delta_pct:+.2f}% "
                f"(predicted {prediction.predicted_delta_pct:+.2f}%, "
                f"margin {margin:+.2f}pp)."
            )
        else:
            verdict = "refuted"
            confirmed = False
            explanation = (
                f"Predicted improvement but actual {actual_delta_pct:+.2f}% "
                f"(regression). Margin {margin:+.2f}pp."
            )
    else:
        # predicted direction = increase (expected worse)
        if actual_delta_pct < 0:
            verdict = "confirmed"
            confirmed = True
            explanation = (
                f"Predicted degradation; actual {actual_delta_pct:+.2f}% confirms."
            )
        else:
            verdict = "refuted"
            confirmed = False
            explanation = (
                f"Predicted degradation but actual improved by "
                f"{actual_delta_pct:+.2f}%."
            )

    return HypothesisVerdict(
        hypothesis_id=prediction.hypothesis_id,
        step=prediction.step,
        prediction=prediction,
        actual_ms=actual_ms,
        actual_delta_pct=actual_delta_pct,
        confirmed=confirmed,
        margin=margin,
        verdict=verdict,
        explanation=explanation,
    )


def build_scorecard(verdicts: List[HypothesisVerdict]) -> HypothesisScorecard:
    """Aggregate individual verdicts into a run-level scorecard."""
    sc = HypothesisScorecard()
    if not verdicts:
        return sc
    sc.total = len(verdicts)
    margins = []
    for v in verdicts:
        sc.verdicts.append(asdict(v))
        if v.verdict == "confirmed":
            sc.confirmed += 1
        elif v.verdict == "refuted":
            sc.refuted += 1
        else:
            sc.inconclusive += 1
        margins.append(v.margin)
        mech = v.prediction.mechanism or "unknown"
        bucket = sc.per_mechanism.setdefault(mech, {"confirmed": 0, "refuted": 0, "inconclusive": 0})
        bucket[v.verdict] = bucket.get(v.verdict, 0) + 1
    sc.avg_margin = sum(margins) / len(margins) if margins else 0.0
    sc.confirmation_rate = sc.confirmed / sc.total if sc.total else 0.0
    return sc
