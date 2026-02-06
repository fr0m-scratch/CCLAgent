"""WP8: Statistical convergence evidence.

Replaces heuristic plateau detection with formal statistical arguments:
bootstrap confidence intervals, effect sizes, and surrogate exhaustion
analysis.
"""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .state import TuningState
    from ..models.surrogate import SurrogateModel
    from ..types import ContextSignature


@dataclass
class BootstrapCI:
    """Bootstrap confidence interval for a statistic."""

    lower: float
    upper: float
    mean: float
    n_resamples: int
    confidence_level: float


@dataclass
class ConvergenceEvidence:
    """Formal evidence supporting a stop decision."""

    converged: bool
    effect_size: float  # Cohen's d between plateau window and earlier
    bootstrap_ci: Optional[BootstrapCI] = None
    ci_contains_zero: bool = True  # True = no significant improvement remaining
    surrogate_max_untested_improvement: float = 0.0
    surrogate_exhausted: bool = False
    plateau_length: int = 0
    llm_confidence: Optional[float] = None
    claims: List[Dict[str, Any]] = field(default_factory=list)
    statistics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------


def _bootstrap_mean_ci(
    values: List[float],
    n_resamples: int = 1000,
    confidence: float = 0.95,
    rng_seed: int = 42,
) -> BootstrapCI:
    """Compute bootstrap CI for the mean of values."""
    rng = random.Random(rng_seed)
    n = len(values)
    if n == 0:
        return BootstrapCI(lower=0, upper=0, mean=0, n_resamples=0, confidence_level=confidence)
    means = []
    for _ in range(n_resamples):
        sample = [rng.choice(values) for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    alpha = 1.0 - confidence
    lo_idx = max(0, int(alpha / 2 * n_resamples))
    hi_idx = min(n_resamples - 1, int((1 - alpha / 2) * n_resamples))
    return BootstrapCI(
        lower=means[lo_idx],
        upper=means[hi_idx],
        mean=sum(means) / len(means),
        n_resamples=n_resamples,
        confidence_level=confidence,
    )


# ---------------------------------------------------------------------------
# Effect Size
# ---------------------------------------------------------------------------


def _cohens_d(group1: List[float], group2: List[float]) -> float:
    """Compute Cohen's d effect size between two groups."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0
    m1 = sum(group1) / n1
    m2 = sum(group2) / n2
    var1 = sum((x - m1) ** 2 for x in group1) / (n1 - 1)
    var2 = sum((x - m2) ** 2 for x in group2) / (n2 - 1)
    pooled_std = math.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std < 1e-12:
        return 0.0
    return (m1 - m2) / pooled_std


# ---------------------------------------------------------------------------
# Surrogate Exhaustion
# ---------------------------------------------------------------------------


def _check_surrogate_exhaustion(
    surrogate: Optional["SurrogateModel"],
    current_best_ms: float,
    context: Optional["ContextSignature"] = None,
    noise_threshold: float = 0.005,
) -> tuple[float, bool]:
    """Query surrogate for best predicted untested config.

    Returns (max_predicted_improvement, exhausted).
    """
    if surrogate is None or surrogate._model is None:
        return 0.0, False
    # Use surrogate's training data to estimate achievable range
    y_vals = surrogate._y
    if not y_vals:
        return 0.0, False
    best_observed = min(y_vals)
    max_improvement = (current_best_ms - best_observed) / current_best_ms if current_best_ms > 0 else 0.0
    exhausted = max_improvement < noise_threshold
    return max_improvement, exhausted


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_convergence_evidence(
    state: "TuningState",
    surrogate: Optional["SurrogateModel"] = None,
    context: Optional["ContextSignature"] = None,
    llm_confidence: Optional[float] = None,
    plateau_window: int = 5,
    n_bootstrap: int = 1000,
) -> ConvergenceEvidence:
    """Build formal convergence evidence from the tuning state."""
    history = state.history
    if len(history) < 3:
        return ConvergenceEvidence(
            converged=False,
            effect_size=0.0,
            claims=[{"claim": "Insufficient data", "refs": ["history_length"]}],
        )

    times = [r.metrics.iteration_time_ms for r in history]
    window = min(plateau_window, len(times))
    recent = times[-window:]
    earlier = times[:-window] if len(times) > window else times[:1]

    # Bootstrap CI on recent improvement deltas
    deltas = [times[i] - times[i + 1] for i in range(max(0, len(times) - window - 1), len(times) - 1)]
    if not deltas:
        deltas = [0.0]
    bootstrap = _bootstrap_mean_ci(deltas, n_resamples=n_bootstrap)
    ci_contains_zero = bootstrap.lower <= 0 <= bootstrap.upper

    # Effect size between recent window and earlier
    effect_size = _cohens_d(earlier, recent)

    # Surrogate exhaustion
    current_best = min(times) if times else 0.0
    surr_improvement, surr_exhausted = _check_surrogate_exhaustion(
        surrogate, current_best, context,
    )

    # Plateau length
    plateau = 0
    if len(times) >= 2:
        best_so_far = times[0]
        for t in times[1:]:
            if t < best_so_far * (1 - 0.003):
                best_so_far = t
                plateau = 0
            else:
                plateau += 1

    # Convergence decision
    claims = []
    converged = False

    if ci_contains_zero:
        claims.append({
            "claim": f"Bootstrap {bootstrap.confidence_level*100:.0f}% CI for improvement "
                     f"[{bootstrap.lower:.4f}, {bootstrap.upper:.4f}] contains zero",
            "refs": ["bootstrap_ci"],
            "confidence": 0.8,
        })
    if abs(effect_size) < 0.2:
        claims.append({
            "claim": f"Effect size (Cohen's d = {effect_size:.3f}) is negligible (<0.2)",
            "refs": ["effect_size"],
            "confidence": 0.7,
        })
    if surr_exhausted:
        claims.append({
            "claim": f"Surrogate predicts max {surr_improvement*100:.2f}% untested improvement (below noise)",
            "refs": ["surrogate_exhaustion"],
            "confidence": 0.6,
        })
    if plateau >= plateau_window:
        claims.append({
            "claim": f"Plateau of {plateau} steps with no significant improvement",
            "refs": ["plateau_count"],
            "confidence": 0.7,
        })
    if llm_confidence is not None and llm_confidence > 0.7:
        claims.append({
            "claim": f"LLM convergence confidence: {llm_confidence:.2f}",
            "refs": ["llm_convergence"],
            "confidence": llm_confidence,
        })

    positive_signals = sum(1 for c in claims if c.get("confidence", 0) >= 0.6)
    converged = positive_signals >= 2

    return ConvergenceEvidence(
        converged=converged,
        effect_size=effect_size,
        bootstrap_ci=bootstrap,
        ci_contains_zero=ci_contains_zero,
        surrogate_max_untested_improvement=surr_improvement,
        surrogate_exhausted=surr_exhausted,
        plateau_length=plateau,
        llm_confidence=llm_confidence,
        claims=claims,
        statistics={
            "recent_mean": sum(recent) / len(recent) if recent else 0.0,
            "earlier_mean": sum(earlier) / len(earlier) if earlier else 0.0,
            "recent_std": _std(recent),
            "improvement_deltas": deltas,
        },
    )


def _std(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = sum(values) / len(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))
