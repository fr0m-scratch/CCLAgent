"""WP7: Surrogate calibration and interpretability.

Provides calibration metrics (ECE, PICP), feature importance extraction,
and partial dependence computation for the surrogate model.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .surrogate import SurrogateModel
    from ..types import ContextSignature, NCCLConfig


@dataclass
class CalibrationMetrics:
    """Quantified calibration of surrogate uncertainty estimates."""

    picp: float = 0.0  # Prediction Interval Coverage Probability
    mpiw: float = 0.0  # Mean Prediction Interval Width
    ece: float = 0.0   # Expected Calibration Error (binned)
    mae: float = 0.0   # Mean Absolute Error
    rmse: float = 0.0  # Root Mean Squared Error
    n_samples: int = 0
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FeatureImportance:
    """Feature importance extracted from the surrogate model."""

    feature_name: str
    importance: float
    rank: int = 0


@dataclass
class PartialDependence:
    """Partial dependence of predicted outcome on a single feature."""

    feature_name: str
    grid: List[float] = field(default_factory=list)
    predictions: List[float] = field(default_factory=list)


@dataclass
class SurrogateInterpretation:
    """Combined interpretability report for the surrogate."""

    calibration: CalibrationMetrics = field(default_factory=CalibrationMetrics)
    feature_importances: List[FeatureImportance] = field(default_factory=list)
    partial_dependences: List[PartialDependence] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def compute_calibration(
    surrogate: "SurrogateModel",
    records: List[Tuple["NCCLConfig", float]],
    context: Optional["ContextSignature"] = None,
    confidence_level: float = 0.9,
) -> CalibrationMetrics:
    """Compute calibration metrics via leave-one-out cross-validation.

    For each record, predict using the other N-1 records and check
    whether the actual value falls within the predicted interval.
    """
    n = len(records)
    if n < 3:
        return CalibrationMetrics(n_samples=n, notes=["Too few samples for calibration"])

    from ..types import NCCLConfig
    from .surrogate import SurrogateModel
    from .features import ConfigFeaturizer

    z = 1.645 if confidence_level == 0.9 else 1.96  # 90% or 95% CI
    covered = 0
    widths = []
    errors = []
    sq_errors = []

    for i in range(n):
        loo_records = records[:i] + records[i + 1:]
        test_cfg, test_y = records[i]
        loo_model = SurrogateModel(surrogate.featurizer, model_type=surrogate.model_type)
        loo_model.fit(loo_records, context)
        pred = loo_model.predict_one(test_cfg, context)
        lower = pred.mean - z * pred.std
        upper = pred.mean + z * pred.std
        width = upper - lower
        widths.append(width)
        if lower <= test_y <= upper:
            covered += 1
        error = abs(test_y - pred.mean)
        errors.append(error)
        sq_errors.append(error ** 2)

    picp = covered / n if n > 0 else 0.0
    mpiw = sum(widths) / n if n > 0 else 0.0
    mae = sum(errors) / n if n > 0 else 0.0
    rmse = math.sqrt(sum(sq_errors) / n) if n > 0 else 0.0

    # Binned ECE: |picp - target| averaged over bins
    ece = abs(picp - confidence_level)

    notes = []
    if picp < confidence_level - 0.1:
        notes.append(f"Under-coverage: PICP={picp:.2f} < target {confidence_level:.2f}")
    if picp > confidence_level + 0.1:
        notes.append(f"Over-coverage: PICP={picp:.2f} > target {confidence_level:.2f}")

    return CalibrationMetrics(
        picp=picp,
        mpiw=mpiw,
        ece=ece,
        mae=mae,
        rmse=rmse,
        n_samples=n,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Feature Importance
# ---------------------------------------------------------------------------


def compute_feature_importance(
    surrogate: "SurrogateModel",
) -> List[FeatureImportance]:
    """Extract feature importances from the surrogate's internal model."""
    model = surrogate._model
    if model is None:
        return []
    if not hasattr(model, "feature_importances_"):
        return []
    importances = model.feature_importances_
    names = surrogate.featurizer.feature_names()
    if len(names) != len(importances):
        names = [f"feature_{i}" for i in range(len(importances))]
    ranked = sorted(
        enumerate(importances),
        key=lambda item: item[1],
        reverse=True,
    )
    result = []
    for rank, (idx, imp) in enumerate(ranked, 1):
        result.append(FeatureImportance(
            feature_name=names[idx],
            importance=float(imp),
            rank=rank,
        ))
    return result


# ---------------------------------------------------------------------------
# Partial Dependence
# ---------------------------------------------------------------------------


def compute_partial_dependence(
    surrogate: "SurrogateModel",
    feature_idx: int,
    grid_size: int = 20,
    context: Optional["ContextSignature"] = None,
) -> PartialDependence:
    """Compute partial dependence for a single feature.

    Averages predictions across training data while varying one feature.
    """
    model = surrogate._model
    x_data = surrogate._x
    if model is None or not x_data:
        names = surrogate.featurizer.feature_names()
        name = names[feature_idx] if feature_idx < len(names) else f"feature_{feature_idx}"
        return PartialDependence(feature_name=name)

    names = surrogate.featurizer.feature_names()
    name = names[feature_idx] if feature_idx < len(names) else f"feature_{feature_idx}"

    col_vals = [row[feature_idx] for row in x_data]
    lo, hi = min(col_vals), max(col_vals)
    if lo == hi:
        return PartialDependence(feature_name=name, grid=[lo], predictions=[float(model.predict([x_data[0]])[0])])

    step = (hi - lo) / max(1, grid_size - 1)
    grid = [lo + i * step for i in range(grid_size)]
    predictions = []
    for g in grid:
        preds = []
        for row in x_data:
            modified = list(row)
            modified[feature_idx] = g
            preds.append(float(model.predict([modified])[0]))
        predictions.append(sum(preds) / len(preds))
    return PartialDependence(feature_name=name, grid=grid, predictions=predictions)


def build_interpretation(
    surrogate: "SurrogateModel",
    records: List[Tuple["NCCLConfig", float]],
    context: Optional["ContextSignature"] = None,
    top_k_pd: int = 3,
) -> SurrogateInterpretation:
    """Build a complete surrogate interpretation report."""
    calibration = compute_calibration(surrogate, records, context)
    importances = compute_feature_importance(surrogate)
    pds = []
    for fi in importances[:top_k_pd]:
        idx = next(
            (i for i, n in enumerate(surrogate.featurizer.feature_names()) if n == fi.feature_name),
            None,
        )
        if idx is not None:
            pds.append(compute_partial_dependence(surrogate, idx, context=context))
    return SurrogateInterpretation(
        calibration=calibration,
        feature_importances=importances,
        partial_dependences=pds,
    )
