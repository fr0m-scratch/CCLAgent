from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional
import json

from ..types import ContextSignature, NCCLConfig


@dataclass
class TwinPrediction:
    mean_ms: float
    std_ms: float
    prior_delta_ms: float = 0.0
    calibration_bias_ms: float = 0.0
    calibration_scale: float = 1.0
    components: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mean_ms": float(self.mean_ms),
            "std_ms": float(self.std_ms),
            "prior_delta_ms": float(self.prior_delta_ms),
            "calibration_bias_ms": float(self.calibration_bias_ms),
            "calibration_scale": float(self.calibration_scale),
            "components": dict(self.components),
        }


class DigitalTwinModel:
    """Lightweight mechanism-aware prior with online calibration."""

    def __init__(self, *, ema_alpha: float = 0.2) -> None:
        self.ema_alpha = min(1.0, max(0.01, float(ema_alpha)))
        self.calibration_bias_ms = 0.0
        self.calibration_scale = 1.0
        self.observations = 0

    def estimate(
        self,
        *,
        config: NCCLConfig,
        surrogate_mean_ms: float,
        surrogate_std_ms: float,
        context: ContextSignature | None = None,
        profiler_summary: Optional[Dict[str, Any]] = None,
        topology_signature: Optional[Dict[str, Any]] = None,
    ) -> TwinPrediction:
        base = float(max(1e-9, surrogate_mean_ms))
        std = max(0.0, float(surrogate_std_ms))

        prior_delta = self._prior_delta_ms(
            config=config,
            context=context,
            profiler_summary=profiler_summary,
            topology_signature=topology_signature,
        )
        with_prior = max(1e-6, base + prior_delta)
        calibrated = max(1e-6, (with_prior + self.calibration_bias_ms) * self.calibration_scale)

        return TwinPrediction(
            mean_ms=calibrated,
            std_ms=std,
            prior_delta_ms=prior_delta,
            calibration_bias_ms=self.calibration_bias_ms,
            calibration_scale=self.calibration_scale,
            components={
                "surrogate_mean_ms": base,
                "with_prior_ms": with_prior,
            },
        )

    def update(
        self,
        *,
        observed_ms: float,
        predicted_ms: float,
    ) -> None:
        observed = float(max(1e-9, observed_ms))
        predicted = float(max(1e-9, predicted_ms))

        err = observed - predicted
        self.calibration_bias_ms = (1.0 - self.ema_alpha) * self.calibration_bias_ms + self.ema_alpha * err

        ratio = observed / predicted
        self.calibration_scale = (1.0 - self.ema_alpha) * self.calibration_scale + self.ema_alpha * ratio
        self.calibration_scale = min(3.0, max(0.25, self.calibration_scale))
        self.observations += 1

    def snapshot(self) -> Dict[str, Any]:
        return {
            "schema_version": "1.0",
            "ema_alpha": self.ema_alpha,
            "calibration_bias_ms": self.calibration_bias_ms,
            "calibration_scale": self.calibration_scale,
            "observations": self.observations,
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.snapshot(), indent=2), encoding="utf-8")

    def load(self, path: str | Path) -> None:
        path = Path(path)
        if not path.exists():
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return
        if not isinstance(payload, dict):
            return
        self.ema_alpha = float(payload.get("ema_alpha", self.ema_alpha))
        self.calibration_bias_ms = float(payload.get("calibration_bias_ms", self.calibration_bias_ms))
        self.calibration_scale = float(payload.get("calibration_scale", self.calibration_scale))
        self.observations = int(payload.get("observations", self.observations))

    def _prior_delta_ms(
        self,
        *,
        config: NCCLConfig,
        context: ContextSignature | None,
        profiler_summary: Optional[Dict[str, Any]],
        topology_signature: Optional[Dict[str, Any]],
    ) -> float:
        delta = 0.0
        params = config.params

        algo = str(params.get("NCCL_ALGO") or "").upper()
        proto = str(params.get("NCCL_PROTO") or "").upper()
        channels = _safe_int(params.get("NCCL_MAX_NCHANNELS"))

        nodes = int(context.nodes) if context is not None else 1
        if algo == "TREE" and nodes > 1:
            delta -= 8.0
        if algo == "RING" and nodes == 1:
            delta -= 4.0
        if proto == "LL" and nodes <= 2:
            delta -= 3.0
        if proto == "SIMPLE" and nodes >= 4:
            delta -= 2.0
        if channels is not None and channels > 32:
            delta += 7.0

        if isinstance(profiler_summary, dict):
            p95 = _safe_float(profiler_summary.get("p95_dur_us"))
            p50 = _safe_float(profiler_summary.get("p50_dur_us"))
            if p95 is not None and p50 is not None and p50 > 0 and p95 / p50 > 2.0:
                delta += 5.0

        if isinstance(topology_signature, dict):
            nic_count = _safe_int(topology_signature.get("nic_count"))
            gpu_count = _safe_int(topology_signature.get("gpu_count"))
            if nic_count is not None and nic_count <= 1 and nodes > 1:
                delta += 6.0
            if gpu_count is not None and channels is not None and channels > max(8, gpu_count):
                delta += 3.0

        return delta


def _safe_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
