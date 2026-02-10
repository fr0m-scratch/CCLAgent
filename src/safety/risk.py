from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import json
from pathlib import Path

from ..types import ContextSignature, NCCLConfig, SafetyConfig


@dataclass
class RiskScore:
    risk_score: float
    risk_level: str
    reasons: List[str]


@dataclass
class RiskDecomposition:
    """Decomposed risk into distinct sources."""

    parameter_risk: float = 0.0  # risk from individual param values
    combination_risk: float = 0.0  # risk from param interactions
    novelty_risk: float = 0.0  # risk from untested configurations
    components: Dict[str, float] = field(default_factory=dict)

    @property
    def total(self) -> float:
        return min(1.0, self.parameter_risk + self.combination_risk + self.novelty_risk)


@dataclass
class RiskBudgetState:
    """Track cumulative risk across a tuning run."""

    total_budget: float = 1.0
    consumed: float = 0.0
    steps_taken: int = 0
    per_step: List[float] = field(default_factory=list)

    @property
    def remaining(self) -> float:
        return max(0.0, self.total_budget - self.consumed)

    @property
    def utilization_pct(self) -> float:
        return (self.consumed / self.total_budget * 100.0) if self.total_budget > 0 else 0.0

    def record_step(self, risk_score: float) -> None:
        self.per_step.append(risk_score)
        self.consumed += risk_score
        self.steps_taken += 1

    def can_proceed(self, proposed_risk: float) -> bool:
        return (self.consumed + proposed_risk) <= self.total_budget


class RiskScorer:
    def __init__(self, safety: SafetyConfig):
        self.safety = safety
        self._known_bad: List[Dict[str, Any]] = []
        if safety.known_bad_combos_path:
            try:
                payload = json.loads(Path(safety.known_bad_combos_path).read_text(encoding="utf-8"))
                if isinstance(payload, list):
                    self._known_bad = [item for item in payload if isinstance(item, dict)]
            except Exception:
                self._known_bad = []

    def score(
        self,
        config: NCCLConfig,
        context: Optional[ContextSignature] = None,
        avoid_rules: Optional[List[Dict[str, Any]]] = None,
    ) -> RiskScore:
        score = 0.0
        reasons: List[str] = []
        max_channels = self.safety.max_channels_safe
        min_buffsize = self.safety.min_buffsize_safe

        if "NCCL_MAX_NCHANNELS" in config.params:
            try:
                channels = int(config.params["NCCL_MAX_NCHANNELS"])
                if channels > max_channels:
                    score += 0.4
                    reasons.append("max_channels_exceeds_safe")
            except (TypeError, ValueError):
                score += 0.2
                reasons.append("max_channels_invalid")
        if "NCCL_BUFFSIZE" in config.params:
            try:
                buffsize = int(config.params["NCCL_BUFFSIZE"])
                if buffsize < min_buffsize:
                    score += 0.3
                    reasons.append("buffsize_below_safe")
            except (TypeError, ValueError):
                score += 0.2
                reasons.append("buffsize_invalid")

        for param, envelope in (self.safety.safe_envelope or {}).items():
            if param not in config.params:
                continue
            value = config.params.get(param)
            min_v = envelope.get("min")
            max_v = envelope.get("max")
            allowed = envelope.get("allowed")
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                numeric = None
            if allowed is not None and value not in allowed:
                score += 0.4
                reasons.append(f"envelope_violation:{param}")
            if numeric is not None:
                if min_v is not None and numeric < float(min_v):
                    score += 0.3
                    reasons.append(f"envelope_below:{param}")
                if max_v is not None and numeric > float(max_v):
                    score += 0.3
                    reasons.append(f"envelope_above:{param}")

        if self._known_bad:
            for combo in self._known_bad:
                if all(config.params.get(k) == v for k, v in combo.items()):
                    score += 0.6
                    reasons.append("known_bad_combo")
                    break

        if avoid_rules:
            for rule in avoid_rules:
                for key, value in rule.items():
                    if config.params.get(key) == value:
                        score += 0.4
                        reasons.append("matches_avoid_rule")
                        break

        if context is not None:
            topology = str(getattr(context, "topology", "") or "").lower()
            extra = getattr(context, "extra", {}) if hasattr(context, "extra") else {}
            if topology in ("unknown", ""):
                score += 0.05
                reasons.append("topology_unknown")
            if isinstance(extra, dict):
                if bool(extra.get("plugin_active")):
                    score += 0.1
                    reasons.append("plugin_override_active")
                if bool(extra.get("topology_signature_mismatch")):
                    score += 0.3
                    reasons.append("topology_signature_mismatch")

        risk_level = "low"
        if score >= 0.6:
            risk_level = "high"
        elif score >= 0.3:
            risk_level = "med"
        return RiskScore(risk_score=min(1.0, score), risk_level=risk_level, reasons=reasons)

    def decompose(
        self,
        config: NCCLConfig,
        context: Optional[ContextSignature] = None,
        avoid_rules: Optional[List[Dict[str, Any]]] = None,
        memory_configs: Optional[List[Dict[str, Any]]] = None,
    ) -> RiskDecomposition:
        """Decompose risk into parameter, combination, and novelty sources."""
        param_risk = 0.0
        combo_risk = 0.0
        novelty_risk = 0.0
        components: Dict[str, float] = {}

        # Parameter risk: envelope violations
        for param, envelope in (self.safety.safe_envelope or {}).items():
            if param not in config.params:
                continue
            value = config.params.get(param)
            allowed = envelope.get("allowed")
            min_v = envelope.get("min")
            max_v = envelope.get("max")
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                numeric = None
            if allowed is not None and value not in allowed:
                param_risk += 0.3
                components[f"envelope_violation:{param}"] = 0.3
            if numeric is not None:
                if min_v is not None and numeric < float(min_v):
                    param_risk += 0.2
                    components[f"below_min:{param}"] = 0.2
                if max_v is not None and numeric > float(max_v):
                    param_risk += 0.2
                    components[f"above_max:{param}"] = 0.2

        # Combination risk: known bad combos + avoid rules
        if self._known_bad:
            for combo in self._known_bad:
                if all(config.params.get(k) == v for k, v in combo.items()):
                    combo_risk += 0.5
                    components["known_bad_combo"] = 0.5
                    break
        if avoid_rules:
            for rule in avoid_rules:
                for key, value in rule.items():
                    if config.params.get(key) == value:
                        combo_risk += 0.3
                        components["avoid_rule_match"] = 0.3
                        break

        if context is not None:
            topology = str(getattr(context, "topology", "") or "").lower()
            extra = getattr(context, "extra", {}) if hasattr(context, "extra") else {}
            if topology in ("unknown", ""):
                combo_risk += 0.05
                components["topology_unknown"] = 0.05
            if isinstance(extra, dict):
                if bool(extra.get("plugin_active")):
                    combo_risk += 0.1
                    components["plugin_override_active"] = 0.1
                if bool(extra.get("topology_signature_mismatch")):
                    combo_risk += 0.3
                    components["topology_signature_mismatch"] = 0.3

        # Novelty risk: how far from previously tested configs
        if memory_configs:
            overlap = 0
            total = len(config.params)
            for prev in memory_configs:
                matching = sum(1 for k, v in config.params.items() if prev.get(k) == v)
                overlap = max(overlap, matching)
            novelty = 1.0 - (overlap / max(1, total))
            novelty_risk = novelty * 0.3
            components["novelty"] = novelty_risk
        else:
            novelty_risk = 0.15
            components["novelty_no_history"] = 0.15

        return RiskDecomposition(
            parameter_risk=min(1.0, param_risk),
            combination_risk=min(1.0, combo_risk),
            novelty_risk=min(1.0, novelty_risk),
            components=components,
        )
