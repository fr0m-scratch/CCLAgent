from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import json
from pathlib import Path

from ..types import ContextSignature, NCCLConfig, SafetyConfig


@dataclass
class RiskScore:
    risk_score: float
    risk_level: str
    reasons: List[str]


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

        risk_level = "low"
        if score >= 0.6:
            risk_level = "high"
        elif score >= 0.3:
            risk_level = "med"
        return RiskScore(risk_score=min(1.0, score), risk_level=risk_level, reasons=reasons)
