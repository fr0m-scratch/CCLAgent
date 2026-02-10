from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from ..types import SearchCandidate


@dataclass
class SafeBOConfig:
    beta: float = 1.5
    risk_threshold: float = 0.7
    risk_penalty: float = 250.0


class SafeBO:
    """Constrained BO-style ranking over candidate mean/std + risk."""

    def __init__(self, config: SafeBOConfig | None = None) -> None:
        self.config = config or SafeBOConfig()

    def acquisition(self, candidate: SearchCandidate) -> float:
        mean = float(candidate.predicted_time_ms)
        std = float(candidate.uncertainty or 0.0)
        value = mean - self.config.beta * std

        risk = candidate.risk_score if candidate.risk_score is not None else 0.0
        if risk > self.config.risk_threshold:
            # Hard penalty outside safe envelope.
            value += self.config.risk_penalty * (1.0 + (risk - self.config.risk_threshold))
        else:
            value += self.config.risk_penalty * 0.05 * risk
        return value

    def rank(self, candidates: List[SearchCandidate]) -> List[SearchCandidate]:
        scored: List[Tuple[float, SearchCandidate]] = []
        for candidate in candidates:
            scored.append((self.acquisition(candidate), candidate))
        scored.sort(key=lambda item: item[0])

        ranked: List[SearchCandidate] = []
        for score, candidate in scored:
            candidate.rationale = (
                f"safe_bo acquisition={score:.4f} mean={candidate.predicted_time_ms:.4f} "
                f"std={candidate.uncertainty:.4f} risk={candidate.risk_score}"
            )
            ranked.append(candidate)
        return ranked

    def diagnostics(self, candidates: List[SearchCandidate]) -> List[Dict[str, float]]:
        out: List[Dict[str, float]] = []
        for candidate in candidates:
            out.append(
                {
                    "acquisition": self.acquisition(candidate),
                    "predicted_time_ms": float(candidate.predicted_time_ms),
                    "uncertainty": float(candidate.uncertainty or 0.0),
                    "risk_score": float(candidate.risk_score or 0.0),
                }
            )
        return out
