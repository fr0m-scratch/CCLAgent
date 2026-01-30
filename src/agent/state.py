from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from typing import Dict, List, Optional, Sequence, Tuple

from ..types import NCCLConfig, ParameterSpace, TuningBudget, TuningRecord


@dataclass
class TuningState:
    budget: TuningBudget
    best_record: Optional[TuningRecord] = None
    history: List[TuningRecord] = field(default_factory=list)
    plateau_count: int = 0

    def record(self, record: TuningRecord) -> None:
        self.history.append(record)
        if self.best_record is None:
            self.best_record = record
            return
        if record.metrics.iteration_time < self.best_record.metrics.iteration_time:
            improvement = (
                (self.best_record.metrics.iteration_time - record.metrics.iteration_time)
                / max(1e-9, self.best_record.metrics.iteration_time)
            )
            if improvement < self.budget.min_improvement:
                self.plateau_count += 1
            else:
                self.plateau_count = 0
            self.best_record = record
        else:
            self.plateau_count += 1

    @property
    def should_stop(self) -> bool:
        return self.plateau_count >= self.budget.patience


class HistorySurrogate:
    def __init__(self) -> None:
        self._records: Dict[str, float] = {}

    def update(self, config: NCCLConfig, iteration_time: float) -> None:
        key = self._key(config)
        self._records[key] = iteration_time

    def predict(self, config: NCCLConfig, default: float = 1.0) -> float:
        return self._records.get(self._key(config), default)

    def _key(self, config: NCCLConfig) -> str:
        return "|".join(f"{k}={v}" for k, v in sorted(config.params.items()))


class SurrogateModel:
    def __init__(
        self,
        parameter_space: ParameterSpace,
        rng_seed: int = 19,
        k_neighbors: int = 5,
        distance_eps: float = 1e-6,
    ) -> None:
        self.parameter_space = parameter_space
        self._rng = random.Random(rng_seed)
        self.k_neighbors = max(1, k_neighbors)
        self.distance_eps = max(1e-12, distance_eps)
        self._records: List[Tuple[List[float], float, NCCLConfig]] = []

    def update(self, config: NCCLConfig, iteration_time: float) -> None:
        x = self._encode(config)
        self._records.append((x, iteration_time, config))

    def predict(self, config: NCCLConfig, default: float = 1.0) -> float:
        if not self._records:
            return default
        x_star = self._encode(config)
        distances: List[Tuple[float, float]] = []
        for x_i, iteration_time, _ in self._records:
            distance = self._distance(x_star, x_i)
            if distance <= self.distance_eps:
                return iteration_time
            distances.append((distance, iteration_time))
        distances.sort(key=lambda item: item[0])
        neighbors = distances[: self.k_neighbors]
        if not neighbors:
            return default
        weighted_sum = 0.0
        weight_total = 0.0
        for distance, iteration_time in neighbors:
            weight = 1.0 / (distance + self.distance_eps)
            weighted_sum += iteration_time * weight
            weight_total += weight
        if weight_total <= 0.0:
            return default
        return weighted_sum / weight_total

    def suggest(
        self,
        base_config: NCCLConfig,
        focus_params: List[str],
        parameter_space: ParameterSpace,
        budget: int,
    ) -> Optional[NCCLConfig]:
        if not self._records:
            return None
        candidates: List[NCCLConfig] = []
        pool = max(4, min(20, budget * 2))
        for _ in range(pool):
            chooser = lambda params: self._rng.choice(params)
            mutated_params = parameter_space.mutate(
                base_config.params, focus_params=focus_params, chooser=chooser
            )
            candidates.append(NCCLConfig(params=mutated_params))
        best_candidate = None
        best_score = None
        for candidate in candidates:
            score = self.predict(candidate, default=1.0)
            if best_score is None or score < best_score:
                best_score = score
                best_candidate = candidate
        return best_candidate

    def _encode(self, config: NCCLConfig) -> List[float]:
        vector: List[float] = []
        for name, spec in sorted(self.parameter_space.specs.items()):
            value = config.params.get(name, spec.default)
            if value is None:
                value = spec.default
            if spec.kind == "bool":
                vector.append(1.0 if bool(value) else 0.0)
                continue
            if spec.kind == "enum":
                choices = spec.choices or []
                if not choices:
                    vector.append(0.0)
                else:
                    idx = choices.index(value) if value in choices else 0
                    denom = max(1, len(choices) - 1)
                    vector.append(idx / denom)
                continue
            if spec.kind in ("int", "float"):
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    numeric = 0.0
                if spec.min_value is not None and spec.max_value is not None and spec.max_value != spec.min_value:
                    vector.append((numeric - spec.min_value) / (spec.max_value - spec.min_value))
                else:
                    vector.append(numeric)
                continue
            vector.append(0.0)
        return vector

    def _distance(self, a: Sequence[float], b: Sequence[float]) -> float:
        sq = 0.0
        for x, y in zip(a, b):
            diff = x - y
            sq += diff * diff
        return math.sqrt(sq)
