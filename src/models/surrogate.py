from __future__ import annotations

import math
import pickle
from dataclasses import dataclass
from typing import List, Optional, Sequence

from ..types import ContextSignature, NCCLConfig
from .features import ConfigFeaturizer


@dataclass
class Prediction:
    config: NCCLConfig
    mean: float
    std: float


class SurrogateModel:
    def __init__(self, featurizer: ConfigFeaturizer, model_type: str = "rf", k_neighbors: int = 5) -> None:
        self.featurizer = featurizer
        self.model_type = model_type
        self.k_neighbors = max(1, k_neighbors)
        self._model = None
        self._x: List[List[float]] = []
        self._y: List[float] = []

    def fit(self, records: List[tuple[NCCLConfig, float]], context: ContextSignature | None = None) -> None:
        self._x = [self.featurizer.encode(cfg, context) for cfg, _ in records]
        self._y = [value for _, value in records]
        self._model = self._train_model(self._x, self._y)

    def add_record(
        self,
        config: NCCLConfig,
        iteration_time_ms: float,
        context: ContextSignature | None = None,
    ) -> None:
        self._x.append(self.featurizer.encode(config, context))
        self._y.append(iteration_time_ms)

    def update(
        self,
        config: NCCLConfig,
        iteration_time_ms: float,
        context: ContextSignature | None = None,
        *,
        refit: bool = False,
        min_records: int = 2,
    ) -> None:
        self.add_record(config, iteration_time_ms, context=context)
        if refit or (self._model is None and len(self._y) >= min_records):
            self._model = self._train_model(self._x, self._y)

    def predict(self, configs: Sequence[NCCLConfig], context: ContextSignature | None = None) -> List[Prediction]:
        if not configs:
            return []
        if self._model is None:
            if not self._x:
                return [Prediction(config=cfg, mean=1.0, std=1.0) for cfg in configs]
            features = [self.featurizer.encode(cfg, context) for cfg in configs]
            return self._knn_predictions(configs, features)
        features = [self.featurizer.encode(cfg, context) for cfg in configs]
        return self._predict_with_uncertainty(configs, features)

    def predict_one(self, config: NCCLConfig, context: ContextSignature | None = None) -> Prediction:
        return self.predict([config], context=context)[0]

    def save(self, path: str) -> None:
        payload = {
            "model_type": self.model_type,
            "model": self._model,
            "x": self._x,
            "y": self._y,
        }
        with open(path, "wb") as handle:
            pickle.dump(payload, handle)

    def load(self, path: str) -> None:
        with open(path, "rb") as handle:
            payload = pickle.load(handle)
        self.model_type = payload.get("model_type", self.model_type)
        self._model = payload.get("model")
        self._x = payload.get("x", [])
        self._y = payload.get("y", [])

    def _train_model(self, x: List[List[float]], y: List[float]):
        if self.model_type == "rf":
            try:
                from sklearn.ensemble import RandomForestRegressor
            except Exception:
                return None
            model = RandomForestRegressor(n_estimators=50, random_state=7)
            model.fit(x, y)
            return model
        return None

    def _predict_with_uncertainty(self, configs: Sequence[NCCLConfig], features: List[List[float]]):
        if self._model is None:
            return [Prediction(config=cfg, mean=1.0, std=1.0) for cfg in configs]
        try:
            import numpy as np
        except Exception:
            np = None
        if hasattr(self._model, "estimators_"):
            preds = []
            for feat, cfg in zip(features, configs):
                tree_preds = [est.predict([feat])[0] for est in self._model.estimators_]
                mean = float(sum(tree_preds) / len(tree_preds))
                variance = sum((p - mean) ** 2 for p in tree_preds) / max(1, len(tree_preds))
                std = math.sqrt(variance)
                preds.append(Prediction(config=cfg, mean=mean, std=std))
            return preds
        values = self._model.predict(features)
        return [Prediction(config=cfg, mean=float(val), std=0.0) for cfg, val in zip(configs, values)]

    def _knn_predictions(self, configs: Sequence[NCCLConfig], features: List[List[float]]) -> List[Prediction]:
        predictions: List[Prediction] = []
        if not self._x or not self._y:
            return [Prediction(config=cfg, mean=1.0, std=1.0) for cfg in configs]
        for cfg, feat in zip(configs, features):
            distances = []
            for x_i, y_i in zip(self._x, self._y):
                dist = self._distance(feat, x_i)
                distances.append((dist, y_i))
            distances.sort(key=lambda item: item[0])
            neighbors = distances[: self.k_neighbors]
            if not neighbors:
                predictions.append(Prediction(config=cfg, mean=1.0, std=1.0))
                continue
            weights = [1.0 / (d + 1e-9) for d, _ in neighbors]
            weighted_sum = sum(w * y for w, (_, y) in zip(weights, neighbors))
            total = sum(weights)
            mean = weighted_sum / total if total > 0 else sum(y for _, y in neighbors) / len(neighbors)
            variance = sum((y - mean) ** 2 for _, y in neighbors) / max(1, len(neighbors))
            predictions.append(Prediction(config=cfg, mean=mean, std=math.sqrt(variance)))
        return predictions

    def _distance(self, a: Sequence[float], b: Sequence[float]) -> float:
        sq = 0.0
        for x, y in zip(a, b):
            diff = x - y
            sq += diff * diff
        return math.sqrt(sq)
