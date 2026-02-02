from __future__ import annotations

import json
import hashlib
import random
from datetime import datetime
from pathlib import Path
from typing import List

from ..memory import MemoryStore
from ..types import ContextSignature, NCCLConfig, SurrogateConfig
from .features import ConfigFeaturizer
from .surrogate import SurrogateModel


def export_dataset(records: List[dict], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def train_surrogate_model(
    records: List[dict],
    context: ContextSignature,
    parameter_space,
    config: SurrogateConfig,
    model_path: str,
) -> SurrogateModel:
    featurizer = ConfigFeaturizer(parameter_space)
    model = SurrogateModel(featurizer, model_type=config.model_type)
    pairs = []
    for record in records:
        cfg = NCCLConfig(params=record.get("config", {}))
        metrics = record.get("metrics", {})
        iteration_time_ms = metrics.get("iteration_time_ms")
        if iteration_time_ms is None:
            continue
        pairs.append((cfg, iteration_time_ms))
    validation = {}
    if pairs:
        rng = random.Random(7)
        shuffled = pairs[:]
        rng.shuffle(shuffled)
        split = max(1, int(len(shuffled) * 0.8))
        train_pairs = shuffled[:split]
        test_pairs = shuffled[split:]
        model.fit(train_pairs, context=context)
        if test_pairs:
            preds = model.predict([cfg for cfg, _ in test_pairs], context=context)
            errors = [pred.mean - actual for pred, (_, actual) in zip(preds, test_pairs)]
            mae = sum(abs(err) for err in errors) / max(1, len(errors))
            rmse = (sum(err * err for err in errors) / max(1, len(errors))) ** 0.5
            validation = {"mae": mae, "rmse": rmse, "count": len(errors)}
        model.fit(pairs, context=context)
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        model.save(model_path)
        meta = {
            "schema_version": "1.0",
            "model_type": config.model_type,
            "trained_at": datetime.utcnow().isoformat() + "Z",
            "record_count": len(pairs),
            "dataset_fingerprint": _fingerprint(records),
            "validation": validation,
        }
        meta_path = model_path.replace(".pkl", ".json")
        Path(meta_path).parent.mkdir(parents=True, exist_ok=True)
        with open(meta_path, "w", encoding="utf-8") as handle:
            json.dump(meta, handle, indent=2)
    return model


def _fingerprint(records: List[dict]) -> str:
    hasher = hashlib.md5()
    for record in records:
        hasher.update(json.dumps(record, sort_keys=True).encode("utf-8"))
    return hasher.hexdigest()
