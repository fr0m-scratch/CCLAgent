from __future__ import annotations

import json
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

from ..types import ContextSignature
from .state import TuningState


def distill_rules(state: TuningState, context: ContextSignature) -> List[Dict]:
    if not state.best_record or not state.history:
        return []
    baseline = state.history[0].metrics.iteration_time_ms
    best = state.best_record.metrics.iteration_time_ms
    improvement = (baseline - best) / max(1e-9, baseline)
    base_config = state.history[0].action.config
    best_config = state.best_record.action.config
    patch = {}
    for key, value in best_config.params.items():
        if base_config.params.get(key) != value:
            patch[key] = value
    if not patch:
        return []
    evidence = {
        "records": [record.step for record in state.history],
        "baseline_ms": baseline,
        "best_ms": best,
        "improvement": improvement,
        "history_count": len(state.history),
    }
    confidence = min(1.0, max(0.1, improvement * 2.0))
    return [
        {
            "id": str(uuid.uuid4()),
            "context": asdict(context),
            "config_patch": patch,
            "improvement": improvement,
            "confidence": confidence,
            "evidence": evidence,
        }
    ]


def persist_rules(path: str, rules: List[Dict]) -> None:
    if not rules:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        for rule in rules:
            handle.write(json.dumps(rule) + "\n")


def persist_avoid_rules(path: str, avoids: List[Dict]) -> None:
    if not avoids:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        for rule in avoids:
            handle.write(json.dumps(rule) + "\n")
