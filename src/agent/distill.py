from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

from ..types import ContextSignature
from .state import TuningState


def distill_semantic_rules(state: TuningState, context: ContextSignature) -> List[Dict]:
    if not state.best_record or not state.history:
        return []
    baseline = state.history[0].metrics.iteration_time_ms
    best = state.best_record.metrics.iteration_time_ms
    improvement = (baseline - best) / max(1e-9, baseline)
    base_config = state.history[0].action.config
    best_config = state.best_record.action.config
    rules: List[Dict] = []
    for key, value in best_config.params.items():
        if base_config.params.get(key) == value:
            continue
        rules.append(
            {
                "schema_version": "1.0",
                "rule_id": f"{key}_{state.best_record.step}",
                "context": asdict(context),
                "condition": {"param": key, "context": asdict(context)},
                "action": {"set": {key: value}},
                "effect": {
                    "metric": "iteration_time_ms",
                    "improvement": improvement,
                },
                "evidence_refs": [f"metric:{state.best_record.step}:primary"],
                "risk": {"level": "unknown"},
            }
        )
    return rules


def persist_rules(path: str, rules: List[Dict]) -> None:
    if not rules:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        for rule in rules:
            handle.write(json.dumps(rule) + "\n")


def persist_report(path: str, rules: List[Dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("Distillation report\n\n")
        handle.write(f"Rules: {len(rules)}\n")
