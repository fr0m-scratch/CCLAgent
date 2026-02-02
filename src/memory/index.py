from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Dict

from ..utils import tokenize


def _numeric_similarity(a: float | int | None, b: float | int | None) -> float:
    if a is None or b is None:
        return 0.0
    try:
        a_val = float(a)
        b_val = float(b)
    except (TypeError, ValueError):
        return 0.0
    denom = max(abs(a_val), abs(b_val), 1.0)
    return max(0.0, 1.0 - abs(a_val - b_val) / denom)


def _categorical_similarity(a: Any, b: Any) -> float:
    if a is None or b is None:
        return 0.0
    return 1.0 if a == b else 0.0


def _text_similarity(a: str | None, b: str | None) -> float:
    if not a or not b:
        return 0.0
    tokens_a = set(tokenize(a))
    tokens_b = set(tokenize(b))
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / max(1, len(tokens_a | tokens_b))


def context_similarity(rule_context: Dict[str, Any], target: Dict[str, Any]) -> float:
    weights = {
        "workload": 0.25,
        "workload_kind": 0.1,
        "topology": 0.2,
        "scale": 0.1,
        "nodes": 0.15,
        "model": 0.1,
        "framework": 0.05,
        "gpus_per_node": 0.05,
        "gpu_type": 0.05,
        "network": 0.05,
        "nic_count": 0.05,
    }
    score = 0.0
    total = 0.0
    for key, weight in weights.items():
        total += weight
        if key in ("nodes", "gpus_per_node"):
            score += weight * _numeric_similarity(rule_context.get(key), target.get(key))
        elif key in ("model", "framework"):
            score += weight * _text_similarity(str(rule_context.get(key) or ""), str(target.get(key) or ""))
        else:
            score += weight * _categorical_similarity(rule_context.get(key), target.get(key))
    if total <= 0.0:
        return 0.0
    return score / total


def recency_decay(timestamp: str | None, half_life_days: float) -> float:
    if not timestamp:
        return 1.0
    try:
        created = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        now = datetime.utcnow().astimezone(created.tzinfo)
        age_days = (now - created).total_seconds() / 86400.0
    except Exception:
        return 1.0
    if half_life_days <= 0:
        return 1.0
    return math.exp(-age_days / half_life_days)


def rule_score(rule, context: Dict[str, Any], half_life_days: float) -> float:
    similarity = context_similarity(rule.context, context)
    decay = recency_decay(rule.last_used or rule.created_at, half_life_days)
    quality = max(0.1, rule.success_rate) * rule.confidence
    return similarity * decay * quality
