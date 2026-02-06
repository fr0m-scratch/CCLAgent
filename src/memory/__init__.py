from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
import uuid
from typing import Any, Dict, List, Optional

from .schema import MemorySchema, Rule, SurrogateRecord
from .index import rule_score, context_similarity
from ..types import ContextSignature, NCCLConfig, Metrics, MemoryConfig, RunContext
from ..utils import read_json, write_json


class MemoryStore:
    def __init__(self, config: MemoryConfig, run_context: Optional[RunContext] = None):
        self.config = config
        self.path = config.path
        self.rules: List[Rule] = []
        self.surrogates: List[SurrogateRecord] = []
        self.schema_version = MemorySchema().schema_version
        self.run_context = run_context
        self._load()

    def _load(self) -> None:
        try:
            payload = read_json(self.path)
        except FileNotFoundError:
            return
        except json.JSONDecodeError:
            # Empty or partially-written memory files should not crash startup.
            self.save()
            return
        schema_version = payload.get("schema_version")
        if not schema_version:
            self._load_v1(payload)
            return
        self.schema_version = schema_version
        for rule in payload.get("rules", []):
            self.rules.append(Rule(**rule))
        for record in payload.get("surrogates", []):
            self.surrogates.append(SurrogateRecord(**record))

    def _load_v1(self, payload: Dict[str, Any]) -> None:
        for rule in payload.get("rules", []):
            self.rules.append(
                Rule(
                    id=str(uuid.uuid4()),
                    context=rule.get("context", {}),
                    config_patch=rule.get("config_patch", {}),
                    improvement=rule.get("improvement", 0.0),
                    evidence=rule.get("evidence", {}),
                    source=rule.get("source", "legacy"),
                )
            )
        for record in payload.get("surrogates", []):
            self.surrogates.append(SurrogateRecord(**record))

    def save(self) -> None:
        payload = {
            "schema_version": self.schema_version,
            "rules": [asdict(rule) for rule in self.rules],
            "surrogates": [asdict(record) for record in self.surrogates],
        }
        write_json(self.path, payload)

    def add_rule(
        self,
        context: ContextSignature,
        config_patch: Dict[str, Any],
        improvement: float,
        evidence: Dict[str, Any] | None = None,
        rule_type: str = "positive",
    ) -> Rule:
        rule = Rule(
            id=str(uuid.uuid4()),
            context=asdict(context),
            config_patch=config_patch,
            improvement=improvement,
            evidence=evidence or {},
            rule_type=rule_type,
        )
        self.rules.append(rule)
        self._handle_conflicts(rule)
        return rule

    def add_avoid_rule(self, context: ContextSignature, config_patch: Dict[str, Any], evidence: Dict[str, Any]) -> Rule:
        return self.add_rule(context, config_patch, improvement=0.0, evidence=evidence, rule_type="avoid")

    def add_surrogate_record(self, context: ContextSignature, config: NCCLConfig, metrics: Metrics) -> None:
        self.surrogates.append(
            SurrogateRecord(
                context=asdict(context),
                config=config.params,
                metrics=metrics.__dict__,
            )
        )

    def retrieve_rules(self, context: ContextSignature, top_k: int = 5, include_negative: bool = False) -> List[Rule]:
        context_dict = asdict(context)
        scored = []
        for rule in self.rules:
            if rule.rule_type == "avoid" and not include_negative:
                continue
            score = rule_score(rule, context_dict, self.config.half_life_days)
            scored.append((score, rule))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [rule for score, rule in scored[:top_k] if score > 0]

    def retrieve_rules_with_scores(
        self, context: ContextSignature, top_k: int = 5, include_negative: bool = False
    ) -> List[Dict[str, Any]]:
        context_dict = asdict(context)
        scored = []
        for rule in self.rules:
            if rule.rule_type == "avoid" and not include_negative:
                continue
            score = rule_score(rule, context_dict, self.config.half_life_days)
            scored.append((score, rule))
        scored.sort(key=lambda item: item[0], reverse=True)
        out: List[Dict[str, Any]] = []
        for score, rule in scored[:top_k]:
            if score <= 0:
                continue
            out.append({"rule": rule, "score": score})
        return out

    def mark_rule_usage(self, rule_id: str, success: bool) -> None:
        for rule in self.rules:
            if rule.id != rule_id:
                continue
            rule.tries += 1
            if success:
                rule.wins += 1
                rule.confidence = min(1.0, rule.confidence + 0.05)
            else:
                rule.confidence = max(0.1, rule.confidence - 0.05)
            rule.last_used = datetime.utcnow().isoformat() + "Z"
            break

    def _handle_conflicts(self, new_rule: Rule) -> None:
        for rule in self.rules:
            if rule.id == new_rule.id:
                continue
            if not self._conflicts(rule, new_rule):
                continue
            rule.confidence = max(0.1, rule.confidence * 0.9)
            new_rule.confidence = max(0.1, new_rule.confidence * 0.9)

    def _conflicts(self, a: Rule, b: Rule) -> bool:
        if context_similarity(a.context, b.context) < 0.6:
            return False
        for key, value in a.config_patch.items():
            if key in b.config_patch and b.config_patch[key] != value:
                return True
        return False
