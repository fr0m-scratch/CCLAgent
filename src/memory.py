from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

from .types import ContextSignature, NCCLConfig, Metrics
from .utils import read_json, write_json


@dataclass
class Rule:
    context: Dict[str, Any]
    config_patch: Dict[str, Any]
    improvement: float
    evidence: Dict[str, Any] = field(default_factory=dict)
    source: str = "online_tuning"


@dataclass
class SurrogateRecord:
    context: Dict[str, Any]
    config: Dict[str, Any]
    metrics: Dict[str, Any]


class MemoryStore:
    def __init__(self, path: str):
        self.path = path
        self.rules: List[Rule] = []
        self.surrogates: List[SurrogateRecord] = []
        self._load()

    def _load(self) -> None:
        try:
            payload = read_json(self.path)
        except FileNotFoundError:
            return
        for rule in payload.get("rules", []):
            self.rules.append(Rule(**rule))
        for record in payload.get("surrogates", []):
            self.surrogates.append(SurrogateRecord(**record))

    def save(self) -> None:
        payload = {
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
    ) -> None:
        self.rules.append(
            Rule(
                context=asdict(context),
                config_patch=config_patch,
                improvement=improvement,
                evidence=evidence or {},
            )
        )

    def add_surrogate_record(self, context: ContextSignature, config: NCCLConfig, metrics: Metrics) -> None:
        self.surrogates.append(
            SurrogateRecord(
                context=asdict(context),
                config=config.params,
                metrics=asdict(metrics),
            )
        )

    def get_rules(self, context: ContextSignature) -> List[Rule]:
        context_dict = asdict(context)
        filtered = []
        for rule in self.rules:
            if self._context_match(rule.context, context_dict):
                filtered.append(rule)
        return sorted(filtered, key=lambda r: r.improvement, reverse=True)

    def _context_match(self, rule_context: Dict[str, Any], target: Dict[str, Any]) -> bool:
        for key in ("workload", "topology", "scale", "nodes"):
            if rule_context.get(key) != target.get(key):
                return False
        return True
