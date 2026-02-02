from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional


@dataclass
class Rule:
    id: str
    context: Dict[str, Any]
    config_patch: Dict[str, Any]
    improvement: float
    evidence: Dict[str, Any] = field(default_factory=dict)
    source: str = "online_tuning"
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    last_used: Optional[str] = None
    tries: int = 0
    wins: int = 0
    confidence: float = 0.5
    rule_type: str = "positive"  # positive or avoid

    @property
    def success_rate(self) -> float:
        if self.tries <= 0:
            return 0.0
        return self.wins / self.tries


@dataclass
class SurrogateRecord:
    context: Dict[str, Any]
    config: Dict[str, Any]
    metrics: Dict[str, Any]
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")


@dataclass
class MemorySchema:
    schema_version: str = "2.0"
