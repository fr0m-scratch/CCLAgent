from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List
import uuid


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_evidence_id() -> str:
    return str(uuid.uuid4())


@dataclass
class Evidence:
    id: str
    kind: str
    source: str
    payload: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "source": self.source,
            "payload": self.payload,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Evidence":
        return cls(
            id=str(payload.get("id") or new_evidence_id()),
            kind=str(payload.get("kind") or "unknown"),
            source=str(payload.get("source") or "unknown"),
            payload=payload.get("payload") if isinstance(payload.get("payload"), dict) else {},
            created_at=str(payload.get("created_at") or utc_now_iso()),
        )


@dataclass
class Claim:
    text: str
    confidence: float = 0.5
    evidence_refs: List[str] = field(default_factory=list)


@dataclass
class Decision:
    action: str
    params_delta: Dict[str, Any] = field(default_factory=dict)
    expected_effect: Dict[str, Any] = field(default_factory=dict)
    claims: List[Claim] = field(default_factory=list)
    risk_budget_used: float = 0.0
    rollback_plan: Dict[str, Any] = field(default_factory=dict)
