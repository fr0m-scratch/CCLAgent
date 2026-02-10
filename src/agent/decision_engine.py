from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TunerContext:
    req_id: str
    coll_type: str
    bytes: int
    nranks: int
    topo_sig: str = "unknown"
    comm_hash: str = ""
    constraints: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TunerResponse:
    req_id: str
    status: str
    override: Dict[str, Any] = field(default_factory=dict)
    source: str = "fallback"
    reasons: List[str] = field(default_factory=list)
    evidence_refs: List[str] = field(default_factory=list)
    fallback_used: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "req_id": self.req_id,
            "status": self.status,
            "override": dict(self.override),
            "source": self.source,
            "reasons": list(self.reasons),
            "evidence_refs": list(self.evidence_refs),
            "fallback_used": bool(self.fallback_used),
        }


class DecisionEngine:
    """Rule-first collective tuner decision engine."""

    def __init__(self, *, rules: Optional[List[Dict[str, Any]]] = None, surrogate: Any = None) -> None:
        self.rules = list(rules or [])
        self.surrogate = surrogate

    def decide_for_collective(self, ctx: TunerContext) -> TunerResponse:
        for rule in self.rules:
            if not self._matches(rule, ctx):
                continue
            override = rule.get("override") if isinstance(rule.get("override"), dict) else {}
            return TunerResponse(
                req_id=ctx.req_id,
                status="ok",
                override=override,
                source="rule",
                reasons=[str(rule.get("name") or "rule_match")],
                evidence_refs=[f"rule:{str(rule.get('name') or 'rule_match')}"],
                fallback_used=False,
            )

        if self.surrogate is not None and hasattr(self.surrogate, "decide_for_collective"):
            out = self.surrogate.decide_for_collective(ctx)
            if isinstance(out, dict):
                return TunerResponse(
                    req_id=ctx.req_id,
                    status="ok",
                    override=out,
                    source="surrogate",
                    reasons=["surrogate"],
                    evidence_refs=["model:surrogate"],
                    fallback_used=False,
                )

        return TunerResponse(
            req_id=ctx.req_id,
            status="ok",
            override={},
            source="fallback",
            reasons=["no_rule_match"],
            evidence_refs=[],
            fallback_used=True,
        )

    def decide_global_env_delta(self, *, recent_metrics: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        # Placeholder for future policy. Keep deterministic and safe by default.
        _ = recent_metrics
        return {}

    def _matches(self, rule: Dict[str, Any], ctx: TunerContext) -> bool:
        coll = rule.get("coll_type")
        if coll is not None and str(coll).lower() != ctx.coll_type.lower():
            return False

        min_bytes = _to_int(rule.get("min_bytes"))
        if min_bytes is not None and ctx.bytes < min_bytes:
            return False

        max_bytes = _to_int(rule.get("max_bytes"))
        if max_bytes is not None and ctx.bytes > max_bytes:
            return False

        min_ranks = _to_int(rule.get("min_ranks"))
        if min_ranks is not None and ctx.nranks < min_ranks:
            return False

        max_ranks = _to_int(rule.get("max_ranks"))
        if max_ranks is not None and ctx.nranks > max_ranks:
            return False

        topo_contains = rule.get("topo_contains")
        if topo_contains is not None:
            token = str(topo_contains).lower().strip()
            if token and token not in str(ctx.topo_sig).lower():
                return False

        return True


def _to_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None
