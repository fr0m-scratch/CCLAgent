from __future__ import annotations

from typing import Any, Dict, List

from .store import EvidenceStore, parse_evidence_ref


CONTRACT_PREFIXES = (
    "decision.",
    "hypothesis.",
    "postrun.distill",
)


def validate_contract(trace_events: List[Dict[str, Any]], evidence_store: EvidenceStore) -> List[str]:
    violations: List[str] = []
    for idx, event in enumerate(trace_events):
        if not isinstance(event, dict):
            violations.append(f"event_not_dict:{idx}")
            continue
        event_type = str(event.get("type") or "")
        if not _requires_contract(event_type):
            continue

        refs = event.get("refs") if isinstance(event.get("refs"), list) else []
        if not refs:
            violations.append(f"missing_refs:{idx}:{event_type}")
            continue

        evidence_refs = [str(ref) for ref in refs if parse_evidence_ref(str(ref)) is not None]
        if not evidence_refs:
            # Transitional mode: allow non-evidence refs while the repository
            # progressively migrates event refs to evidence:* IDs.
            continue

        for ref in evidence_refs:
            if evidence_store.get(ref) is None:
                violations.append(f"unresolved_evidence_ref:{idx}:{event_type}:{ref}")
    return violations


def _requires_contract(event_type: str) -> bool:
    return any(event_type.startswith(prefix) for prefix in CONTRACT_PREFIXES)
