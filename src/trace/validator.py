from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from .reader import read_events


CRITICAL_REF_EVENT_TYPES = {
    "decision.select_action",
    "search.prune",
    "stop.decision",
    "postrun.distill.rule",
}

REQUIRED_EVENT_FIELDS = {
    "schema_version",
    "event_id",
    "ts",
    "run_id",
    "phase",
    "step",
    "actor",
    "type",
    "payload",
    "refs",
    "causal_refs",
    "quality_flags",
    "status",
}


@dataclass
class TraceValidationReport:
    path: str
    total_events: int = 0
    schema_errors: List[str] = field(default_factory=list)
    ref_errors: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.schema_errors and not self.ref_errors

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "total_events": self.total_events,
            "schema_errors": list(self.schema_errors),
            "ref_errors": list(self.ref_errors),
            "ok": self.ok,
        }


def validate_event_schema(event: Any) -> List[str]:
    errors: List[str] = []
    if not isinstance(event, dict):
        return ["event_not_dict"]
    for key in REQUIRED_EVENT_FIELDS:
        if key not in event:
            errors.append(f"missing:{key}")
    if "event_id" in event and not isinstance(event.get("event_id"), str):
        errors.append("invalid_type:event_id")
    if "payload" in event and not isinstance(event.get("payload"), dict):
        errors.append("invalid_type:payload")
    if "refs" in event:
        errors.extend(_validate_string_list(event.get("refs"), key="refs"))
    if "causal_refs" in event:
        errors.extend(_validate_string_list(event.get("causal_refs"), key="causal_refs"))
    if "quality_flags" in event:
        errors.extend(_validate_string_list(event.get("quality_flags"), key="quality_flags"))
    return errors


def validate_event_refs(event: Any) -> List[str]:
    errors: List[str] = []
    if not isinstance(event, dict):
        return ["event_not_dict"]
    event_type = str(event.get("type", ""))
    refs = event.get("refs") if isinstance(event.get("refs"), list) else []
    if event_type in CRITICAL_REF_EVENT_TYPES and not refs:
        errors.append(f"refs_required:{event_type}")
    if event_type == "llm.call":
        if not refs:
            errors.append("refs_required:llm.call")
        elif not any(str(item).startswith("llm:call_") for item in refs):
            errors.append("refs_missing_llm_call")
    return errors


def validate_trace_file(path: str | Path) -> TraceValidationReport:
    path = Path(path)
    report = TraceValidationReport(path=str(path))
    if not path.exists():
        report.schema_errors.append("missing_trace_file")
        return report
    for line_no, event in enumerate(read_events(str(path)), start=1):
        report.total_events += 1
        schema_errors = validate_event_schema(event)
        for error in schema_errors:
            report.schema_errors.append(f"line:{line_no}:{error}")
        ref_errors = validate_event_refs(event)
        for error in ref_errors:
            report.ref_errors.append(f"line:{line_no}:{error}")
    return report


def _validate_string_list(value: Any, *, key: str) -> List[str]:
    errors: List[str] = []
    if not isinstance(value, list):
        return [f"invalid_type:{key}"]
    for item in value:
        if not isinstance(item, str):
            errors.append(f"invalid_item_type:{key}")
            break
    return errors
