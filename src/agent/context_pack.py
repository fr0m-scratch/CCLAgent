from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Optional

from ..types import ContextSignature, WorkloadSpec
from ..utils import write_json


SCHEMA_VERSION = "1.0"


def build_context_pack(
    *,
    phase: str,
    step: Optional[int],
    workload: WorkloadSpec,
    context: ContextSignature,
    observations: Dict[str, Any] | None = None,
    memory_rules: List[Dict[str, Any]] | None = None,
    rag_chunks: List[Dict[str, Any]] | None = None,
    surrogate: Dict[str, Any] | None = None,
    constraints: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "phase": phase,
        "step": step,
        "workload": asdict(workload),
        "context_signature": asdict(context),
        "observations": observations or {},
        "retrieval": {
            "memory_rules": memory_rules or [],
            "rag_chunks": rag_chunks or [],
        },
        "models": surrogate or {},
        "constraints": constraints or {},
    }


def write_context_pack(path: str, payload: Dict[str, Any]) -> None:
    write_json(path, payload)
