from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional
import json

from .schema import Evidence, new_evidence_id


def as_evidence_ref(evidence_id: str) -> str:
    return f"evidence:{evidence_id}"


def parse_evidence_ref(ref: str) -> str | None:
    text = str(ref)
    if not text.startswith("evidence:"):
        return None
    eid = text.split(":", 1)[1].strip()
    return eid or None


class EvidenceStore:
    def __init__(self, evidences: Iterable[Evidence] | None = None) -> None:
        self._by_id: Dict[str, Evidence] = {}
        self._by_kind: Dict[str, List[str]] = {}
        if evidences is not None:
            for evidence in evidences:
                self.add_evidence(evidence)

    def add_evidence(self, evidence: Evidence) -> str:
        evidence_id = str(evidence.id or new_evidence_id())
        if not evidence.id:
            evidence.id = evidence_id
        self._by_id[evidence_id] = evidence
        kind = str(evidence.kind or "unknown")
        if kind not in self._by_kind:
            self._by_kind[kind] = []
        if evidence_id not in self._by_kind[kind]:
            self._by_kind[kind].append(evidence_id)
        return as_evidence_ref(evidence_id)

    def get(self, ref_or_id: str) -> Optional[Evidence]:
        evidence_id = parse_evidence_ref(ref_or_id) or str(ref_or_id)
        return self._by_id.get(evidence_id)

    def list_by_kind(self, kind: str) -> List[Evidence]:
        out: List[Evidence] = []
        for evidence_id in self._by_kind.get(kind, []):
            evidence = self._by_id.get(evidence_id)
            if evidence is not None:
                out.append(evidence)
        return out

    def all(self) -> List[Evidence]:
        return list(self._by_id.values())

    def flush(self, run_dir: str | Path) -> Path:
        run_dir = Path(run_dir)
        whitebox_dir = run_dir / "whitebox"
        whitebox_dir.mkdir(parents=True, exist_ok=True)
        out_path = whitebox_dir / "evidence.jsonl"
        with out_path.open("w", encoding="utf-8") as handle:
            for evidence in self.all():
                handle.write(json.dumps(evidence.to_dict(), ensure_ascii=False) + "\n")
        return out_path

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "EvidenceStore":
        path = Path(path)
        if not path.exists():
            return cls()
        evidences: List[Evidence] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if not text:
                    continue
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    evidences.append(Evidence.from_dict(payload))
        return cls(evidences)
