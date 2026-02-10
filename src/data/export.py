from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import List


_STEP_RE = re.compile(r"^step_(\d+)$")


def _load_steps(run_dir: str) -> List[dict]:
    steps_dir = Path(run_dir) / "steps"
    records = []
    for path in sorted(steps_dir.glob("step_*.json")):
        if not _STEP_RE.match(path.stem):
            continue
        records.append(json.loads(path.read_text(encoding="utf-8")))
    return records


def export_sft_dataset(run_dir: str, out_path: str) -> None:
    run_context_path = Path(run_dir) / "run_context.json"
    context = {}
    if run_context_path.exists():
        context = json.loads(run_context_path.read_text(encoding="utf-8"))
    records = _load_steps(run_dir)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "type": "sft",
                    "created_at": datetime.utcnow().isoformat() + "Z",
                    "run_dir": run_dir,
                }
            )
            + "\n"
        )
        for record in records:
            step = record.get("step")
            context_pack = _load_step_artifact(run_dir, step, "context_pack")
            decision_record = _load_step_artifact(run_dir, step, "decision_record")
            decision_bundle = _load_step_artifact(run_dir, step, "decision_bundle")
            payload = {
                "context": context,
                "context_pack": context_pack,
                "metrics": record.get("metrics", {}),
                "action": record.get("action", {}),
                "decision_record": decision_record,
                "decision_bundle": decision_bundle,
                "evidence_refs": _extract_refs(decision_bundle),
            }
            handle.write(json.dumps(payload) + "\n")


def export_rl_dataset(run_dir: str, out_path: str) -> None:
    records = _load_steps(run_dir)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "type": "rl",
                    "created_at": datetime.utcnow().isoformat() + "Z",
                    "run_dir": run_dir,
                }
            )
            + "\n"
        )
        for idx, record in enumerate(records[:-1]):
            next_record = records[idx + 1]
            reward = 0.0
            try:
                reward = record["metrics"]["iteration_time_ms"] - next_record["metrics"]["iteration_time_ms"]
            except Exception:
                reward = 0.0
            step = record.get("step")
            context_pack = _load_step_artifact(run_dir, step, "context_pack")
            decision_bundle = _load_step_artifact(run_dir, step, "decision_bundle")
            payload = {
                "state": {
                    "metrics": record.get("metrics", {}),
                    "context_pack": context_pack,
                },
                "action": record.get("action", {}),
                "reward": reward,
                "next_state": {
                    "metrics": next_record.get("metrics", {}),
                    "context_pack": _load_step_artifact(run_dir, next_record.get("step"), "context_pack"),
                },
                "decision_bundle": decision_bundle,
                "evidence_refs": _extract_refs(decision_bundle),
            }
            handle.write(json.dumps(payload) + "\n")


def _load_step_artifact(run_dir: str, step: int | None, suffix: str) -> dict:
    if step is None:
        return {}
    path = Path(run_dir) / "steps" / f"step_{step}_{suffix}.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _extract_refs(payload: dict) -> list:
    if not isinstance(payload, dict):
        return []
    refs = payload.get("call_chain")
    if isinstance(refs, list):
        return refs
    selected = payload.get("chosen_action", {}).get("selected_candidate_ref") if isinstance(payload.get("chosen_action"), dict) else None
    return [selected] if isinstance(selected, str) else []
