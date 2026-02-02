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
            payload = {
                "context": context,
                "metrics": record.get("metrics", {}),
                "action": record.get("action", {}),
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
            payload = {
                "state": record.get("metrics", {}),
                "action": record.get("action", {}),
                "reward": reward,
                "next_state": next_record.get("metrics", {}),
            }
            handle.write(json.dumps(payload) + "\n")
