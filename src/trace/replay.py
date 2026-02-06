"""WP10b: Deterministic replay engine.

Loads trace events and step artifacts from a completed run, then
replays the decision sequence to verify consistency.  Mismatches
between original and replayed decisions indicate non-determinism
or configuration drift.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .events import TraceEvent
from .writer import JsonlTraceReader


@dataclass
class ReplayMismatch:
    """A single mismatch between original and replayed decisions."""

    step: int
    field: str
    original: Any
    replayed: Any
    severity: str = "info"  # info, warning, error


@dataclass
class ReplayReport:
    """Result of replaying a tuning run."""

    run_id: str
    total_steps: int = 0
    steps_replayed: int = 0
    action_match_rate: float = 0.0
    config_match_rate: float = 0.0
    mismatches: List[ReplayMismatch] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_run_artifacts(run_dir: str) -> Dict[str, Any]:
    """Load all step artifacts and trace events from a run directory."""
    rd = Path(run_dir)
    artifacts: Dict[str, Any] = {
        "steps": {},
        "decisions": {},
        "events": [],
        "convergence": {},
        "run_context": {},
    }

    # Load step artifacts
    steps_dir = rd / "steps"
    if steps_dir.is_dir():
        for f in sorted(steps_dir.glob("step_*.json")):
            name = f.stem
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                if "_decision" in name:
                    step_num = name.split("_")[1]
                    artifacts["decisions"][step_num] = data
                else:
                    step_num = name.split("_")[1]
                    artifacts["steps"][step_num] = data
            except (json.JSONDecodeError, IndexError):
                continue

    # Load trace events
    events_path = rd / "events.jsonl"
    if events_path.exists():
        reader = JsonlTraceReader(str(events_path))
        artifacts["events"] = reader.read_all()

    # Load convergence
    conv_path = rd / "postrun" / "convergence.json"
    if conv_path.exists():
        try:
            artifacts["convergence"] = json.loads(conv_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass

    # Load run context
    ctx_path = rd / "run_context.json"
    if ctx_path.exists():
        try:
            artifacts["run_context"] = json.loads(ctx_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass

    return artifacts


def replay_decisions(artifacts: Dict[str, Any]) -> ReplayReport:
    """Replay the decision sequence and check for consistency.

    Compares each step's action.kind and config against the decision
    record to verify the decision pipeline produced consistent outputs.
    """
    run_id = artifacts.get("run_context", {}).get("run_id", "unknown")
    steps = artifacts.get("steps", {})
    decisions = artifacts.get("decisions", {})

    report = ReplayReport(run_id=run_id)
    report.total_steps = len(steps)

    action_matches = 0
    config_matches = 0
    replayed = 0

    for step_key in sorted(steps.keys(), key=lambda x: int(x) if x.isdigit() else 0):
        step_data = steps[step_key]
        decision_data = decisions.get(step_key, {})
        if not decision_data:
            report.notes.append(f"Step {step_key}: no decision record found")
            continue

        replayed += 1
        step_action = step_data.get("action", {})
        decision_action = decision_data.get("chosen_action", {})

        # Check action kind match
        step_kind = step_action.get("kind", "")
        decision_kind = decision_action.get("kind", "")
        if step_kind == decision_kind:
            action_matches += 1
        else:
            report.mismatches.append(ReplayMismatch(
                step=int(step_key) if step_key.isdigit() else 0,
                field="action.kind",
                original=step_kind,
                replayed=decision_kind,
                severity="warning",
            ))

        # Check config match
        step_config = step_action.get("config", {})
        decision_config = decision_action.get("config", {})
        if step_config == decision_config:
            config_matches += 1
        else:
            diff_keys = set(step_config.keys()) ^ set(decision_config.keys())
            changed = {
                k for k in set(step_config) & set(decision_config)
                if step_config[k] != decision_config[k]
            }
            if diff_keys or changed:
                report.mismatches.append(ReplayMismatch(
                    step=int(step_key) if step_key.isdigit() else 0,
                    field="action.config",
                    original={"diff_keys": list(diff_keys), "changed": list(changed)},
                    replayed=None,
                    severity="error" if changed else "info",
                ))

    report.steps_replayed = replayed
    report.action_match_rate = action_matches / replayed if replayed > 0 else 0.0
    report.config_match_rate = config_matches / replayed if replayed > 0 else 0.0

    if report.action_match_rate < 1.0:
        report.notes.append(
            f"Action mismatch detected: {action_matches}/{replayed} steps match."
        )
    if report.config_match_rate < 1.0:
        report.notes.append(
            f"Config mismatch detected: {config_matches}/{replayed} steps match."
        )

    return report
