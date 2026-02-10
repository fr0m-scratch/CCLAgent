#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent.decision_bundle import validate_decision_bundle
from src.trace import validate_trace_file, validate_whitebox_contract


REQUIRED = [
    "trace/events.jsonl",
    "offline/context_snapshot.json",
]


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: validate_run_dir.py <artifacts_run_dir>")
        sys.exit(1)

    run_dir = Path(sys.argv[1])
    failed = False

    missing = []
    for rel in REQUIRED:
        path = run_dir / rel
        if not path.exists():
            missing.append(rel)
    if missing:
        failed = True
        print("Missing:")
        for item in missing:
            print(f"- {item}")

    trace_path = run_dir / "trace" / "events.jsonl"
    trace_report = validate_trace_file(trace_path)
    if not trace_report.ok:
        failed = True
        print("Trace validation failed:")
        for item in trace_report.schema_errors[:50]:
            print(f"- {item}")
        for item in trace_report.ref_errors[:50]:
            print(f"- {item}")

    steps_dir = run_dir / "steps"
    bundle_paths = sorted(steps_dir.glob("step_*_decision_bundle.json"))
    if not bundle_paths:
        failed = True
        print("No decision bundles found")
    else:
        for path in bundle_paths:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                failed = True
                print(f"{path.name}: invalid_json ({exc})")
                continue
            errors = validate_decision_bundle(payload)
            if errors:
                failed = True
                print(f"{path.name}:")
                for item in errors:
                    print(f"- {item}")

            # Cross-check selected candidate against candidates trace if present.
            step = payload.get("step") if isinstance(payload, dict) else None
            chosen = payload.get("chosen_action", {}) if isinstance(payload, dict) else {}
            selected_ref = chosen.get("selected_candidate_ref") if isinstance(chosen, dict) else None
            if isinstance(step, int) and isinstance(selected_ref, str) and selected_ref:
                trace_candidates_path = steps_dir / f"step_{step}_candidates_trace.json"
                if trace_candidates_path.exists():
                    try:
                        trace_payload = json.loads(trace_candidates_path.read_text(encoding="utf-8"))
                        trace_candidates = trace_payload.get("candidates") if isinstance(trace_payload.get("candidates"), list) else []
                        trace_refs = {
                            f"candidate:{step}:{item.get('candidate_id')}"
                            for item in trace_candidates
                            if isinstance(item, dict) and item.get("candidate_id") is not None
                        }
                        if selected_ref not in trace_refs:
                            failed = True
                            print(f"{path.name}: selected_ref_not_in_candidates_trace")
                    except Exception as exc:
                        failed = True
                        print(f"{path.name}: invalid_candidates_trace_json ({exc})")

    whitebox_violations = validate_whitebox_contract(run_dir)
    if whitebox_violations:
        failed = True
        print("Whitebox contract violations:")
        for item in whitebox_violations[:100]:
            print(f"- {item}")

    if failed:
        sys.exit(1)
    print("OK")


if __name__ == "__main__":
    main()
