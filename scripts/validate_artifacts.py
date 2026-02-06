#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent.decision_bundle import validate_decision_bundle
from src.trace import validate_trace_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate key artifact schemas and refs.")
    parser.add_argument("--run-dir", required=True, help="Path to a single run directory under artifacts.")
    parser.add_argument(
        "--check",
        default="all",
        choices=("all", "trace", "decision-bundle"),
        help="Validation target.",
    )
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"Run dir does not exist: {run_dir}")
        sys.exit(1)

    failed = False
    if args.check in ("all", "trace"):
        failed = _check_trace(run_dir) or failed
    if args.check in ("all", "decision-bundle"):
        failed = _check_decision_bundle(run_dir) or failed
    if failed:
        sys.exit(1)
    print("OK")


def _check_trace(run_dir: Path) -> bool:
    trace_path = run_dir / "trace" / "events.jsonl"
    report = validate_trace_file(trace_path)
    print(f"[trace] total_events={report.total_events} ok={report.ok}")
    if report.schema_errors:
        print("[trace] schema_errors:")
        for item in report.schema_errors[:50]:
            print(f"  - {item}")
    if report.ref_errors:
        print("[trace] ref_errors:")
        for item in report.ref_errors[:50]:
            print(f"  - {item}")
    return not report.ok


def _check_decision_bundle(run_dir: Path) -> bool:
    steps_dir = run_dir / "steps"
    paths = sorted(steps_dir.glob("step_*_decision_bundle.json"))
    if not paths:
        print("[decision-bundle] no decision bundle files found")
        return True
    has_error = False
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[decision-bundle] {path.name}: invalid_json ({exc})")
            has_error = True
            continue
        errors = validate_decision_bundle(payload)
        if errors:
            has_error = True
            print(f"[decision-bundle] {path.name}:")
            for item in errors:
                print(f"  - {item}")
    if not has_error:
        print(f"[decision-bundle] validated {len(paths)} file(s)")
    return has_error


if __name__ == "__main__":
    main()
