#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent.decision_bundle import validate_decision_bundle
from src.trace import validate_trace_file, validate_whitebox_contract

NUMERIC_CANDIDATE_REF = re.compile(r"^candidate:(\d+):(\d+_\d+)$")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate key artifact schemas and refs.")
    parser.add_argument("--run-dir", required=True, help="Path to a single run directory under artifacts.")
    parser.add_argument(
        "--check",
        default="all",
        choices=("all", "trace", "decision-bundle", "whitebox-contract"),
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
    if args.check in ("all", "whitebox-contract"):
        failed = _check_whitebox_contract(run_dir) or failed
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
        errors.extend(_validate_bundle_alignment(run_dir, payload))
        if errors:
            has_error = True
            print(f"[decision-bundle] {path.name}:")
            for item in errors:
                print(f"  - {item}")
    if not has_error:
        print(f"[decision-bundle] validated {len(paths)} file(s)")
    return has_error


def _check_whitebox_contract(run_dir: Path) -> bool:
    violations = validate_whitebox_contract(run_dir)
    if not violations:
        print("[whitebox-contract] OK")
        return False
    print("[whitebox-contract] violations:")
    for item in violations[:100]:
        print(f"  - {item}")
    return True


def _validate_bundle_alignment(run_dir: Path, payload: dict) -> list[str]:
    errors: list[str] = []
    step = payload.get("step")
    if not isinstance(step, int):
        errors.append("invalid_step")
        return errors

    chosen = payload.get("chosen_action", {}) if isinstance(payload.get("chosen_action"), dict) else {}
    chosen_ref = chosen.get("selected_candidate_ref")
    candidates = payload.get("candidates_considered") if isinstance(payload.get("candidates_considered"), list) else []
    candidate_refs = {
        item.get("candidate_ref")
        for item in candidates
        if isinstance(item, dict) and isinstance(item.get("candidate_ref"), str)
    }
    if candidates and isinstance(chosen_ref, str) and chosen_ref not in candidate_refs:
        errors.append("chosen_selected_candidate_ref_not_in_bundle_candidates")

    trace_path = run_dir / "steps" / f"step_{step}_candidates_trace.json"
    if trace_path.exists():
        try:
            trace_payload = json.loads(trace_path.read_text(encoding="utf-8"))
        except Exception:
            errors.append("invalid_candidates_trace_json")
            trace_payload = {}
        trace_candidates = trace_payload.get("candidates") if isinstance(trace_payload.get("candidates"), list) else []
        trace_refs = {
            f"candidate:{step}:{item.get('candidate_id')}"
            for item in trace_candidates
            if isinstance(item, dict) and item.get("candidate_id") is not None
        }
        if isinstance(chosen_ref, str) and chosen_ref and chosen_ref not in trace_refs:
            errors.append("chosen_selected_candidate_ref_not_in_candidates_trace")

    if chosen.get("kind") == "numeric":
        if not isinstance(chosen_ref, str) or not chosen_ref:
            errors.append("missing_numeric_selected_candidate_ref")
        else:
            match = NUMERIC_CANDIDATE_REF.match(chosen_ref)
            if not match:
                errors.append("invalid_numeric_selected_candidate_ref_format")
            else:
                ref_step = int(match.group(1))
                if ref_step != step:
                    errors.append("numeric_selected_candidate_ref_step_mismatch")

    return errors


if __name__ == "__main__":
    main()
