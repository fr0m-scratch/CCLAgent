#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: export_run_report.py <artifacts_run_dir> [output.md]")
        sys.exit(1)
    run_dir = Path(sys.argv[1])
    output = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    report = []
    report.append(f"# Run report: {run_dir.name}\n")
    context = run_dir / "offline" / "context_snapshot.json"
    if context.exists():
        report.append("## Context\n")
        report.append(context.read_text())
    convergence = run_dir / "postrun" / "convergence.json"
    if convergence.exists():
        report.append("## Convergence\n")
        report.append(convergence.read_text())
    rules = run_dir / "postrun" / "rules_distilled.jsonl"
    if rules.exists():
        report.append("## Distilled Rules\n")
        report.append(rules.read_text())

    report.append("## Online Observability\n")
    steps_dir = run_dir / "steps"
    debug_files = sorted(steps_dir.glob("step_*_nccl_debug_summary.json"))
    profiler_files = sorted(steps_dir.glob("step_*_profiler_summary.json"))
    failure_files = sorted(steps_dir.glob("step_*_failure_mode.json"))
    report.append(
        json.dumps(
            {
                "nccl_debug_steps": len(debug_files),
                "profiler_steps": len(profiler_files),
                "failure_mode_steps": len(failure_files),
            },
            indent=2,
        )
    )

    rep_files = sorted(steps_dir.glob("step_*_replicate_summary.json"))
    if rep_files:
        ci_values = []
        for path in rep_files:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            value = payload.get("ci95_ms")
            try:
                ci_values.append(float(value))
            except (TypeError, ValueError):
                continue
        if ci_values:
            report.append("## Replicate Significance\n")
            report.append(
                json.dumps(
                    {
                        "steps_with_replicates": len(rep_files),
                        "mean_ci95_ms": sum(ci_values) / max(1, len(ci_values)),
                        "max_ci95_ms": max(ci_values),
                        "min_ci95_ms": min(ci_values),
                    },
                    indent=2,
                )
            )

    whitebox = run_dir / "whitebox" / "evidence.jsonl"
    if whitebox.exists():
        lines = [line for line in whitebox.read_text(encoding="utf-8").splitlines() if line.strip()]
        report.append("## Whitebox Evidence\n")
        report.append(json.dumps({"evidence_count": len(lines)}, indent=2))

    twin = run_dir / "models" / "twin.json"
    if twin.exists():
        report.append("## Digital Twin\n")
        report.append(twin.read_text())
    output_text = "\n\n".join(report)
    if output:
        output.write_text(output_text, encoding="utf-8")
    else:
        print(output_text)


if __name__ == "__main__":
    main()
