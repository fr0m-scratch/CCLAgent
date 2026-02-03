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
    output_text = "\n\n".join(report)
    if output:
        output.write_text(output_text, encoding="utf-8")
    else:
        print(output_text)


if __name__ == "__main__":
    main()
