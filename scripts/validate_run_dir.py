#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


REQUIRED = [
    "trace/events.jsonl",
    "offline/context_snapshot.json",
]


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: validate_run_dir.py <artifacts_run_dir>")
        sys.exit(1)
    run_dir = Path(sys.argv[1])
    missing = []
    for rel in REQUIRED:
        path = run_dir / rel
        if not path.exists():
            missing.append(rel)
    if missing:
        print("Missing:")
        for item in missing:
            print(f"- {item}")
        sys.exit(1)
    print("OK")


if __name__ == "__main__":
    main()
