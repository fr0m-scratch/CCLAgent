#!/usr/bin/env python3
from __future__ import annotations

import sys
import json
from pathlib import Path


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: replay_trace.py <artifacts_run_dir>")
        sys.exit(1)
    run_dir = Path(sys.argv[1])
    trace_path = run_dir / "trace" / "events.jsonl"
    if not trace_path.exists():
        print("No trace found")
        sys.exit(1)
    with trace_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            print(f"[{event.get('phase')}] step={event.get('step')} type={event.get('type')} status={event.get('status')}")


if __name__ == "__main__":
    main()
