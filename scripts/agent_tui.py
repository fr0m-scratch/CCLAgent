#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    parser = argparse.ArgumentParser(description="CCLAgent TUI dashboard")
    parser.add_argument("--artifacts-root", default="artifacts")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--provider", default="ollama")
    parser.add_argument("--model", default="deepseek-r1:8b")
    parser.add_argument("--env-file", default=".env.local")
    parser.add_argument("--poll", type=float, default=1.0)
    parser.add_argument("--tail", type=int, default=200)
    args = parser.parse_args()

    try:
        from src.tui.app import AgentDashboard
    except RuntimeError as exc:
        print(str(exc))
        print("Install with: pip install textual rich")
        sys.exit(1)

    app = AgentDashboard(
        artifacts_root=args.artifacts_root,
        run_dir=args.run_dir,
        provider=args.provider,
        model=args.model,
        env_file=args.env_file,
        poll_interval=args.poll,
        tail_lines=args.tail,
    )
    app.run()


if __name__ == "__main__":
    main()
