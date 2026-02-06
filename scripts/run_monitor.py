#!/usr/bin/env python3
"""
Run the comprehensive CCL Agent Monitor TUI.

Usage:
    python -m scripts.run_monitor [--run-dir <path>] [--artifacts <path>]
    
Examples:
    # Monitor most recent run
    python -m scripts.run_monitor
    
    # Monitor specific run
    python -m scripts.run_monitor --run-dir artifacts/2024-01-15_12-30-00
    
Keyboard shortcuts:
    1-6     Switch panels (LLM, Reasoning, Tools, Perf, Context, Trace)
    j/k     Navigate to next/previous step
    l       Toggle live mode
    r       Refresh
    q       Quit
"""
import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tui.monitor import AgentMonitor


def main():
    parser = argparse.ArgumentParser(
        description="CCL Agent Monitor - White-box visibility into agent operations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Path to specific run directory to monitor"
    )
    parser.add_argument(
        "--artifacts",
        type=str,
        default="artifacts",
        help="Root artifacts directory (default: artifacts)"
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=1.0,
        help="Polling interval in seconds for live mode (default: 1.0)"
    )
    parser.add_argument(
        "--env-file",
        type=str,
        default=".env.local",
        help="Environment file to load (default: .env.local)"
    )
    
    args = parser.parse_args()
    
    app = AgentMonitor(
        artifacts_root=args.artifacts,
        run_dir=args.run_dir,
        poll_interval=args.poll_interval,
        env_file=args.env_file,
    )
    app.run()


if __name__ == "__main__":
    main()
