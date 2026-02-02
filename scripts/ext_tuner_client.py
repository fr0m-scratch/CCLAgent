#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time

from src.tools.tuner_plugin_protocol import FileTunerProtocol


def run_command(cmd: str) -> dict:
    result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
    try:
        return json.loads(result.stdout.strip())
    except json.JSONDecodeError:
        return {"iteration_time_ms": 1000.0, "success": True, "raw": {"stdout": result.stdout}}


def main() -> None:
    parser = argparse.ArgumentParser(description="Ext-tuner client shim")
    parser.add_argument("--session-dir", default=os.getenv("CCL_TUNER_SESSION_DIR", "."))
    parser.add_argument("--cmd", default="")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--simulate", action="store_true")
    args = parser.parse_args()

    proto = FileTunerProtocol(args.session_dir)
    for step in range(args.steps):
        with open(os.path.join(args.session_dir, "request.json"), "w", encoding="utf-8") as handle:
            json.dump({"type": "GET_CONFIG", "step": step}, handle)
        response = proto.read_response()
        if response.get("type") == "STOP":
            break

        if args.simulate or not args.cmd:
            metrics = {
                "iteration_time_ms": 1000.0 - step * 10.0,
                "success": True,
                "raw": {"simulated": True, "step": step},
            }
        else:
            metrics = run_command(args.cmd)
            if "success" not in metrics:
                metrics["success"] = True
        with open(os.path.join(args.session_dir, "metrics.json"), "w", encoding="utf-8") as handle:
            json.dump({"type": "REPORT_METRICS", "step": step, "metrics": metrics}, handle)
        response = proto.read_response()
        if response.get("type") == "STOP":
            break
        time.sleep(0.1)


if __name__ == "__main__":
    main()
