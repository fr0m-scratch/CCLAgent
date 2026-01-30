from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from ..types import MicrobenchResult, ParameterSpace, WorkloadSpec
from ..utils import setup_logger


logger = setup_logger("cclagent.microbench")


@dataclass
class MicrobenchConfig:
    command: Optional[List[str]] = None
    timeout_s: int = 900
    dry_run: bool = True


class MicrobenchRunner:
    def __init__(self, config: MicrobenchConfig, executor: Optional[Callable[..., MicrobenchResult]] = None):
        self.config = config
        self.executor = executor

    def run(self, workload: WorkloadSpec, parameter_space: ParameterSpace) -> MicrobenchResult:
        if self.executor:
            return self.executor(workload, parameter_space)
        if self.config.dry_run or not self.config.command:
            important = list(parameter_space.specs.keys())[:4]
            signals = {"bandwidth": 1.0, "latency": 1.0}
            return MicrobenchResult(important_params=important, signals=signals, raw={})

        start = time.time()
        try:
            result = subprocess.run(
                self.config.command,
                check=True,
                capture_output=True,
                text=True,
                timeout=self.config.timeout_s,
            )
        except subprocess.SubprocessError as exc:
            logger.error("Microbench failed: %s", exc)
            return MicrobenchResult()

        raw_output = result.stdout.strip()
        elapsed = time.time() - start
        try:
            payload = json.loads(raw_output)
        except json.JSONDecodeError:
            payload = {"raw": raw_output}
        payload["elapsed_s"] = elapsed
        important = payload.get("important_params", [])
        signals = payload.get("signals", {})
        return MicrobenchResult(important_params=important, signals=signals, raw=payload)
