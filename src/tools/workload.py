from __future__ import annotations

import hashlib
import os
import subprocess
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from ..types import Metrics, NCCLConfig, WorkloadSpec
from ..utils import setup_logger


logger = setup_logger("cclagent.workload")


@dataclass
class WorkloadRunConfig:
    timeout_s: int = 3600
    dry_run: bool = True


class WorkloadRunner:
    def __init__(
        self,
        config: WorkloadRunConfig,
        metrics_parser: Optional[Callable[[str], Metrics]] = None,
    ) -> None:
        self.config = config
        self.metrics_parser = metrics_parser

    def run(
        self,
        workload: WorkloadSpec,
        config: NCCLConfig,
        step: int,
        env_overrides: Optional[Dict[str, str]] = None,
    ) -> Metrics:
        if self.config.dry_run or not workload.command:
            return self._simulate_metrics(config, step)

        env = os.environ.copy()
        env.update(workload.env)
        env.update({k: str(v) for k, v in config.params.items()})
        if env_overrides:
            env.update(env_overrides)

        start = time.time()
        try:
            result = subprocess.run(
                workload.command,
                check=True,
                capture_output=True,
                text=True,
                timeout=self.config.timeout_s,
                env=env,
            )
        except subprocess.SubprocessError as exc:
            logger.error("Workload failed: %s", exc)
            return Metrics(iteration_time=float("inf"), errors=1, extras={"error": str(exc)})

        raw_output = result.stdout.strip()
        if self.metrics_parser:
            metrics = self.metrics_parser(raw_output)
        else:
            elapsed = time.time() - start
            metrics = Metrics(iteration_time=elapsed, extras={"raw": raw_output})
        return metrics

    def _simulate_metrics(self, config: NCCLConfig, step: int) -> Metrics:
        seed = self._seed_from_config(config)
        base = 1.0 + (seed % 100) / 500.0
        improvement = (step + 1) * 0.01
        iter_time = max(0.1, base - improvement)
        comm_time = iter_time * 0.4
        bandwidth = 100.0 + (seed % 50)
        return Metrics(
            iteration_time=iter_time,
            comm_time=comm_time,
            bandwidth=bandwidth,
            errors=0,
            extras={"simulated": True, "seed": seed},
        )

    def _seed_from_config(self, config: NCCLConfig) -> int:
        payload = "|".join(f"{k}={v}" for k, v in sorted(config.params.items()))
        digest = hashlib.md5(payload.encode("utf-8")).hexdigest()
        return int(digest[:6], 16)
