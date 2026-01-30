from __future__ import annotations

import hashlib
import os
import subprocess
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..types import Metrics, NCCLConfig, WorkloadSpec
from ..utils import setup_logger


logger = setup_logger("cclagent.training")


@dataclass
class TrainingJobConfig:
    command: List[str] = field(default_factory=list)
    timeout_s: int = 7200
    dry_run: bool = True


class TrainingJobRunner:
    def __init__(self, config: TrainingJobConfig) -> None:
        self.config = config

    def run(
        self,
        workload: WorkloadSpec,
        config: NCCLConfig,
        env: Optional[Dict[str, str]] = None,
        *,
        step: int = 0,
        env_overrides: Optional[Dict[str, str]] = None,
    ) -> Metrics:
        cmd = self.config.command or workload.command
        if self.config.dry_run or not cmd:
            return self._simulate_metrics(config, step)

        merged_env = dict(env) if env is not None else os.environ.copy()
        merged_env.update(workload.env)
        merged_env.update({k: str(v) for k, v in config.params.items()})
        if env_overrides:
            merged_env.update(env_overrides)

        start = time.time()
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=self.config.timeout_s,
                env=merged_env,
            )
        except subprocess.SubprocessError as exc:
            logger.error("Training job failed: %s", exc)
            return Metrics(iteration_time=float("inf"), errors=1, extras={"error": str(exc)})

        elapsed = time.time() - start
        return Metrics(iteration_time=elapsed, extras={"raw": result.stdout.strip()})

    def _simulate_metrics(self, config: NCCLConfig, step: int) -> Metrics:
        seed = self._seed_from_config(config)
        base = 1.2 + (seed % 100) / 400.0
        improvement = (step + 1) * 0.015
        iter_time = max(0.2, base - improvement)
        comm_time = iter_time * 0.45
        bandwidth = 90.0 + (seed % 60)
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
