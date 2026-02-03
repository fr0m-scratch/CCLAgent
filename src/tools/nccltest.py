from __future__ import annotations

import subprocess
import random
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..types import Metrics, NCCLConfig, RunContext
from ..utils import artifact_path, setup_logger, write_json
from .base import ToolExecutionError


logger = setup_logger("cclagent.nccltest")


@dataclass
class NcclTestConfig:
    binary: str = "all_reduce_perf"
    args: List[str] = field(default_factory=list)
    timeout_s: int = 600
    dry_run: bool = True
    allow_fallback: bool = True


class NcclTestRunner:
    def __init__(self, config: NcclTestConfig, run_context: Optional[RunContext] = None) -> None:
        self.config = config
        self.run_context = run_context

    def run(
        self,
        config: NCCLConfig,
        env: Optional[Dict[str, str]] = None,
        extra_args: Optional[List[str]] = None,
    ) -> Metrics:
        if self.config.dry_run:
            iter_count = _iter_count_from_env(env)
            iter_samples = _simulate_iter_samples(1000.0, iter_count, 7)
            iter_mean = sum(iter_samples) / max(1, len(iter_samples))
            variance = sum((v - iter_mean) ** 2 for v in iter_samples) / max(1, len(iter_samples))
            iter_std = math.sqrt(variance)
            metrics = Metrics(
                iteration_time_ms=iter_mean,
                algbw_gbps=100.0,
                success=True,
                raw={
                    "dry_run": True,
                    "iteration_count": iter_count,
                    "iter_samples_ms": iter_samples,
                    "iter_mean_ms": iter_mean,
                    "iter_std_ms": iter_std,
                },
            )
            self._persist_metrics(metrics)
            return metrics

        cmd = [self.config.binary] + self.config.args + (extra_args or [])
        merged_env = None
        if env is not None:
            merged_env = dict(env)
            merged_env.update({k: str(v) for k, v in config.params.items()})

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
            logger.error("nccl-tests failed: %s", exc)
            if self.config.allow_fallback:
                metrics = Metrics(
                    iteration_time_ms=float("inf"),
                    success=False,
                    failure_reason=str(exc),
                    raw={"error": str(exc)},
                )
                self._persist_metrics(metrics)
                return metrics
            raise ToolExecutionError(f"nccl-tests failed: {exc}") from exc

        output = result.stdout.strip()
        metrics = Metrics(iteration_time_ms=1000.0, algbw_gbps=None, success=True, raw={"raw": output})
        self._persist_metrics(metrics)
        return metrics

    def _persist_metrics(self, metrics: Metrics) -> None:
        if not self.run_context:
            return
        write_json(artifact_path(self.run_context, "offline", "nccltest_metrics.json"), metrics.__dict__)


def _iter_count_from_env(env: Optional[Dict[str, str]]) -> int:
    if env is None:
        return 200
    value = env.get("CCL_SIMULATE_ITERS")
    if value is None:
        return 200
    try:
        return max(1, int(value))
    except ValueError:
        return 200


def _simulate_iter_samples(mean_ms: float, count: int, seed: int) -> list[float]:
    rng = random.Random(seed)
    samples: list[float] = []
    sigma = max(1.0, mean_ms * 0.02)
    for _ in range(count):
        value = rng.gauss(mean_ms, sigma)
        samples.append(max(1.0, value))
    return samples
