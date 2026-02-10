from __future__ import annotations

import subprocess
import random
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..types import Metrics, NCCLConfig, RunContext
from ..utils import artifact_path, setup_logger, write_json
from .base import ToolExecutionError
from .metrics import MetricsCollector
from ..types import MetricsConfig


logger = setup_logger("cclagent.nccltest")


@dataclass
class NcclTestConfig:
    binary: str = "all_reduce_perf"
    args: List[str] = field(default_factory=list)
    timeout_s: int = 600
    dry_run: bool = True
    parse_mode: str = "nccltests_v1"
    replicates: int = 1
    allow_fallback: bool = True


class NcclTestRunner:
    def __init__(self, config: NcclTestConfig, run_context: Optional[RunContext] = None) -> None:
        self.config = config
        self.run_context = run_context
        self.metrics_parser = MetricsCollector(MetricsConfig(parse_mode=self.config.parse_mode))

    def run(
        self,
        config: NCCLConfig,
        env: Optional[Dict[str, str]] = None,
        extra_args: Optional[List[str]] = None,
    ) -> Metrics:
        replicates = _replicates_from_env(env, default=self.config.replicates)
        if self.config.dry_run:
            metrics_list: List[Metrics] = []
            sample_ms: List[float] = []
            for rep in range(replicates):
                iter_count = _iter_count_from_env(env)
                iter_samples = _simulate_iter_samples(1000.0, iter_count, 7 + rep)
                iter_mean = sum(iter_samples) / max(1, len(iter_samples))
                variance = sum((v - iter_mean) ** 2 for v in iter_samples) / max(1, len(iter_samples))
                iter_std = math.sqrt(variance)
                sample_ms.append(iter_mean)
                metrics_list.append(
                    Metrics(
                        iteration_time_ms=iter_mean,
                        algbw_gbps=100.0,
                        success=True,
                        raw={
                            "dry_run": True,
                            "replicate": rep,
                            "iteration_count": iter_count,
                            "iter_samples_ms": iter_samples,
                            "iter_mean_ms": iter_mean,
                            "iter_std_ms": iter_std,
                        },
                    )
                )
            metrics = _aggregate_replicates(metrics_list)
            self._persist_metrics(metrics)
            return metrics

        cmd = [self.config.binary] + self.config.args + (extra_args or [])
        merged_env = None
        if env is not None:
            merged_env = dict(env)
            merged_env.update({k: str(v) for k, v in config.params.items()})

        metrics_list: List[Metrics] = []
        outputs: List[str] = []
        for rep in range(replicates):
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
            outputs.append(output)
            parsed = self.metrics_parser.parse(output, parse_mode=self.config.parse_mode)
            parsed.raw["replicate"] = rep
            metrics_list.append(parsed)

        metrics = _aggregate_replicates(metrics_list)
        metrics.raw["raw_outputs"] = outputs
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


def _replicates_from_env(env: Optional[Dict[str, str]], *, default: int = 1) -> int:
    if env is None:
        return max(1, int(default))
    value = env.get("CCL_REPLICATES")
    if value is None:
        return max(1, int(default))
    try:
        return max(1, int(value))
    except ValueError:
        return max(1, int(default))


def _simulate_iter_samples(mean_ms: float, count: int, seed: int) -> list[float]:
    rng = random.Random(seed)
    samples: list[float] = []
    sigma = max(1.0, mean_ms * 0.02)
    for _ in range(count):
        value = rng.gauss(mean_ms, sigma)
        samples.append(max(1.0, value))
    return samples


def _aggregate_replicates(metrics_list: List[Metrics]) -> Metrics:
    if not metrics_list:
        return Metrics(iteration_time_ms=float("inf"), success=False, failure_reason="no_metrics")
    if len(metrics_list) == 1:
        metric = metrics_list[0]
        metric.raw["replicate_count"] = 1
        return metric
    sample_ms = [item.iteration_time_ms for item in metrics_list]
    mean_ms = sum(sample_ms) / max(1, len(sample_ms))
    variance = sum((v - mean_ms) ** 2 for v in sample_ms) / max(1, len(sample_ms))
    std_ms = math.sqrt(variance)
    ci95 = 1.96 * std_ms / math.sqrt(max(1, len(sample_ms)))
    first = metrics_list[0]
    return Metrics(
        iteration_time_ms=mean_ms,
        throughput=first.throughput,
        comm_time_ms=first.comm_time_ms,
        busbw_gbps=first.busbw_gbps,
        algbw_gbps=first.algbw_gbps,
        loss=first.loss,
        error_budget=first.error_budget,
        success=all(item.success for item in metrics_list),
        failure_reason=None if all(item.success for item in metrics_list) else "replicate_failure",
        raw={
            "replicate_count": len(metrics_list),
            "replicate_samples_ms": sample_ms,
            "replicate_mean_ms": mean_ms,
            "replicate_std_ms": std_ms,
            "replicate_ci95_ms": ci95,
        },
        schema_version=first.schema_version,
    )
