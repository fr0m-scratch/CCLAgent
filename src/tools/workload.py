from __future__ import annotations

import hashlib
import os
import subprocess
import time
from dataclasses import dataclass
import random
import math
from typing import Any, Callable, Dict, Optional
from pathlib import Path

from ..types import Metrics, NCCLConfig, RunContext, WorkloadSpec
from .launchers import build_mpi_command, build_slurm_command, build_torchrun_command
from ..utils import artifact_path, setup_logger, write_json
from .base import ToolExecutionError


logger = setup_logger("cclagent.workload")


@dataclass
class WorkloadRunConfig:
    timeout_s: int = 3600
    dry_run: bool = True
    allow_fallback: bool = True


class WorkloadRunner:
    def __init__(
        self,
        config: WorkloadRunConfig,
        metrics_parser: Optional[Callable[[str], Metrics]] = None,
        run_context: Optional[RunContext] = None,
    ) -> None:
        self.config = config
        self.metrics_parser = metrics_parser
        self.run_context = run_context

    def run(
        self,
        workload: WorkloadSpec,
        config: NCCLConfig,
        step: int,
        env_overrides: Optional[Dict[str, str]] = None,
        execution_env: Optional[Dict[str, str]] = None,
        command: Optional[list[str]] = None,
        artifact_subdir: str = "steps",
    ) -> Metrics:
        if self.config.dry_run or not workload.command:
            sleep_sec = self._sleep_from_env(env_overrides, workload.env)
            if sleep_sec > 0:
                time.sleep(sleep_sec)
            metrics = self._simulate_metrics(config, step, env_overrides, workload.env)
            stdout = self._build_simulated_log(metrics.raw)
            self._persist_logs(step, stdout=stdout, stderr="", artifact_subdir=artifact_subdir)
            self._persist_metrics(metrics, step, artifact_subdir=artifact_subdir)
            return metrics

        env = dict(execution_env) if execution_env is not None else os.environ.copy()
        env.update(workload.env)
        env.update({k: str(v) for k, v in config.params.items()})
        if env_overrides:
            env.update(env_overrides)

        cmd = command or workload.command
        cmd = self._select_command(workload, cmd)

        start = time.time()
        timeout = self.config.timeout_s
        if workload.eval_mode == "short" and workload.eval_timeout_sec:
            timeout = workload.eval_timeout_sec
        if env_overrides and env_overrides.get("CCL_EVAL_TIMEOUT_SEC"):
            try:
                timeout = int(env_overrides["CCL_EVAL_TIMEOUT_SEC"])
            except ValueError:
                pass
        replicates = self._replicates_from_env(env_overrides, workload.env)
        try:
            if replicates <= 1:
                result = subprocess.run(
                    cmd,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    env=env,
                )
                outputs = [result.stdout.strip()]
                stdout_logs = [result.stdout]
                stderr_logs = [result.stderr]
            else:
                outputs = []
                stdout_logs = []
                stderr_logs = []
                for _rep in range(replicates):
                    result = subprocess.run(
                        cmd,
                        check=True,
                        capture_output=True,
                        text=True,
                        timeout=timeout,
                        env=env,
                    )
                    outputs.append(result.stdout.strip())
                    stdout_logs.append(result.stdout)
                    stderr_logs.append(result.stderr)
        except subprocess.SubprocessError as exc:
            logger.error("Workload failed: %s", exc)
            if self.config.allow_fallback:
                metrics = Metrics(
                    iteration_time_ms=float("inf"),
                    success=False,
                    failure_reason=str(exc),
                    raw={"error": str(exc)},
                )
                self._persist_metrics(metrics, step)
                return metrics
            raise ToolExecutionError(f"workload failed: {exc}") from exc

        if self.metrics_parser:
            metrics_list = [self.metrics_parser(raw_output) for raw_output in outputs]
            metrics = _aggregate_metrics(metrics_list)
        else:
            elapsed = time.time() - start
            metrics = Metrics(iteration_time_ms=elapsed * 1000.0, raw={"raw": outputs[0] if outputs else ""})
        metrics.raw["replicates"] = replicates
        if replicates > 1:
            sample_ms = [item.iteration_time_ms for item in metrics_list] if self.metrics_parser else [metrics.iteration_time_ms]
            mean_ms = sum(sample_ms) / max(1, len(sample_ms))
            variance = sum((v - mean_ms) ** 2 for v in sample_ms) / max(1, len(sample_ms))
            std_ms = math.sqrt(variance)
            ci95 = 1.96 * std_ms / math.sqrt(max(1, len(sample_ms)))
            metrics.raw["replicate_samples_ms"] = sample_ms
            metrics.raw["replicate_mean_ms"] = mean_ms
            metrics.raw["replicate_std_ms"] = std_ms
            metrics.raw["replicate_ci95_ms"] = ci95

        if self.run_context:
            self._persist_logs(step, stdout="\n".join(stdout_logs), stderr="\n".join(stderr_logs), artifact_subdir=artifact_subdir)
            if replicates > 1:
                for idx, (out, err) in enumerate(zip(stdout_logs, stderr_logs)):
                    self._persist_logs(
                        step,
                        stdout=out,
                        stderr=err,
                        artifact_subdir=f"{artifact_subdir}/step_{step}_replicate_{idx}",
                    )
                write_json(
                    artifact_path(self.run_context, artifact_subdir, f"step_{step}_replicate_summary.json"),
                    {
                        "schema_version": "1.0",
                        "replicates": replicates,
                        "sample_ms": metrics.raw.get("replicate_samples_ms", []),
                        "mean_ms": metrics.raw.get("replicate_mean_ms"),
                        "std_ms": metrics.raw.get("replicate_std_ms"),
                        "ci95_ms": metrics.raw.get("replicate_ci95_ms"),
                    },
                )
            write_json(
                artifact_path(self.run_context, artifact_subdir, f"workload_cmd_step_{step}.json"),
                {
                    "command": cmd,
                    "launcher": workload.launcher,
                    "launcher_args": workload.launcher_args,
                    "env_overrides": env_overrides or {},
                    "replicates": replicates,
                },
            )

        self._persist_metrics(metrics, step, artifact_subdir=artifact_subdir)
        return metrics

    def _persist_metrics(self, metrics: Metrics, step: int, artifact_subdir: str = "steps") -> None:
        if not self.run_context:
            return
        write_json(artifact_path(self.run_context, artifact_subdir, f"step_{step}_metrics.json"), metrics.__dict__)

    def _persist_logs(self, step: int, stdout: str, stderr: str, artifact_subdir: str = "steps") -> None:
        if not self.run_context:
            return
        stdout_path = artifact_path(self.run_context, artifact_subdir, f"step_{step}_stdout.log")
        stderr_path = artifact_path(self.run_context, artifact_subdir, f"step_{step}_stderr.log")
        Path(stdout_path).parent.mkdir(parents=True, exist_ok=True)
        Path(stderr_path).parent.mkdir(parents=True, exist_ok=True)
        with open(stdout_path, "w", encoding="utf-8") as handle:
            handle.write(stdout)
        with open(stderr_path, "w", encoding="utf-8") as handle:
            handle.write(stderr)


    def _select_command(self, workload: WorkloadSpec, default_cmd: list[str]) -> list[str]:
        launcher = (workload.launcher or "local").lower()
        if launcher == "torchrun":
            return build_torchrun_command(workload)
        if launcher in ("slurm", "srun"):
            return build_slurm_command(workload)
        if launcher in ("mpi", "mpirun"):
            return build_mpi_command(workload)
        return default_cmd

    def _simulate_metrics(
        self,
        config: NCCLConfig,
        step: int,
        env_overrides: Optional[Dict[str, str]],
        workload_env: Dict[str, str],
    ) -> Metrics:
        seed = self._seed_from_config(config)
        base = 1.0 + (seed % 100) / 500.0
        improvement = (step + 1) * 0.01
        effects = self._simulate_effects(config)
        iter_time = max(0.1, base - improvement + effects["iter_adjust"])
        comm_time = iter_time * 0.4
        bandwidth = (100.0 + (seed % 50)) * (1.0 + effects["bw_adjust"])
        bandwidth = max(1.0, bandwidth)
        iter_count = self._iter_count_from_env(env_overrides, workload_env)
        iter_samples = self._simulate_iter_samples(iter_time * 1000.0, iter_count, seed + step)
        iter_mean = sum(iter_samples) / max(1, len(iter_samples))
        variance = sum((v - iter_mean) ** 2 for v in iter_samples) / max(1, len(iter_samples))
        iter_std = math.sqrt(variance)
        simulated_total_ms = iter_mean * iter_count
        return Metrics(
            iteration_time_ms=iter_mean,
            comm_time_ms=comm_time * 1000.0,
            algbw_gbps=bandwidth,
            busbw_gbps=bandwidth * 0.9,
            success=True,
            raw={
                "simulated": True,
                "seed": seed,
                "simulated_effects": effects["details"],
                "iteration_count": iter_count,
                "iter_samples_ms": iter_samples,
                "iter_mean_ms": iter_mean,
                "iter_std_ms": iter_std,
                "simulated_total_ms": simulated_total_ms,
            },
        )

    def _sleep_from_env(self, env_overrides: Optional[Dict[str, str]], workload_env: Dict[str, str]) -> float:
        for source in (env_overrides or {}, workload_env, dict(os.environ)):
            value = source.get("CCL_SIMULATE_SLEEP_SEC")
            if value is None:
                continue
            try:
                return float(value)
            except ValueError:
                continue
        return 0.0

    def _iter_count_from_env(self, env_overrides: Optional[Dict[str, str]], workload_env: Dict[str, str]) -> int:
        for source in (env_overrides or {}, workload_env, dict(os.environ)):
            value = source.get("CCL_SIMULATE_ITERS")
            if value is None:
                continue
            try:
                return max(1, int(value))
            except ValueError:
                continue
        return 200

    def _replicates_from_env(self, env_overrides: Optional[Dict[str, str]], workload_env: Dict[str, str]) -> int:
        for source in (env_overrides or {}, workload_env, dict(os.environ)):
            value = source.get("CCL_REPLICATES")
            if value is None:
                continue
            try:
                return max(1, int(value))
            except ValueError:
                continue
        return 1

    def _simulate_iter_samples(self, mean_ms: float, count: int, seed: int) -> list[float]:
        rng = random.Random(seed)
        samples: list[float] = []
        sigma = max(1.0, mean_ms * 0.02)
        for _ in range(count):
            value = rng.gauss(mean_ms, sigma)
            samples.append(max(1.0, value))
        return samples

    def _build_simulated_log(self, raw: Dict[str, Any]) -> str:
        if not isinstance(raw, dict):
            return "dry_run"
        samples = raw.get("iter_samples_ms") or []
        count = raw.get("iteration_count") or len(samples)
        header = f"simulated workload: iterations={count}"
        lines = [header]
        max_lines = 5000
        for idx, value in enumerate(samples[:max_lines], start=1):
            lines.append(f"iter={idx} time_ms={value:.3f}")
        if len(samples) > max_lines:
            lines.append(f"... truncated {len(samples) - max_lines} iterations ...")
        return "\n".join(lines)

    def _seed_from_config(self, config: NCCLConfig) -> int:
        payload = "|".join(f"{k}={v}" for k, v in sorted(config.params.items()))
        digest = hashlib.md5(payload.encode("utf-8")).hexdigest()
        return int(digest[:6], 16)

    def _simulate_effects(self, config: NCCLConfig) -> Dict[str, Any]:
        params = config.params
        iter_adjust = 0.0
        bw_adjust = 0.0
        details: Dict[str, Any] = {}

        algo = params.get("NCCL_ALGO")
        if algo == "TREE":
            iter_adjust -= 0.03
            bw_adjust += 0.05
        elif algo == "COLLNET":
            iter_adjust -= 0.015
            bw_adjust += 0.03

        proto = params.get("NCCL_PROTO")
        if proto == "LL":
            iter_adjust -= 0.01
            bw_adjust += 0.02
        elif proto == "LL128":
            iter_adjust -= 0.005
            bw_adjust += 0.01

        nthreads = self._safe_int(params.get("NCCL_NTHREADS"))
        if nthreads:
            if 256 <= nthreads <= 384:
                iter_adjust -= 0.005
                bw_adjust += 0.01
            elif nthreads > 512:
                iter_adjust += 0.01

        buffsize = self._safe_int(params.get("NCCL_BUFFSIZE"))
        if buffsize:
            if buffsize < (1 << 20):
                iter_adjust += 0.04
                bw_adjust -= 0.05
            elif buffsize > (1 << 25):
                iter_adjust += 0.01
            else:
                iter_adjust -= 0.005

        max_channels = self._safe_int(params.get("NCCL_MAX_NCHANNELS"))
        if max_channels:
            if 4 <= max_channels <= 16:
                iter_adjust -= 0.005
                bw_adjust += 0.01
            elif max_channels > 32:
                iter_adjust += 0.01
                bw_adjust -= 0.01

        details["algo"] = algo
        details["proto"] = proto
        details["nthreads"] = nthreads
        details["buffsize"] = buffsize
        details["max_channels"] = max_channels
        details["iter_adjust"] = iter_adjust
        details["bw_adjust"] = bw_adjust
        return {"iter_adjust": iter_adjust, "bw_adjust": bw_adjust, "details": details}

    def _safe_int(self, value: Any) -> Optional[int]:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None


def _aggregate_metrics(metrics_list: list[Metrics]) -> Metrics:
    if not metrics_list:
        return Metrics(iteration_time_ms=float("inf"), success=False, failure_reason="no_metrics")
    if len(metrics_list) == 1:
        return metrics_list[0]
    sample_ms = [item.iteration_time_ms for item in metrics_list]
    mean_ms = sum(sample_ms) / max(1, len(sample_ms))
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
        raw={"replicate_count": len(metrics_list)},
        schema_version=first.schema_version,
    )
