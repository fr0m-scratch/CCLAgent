from __future__ import annotations

import hashlib
import os
import subprocess
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..types import Metrics, NCCLConfig, RunContext, WorkloadSpec
from .launchers import build_mpi_command, build_slurm_command, build_torchrun_command
from ..utils import artifact_path, setup_logger, write_json
from .base import ToolExecutionError


logger = setup_logger("cclagent.training")


@dataclass
class TrainingJobConfig:
    command: List[str] = field(default_factory=list)
    timeout_s: int = 7200
    dry_run: bool = True
    allow_fallback: bool = True


class TrainingJobRunner:
    def __init__(self, config: TrainingJobConfig, run_context: Optional[RunContext] = None) -> None:
        self.config = config
        self.run_context = run_context

    def run(
        self,
        workload: WorkloadSpec,
        config: NCCLConfig,
        env: Optional[Dict[str, str]] = None,
        *,
        step: int = 0,
        env_overrides: Optional[Dict[str, str]] = None,
        command: Optional[List[str]] = None,
    ) -> Metrics:
        cmd = command or self.config.command or workload.command
        cmd = self._select_command(workload, cmd)
        if self.config.dry_run or not cmd:
            metrics = self._simulate_metrics(config, step)
            self._persist_logs(step, stdout="dry_run", stderr="")
            self._persist_metrics(metrics, step)
            return metrics

        merged_env = dict(env) if env is not None else os.environ.copy()
        merged_env.update(workload.env)
        merged_env.update({k: str(v) for k, v in config.params.items()})
        if env_overrides:
            merged_env.update(env_overrides)

        start = time.time()
        timeout = self.config.timeout_s
        if workload.eval_mode == "short" and workload.eval_timeout_sec:
            timeout = workload.eval_timeout_sec
        if env_overrides and env_overrides.get("CCL_EVAL_TIMEOUT_SEC"):
            try:
                timeout = int(env_overrides["CCL_EVAL_TIMEOUT_SEC"])
            except ValueError:
                pass
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=merged_env,
            )
        except subprocess.SubprocessError as exc:
            logger.error("Training job failed: %s", exc)
            if self.config.allow_fallback:
                metrics = Metrics(
                    iteration_time_ms=float("inf"),
                    success=False,
                    failure_reason=str(exc),
                    raw={"error": str(exc)},
                )
                self._persist_metrics(metrics, step)
                return metrics
            raise ToolExecutionError(f"training job failed: {exc}") from exc

        elapsed = time.time() - start
        metrics = Metrics(
            iteration_time_ms=elapsed * 1000.0,
            success=True,
            raw={"raw": result.stdout.strip()},
        )

        if self.run_context:
            self._persist_logs(step, stdout=result.stdout, stderr=result.stderr)
            write_json(
                artifact_path(self.run_context, "steps", f"training_cmd_step_{step}.json"),
                {
                    "command": cmd,
                    "launcher": workload.launcher,
                    "launcher_args": workload.launcher_args,
                    "env_overrides": env_overrides or {},
                },
            )

        self._persist_metrics(metrics, step)
        return metrics

    def _persist_metrics(self, metrics: Metrics, step: int) -> None:
        if not self.run_context:
            return
        write_json(artifact_path(self.run_context, "steps", f"step_{step}_metrics.json"), metrics.__dict__)

    def _persist_logs(self, step: int, stdout: str, stderr: str) -> None:
        if not self.run_context:
            return
        stdout_path = artifact_path(self.run_context, "steps", f"step_{step}_stdout.log")
        stderr_path = artifact_path(self.run_context, "steps", f"step_{step}_stderr.log")
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

    def _simulate_metrics(self, config: NCCLConfig, step: int) -> Metrics:
        seed = self._seed_from_config(config)
        base = 1.2 + (seed % 100) / 400.0
        improvement = (step + 1) * 0.015
        iter_time = max(0.2, base - improvement)
        comm_time = iter_time * 0.45
        bandwidth = 90.0 + (seed % 60)
        return Metrics(
            iteration_time_ms=iter_time * 1000.0,
            comm_time_ms=comm_time * 1000.0,
            algbw_gbps=bandwidth,
            busbw_gbps=bandwidth * 0.88,
            success=True,
            raw={"simulated": True, "seed": seed},
        )

    def _seed_from_config(self, config: NCCLConfig) -> int:
        payload = "|".join(f"{k}={v}" for k, v in sorted(config.params.items()))
        digest = hashlib.md5(payload.encode("utf-8")).hexdigest()
        return int(digest[:6], 16)
