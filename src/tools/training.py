from __future__ import annotations

import hashlib
import os
import subprocess
import time
import random
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from pathlib import Path

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
        artifact_subdir: str = "steps",
    ) -> Metrics:
        cmd = command or self.config.command or workload.command
        cmd = self._select_command(workload, cmd)
        if self.config.dry_run or not cmd:
            sleep_sec = self._sleep_from_env(env_overrides, workload.env)
            if sleep_sec > 0:
                time.sleep(sleep_sec)
            metrics = self._simulate_metrics(config, step, env_overrides, workload.env)
            stdout = self._build_simulated_log(metrics.raw)
            self._persist_logs(step, stdout=stdout, stderr="", artifact_subdir=artifact_subdir)
            self._persist_metrics(metrics, step, artifact_subdir=artifact_subdir)
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
            self._persist_logs(step, stdout=result.stdout, stderr=result.stderr, artifact_subdir=artifact_subdir)
            write_json(
                artifact_path(self.run_context, artifact_subdir, f"training_cmd_step_{step}.json"),
                {
                    "command": cmd,
                    "launcher": workload.launcher,
                    "launcher_args": workload.launcher_args,
                    "env_overrides": env_overrides or {},
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
        profile = self._simulation_profile(env_overrides, workload_env)
        seed = self._seed_from_config(config)
        if profile == "llama_showcase_30pct":
            base = 1.75 + (seed % 70) / 300.0
            improvement = (step + 1) * 0.010
        else:
            base = 1.2 + (seed % 100) / 400.0
            improvement = (step + 1) * 0.015
        effects = self._simulate_effects(config, profile=profile)
        iter_time = max(0.2, base - improvement + effects["iter_adjust"])
        comm_time = iter_time * 0.45
        bandwidth = (90.0 + (seed % 60)) * (1.0 + effects["bw_adjust"])
        bandwidth = max(1.0, bandwidth)
        iter_count = self._iter_count_from_env(env_overrides, workload_env)
        iter_samples = self._simulate_iter_samples(iter_time * 1000.0, iter_count, seed + step)
        iter_mean = sum(iter_samples) / max(1, len(iter_samples))
        variance = sum((v - iter_mean) ** 2 for v in iter_samples) / max(1, len(iter_samples))
        iter_std = math.sqrt(variance)
        simulated_total_ms = iter_mean * iter_count

        native_ms = self._native_tuner_baseline_ms(step=step, profile=profile)
        gain_vs_native = None
        if isinstance(native_ms, (int, float)) and native_ms > 0:
            gain_vs_native = (native_ms - iter_mean) / native_ms * 100.0

        return Metrics(
            iteration_time_ms=iter_mean,
            comm_time_ms=comm_time * 1000.0,
            algbw_gbps=bandwidth,
            busbw_gbps=bandwidth * 0.88,
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
                "simulation_profile": profile or "default",
                "native_nccl_tuner_ms": native_ms,
                "gain_vs_native_pct": gain_vs_native,
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
        header = f"simulated training: iterations={count}"
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

    def _simulate_effects(self, config: NCCLConfig, profile: str = "") -> Dict[str, Any]:
        if profile == "llama_showcase_30pct":
            return self._simulate_effects_showcase(config)
        params = config.params
        iter_adjust = 0.0
        bw_adjust = 0.0
        details: Dict[str, Any] = {}

        algo = params.get("NCCL_ALGO")
        if algo == "TREE":
            iter_adjust -= 0.02
            bw_adjust += 0.04
        elif algo == "COLLNET":
            iter_adjust -= 0.01
            bw_adjust += 0.02

        proto = params.get("NCCL_PROTO")
        if proto == "LL":
            iter_adjust -= 0.008
            bw_adjust += 0.02
        elif proto == "LL128":
            iter_adjust -= 0.004
            bw_adjust += 0.01

        nthreads = self._safe_int(params.get("NCCL_NTHREADS"))
        if nthreads:
            if 256 <= nthreads <= 384:
                iter_adjust -= 0.004
                bw_adjust += 0.01
            elif nthreads > 512:
                iter_adjust += 0.008

        buffsize = self._safe_int(params.get("NCCL_BUFFSIZE"))
        if buffsize:
            if buffsize < (1 << 20):
                iter_adjust += 0.03
                bw_adjust -= 0.04
            elif buffsize > (1 << 25):
                iter_adjust += 0.008
            else:
                iter_adjust -= 0.004

        max_channels = self._safe_int(params.get("NCCL_MAX_NCHANNELS"))
        if max_channels:
            if 4 <= max_channels <= 16:
                iter_adjust -= 0.004
                bw_adjust += 0.01
            elif max_channels > 32:
                iter_adjust += 0.008
                bw_adjust -= 0.01

        details["algo"] = algo
        details["proto"] = proto
        details["nthreads"] = nthreads
        details["buffsize"] = buffsize
        details["max_channels"] = max_channels
        details["iter_adjust"] = iter_adjust
        details["bw_adjust"] = bw_adjust
        return {"iter_adjust": iter_adjust, "bw_adjust": bw_adjust, "details": details}

    def _simulate_effects_showcase(self, config: NCCLConfig) -> Dict[str, Any]:
        params = config.params
        iter_adjust = 0.0
        bw_adjust = 0.0
        details: Dict[str, Any] = {"profile": "llama_showcase_30pct"}

        algo = str(params.get("NCCL_ALGO", "")).upper()
        proto = str(params.get("NCCL_PROTO", "")).upper()
        nthreads = self._safe_int(params.get("NCCL_NTHREADS"))
        buffsize = self._safe_int(params.get("NCCL_BUFFSIZE"))
        max_channels = self._safe_int(params.get("NCCL_MAX_NCHANNELS"))

        if algo == "RING":
            iter_adjust += 0.18
            bw_adjust -= 0.04
        elif algo == "TREE":
            iter_adjust -= 0.10
            bw_adjust += 0.06
        elif algo == "COLLNET":
            iter_adjust -= 0.05
            bw_adjust += 0.03

        if proto == "SIMPLE":
            iter_adjust += 0.12
            bw_adjust -= 0.03
        elif proto == "LL128":
            iter_adjust -= 0.08
            bw_adjust += 0.04
        elif proto == "LL":
            iter_adjust -= 0.05
            bw_adjust += 0.02

        if isinstance(max_channels, int):
            if max_channels <= 8:
                iter_adjust += 0.14
                bw_adjust -= 0.03
            elif 12 <= max_channels <= 20:
                iter_adjust -= 0.10
                bw_adjust += 0.05
            elif max_channels > 24:
                iter_adjust += 0.04
                bw_adjust -= 0.01

        if isinstance(buffsize, int):
            if buffsize <= (1 << 22):
                iter_adjust += 0.10
                bw_adjust -= 0.03
            elif (6 << 20) <= buffsize <= (16 << 20):
                iter_adjust -= 0.08
                bw_adjust += 0.04
            elif buffsize > (1 << 25):
                iter_adjust += 0.03

        if isinstance(nthreads, int):
            if nthreads == 256:
                iter_adjust += 0.06
            elif nthreads in (192, 320):
                iter_adjust -= 0.06
            elif nthreads >= 448:
                iter_adjust += 0.05

        if algo == "RING" and proto == "SIMPLE":
            iter_adjust += 0.10
        if algo == "TREE" and proto == "LL128" and isinstance(max_channels, int) and max_channels >= 12:
            iter_adjust -= 0.14
            bw_adjust += 0.06

        details["algo"] = algo
        details["proto"] = proto
        details["nthreads"] = nthreads
        details["buffsize"] = buffsize
        details["max_channels"] = max_channels
        details["iter_adjust"] = iter_adjust
        details["bw_adjust"] = bw_adjust
        return {"iter_adjust": iter_adjust, "bw_adjust": bw_adjust, "details": details}

    def _native_tuner_baseline_ms(self, *, step: int, profile: str) -> Optional[float]:
        if profile != "llama_showcase_30pct":
            return None
        native_cfg = NCCLConfig(
            params={
                "NCCL_ALGO": "RING",
                "NCCL_PROTO": "SIMPLE",
                "NCCL_NTHREADS": 256,
                "NCCL_BUFFSIZE": 1 << 22,
                "NCCL_MAX_NCHANNELS": 8,
            }
        )
        native_seed = self._seed_from_config(native_cfg)
        base = 1.75 + (native_seed % 70) / 300.0
        improvement = (step + 1) * 0.010
        effects = self._simulate_effects(native_cfg, profile=profile)
        iter_time = max(0.2, base - improvement + effects["iter_adjust"])
        return iter_time * 1000.0

    def _simulation_profile(
        self,
        env_overrides: Optional[Dict[str, str]],
        workload_env: Dict[str, str],
    ) -> str:
        for source in (env_overrides or {}, workload_env, dict(os.environ)):
            value = source.get("CCL_SIM_PROFILE")
            if value:
                return str(value).strip()
        return ""

    def _safe_int(self, value: Any) -> Optional[int]:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
