from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Optional
from pathlib import Path

from ..observability import FailureModeDetector
from ..types import CompiledConfig, NCCLConfig, RunContext, WorkloadSpec
from ..utils import artifact_path, setup_logger, write_json


logger = setup_logger("cclagent.executor")


class WorkloadExecutor:
    def __init__(self, tools: Any, run_context: Optional[RunContext] = None) -> None:
        self.tools = tools
        self.run_context = run_context
        self.failure_detector = FailureModeDetector()

    def run(
        self,
        workload: WorkloadSpec,
        config: NCCLConfig,
        step: int,
        compiled: Optional[CompiledConfig] = None,
        execution_mode: str = "restart_per_step",
        extra_env: Optional[Dict[str, str]] = None,
        artifact_subdir: str = "steps",
    ):
        apply_result = self.tools.nccl.apply(config)
        if not apply_result.ok:
            logger.warning("Config validation failed: %s", apply_result.errors)
        if apply_result.warnings:
            logger.info("Config warnings: %s", apply_result.warnings)

        env_overrides: Dict[str, str] = {}
        if compiled is not None:
            env_overrides.update(compiled.env)
        if getattr(self.tools, "ext_tuner", None) is not None:
            env_overrides.update(self.tools.ext_tuner.env_overrides())
        if getattr(self.tools, "autoccl", None) is not None:
            env_overrides.update(self.tools.autoccl.env_overrides())
        if getattr(self.tools, "ext_net", None) is not None:
            env_overrides.update(self.tools.ext_net.env_overrides())
        if getattr(self.tools, "nccl_debug", None) is not None:
            env_overrides.update(self.tools.nccl_debug.env_overrides(step=step, artifact_subdir=artifact_subdir))
        if getattr(self.tools, "profiler", None) is not None:
            env_overrides.update(self.tools.profiler.env_overrides())
        if extra_env:
            env_overrides.update(extra_env)

        if self.run_context:
            write_json(
                artifact_path(self.run_context, artifact_subdir, f"step_{step}_final_env.json"),
                {"env": env_overrides},
            )

        runner = self.tools.workload
        workload_kind = getattr(workload, "kind", "workload").lower()
        if workload_kind in ("training", "train") and getattr(self.tools, "training", None) is not None:
            runner = self.tools.training

        metrics = runner.run(workload, config, step=step, env_overrides=env_overrides, artifact_subdir=artifact_subdir)
        sla_result = self.tools.sla.check(metrics)
        metrics.raw["sla_ok"] = sla_result.ok
        metrics.raw["sla_violations"] = sla_result.violations
        metrics.raw["sla_severity"] = sla_result.severity
        metrics.raw["sla_rollback"] = sla_result.rollback_recommended

        stdout_text, stderr_text = self._read_logs(step, artifact_subdir=artifact_subdir)
        if getattr(self.tools, "nccl_debug", None) is not None:
            debug_payload = self.tools.nccl_debug.collect_step(
                step=step,
                stdout_text=stdout_text,
                stderr_text=stderr_text,
                artifact_subdir=artifact_subdir,
            )
            metrics.raw["nccl_debug"] = debug_payload.get("summary", {})
        if getattr(self.tools, "profiler", None) is not None:
            profiler_payload = self.tools.profiler.collect_step(step, artifact_subdir=artifact_subdir)
            metrics.raw["profiler"] = profiler_payload.get("summary", {})

        baseline_ms = metrics.raw.get("baseline_iteration_time_ms")
        try:
            baseline_ms = float(baseline_ms) if baseline_ms is not None else None
        except (TypeError, ValueError):
            baseline_ms = None
        failure_signal = self.failure_detector.detect(
            metrics=metrics,
            stdout_text=stdout_text,
            stderr_text=stderr_text,
            baseline_ms=baseline_ms,
        )
        if failure_signal is not None:
            metrics.raw["failure_mode"] = failure_signal.kind
            metrics.raw["failure_severity"] = failure_signal.severity
            metrics.raw["failure_confidence"] = failure_signal.confidence
            metrics.raw["failure_reasons"] = list(failure_signal.reasons)
            metrics.raw["policy_lane_hint"] = self.failure_detector.policy_lane(failure_signal)
            if self.run_context:
                write_json(
                    artifact_path(self.run_context, artifact_subdir, f"step_{step}_failure_mode.json"),
                    failure_signal.to_dict(),
                )
            if getattr(self.tools, "debug_playbook", None) is not None:
                actions = self.tools.debug_playbook.plan(failure_signal.kind)
                metrics.raw["debug_playbook_actions"] = self.tools.debug_playbook.serialize(actions)
                if self.run_context:
                    write_json(
                        artifact_path(self.run_context, artifact_subdir, f"step_{step}_debug_playbook.json"),
                        {
                            "schema_version": "1.0",
                            "failure_mode": failure_signal.kind,
                            "actions": self.tools.debug_playbook.serialize(actions),
                        },
                    )

        if not sla_result.ok:
            logger.warning("SLA violation: %s", sla_result.violations)
        return metrics
    def run_batch(
        self,
        workload: WorkloadSpec,
        candidates: list[NCCLConfig],
        step: int,
        eval_mode: str = "short",
        concurrency: int = 1,
        artifact_subdir: str = "steps",
        eval_steps_override: int | None = None,
        eval_timeout_override: int | None = None,
    ) -> list[tuple[NCCLConfig, Any]]:
        results: list[tuple[NCCLConfig, Any]] = []

        def _run_one(idx: int, candidate: NCCLConfig) -> tuple[NCCLConfig, Any]:
            config_step = step
            candidate_artifact_subdir = artifact_subdir
            if self.run_context:
                candidate_artifact_subdir = f"{artifact_subdir}/step_{step}_candidate_{idx}"
            env_overrides = {"CCL_EVAL_MODE": eval_mode}
            eval_steps = eval_steps_override if eval_steps_override is not None else workload.eval_steps
            eval_timeout = eval_timeout_override if eval_timeout_override is not None else workload.eval_timeout_sec
            if eval_steps:
                env_overrides["CCL_EVAL_STEPS"] = str(eval_steps)
            if eval_timeout:
                env_overrides["CCL_EVAL_TIMEOUT_SEC"] = str(eval_timeout)
            metrics = self.run(
                workload,
                candidate,
                config_step,
                extra_env=env_overrides,
                artifact_subdir=candidate_artifact_subdir,
            )
            if self.run_context:
                write_json(
                    artifact_path(self.run_context, artifact_subdir, f"step_{step}_candidate_{idx}_metrics.json"),
                    metrics.__dict__,
                )
            return candidate, metrics

        if concurrency <= 1:
            for idx, candidate in enumerate(candidates):
                results.append(_run_one(idx, candidate))
            return results

        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            future_map = {
                pool.submit(_run_one, idx, candidate): (idx, candidate)
                for idx, candidate in enumerate(candidates)
            }
            for future in as_completed(future_map):
                results.append(future.result())
        return results

    def _read_logs(self, step: int, *, artifact_subdir: str) -> tuple[str, str]:
        if self.run_context is None:
            return "", ""
        stdout_path = Path(artifact_path(self.run_context, artifact_subdir, f"step_{step}_stdout.log"))
        stderr_path = Path(artifact_path(self.run_context, artifact_subdir, f"step_{step}_stderr.log"))
        stdout_text = stdout_path.read_text(encoding="utf-8") if stdout_path.exists() else ""
        stderr_text = stderr_path.read_text(encoding="utf-8") if stderr_path.exists() else ""
        return stdout_text, stderr_text
