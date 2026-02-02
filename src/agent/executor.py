from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Optional

from ..types import CompiledConfig, NCCLConfig, RunContext, WorkloadSpec
from ..utils import artifact_path, setup_logger, write_json


logger = setup_logger("cclagent.executor")


class WorkloadExecutor:
    def __init__(self, tools: Any, run_context: Optional[RunContext] = None) -> None:
        self.tools = tools
        self.run_context = run_context

    def run(
        self,
        workload: WorkloadSpec,
        config: NCCLConfig,
        step: int,
        compiled: Optional[CompiledConfig] = None,
        execution_mode: str = "restart_per_step",
        extra_env: Optional[Dict[str, str]] = None,
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
        if extra_env:
            env_overrides.update(extra_env)

        if self.run_context:
            write_json(
                artifact_path(self.run_context, "steps", f"step_{step}_final_env.json"),
                {"env": env_overrides},
            )

        runner = self.tools.workload
        workload_kind = getattr(workload, "kind", "workload").lower()
        if workload_kind in ("training", "train") and getattr(self.tools, "training", None) is not None:
            runner = self.tools.training

        metrics = runner.run(workload, config, step=step, env_overrides=env_overrides)
        sla_result = self.tools.sla.check(metrics)
        metrics.raw["sla_ok"] = sla_result.ok
        metrics.raw["sla_violations"] = sla_result.violations
        metrics.raw["sla_severity"] = sla_result.severity
        metrics.raw["sla_rollback"] = sla_result.rollback_recommended
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
    ) -> list[tuple[NCCLConfig, Any]]:
        results: list[tuple[NCCLConfig, Any]] = []

        def _run_one(idx: int, candidate: NCCLConfig) -> tuple[NCCLConfig, Any]:
            config_step = step * 100 + idx
            env_overrides = {"CCL_EVAL_MODE": eval_mode}
            if workload.eval_steps:
                env_overrides["CCL_EVAL_STEPS"] = str(workload.eval_steps)
            if workload.eval_timeout_sec:
                env_overrides["CCL_EVAL_TIMEOUT_SEC"] = str(workload.eval_timeout_sec)
            metrics = self.run(workload, candidate, config_step, extra_env=env_overrides)
            if self.run_context:
                write_json(
                    artifact_path(self.run_context, "steps", f"step_{step}_candidate_{idx}_metrics.json"),
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
