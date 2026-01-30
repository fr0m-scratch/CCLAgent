from __future__ import annotations

from typing import Any

from ..types import NCCLConfig, WorkloadSpec
from ..utils import setup_logger


logger = setup_logger("cclagent.executor")


class WorkloadExecutor:
    def __init__(self, tools: Any) -> None:
        self.tools = tools

    def run(self, workload: WorkloadSpec, config: NCCLConfig, step: int):
        apply_result = self.tools.nccl.apply(config)
        if not apply_result.ok:
            logger.warning("Config validation failed: %s", apply_result.errors)
        env_overrides = {}
        if getattr(self.tools, "ext_tuner", None) is not None:
            env_overrides.update(self.tools.ext_tuner.env_overrides())
        if getattr(self.tools, "autoccl", None) is not None:
            env_overrides.update(self.tools.autoccl.env_overrides())
        if getattr(self.tools, "ext_net", None) is not None:
            env_overrides.update(self.tools.ext_net.env_overrides())
        runner = self.tools.workload
        workload_kind = getattr(workload, "kind", "workload").lower()
        if workload_kind in ("training", "train") and getattr(self.tools, "training", None) is not None:
            runner = self.tools.training
        metrics = runner.run(workload, config, step=step, env_overrides=env_overrides or None)
        sla_result = self.tools.sla.check(metrics)
        metrics.extras["sla_ok"] = sla_result.ok
        if sla_result.reason:
            metrics.extras["sla_reason"] = sla_result.reason
        if not sla_result.ok:
            logger.warning("SLA violation: %s", sla_result.reason)
        return metrics
