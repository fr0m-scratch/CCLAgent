from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..types import Metrics, NCCLConfig
from ..utils import setup_logger


logger = setup_logger("cclagent.nccltest")


@dataclass
class NcclTestConfig:
    binary: str = "all_reduce_perf"
    args: List[str] = field(default_factory=list)
    timeout_s: int = 600
    dry_run: bool = True


class NcclTestRunner:
    def __init__(self, config: NcclTestConfig) -> None:
        self.config = config

    def run(
        self,
        config: NCCLConfig,
        env: Optional[Dict[str, str]] = None,
        extra_args: Optional[List[str]] = None,
    ) -> Metrics:
        if self.config.dry_run:
            return Metrics(iteration_time=1.0, bandwidth=100.0, extras={"dry_run": True})

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
            return Metrics(iteration_time=float("inf"), errors=1, extras={"error": str(exc)})

        output = result.stdout.strip()
        return Metrics(iteration_time=1.0, bandwidth=None, extras={"raw": output})
