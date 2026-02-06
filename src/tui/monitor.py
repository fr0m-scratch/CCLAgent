from __future__ import annotations

from .workbench import AgentWorkbench


class AgentMonitor(AgentWorkbench):
    """Inspect/monitor mode app for completed or ongoing runs."""

    def __init__(
        self,
        artifacts_root: str = "artifacts",
        run_dir: str | None = None,
        poll_interval: float = 1.0,
        env_file: str = ".env.local",
    ) -> None:
        super().__init__(
            bridge=None,
            artifacts_root=artifacts_root,
            run_dir=run_dir,
            poll_interval=poll_interval,
            env_file=env_file,
            live_mode=False,
        )


__all__ = ["AgentMonitor"]
