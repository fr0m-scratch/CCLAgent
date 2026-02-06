from __future__ import annotations

from typing import TYPE_CHECKING

from .workbench import AgentWorkbench

if TYPE_CHECKING:
    from ..runner import AgentBridge


class LiveAgentMonitor(AgentWorkbench):
    """Live mode app wired to AgentBridge for bidirectional interaction."""

    def __init__(
        self,
        bridge: "AgentBridge",
        env_file: str = ".env.local",
        poll_interval: float = 0.5,
    ) -> None:
        super().__init__(
            bridge=bridge,
            artifacts_root="artifacts",
            run_dir=None,
            poll_interval=poll_interval,
            env_file=env_file,
            live_mode=True,
        )


__all__ = ["LiveAgentMonitor"]
