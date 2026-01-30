from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class ExtNetConfig:
    plugin_name: Optional[str] = None
    plugin_path: Optional[str] = None
    net_name: Optional[str] = None
    extra_env: Dict[str, str] = field(default_factory=dict)


class ExtNetBridge:
    def __init__(self, config: Optional[ExtNetConfig] = None) -> None:
        self.config = config or ExtNetConfig()

    def build_env(self, base_env: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        env = dict(base_env or os.environ)
        if self.config.plugin_name:
            env["NCCL_NET_PLUGIN"] = self.config.plugin_name
        if self.config.net_name:
            env["NCCL_NET"] = self.config.net_name
        if self.config.plugin_path:
            current = env.get("LD_LIBRARY_PATH", "")
            env["LD_LIBRARY_PATH"] = f"{self.config.plugin_path}:{current}" if current else self.config.plugin_path
        env.update(self.config.extra_env)
        return env

    def env_overrides(self) -> Dict[str, str]:
        return self.build_env(base_env={})
