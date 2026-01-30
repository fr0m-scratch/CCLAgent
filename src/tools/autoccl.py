from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..types import NCCLConfig
from ..utils import write_json


@dataclass
class AutoCCLRuntimeConfig:
    ld_preload: Optional[str] = None
    ld_library_path: Optional[str] = None
    tuner_plugin: Optional[str] = None
    coordinator: Optional[str] = None
    world_size: Optional[int] = None
    role: Optional[str] = None
    extra_env: Dict[str, str] = field(default_factory=dict)


class AutoCCLBridge:
    def __init__(self, config: Optional[AutoCCLRuntimeConfig] = None) -> None:
        self.config = config or AutoCCLRuntimeConfig()

    def build_env(self, base_env: Optional[Dict[str, str]] = None, role: Optional[str] = None) -> Dict[str, str]:
        env = dict(base_env or os.environ)
        if self.config.ld_preload:
            env["LD_PRELOAD"] = self.config.ld_preload
        if self.config.ld_library_path:
            env["LD_LIBRARY_PATH"] = self.config.ld_library_path
        if self.config.tuner_plugin:
            env["NCCL_TUNER_PLUGIN"] = self.config.tuner_plugin
        if self.config.coordinator:
            env["TUNER_COORDINATOR"] = self.config.coordinator
        if self.config.world_size is not None:
            env["TUNER_WORLDSIZE"] = str(self.config.world_size)
        if role or self.config.role:
            env["TUNER_ROLE"] = role or self.config.role
        env.update(self.config.extra_env)
        return env

    def env_overrides(self, role: Optional[str] = None) -> Dict[str, str]:
        return self.build_env(base_env={}, role=role)

    def export_candidate(self, config: NCCLConfig, path: str) -> None:
        write_json(path, {"candidate": config.params, "metadata": config.metadata})

    def to_tuner_envs(self, config: NCCLConfig) -> Dict[str, int]:
        tuner_envs: Dict[str, int] = {}
        for key, value in config.params.items():
            if isinstance(value, bool):
                tuner_envs[f"tuner_{key.lower()}"] = int(value)
            elif isinstance(value, int):
                tuner_envs[f"tuner_{key.lower()}"] = value
        return tuner_envs


class AutoCCLCandidateProvider:
    def __init__(self, wrapper_path: Optional[str] = None) -> None:
        self.wrapper_path = wrapper_path
        self._wrapper = None

    def _load_wrapper(self):
        if self._wrapper:
            return self._wrapper
        if not self.wrapper_path:
            return None
        spec = importlib.util.spec_from_file_location("autoccl_wrapper", self.wrapper_path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
            self._wrapper = module.NCCLCandidateWrapper()
        except Exception:
            return None
        return self._wrapper

    def get_valid_candidates(
        self,
        nrank: int,
        nnode: int,
        coll: int,
        size: int,
        tuner_envs: Dict[str, int],
        scale2: bool = True,
    ) -> List[List[int]]:
        wrapper = self._load_wrapper()
        if wrapper is None:
            return []
        return wrapper.nccl_get_valid_candidates(nrank, nnode, coll, size, tuner_envs, scale2)
