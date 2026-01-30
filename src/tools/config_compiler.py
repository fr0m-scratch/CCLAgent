from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from ..types import NCCLConfig, ParameterSpace


@dataclass
class CompileResult:
    ok: bool
    env: Dict[str, str]
    errors: List[str]


class ConfigCompiler:
    def __init__(self, parameter_space: ParameterSpace) -> None:
        self.parameter_space = parameter_space

    def compile(self, config: NCCLConfig) -> CompileResult:
        errors = self.parameter_space.validate(config.params)
        env = {k: str(v) for k, v in config.params.items()}
        return CompileResult(ok=not errors, env=env, errors=errors)
