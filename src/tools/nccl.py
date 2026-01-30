from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from ..types import NCCLConfig
from .config_compiler import ConfigCompiler


@dataclass
class NCCLApplyResult:
    ok: bool
    env: Dict[str, str]
    errors: list[str]


class NCCLInterface:
    def __init__(self, compiler: ConfigCompiler) -> None:
        self.compiler = compiler

    def apply(self, config: NCCLConfig) -> NCCLApplyResult:
        result = self.compiler.compile(config)
        return NCCLApplyResult(ok=result.ok, env=result.env, errors=result.errors)
