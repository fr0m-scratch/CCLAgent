from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from ..types import CompiledConfig, NCCLConfig, ParameterSpace, SafetyConfig
from ..safety.risk import RiskScorer


@dataclass
class CompileResult:
    ok: bool
    env: Dict[str, str]
    errors: List[str]
    warnings: List[str]


class ConfigCompiler:
    def __init__(self, parameter_space: ParameterSpace, safety: Optional[SafetyConfig] = None) -> None:
        self.parameter_space = parameter_space
        self.safety = safety or SafetyConfig()

    def compile(self, config: NCCLConfig) -> CompileResult:
        errors = self.parameter_space.validate(config.params)
        warnings = self._safety_warnings(config)
        env = {k: str(v) for k, v in config.params.items()}
        return CompileResult(ok=not errors, env=env, errors=errors, warnings=warnings)

    def compile_patch(self, base: NCCLConfig, patch: Dict[str, str | int | float]) -> CompileResult:
        merged = dict(base.params)
        merged.update(patch)
        return self.compile(NCCLConfig(params=merged))


    def compile_hypothesis(self, base: NCCLConfig, patch: Dict[str, Any]) -> CompiledConfig:
        result = self.compile_patch(base, patch)
        merged = dict(base.params)
        merged.update(patch)
        risk = RiskScorer(self.safety).score(NCCLConfig(params=merged))
        risk_score = risk.risk_score
        return CompiledConfig(
            config=NCCLConfig(params=merged),
            env=result.env,
            warnings=result.warnings,
            risk_score=risk_score,
            risk_reasons=risk.reasons,
        )

    def _safety_warnings(self, config: NCCLConfig) -> List[str]:
        warnings: List[str] = []
        max_channels = self.safety.max_channels_safe
        min_buffsize = self.safety.min_buffsize_safe
        if "NCCL_MAX_NCHANNELS" in config.params:
            try:
                channels = int(config.params["NCCL_MAX_NCHANNELS"])
                if channels > max_channels:
                    warnings.append("NCCL_MAX_NCHANNELS exceeds safety envelope")
            except (TypeError, ValueError):
                warnings.append("NCCL_MAX_NCHANNELS is not an int")
        if "NCCL_BUFFSIZE" in config.params:
            try:
                buffsize = int(config.params["NCCL_BUFFSIZE"])
                if buffsize < min_buffsize:
                    warnings.append("NCCL_BUFFSIZE below safety envelope")
            except (TypeError, ValueError):
                warnings.append("NCCL_BUFFSIZE is not an int")
        for param, envelope in (self.safety.safe_envelope or {}).items():
            if param not in config.params:
                continue
            value = config.params.get(param)
            allowed = envelope.get("allowed")
            min_v = envelope.get("min")
            max_v = envelope.get("max")
            if allowed is not None and value not in allowed:
                warnings.append(f"{param} not in safe envelope allowed set")
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                numeric = None
            if numeric is not None:
                if min_v is not None and numeric < float(min_v):
                    warnings.append(f"{param} below safe envelope minimum")
                if max_v is not None and numeric > float(max_v):
                    warnings.append(f"{param} above safe envelope maximum")
        return warnings
