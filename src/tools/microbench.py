from __future__ import annotations

import json
import os
import subprocess
import time
import math
from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, List, Optional

from ..types import (
    ImportantParam,
    MicrobenchResult,
    MicrobenchSettings,
    MicrobenchSignal,
    ParameterSpace,
    RunContext,
    WorkloadSpec,
)
from ..utils import artifact_path, setup_logger, write_json
from .base import ToolExecutionError


logger = setup_logger("cclagent.microbench")


@dataclass
class MicrobenchConfig:
    mode: str = "dry"
    command_template: Optional[List[str]] = None
    parse_schema: str = "cclinsight_v1"
    timeout_s: int = 900
    dry_run: bool = True
    env: Optional[Dict[str, str]] = None
    repetitions: int = 1
    allow_fallback: bool = True

    @classmethod
    def from_settings(cls, settings: MicrobenchSettings, *, dry_run: bool) -> "MicrobenchConfig":
        return cls(
            mode=settings.mode,
            command_template=settings.command_template or None,
            parse_schema=settings.parse_schema,
            timeout_s=settings.timeout_sec,
            dry_run=dry_run,
            env=settings.env or None,
            repetitions=max(1, settings.repetitions),
            allow_fallback=settings.allow_fallback,
        )


class MicrobenchRunner:
    def __init__(
        self,
        config: MicrobenchConfig,
        executor: Optional[Callable[..., MicrobenchResult]] = None,
        run_context: Optional[RunContext] = None,
    ):
        self.config = config
        self.executor = executor
        self.run_context = run_context

    def run(self, workload: WorkloadSpec, parameter_space: ParameterSpace) -> MicrobenchResult:
        if self.executor:
            return self.executor(workload, parameter_space)
        if self.config.dry_run or self.config.mode == "dry" or not self.config.command_template:
            return self._simulate(parameter_space)

        try:
            return self._run_real(workload, parameter_space)
        except ToolExecutionError as exc:
            if self.config.allow_fallback:
                logger.warning("Microbench failed (%s). Falling back to dry-run.", exc)
                result = self._simulate(parameter_space)
                result.raw["fallback_reason"] = str(exc)
                return result
            raise

    def _simulate(self, parameter_space: ParameterSpace) -> MicrobenchResult:
        important = [
            ImportantParam(param=name, importance=0.5, reason="dry-run default")
            for name in list(parameter_space.specs.keys())[:4]
        ]
        signals = [
            MicrobenchSignal(name="bandwidth", value=1.0, unit="GB/s", confidence=0.2, source="dry"),
            MicrobenchSignal(name="latency", value=1.0, unit="us", confidence=0.2, source="dry"),
        ]
        return MicrobenchResult(important_params=important, signals=signals, raw={"dry_run": True})

    def _run_real(self, workload: WorkloadSpec, parameter_space: ParameterSpace) -> MicrobenchResult:
        command = self._build_command(workload)
        env = os.environ.copy()
        env["CCLAGENT_PARAM_LIST"] = ",".join(parameter_space.specs.keys())
        if self.config.env:
            env.update(self.config.env)
        stdout_path = ""
        stderr_path = ""
        raw_payloads: List[Dict[str, Any]] = []
        start = time.time()

        for rep in range(max(1, self.config.repetitions)):
            logger.info("Running microbench repetition %d/%d", rep + 1, self.config.repetitions)
            try:
                result = subprocess.run(
                    command,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=self.config.timeout_s,
                    env=env,
                )
            except subprocess.SubprocessError as exc:
                raise ToolExecutionError(f"microbench failed: {exc}") from exc

            stdout = result.stdout.strip()
            stderr = result.stderr.strip()
            if self.run_context:
                stdout_path = artifact_path(self.run_context, "offline", f"microbench_stdout_{rep}.log")
                stderr_path = artifact_path(self.run_context, "offline", f"microbench_stderr_{rep}.log")
                with open(stdout_path, "w", encoding="utf-8") as handle:
                    handle.write(stdout)
                with open(stderr_path, "w", encoding="utf-8") as handle:
                    handle.write(stderr)

            try:
                payload = json.loads(stdout)
            except json.JSONDecodeError as exc:
                raise ToolExecutionError("microbench output is not valid JSON") from exc
            raw_payloads.append(payload)

        runtime = time.time() - start
        merged = self._merge_payloads(raw_payloads)
        important_params = self._parse_important(merged)
        signals = self._parse_signals(merged)
        result = MicrobenchResult(
            important_params=important_params,
            signals=signals,
            raw_path=stdout_path,
            command=command,
            runtime_sec=runtime,
            raw={
                "payloads": raw_payloads,
                "stderr_path": stderr_path,
            },
        )
        if self.run_context:
            write_json(artifact_path(self.run_context, "offline", "microbench_result.json"), asdict(result))
        return result

    def _build_command(self, workload: WorkloadSpec) -> List[str]:
        template = self.config.command_template or []
        if not template:
            return []
        context = {
            "workload": workload.name,
            "nodes": workload.nodes,
            "topology": workload.topology,
            "scale": workload.scale,
        }
        return [part.format(**context) for part in template]

    def _merge_payloads(self, payloads: List[Dict[str, Any]]) -> Dict[str, Any]:
        if len(payloads) == 1:
            return payloads[0]
        merged: Dict[str, Any] = {"repetitions": len(payloads), "payloads": payloads}
        param_scores: Dict[str, List[float]] = {}
        param_reasons: Dict[str, str] = {}
        param_evidence: Dict[str, List[Dict[str, Any]]] = {}
        for payload in payloads:
            for item in payload.get("important_params", []) or []:
                name = item.get("name") or item.get("param")
                if not name:
                    continue
                score = item.get("score", item.get("importance"))
                if score is None:
                    continue
                try:
                    score_val = float(score)
                except (TypeError, ValueError):
                    continue
                param_scores.setdefault(name, []).append(score_val)
                if name not in param_reasons and item.get("reason"):
                    param_reasons[name] = item.get("reason")
                param_evidence.setdefault(name, []).append(item.get("evidence", {}))

        merged_params: List[Dict[str, Any]] = []
        for name, scores in param_scores.items():
            mean = sum(scores) / len(scores)
            variance = sum((s - mean) ** 2 for s in scores) / max(1, len(scores))
            stdev = math.sqrt(variance)
            merged_params.append(
                {
                    "name": name,
                    "score": mean,
                    "reason": param_reasons.get(name, ""),
                    "evidence": {
                        "samples": scores,
                        "mean": mean,
                        "stdev": stdev,
                        "repetitions": len(scores),
                        "raw_evidence": param_evidence.get(name, []),
                    },
                }
            )
        if merged_params:
            merged["important_params"] = merged_params

        signal_values: Dict[str, List[float]] = {}
        signal_first: Dict[str, Any] = {}
        signal_units: Dict[str, str] = {}
        signal_conf: Dict[str, List[float]] = {}
        signal_raw_values: Dict[str, List[Any]] = {}
        for payload in payloads:
            for item in payload.get("signals", []) or []:
                name = item.get("name")
                if not name:
                    continue
                value = item.get("value")
                signal_raw_values.setdefault(name, []).append(value)
                if name not in signal_first and value is not None:
                    signal_first[name] = value
                unit = item.get("unit")
                if unit:
                    signal_units[name] = unit
                conf = item.get("confidence")
                if conf is not None:
                    try:
                        signal_conf.setdefault(name, []).append(float(conf))
                    except (TypeError, ValueError):
                        pass
                try:
                    signal_values.setdefault(name, []).append(float(value))
                except (TypeError, ValueError):
                    continue

        merged_signals: List[Dict[str, Any]] = []
        for name in set(signal_raw_values.keys()):
            unit = signal_units.get(name)
            confs = signal_conf.get(name, [])
            confidence = sum(confs) / len(confs) if confs else 0.5
            if name in signal_values:
                vals = signal_values[name]
                mean = sum(vals) / len(vals)
                variance = sum((v - mean) ** 2 for v in vals) / max(1, len(vals))
                stdev = math.sqrt(variance)
                merged_signals.append(
                    {
                        "name": name,
                        "value": mean,
                        "unit": unit,
                        "confidence": confidence,
                        "evidence": {
                            "samples": vals,
                            "mean": mean,
                            "stdev": stdev,
                            "repetitions": len(vals),
                        },
                    }
                )
            else:
                merged_signals.append(
                    {
                        "name": name,
                        "value": signal_first.get(name),
                        "unit": unit,
                        "confidence": confidence,
                        "evidence": {
                            "samples": signal_raw_values.get(name, []),
                            "repetitions": len(signal_raw_values.get(name, [])),
                        },
                    }
                )
        if merged_signals:
            merged["signals"] = merged_signals
        return merged

    def _parse_important(self, payload: Dict[str, Any]) -> List[ImportantParam]:
        items = payload.get("important_params") or []
        important: List[ImportantParam] = []
        for item in items:
            name = item.get("name") or item.get("param")
            if not name:
                continue
            important.append(
                ImportantParam(
                    param=name,
                    importance=float(item.get("score", item.get("importance", 0.0))),
                    reason=item.get("reason", ""),
                    evidence=item.get("evidence", {}),
                )
            )
        return important

    def _parse_signals(self, payload: Dict[str, Any]) -> List[MicrobenchSignal]:
        if self.config.parse_schema == "nccltests_v1":
            signals = []
            for key in ("algbw", "busbw", "time"):
                if key in payload:
                    signals.append(MicrobenchSignal(name=key, value=payload.get(key), unit=None, confidence=0.5, source="nccltests"))
            return signals
        items = payload.get("signals") or []
        signals: List[MicrobenchSignal] = []
        for item in items:
            name = item.get("name")
            if not name:
                continue
            signals.append(
                MicrobenchSignal(
                    name=name,
                    value=item.get("value"),
                    unit=item.get("unit"),
                    confidence=float(item.get("confidence", 0.5)),
                    source=item.get("source", self.config.parse_schema),
                )
            )
        return signals
