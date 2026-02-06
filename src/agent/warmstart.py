from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from ..types import NCCLConfig, WorkloadSpec
from ..utils import artifact_path, write_json
from ..safety.risk import RiskScorer
from ..trace import TraceEmitter, NullTraceEmitter


@dataclass
class WarmStartProbe:
    candidate_id: str
    config: Dict[str, Any]
    risk_score: float
    success: bool
    iteration_time_ms: float
    reason: str


def _build_candidate_configs(program: Dict[str, Any], defaults: Dict[str, Any]) -> List[Dict[str, Any]]:
    configs: List[Dict[str, Any]] = []
    for candidate in program.get("candidates", []):
        patch = candidate.get("patch", {}) if isinstance(candidate, dict) else {}
        merged = dict(defaults)
        if isinstance(patch, dict):
            merged.update(patch)
        configs.append(merged)
    return configs


def run_warm_start_program(
    *,
    program: Dict[str, Any],
    defaults: Dict[str, Any],
    workload: WorkloadSpec,
    executor: Any,
    parameter_space: Any,
    safety: Any,
    run_context: Optional[Any] = None,
    trace: Optional[TraceEmitter] = None,
    max_candidates: int = 3,
    eval_steps: int = 50,
    eval_timeout_sec: int = 300,
    concurrency: int = 1,
) -> Tuple[NCCLConfig, List[WarmStartProbe]]:
    trace = trace or NullTraceEmitter()
    if not program or program.get("mode") != "series":
        return NCCLConfig(params=defaults), []

    candidates = program.get("candidates", [])
    if not isinstance(candidates, list):
        return NCCLConfig(params=defaults), []
    candidates = candidates[: max(1, max_candidates)]

    merged_configs = _build_candidate_configs({"candidates": candidates}, defaults)
    valid_configs: List[NCCLConfig] = []
    valid_candidate_ids: List[str] = []
    risk_scores: Dict[str, float] = {}
    for idx, cfg in enumerate(merged_configs):
        candidate = candidates[idx] if idx < len(candidates) else {}
        candidate_id = candidate.get("id") or f"WS{idx}"
        errors = parameter_space.validate(cfg)
        risk = RiskScorer(safety).score(NCCLConfig(params=cfg))
        risk_scores[candidate_id] = risk.risk_score
        if errors:
            continue
        valid_candidate_ids.append(candidate_id)
        valid_configs.append(NCCLConfig(params=cfg))

    if not valid_configs:
        return NCCLConfig(params=defaults), []

    results = executor.run_batch(
        workload=workload,
        candidates=valid_configs,
        step=0,
        eval_mode="short",
        concurrency=concurrency,
        artifact_subdir="offline/warmstart",
        eval_steps_override=eval_steps,
        eval_timeout_override=eval_timeout_sec,
    )

    probes: List[WarmStartProbe] = []
    for idx, (candidate_cfg, metrics) in enumerate(results):
        candidate_id = valid_candidate_ids[idx] if idx < len(valid_candidate_ids) else f"WS{idx}"
        probes.append(
            WarmStartProbe(
                candidate_id=candidate_id,
                config=candidate_cfg.params,
                risk_score=risk_scores.get(candidate_id, 0.0),
                success=metrics.success,
                iteration_time_ms=metrics.iteration_time_ms,
                reason="ok" if metrics.success else "failed",
            )
        )
        if run_context:
            trace.event(
                run_id=run_context.run_id,
                phase="offline",
                step=None,
                actor="agent",
                type="warmstart.probe.run",
                payload={
                    "candidate_id": candidate_id,
                    "success": metrics.success,
                    "iteration_time_ms": metrics.iteration_time_ms,
                    "risk_score": risk_scores.get(candidate_id, 0.0),
                },
                refs=[f"candidate:warmstart:{candidate_id}"],
            )

    # choose best (min iteration_time_ms among success)
    successful = [p for p in probes if p.success]
    if not successful:
        selected = probes[0]
    else:
        successful.sort(
            key=lambda p: (p.iteration_time_ms, p.risk_score),
        )
        selected = successful[0]

    selected_config = NCCLConfig(params=selected.config)

    if run_context:
        write_json(
            artifact_path(run_context, "offline", "warm_start_probe_results.json"),
            [probe.__dict__ for probe in probes],
        )
        write_json(
            artifact_path(run_context, "offline", "warm_start_probe_decision.json"),
            {
                "schema_version": "1.0",
                "selected_id": selected.candidate_id,
                "iteration_time_ms": selected.iteration_time_ms,
                "risk_score": selected.risk_score,
                "reason": "min_iteration_time_ms",
            },
        )
        trace.event(
            run_id=run_context.run_id,
            phase="offline",
            step=None,
            actor="agent",
            type="warmstart.probe.select",
            payload={
                "selected_id": selected.candidate_id,
                "iteration_time_ms": selected.iteration_time_ms,
                "risk_score": selected.risk_score,
                "reason": "min_iteration_time_ms",
            },
            refs=[f"candidate:warmstart:{selected.candidate_id}"],
        )
    return selected_config, probes
