from __future__ import annotations

import json
import uuid
from typing import Any, Dict, List, Optional

from ..llm import LLMClient, LLMMessage, NullLLMClient
from ..llm.context_window import ContextWindowManager, PromptSection, estimate_tokens
from ..types import ContextSignature, NCCLConfig, ParameterSpace
from .state import TuningState


def distill_rules_llm(
    *,
    state: TuningState,
    context: ContextSignature,
    llm: LLMClient,
    parameter_space: ParameterSpace,
    llm_config: Any,
    microbench_snapshot: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    if isinstance(llm, NullLLMClient):
        return []
    if not state.best_record or not state.history:
        return []

    baseline = state.history[0].metrics.iteration_time_ms
    best = state.best_record.metrics.iteration_time_ms
    improvement = (baseline - best) / max(1e-9, baseline)

    history_payload = [
        {
            "step": rec.step,
            "config": rec.action.config.params,
            "metrics": rec.metrics.__dict__,
            "success": rec.metrics.success,
        }
        for rec in state.history
    ]

    system_text = (
        "You are the post-run distiller for a white-box CCL/NCCL tuning agent.\n\n"
        "Task: produce portable, evidence-backed tuning rules.\n\n"
        "Rules:\n"
        "1) Output MUST be a single JSON object with a \"rules\" array; no markdown.\n"
        "2) Every rule must cite evidence_refs using the provided evidence IDs.\n"
        "3) Do not claim causality beyond the provided effect estimates. Use probabilistic language and confidence.\n"
        "4) Keep rules minimal: prefer rules that change <= 4 parameters.\n"
        "5) Include explicit limitations (\"when NOT to use\") and risk notes.\n"
        "6) Do NOT output chain-of-thought. Provide brief structured claims with refs.\n"
    )

    sections = [
        PromptSection(name="CONTEXT_SIGNATURE", content=json.dumps(context.__dict__, indent=2), priority=0),
        PromptSection(
            name="RUN_SUMMARY_STATS",
            content=json.dumps(
                {
                    "baseline_ms": baseline,
                    "best_ms": best,
                    "improvement": improvement,
                    "history_count": len(state.history),
                },
                indent=2,
            ),
            priority=0,
        ),
        PromptSection(name="CONFIG_HISTORY", content=json.dumps(history_payload, indent=2), priority=0),
        PromptSection(
            name="PARAM_SPECS",
            content=json.dumps({name: spec.__dict__ for name, spec in parameter_space.specs.items()}, indent=2),
            priority=0,
        ),
        PromptSection(
            name="MICROBENCH_SIGNALS",
            content=json.dumps(microbench_snapshot or {}, indent=2),
            priority=1,
            max_tokens=400,
        ),
        PromptSection(
            name="KNOWN_BAD_COMBOS",
            content=json.dumps(getattr(getattr(llm_config, "safety", None), "safe_envelope", {}) if llm_config else {}, indent=2),
            priority=2,
            max_tokens=400,
        ),
    ]

    max_context_tokens = llm_config.llm.max_context_tokens if hasattr(llm_config, "llm") else 8000
    max_response_tokens = llm_config.llm.max_response_tokens if hasattr(llm_config, "llm") else 512
    reserve_tokens = estimate_tokens(system_text) + int(max_response_tokens)
    manager = ContextWindowManager(max_context_tokens, reserve_tokens=reserve_tokens)
    user_text, ctx_meta = manager.build(sections)
    ctx_meta["system_tokens"] = estimate_tokens(system_text)
    ctx_meta["response_token_budget"] = max_response_tokens

    response = llm.complete(
        [
            LLMMessage(role="system", content=system_text),
            LLMMessage(role="user", content=user_text),
        ],
        trace_phase="postrun",
        trace_step=None,
        system_prompt_version=getattr(llm_config.llm, "system_prompt_version_postrun", None)
        if hasattr(llm_config, "llm")
        else None,
        context_window=ctx_meta,
        context_refs=[],
        injected_context_refs=[],
        max_tokens=max_response_tokens,
        temperature=llm_config.llm.temperature if hasattr(llm_config, "llm") else 0.2,
        timeout_s=getattr(llm_config.llm, "online_hard_timeout_s", None) if hasattr(llm_config, "llm") else None,
    )

    parsed = _parse_llm_json(response.content)
    rules = parsed.get("rules", []) if isinstance(parsed, dict) else []
    cleaned: List[Dict[str, Any]] = []
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        action = rule.get("action", {})
        patch = action.get("set", {}) if isinstance(action, dict) else {}
        if not isinstance(patch, dict):
            continue
        if len(patch) > 4:
            continue
        if any(k not in parameter_space.specs for k in patch.keys()):
            continue
        rule_id = rule.get("rule_id") or f"rule_{uuid.uuid4()}"
        cleaned.append(
            {
                "schema_version": "1.0",
                "rule_id": rule_id,
                "context": rule.get("context", context.__dict__),
                "condition": rule.get("condition", {}),
                "action": {"set": patch},
                "effect": rule.get("effect", {}),
                "limitations": rule.get("limitations", []),
                "evidence_refs": rule.get("evidence_refs", []),
                "risk": rule.get("risk", {}),
            }
        )
    return cleaned


def _parse_llm_json(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        return {}
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {}
