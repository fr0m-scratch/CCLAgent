from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from typing import Any, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, Future

from ..llm import LLMClient, LLMMessage
from ..llm.context_window import ContextWindowManager, PromptSection, estimate_tokens
from ..llm.schemas import validate_online_decision_support
from ..trace import TraceEmitter, NullTraceEmitter


@dataclass
class LLMAdvice:
    step: int
    call_id: Optional[str]
    output: Dict[str, Any]
    parse_errors: list[str]
    raw_text: str
    raw_is_valid_json: bool = False
    schema_passed: bool = False
    decision_eligible: bool = False


@dataclass
class JSONParseStatus:
    value: Any
    raw_is_valid_json: bool
    used_partial_extraction: bool = False


class OnlineLLMAdvisor:
    def __init__(
        self,
        llm: LLMClient,
        parameter_space: Any,
        config: Any,
        run_context: Optional[Any] = None,
        trace: Optional[TraceEmitter] = None,
    ) -> None:
        self.llm = llm
        self.parameter_space = parameter_space
        self.config = config
        self.run_context = run_context
        self.trace = trace or NullTraceEmitter()
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._futures: Dict[int, Future] = {}
        self._lock = threading.Lock()

    def should_call(self, step: int) -> bool:
        llm_cfg = getattr(self.config, "llm", None)
        if llm_cfg is None or not getattr(llm_cfg, "online_enabled", False):
            return False
        every = max(1, int(getattr(llm_cfg, "online_call_every_steps", 1)))
        return (step % every) == 0

    def request(
        self,
        *,
        step: int,
        context_pack: Dict[str, Any],
        playbook: list[Dict[str, Any]],
        param_specs: Dict[str, Any],
        recent_history: Dict[str, Any],
        recent_pruning: Dict[str, Any],
    ) -> None:
        if not self.should_call(step):
            return
        with self._lock:
            if step in self._futures:
                return

            system_text = (
                "You are the online decision-support reasoner for a white-box CCL/NCCL tuning agent.\n\n"
                "You MUST:\n"
                "- Output a single JSON object exactly matching the requested schema.\n"
                "- Only propose parameter keys that exist in [PARAM_SPECS].\n"
                "- Keep each hypothesis.patch to <= 4 keys and prefer low-risk changes.\n"
                "- Cite evidence using the provided evidence IDs in refs/evidence_refs.\n"
                "- Every hypothesis summary MUST be concrete and falsifiable: mention bottleneck/mechanism and exact parameter change.\n"
                "- Never use vague summaries like 'apply memory rule', 'tune params', or 'generic optimization'.\n"
                "- For each hypothesis include short reason_claims with explicit evidence refs.\n"
                "- Include convergence decision under key `convergence` with decision in {continue, stop}.\n"
                "- Include `action_preference` in {auto, hypothesis, numeric}.\n"
                "- Use `action_preference` to choose lane this step: hypothesis for mechanism-driven interventions, numeric for continued safe exploration (including latency-hiding when uncertainty is high).\n"
                "- Do NOT output chain-of-thought. Use short claims with refs.\n\n"
                "Expected JSON keys:\n"
                "{\n"
                '  "hypotheses": [...],\n'
                '  "numeric_guidance": {...},\n'
                '  "tool_request": {"name":"none|nccltest.short|workload.short|microbench.reduced","reason":"..."},\n'
                '  "action_preference": "auto|hypothesis|numeric",\n'
                '  "convergence": {\n'
                '    "decision":"continue|stop",\n'
                '    "reason":"...",\n'
                '    "confidence":0.0,\n'
                '    "claims":[{"claim":"...", "refs":["..."]}]\n'
                "  }\n"
                "}\n\n"
                "Objective: reduce iteration_time_ms safely under the provided risk/SLA budgets.\n"
                "Bad example summary: 'apply memory rule'.\n"
                "Good example summary: 'NCCL_MAX_NCHANNELS appears too low for 32-GPU scale; increase channels to reduce ring under-utilization'."
            )
            sections = [
                PromptSection(name="CONTEXT_PACK", content=json.dumps(context_pack, indent=2), priority=0),
                PromptSection(name="RECENT_HISTORY_SUMMARY", content=json.dumps(recent_history, indent=2), priority=0),
                PromptSection(name="CURRENT_PLAYBOOK", content=json.dumps(playbook, indent=2), priority=1),
                PromptSection(name="PARAM_SPECS", content=json.dumps(param_specs, indent=2), priority=0),
                PromptSection(name="RECENT_PRUNING", content=json.dumps(recent_pruning, indent=2), priority=1),
            ]
            max_context_tokens = self.config.llm.max_context_tokens or 8000
            max_response_tokens = self.config.llm.max_response_tokens or 512
            reserve_tokens = estimate_tokens(system_text) + int(max_response_tokens)
            manager = ContextWindowManager(max_context_tokens, reserve_tokens=reserve_tokens)
            user_text, ctx_meta = manager.build(sections)
            ctx_meta["system_tokens"] = estimate_tokens(system_text)
            ctx_meta["response_token_budget"] = max_response_tokens

            messages = [
                LLMMessage(role="system", content=system_text),
                LLMMessage(role="user", content=user_text),
            ]
            context_refs = _extract_context_refs(context_pack)

            def _call_llm() -> LLMAdvice:
                response = self.llm.complete(
                    messages,
                    trace_phase="online",
                    trace_step=step,
                    system_prompt_version=getattr(self.config.llm, "system_prompt_version_online", None)
                    or self.config.llm.system_prompt_version,
                    context_window=ctx_meta,
                    context_refs=context_refs,
                    injected_context_refs=context_refs,
                    max_tokens=self.config.llm.max_response_tokens or 512,
                    temperature=self.config.llm.temperature,
                    timeout_s=getattr(self.config.llm, "online_hard_timeout_s", None),
                    **self._json_response_kwargs(),
                )
                parsed_status = _parse_llm_json_status(response.content)
                parse_errors: list[str] = []
                if not parsed_status.raw_is_valid_json:
                    parse_errors.append("invalid_json")
                if parsed_status.used_partial_extraction:
                    parse_errors.append("partial_json")
                parsed = parsed_status.value
                schema_passed = False
                if not isinstance(parsed, dict):
                    parse_errors.append("schema:output_not_dict")
                    parsed = {}
                else:
                    schema_errors = validate_online_decision_support(parsed, self.parameter_space)
                    parse_errors.extend([f"schema:{item}" for item in schema_errors])
                    schema_passed = len(schema_errors) == 0
                decision_eligible = (
                    parsed_status.raw_is_valid_json
                    and schema_passed
                    and isinstance(parsed, dict)
                    and bool(parsed)
                    and not parse_errors
                )
                call_id = getattr(response, "call_id", None)
                return LLMAdvice(
                    step=step,
                    call_id=call_id,
                    output=parsed,
                    parse_errors=parse_errors,
                    raw_text=response.content,
                    raw_is_valid_json=parsed_status.raw_is_valid_json,
                    schema_passed=schema_passed,
                    decision_eligible=decision_eligible,
                )

            self._futures[step] = self._executor.submit(_call_llm)
            self._trace_async(
                step=step,
                event_type="llm.advice.async_dispatched",
                payload={
                    "context_tokens": ctx_meta.get("total_tokens"),
                    "section_count": len(ctx_meta.get("sections", [])),
                    "response_token_budget": max_response_tokens,
                },
            )

    def try_get(self, *, step: int, timeout_s: float = 0.0) -> Optional[LLMAdvice]:
        with self._lock:
            fut = self._futures.get(step)
        if fut is None:
            return None
        try:
            result = fut.result(timeout=timeout_s)
            with self._lock:
                self._futures.pop(step, None)
            self._trace_async(
                step=step,
                event_type="llm.advice.async_ready",
                payload={"call_id": result.call_id},
            )
            return result
        except Exception:
            return None

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False)

    def collect_ready(self) -> list[LLMAdvice]:
        ready: list[LLMAdvice] = []
        with self._lock:
            items = list(self._futures.items())
        for step, fut in items:
            if not fut.done():
                continue
            try:
                result = fut.result(timeout=0)
            except Exception:
                result = None
            with self._lock:
                self._futures.pop(step, None)
            if result is not None:
                self._trace_async(
                    step=step,
                    event_type="llm.advice.async_ready",
                    payload={"call_id": result.call_id},
                )
                ready.append(result)
        return ready

    def decide_convergence(
        self,
        *,
        step: int,
        context_pack: Dict[str, Any],
        recent_history: Dict[str, Any],
        policy_hints: Dict[str, Any],
    ) -> Optional[LLMAdvice]:
        llm_cfg = getattr(self.config, "llm", None)
        if llm_cfg is None or not getattr(llm_cfg, "convergence_enabled", True):
            return None
        system_text = (
            "You are the convergence decider for an online NCCL tuning run.\n"
            "Output one JSON object only:\n"
            "{\n"
            '  "decision":"continue|stop",\n'
            '  "reason":"short reason",\n'
            '  "confidence":0.0,\n'
            '  "claims":[{"claim":"...", "refs":["..."]}]\n'
            "}\n"
            "Use provided evidence IDs. Prefer continue when evidence is weak."
        )
        sections = [
            PromptSection(name="CONTEXT_PACK", content=json.dumps(context_pack, indent=2), priority=0),
            PromptSection(name="RECENT_HISTORY_SUMMARY", content=json.dumps(recent_history, indent=2), priority=0),
            PromptSection(name="POLICY_HINTS", content=json.dumps(policy_hints, indent=2), priority=0),
        ]
        max_context_tokens = self.config.llm.max_context_tokens or 8000
        max_response_tokens = min(300, self.config.llm.max_response_tokens or 512)
        reserve_tokens = estimate_tokens(system_text) + int(max_response_tokens)
        manager = ContextWindowManager(max_context_tokens, reserve_tokens=reserve_tokens)
        user_text, ctx_meta = manager.build(sections)
        ctx_meta["system_tokens"] = estimate_tokens(system_text)
        ctx_meta["response_token_budget"] = max_response_tokens
        messages = [
            LLMMessage(role="system", content=system_text),
            LLMMessage(role="user", content=user_text),
        ]
        context_refs = _extract_context_refs(context_pack)
        response = self.llm.complete(
            messages,
            trace_phase="online",
            trace_step=step,
            system_prompt_version="online_convergence_v1",
            context_window=ctx_meta,
            context_refs=context_refs,
            injected_context_refs=context_refs,
            max_tokens=max_response_tokens,
            temperature=self.config.llm.temperature,
            timeout_s=getattr(self.config.llm, "online_hard_timeout_s", None),
            **self._json_response_kwargs(),
        )
        parsed_status = _parse_llm_json_status(response.content)
        parse_errors: list[str] = []
        if not parsed_status.raw_is_valid_json:
            parse_errors.append("invalid_json")
        if parsed_status.used_partial_extraction:
            parse_errors.append("partial_json")
        parsed_raw = parsed_status.value
        if not isinstance(parsed_raw, dict):
            parsed_raw = {}
            parse_errors.append("schema:output_not_dict")
        parsed = _coerce_convergence_output(parsed_raw, response.content)
        decision = parsed.get("decision")
        if decision not in ("continue", "stop"):
            parse_errors.append("schema:invalid_convergence_decision")
        confidence = parsed.get("confidence")
        if confidence is not None:
            try:
                score = float(confidence)
                if score < 0.0 or score > 1.0:
                    parse_errors.append("schema:invalid_convergence_confidence")
            except (TypeError, ValueError):
                parse_errors.append("schema:invalid_convergence_confidence")
        output = {"convergence": parsed}
        schema_passed = all(
            not item.startswith("schema:")
            for item in parse_errors
        )
        decision_eligible = parsed_status.raw_is_valid_json and schema_passed and not parse_errors
        return LLMAdvice(
            step=step,
            call_id=getattr(response, "call_id", None),
            output=output,
            parse_errors=parse_errors,
            raw_text=response.content,
            raw_is_valid_json=parsed_status.raw_is_valid_json,
            schema_passed=schema_passed,
            decision_eligible=decision_eligible,
        )

    def _trace_async(self, *, step: int, event_type: str, payload: Dict[str, Any]) -> None:
        if self.run_context is None:
            return
        try:
            self.trace.event(
                run_id=self.run_context.run_id,
                phase="online",
                step=step,
                actor="llm",
                type=event_type,
                payload=payload,
            )
        except Exception:
            # Async telemetry should never break tuning.
            return

    def _json_response_kwargs(self) -> dict[str, Any]:
        provider = str(getattr(self.config.llm, "provider", "")).lower()
        if provider in ("openai", "fireworks", "openai-compatible"):
            return {"response_format": {"type": "json_object"}}
        return {}


def _parse_llm_json_status(text: str) -> JSONParseStatus:
    try:
        return JSONParseStatus(value=json.loads(text), raw_is_valid_json=True)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        return JSONParseStatus(value={}, raw_is_valid_json=False, used_partial_extraction=False)
    try:
        return JSONParseStatus(
            value=json.loads(text[start : end + 1]),
            raw_is_valid_json=False,
            used_partial_extraction=True,
        )
    except json.JSONDecodeError:
        return JSONParseStatus(value={}, raw_is_valid_json=False, used_partial_extraction=False)


def _extract_context_refs(context_pack: Dict[str, Any]) -> list[str]:
    refs: list[str] = []
    retrieval = context_pack.get("retrieval", {}) if isinstance(context_pack, dict) else {}
    for rule in retrieval.get("memory_rules", []) if isinstance(retrieval, dict) else []:
        ref = rule.get("ref") or rule.get("rule_id")
        if ref:
            if not ref.startswith("rule:"):
                ref = f"rule:{ref}"
            refs.append(ref)
    for chunk in retrieval.get("rag_chunks", []) if isinstance(retrieval, dict) else []:
        ref = chunk.get("ref")
        if ref:
            refs.append(ref)
    return refs


def _coerce_convergence_output(parsed: Dict[str, Any], raw_text: str) -> Dict[str, Any]:
    if isinstance(parsed, dict) and parsed.get("decision") in ("continue", "stop"):
        return parsed
    lower = (raw_text or "").lower()
    if "decision" in lower and "stop" in lower and "continue" not in lower:
        decision = "stop"
    elif "decision" in lower and "continue" in lower and "stop" not in lower:
        decision = "continue"
    elif "should stop" in lower or "recommend stop" in lower:
        decision = "stop"
    else:
        decision = "continue"
    reason = "coerced_non_json_response"
    preferred = ("recommend stop", "recommend continue", "should stop", "should continue")
    for line in (raw_text or "").splitlines():
        text = line.strip()
        if not text:
            continue
        low = text.lower()
        if any(token in low for token in preferred):
            reason = text[:220]
            break
    confidence = 0.65
    conf_match = _extract_confidence(lower)
    if conf_match is not None:
        confidence = conf_match
    claims = []
    if reason:
        claims.append({"claim": reason, "refs": []})
    return {
        "decision": decision,
        "reason": reason or "coerced_from_text",
        "confidence": confidence,
        "claims": claims,
    }


def _extract_confidence(text: str) -> float | None:
    import re

    match = re.search(r"confidence\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", text)
    if not match:
        return None
    try:
        value = float(match.group(1))
    except ValueError:
        return None
    if value > 1.0:
        value = value / 100.0
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value
