from __future__ import annotations

import json
import os
import time
import uuid
from typing import Any, Dict, List, Optional

from .base import LLMClient, LLMMessage, LLMResponse
from ..trace import TraceEmitter


class TracedLLMClient(LLMClient):
    def __init__(self, inner: LLMClient, trace: TraceEmitter, artifacts_dir: str, run_id: str) -> None:
        super().__init__(model=inner.model)
        self.inner = inner
        self.trace = trace
        self.artifacts_dir = artifacts_dir
        self.run_id = run_id

    def complete(self, messages: List[LLMMessage], **kwargs: Any) -> LLMResponse:
        call_id = str(uuid.uuid4())
        trace_phase = kwargs.pop("trace_phase", "online")
        trace_step = kwargs.pop("trace_step", None)
        system_prompt_version = kwargs.pop("system_prompt_version", "default")
        context_refs = kwargs.pop("context_refs", [])
        injected_context_refs = kwargs.pop("injected_context_refs", context_refs)
        context_window = kwargs.pop("context_window", None)
        request_kwargs = _safe_serialize_kwargs(kwargs)
        start = time.time()
        response: Optional[LLMResponse] = None
        error: Optional[str] = None
        try:
            response = self.inner.complete(messages, **kwargs)
        except Exception as exc:
            error = str(exc)
        duration_ms = (time.time() - start) * 1000.0
        response_content = response.content if response is not None else ""
        response_model = response.model if response is not None else self.inner.model
        response_raw = _safe_serialize_value(response.raw) if response is not None else None
        payload = {
            "schema_version": "1.0",
            "call_id": call_id,
            "model": response_model,
            "system_prompt_version": system_prompt_version,
            "trace": {"phase": trace_phase, "step": trace_step, "run_id": self.run_id},
            "messages": [message.__dict__ for message in messages],
            "injected_context_refs": injected_context_refs,
            "context_refs": context_refs,
            "context_window": context_window,
            "request_kwargs": request_kwargs,
            "token_estimates": _estimate_tokens(messages, response_content),
            "response": {
                "content": response_content,
                "raw": response_raw,
            },
            "validation_errors": [],
            "duration_ms": duration_ms,
        }
        if error:
            payload["error"] = error
        path = os.path.join(self.artifacts_dir, "llm")
        os.makedirs(path, exist_ok=True)
        call_path = os.path.join(path, f"call_{call_id}.json")
        with open(call_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

        self.trace.event(
            run_id=self.run_id,
            phase=str(trace_phase),
            step=trace_step if isinstance(trace_step, int) else None,
            actor="llm",
            type="llm.call",
            payload={
                "call_id": call_id,
                "model": response_model,
                "call_path": call_path,
            },
            refs=[f"llm:call_{call_id}"],
            duration_ms=duration_ms,
            status="error" if error else "ok",
            error=error,
        )
        if error:
            raise RuntimeError(error)
        return response


def _estimate_tokens(messages: List[LLMMessage], response: str) -> Dict[str, int]:
    prompt_chars = sum(len(m.content) for m in messages)
    response_chars = len(response or "")
    # rough heuristic: 4 chars per token
    return {
        "prompt_tokens_est": int(prompt_chars / 4),
        "response_tokens_est": int(response_chars / 4),
        "total_tokens_est": int((prompt_chars + response_chars) / 4),
    }


def _safe_serialize_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    return {str(k): _safe_serialize_value(v) for k, v in kwargs.items()}


def _safe_serialize_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _safe_serialize_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_serialize_value(v) for v in value]
    return str(value)
