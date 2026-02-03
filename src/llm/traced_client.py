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
        system_prompt_version = kwargs.pop("system_prompt_version", "default")
        context_refs = kwargs.pop("context_refs", [])
        injected_context_refs = kwargs.pop("injected_context_refs", context_refs)
        start = time.time()
        response = self.inner.complete(messages, **kwargs)
        duration_ms = (time.time() - start) * 1000.0
        payload = {
            "schema_version": "1.0",
            "call_id": call_id,
            "model": response.model,
            "system_prompt_version": system_prompt_version,
            "messages": [message.__dict__ for message in messages],
            "injected_context_refs": injected_context_refs,
            "token_estimates": _estimate_tokens(messages, response.content),
            "response": {
                "content": response.content,
            },
            "validation_errors": [],
        }
        path = os.path.join(self.artifacts_dir, "llm")
        os.makedirs(path, exist_ok=True)
        call_path = os.path.join(path, f"call_{call_id}.json")
        with open(call_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

        self.trace.event(
            run_id=self.run_id,
            phase="online",
            step=None,
            actor="llm",
            type="llm.call",
            payload={
                "call_id": call_id,
                "model": response.model,
                "call_path": call_path,
            },
            refs=[f"llm:call_{call_id}"],
            duration_ms=duration_ms,
        )
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
