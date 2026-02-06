from __future__ import annotations

import json
import os
import time
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class LLMMessage:
    role: str
    content: str


@dataclass
class LLMResponse:
    content: str
    model: str
    raw: Any | None = None
    call_id: str | None = None


class LLMError(RuntimeError):
    pass


class LLMClient:
    def __init__(self, model: str) -> None:
        self.model = model

    def complete(self, messages: List[LLMMessage], **kwargs: Any) -> LLMResponse:
        raise NotImplementedError


class NullLLMClient(LLMClient):
    def __init__(self) -> None:
        super().__init__(model="none")

    def complete(self, messages: List[LLMMessage], **kwargs: Any) -> LLMResponse:
        return LLMResponse(content="", model=self.model, raw=None)


class OpenAICompatibleClient(LLMClient):
    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        api_path: str = "/v1/chat/completions",
        timeout_s: int = 60,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> None:
        super().__init__(model=model)
        self.api_key = api_key
        self.base_url = base_url
        self.api_path = api_path
        self.timeout_s = timeout_s
        self.extra_headers = extra_headers or {}

    def complete(
        self,
        messages: List[LLMMessage],
        max_tokens: int = 1024,
        temperature: float = 0.2,
        **kwargs: Any,
    ) -> LLMResponse:
        if not self.base_url:
            raise LLMError("base_url is required for OpenAI-compatible clients")
        if not self.api_key:
            raise LLMError("api_key is required for OpenAI-compatible clients")

        url = self.base_url.rstrip("/") + self.api_path
        timeout_s = kwargs.pop("timeout_s", None)
        payload = {
            "model": self.model,
            "messages": [message.__dict__ for message in messages],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        payload.update(kwargs)

        _trace_llm_event("request", {"model": self.model, "payload": payload, "base_url": self.base_url})

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        headers.update(self.extra_headers)

        response = _post_json(url, payload, headers=headers, timeout_s=timeout_s or self.timeout_s)
        content = _extract_openai_content(response)
        _trace_llm_event("response", {"model": self.model, "content": content, "raw": response})
        return LLMResponse(content=content, model=self.model, raw=response)


def _post_json(url: str, payload: Dict[str, Any], headers: Dict[str, str], timeout_s: int) -> Any:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        body = response.read().decode("utf-8")
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return {"raw": body}


def _extract_openai_content(response: Any) -> str:
    try:
        return response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return ""


def _trace_llm_event(kind: str, payload: Dict[str, Any]) -> None:
    if os.getenv("CCL_LLM_TRACE_STDOUT", "0").lower() in ("1", "true", "yes"):
        try:
            print(f"[LLM_TRACE] {json.dumps({'kind': kind, 'payload': payload})}")
        except Exception:
            pass
    trace_dir = os.getenv("CCL_LLM_TRACE_DIR")
    if not trace_dir:
        return
    try:
        os.makedirs(trace_dir, exist_ok=True)
        path = os.path.join(trace_dir, "llm_trace.jsonl")
        envelope = {
            "ts": time.time(),
            "kind": kind,
            "payload": payload,
        }
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(envelope) + "\n")
    except Exception:
        return
