from __future__ import annotations

import json
import os
import urllib.request
from typing import Any, Dict, List, Optional

from .base import LLMClient, LLMError, LLMMessage, LLMResponse, _trace_llm_event


DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"


class OllamaClient(LLMClient):
    def __init__(
        self,
        model: str,
        base_url: Optional[str] = None,
        keep_alive: Optional[str] = None,
    ) -> None:
        model = model or os.getenv("OLLAMA_MODEL", "")
        if not model:
            raise LLMError("OllamaClient requires a model name (set --model or OLLAMA_MODEL).")
        super().__init__(model=model)
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL") or DEFAULT_OLLAMA_BASE_URL
        self.keep_alive = keep_alive or os.getenv("OLLAMA_KEEP_ALIVE")

    def complete(
        self,
        messages: List[LLMMessage],
        max_tokens: int = 1024,
        temperature: float = 0.2,
        **kwargs: Any,
    ) -> LLMResponse:
        url = self.base_url.rstrip("/") + "/api/chat"
        timeout_s = kwargs.pop("timeout_s", 60)
        options: Dict[str, Any] = dict(kwargs.pop("options", {}) or {})
        if max_tokens is not None:
            options.setdefault("num_predict", max_tokens)
        if temperature is not None:
            options.setdefault("temperature", temperature)
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [message.__dict__ for message in messages],
            "stream": False,
            "options": options,
        }
        if self.keep_alive:
            payload["keep_alive"] = self.keep_alive
        payload.update(kwargs)

        _trace_llm_event("request", {"model": self.model, "payload": payload, "base_url": self.base_url})

        response = _post_json(url, payload, timeout_s=timeout_s)
        content = ""
        try:
            content = response.get("message", {}).get("content", "")
        except AttributeError:
            content = ""
        _trace_llm_event("response", {"model": self.model, "content": content, "raw": response})
        return LLMResponse(content=content, model=self.model, raw=response)


def _post_json(url: str, payload: Dict[str, Any], timeout_s: int = 60) -> Any:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        body = response.read().decode("utf-8")
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return {"raw": body}
