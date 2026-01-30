from __future__ import annotations

import os
from typing import Any, Callable, List, Optional

from .base import LLMClient, LLMError, LLMMessage, LLMResponse


class GeminiClient(LLMClient):
    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        complete_fn: Optional[Callable[[List[LLMMessage], Any], str]] = None,
    ) -> None:
        super().__init__(model=model)
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.complete_fn = complete_fn
        self._client = None
        if self.complete_fn is None:
            try:
                import google.generativeai as genai
            except ImportError:
                return
            if not self.api_key:
                return
            genai.configure(api_key=self.api_key)
            self._client = genai.GenerativeModel(self.model)

    def complete(self, messages: List[LLMMessage], **kwargs: Any) -> LLMResponse:
        if self.complete_fn:
            content = self.complete_fn(messages, **kwargs)
            return LLMResponse(content=content, model=self.model, raw=None)
        if self._client is None:
            raise LLMError(
                "GeminiClient requires google-generativeai or a custom complete_fn"
            )

        prompt = "\n".join(f"{m.role}: {m.content}" for m in messages)
        response = self._client.generate_content(prompt, **kwargs)
        content = getattr(response, "text", "")
        return LLMResponse(content=content, model=self.model, raw=response)
