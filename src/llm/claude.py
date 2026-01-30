from __future__ import annotations

import os
from typing import Any, Callable, List, Optional

from .base import LLMClient, LLMError, LLMMessage, LLMResponse


class ClaudeClient(LLMClient):
    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        complete_fn: Optional[Callable[[List[LLMMessage], Any], str]] = None,
    ) -> None:
        super().__init__(model=model)
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.complete_fn = complete_fn
        self._client = None
        if self.complete_fn is None:
            try:
                import anthropic
            except ImportError:
                return
            if not self.api_key:
                return
            self._client = anthropic.Anthropic(api_key=self.api_key)

    def complete(self, messages: List[LLMMessage], max_tokens: int = 1024, temperature: float = 0.2, **kwargs: Any) -> LLMResponse:
        if self.complete_fn:
            content = self.complete_fn(messages, **kwargs)
            return LLMResponse(content=content, model=self.model, raw=None)
        if self._client is None:
            raise LLMError(
                "ClaudeClient requires the anthropic package or a custom complete_fn"
            )

        payload_messages = [
            {"role": message.role, "content": message.content} for message in messages
        ]
        response = self._client.messages.create(
            model=self.model,
            messages=payload_messages,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )
        content = ""
        try:
            if response.content:
                content = response.content[0].text
        except Exception:
            content = ""
        return LLMResponse(content=content, model=self.model, raw=response)
