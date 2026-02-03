from __future__ import annotations

from typing import Any

from .base import LLMClient, LLMMessage, LLMResponse, NullLLMClient, OpenAICompatibleClient
from .claude import ClaudeClient
from .fireworks import FireworksClient
from .gemini import GeminiClient
from .openai import OpenAIClient
from .traced_client import TracedLLMClient


def create_llm_client(provider: str, model: str, **kwargs: Any) -> LLMClient:
    provider = provider.lower()
    if provider == "openai":
        return OpenAIClient(model=model, **kwargs)
    if provider == "fireworks":
        return FireworksClient(model=model, **kwargs)
    if provider == "claude":
        return ClaudeClient(model=model, **kwargs)
    if provider == "gemini":
        return GeminiClient(model=model, **kwargs)
    if provider == "openai-compatible":
        return OpenAICompatibleClient(model=model, **kwargs)
    if provider == "none":
        return NullLLMClient()
    raise ValueError(f"Unknown LLM provider: {provider}")


__all__ = [
    "LLMClient",
    "LLMMessage",
    "LLMResponse",
    "NullLLMClient",
    "OpenAICompatibleClient",
    "ClaudeClient",
    "FireworksClient",
    "GeminiClient",
    "OpenAIClient",
    "TracedLLMClient",
    "create_llm_client",
]
