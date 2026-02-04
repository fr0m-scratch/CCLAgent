from __future__ import annotations

import os

from .base import OpenAICompatibleClient

DEFAULT_FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"
DEFAULT_FIREWORKS_HEADERS = {
    "User-Agent": "CCLAgent/0.1",
    "Accept": "application/json",
}


class FireworksClient(OpenAICompatibleClient):
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        extra_headers: dict[str, str] | None = None,
    ):
        headers = dict(DEFAULT_FIREWORKS_HEADERS)
        if extra_headers:
            headers.update(extra_headers)
        super().__init__(
            model=model,
            api_key=api_key or os.getenv("FIREWORKS_API_KEY"),
            base_url=base_url or os.getenv("FIREWORKS_BASE_URL") or DEFAULT_FIREWORKS_BASE_URL,
            api_path="/chat/completions",
            extra_headers=headers,
        )
