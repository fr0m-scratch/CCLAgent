from __future__ import annotations

import os

from .base import OpenAICompatibleClient


class FireworksClient(OpenAICompatibleClient):
    def __init__(self, model: str, api_key: str | None = None, base_url: str | None = None):
        super().__init__(
            model=model,
            api_key=api_key or os.getenv("FIREWORKS_API_KEY"),
            base_url=base_url or os.getenv("FIREWORKS_BASE_URL"),
        )
