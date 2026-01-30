import os

from .base import OpenAICompatibleClient


class OpenAIClient(OpenAICompatibleClient):
    def __init__(self, model: str, api_key: str | None = None, base_url: str | None = None):
        super().__init__(
            model=model,
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=base_url or os.getenv("OPENAI_BASE_URL"),
        )
