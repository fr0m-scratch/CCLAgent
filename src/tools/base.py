from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Protocol, runtime_checkable


@dataclass
class ToolResult:
    ok: bool
    data: Dict[str, Any] | None = None
    error: str | None = None


class ToolExecutionError(RuntimeError):
    pass


class Tool:
    name = "tool"

    def run(self, *args: Any, **kwargs: Any) -> ToolResult:
        raise NotImplementedError


@runtime_checkable
class ToolInterface(Protocol):
    name: str

    def run(self, *args: Any, **kwargs: Any) -> ToolResult: ...


@runtime_checkable
class AsyncToolInterface(Protocol):
    name: str

    async def run_async(self, *args: Any, **kwargs: Any) -> ToolResult: ...
