from __future__ import annotations

import time
import uuid
from dataclasses import asdict, is_dataclass
from typing import Any, Callable, Dict, Optional

from ..trace import TraceEmitter


class ToolProxy:
    def __init__(self, name: str, tool: Any, trace: TraceEmitter, run_id: str) -> None:
        self._name = name
        self._tool = tool
        self._trace = trace
        self._run_id = run_id

    def __getattr__(self, attr: str):
        target = getattr(self._tool, attr)
        if not callable(target):
            return target

        def wrapper(*args, **kwargs):
            call_id = str(uuid.uuid4())
            step = _infer_step(kwargs, args)
            payload = {
                "tool": self._name,
                "method": attr,
                "call_id": call_id,
                "args": _summarize_args(args),
                "kwargs": _summarize_kwargs(kwargs),
            }
            self._trace.event(
                run_id=self._run_id,
                phase=_infer_phase(self._name),
                step=step,
                actor="tool",
                type="tool.call",
                payload=payload,
                refs=[f"tool:{step}:{call_id}" if step is not None else f"tool:{call_id}"],
            )
            start = time.time()
            try:
                result = target(*args, **kwargs)
                duration_ms = (time.time() - start) * 1000.0
                self._trace.event(
                    run_id=self._run_id,
                    phase=_infer_phase(self._name),
                    step=step,
                    actor="tool",
                    type="tool.result",
                    payload={
                        "tool": self._name,
                        "method": attr,
                        "call_id": call_id,
                        "result": _summarize_result(result),
                    },
                    refs=[f"tool:{step}:{call_id}" if step is not None else f"tool:{call_id}"],
                    duration_ms=duration_ms,
                )
                return result
            except Exception as exc:  # pragma: no cover - surface to caller
                duration_ms = (time.time() - start) * 1000.0
                self._trace.event(
                    run_id=self._run_id,
                    phase=_infer_phase(self._name),
                    step=step,
                    actor="tool",
                    type="tool.result",
                    payload={
                        "tool": self._name,
                        "method": attr,
                        "call_id": call_id,
                        "error": str(exc),
                    },
                    refs=[f"tool:{step}:{call_id}" if step is not None else f"tool:{call_id}"],
                    status="error",
                    duration_ms=duration_ms,
                )
                raise

        return wrapper


class InstrumentedToolSuite:
    def __init__(self, tools: Any, trace: TraceEmitter, run_id: str) -> None:
        self._tools = tools
        self._trace = trace
        self._run_id = run_id

    def __getattr__(self, item: str):
        value = getattr(self._tools, item)
        if item == "run_context":
            return value
        if value is None:
            return None
        # wrap tool objects
        return ToolProxy(item, value, self._trace, self._run_id)


def _infer_step(kwargs: Dict[str, Any], args: tuple) -> Optional[int]:
    if "step" in kwargs and isinstance(kwargs.get("step"), int):
        return kwargs.get("step")
    for arg in args:
        if isinstance(arg, int):
            return arg
    return None


def _infer_phase(tool_name: str) -> str:
    if tool_name in ("microbench",):
        return "offline"
    if tool_name in (
        "training",
        "workload",
        "metrics",
        "sla",
        "compiler",
        "nccl",
        "numeric_search",
        "nccl_debug",
        "profiler",
        "debug_playbook",
    ):
        return "online"
    return "system"


def _summarize_args(args: tuple) -> list:
    return [_summarize_value(arg) for arg in args]


def _summarize_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    return {k: _summarize_value(v) for k, v in kwargs.items()}


def _summarize_result(result: Any) -> Any:
    return _summarize_value(result)


def _summarize_value(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        out = {}
        for key, val in value.items():
            out[str(key)] = _summarize_value(val)
        return out
    if isinstance(value, (list, tuple)):
        return [_summarize_value(item) for item in value]
    return str(value)
