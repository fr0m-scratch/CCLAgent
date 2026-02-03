from __future__ import annotations

import json
from typing import Any, Dict, Iterable, Iterator, Optional


def read_events(path: str) -> Iterator[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        return


def filter_events(
    events: Iterable[Dict[str, Any]],
    *,
    step: Optional[int] = None,
    phase: Optional[str] = None,
    type: Optional[str] = None,
) -> Iterator[Dict[str, Any]]:
    for event in events:
        if step is not None and event.get("step") != step:
            continue
        if phase is not None and event.get("phase") != phase:
            continue
        if type is not None and event.get("type") != type:
            continue
        yield event
