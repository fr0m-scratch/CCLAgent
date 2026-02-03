from __future__ import annotations

from pathlib import Path
from typing import Dict, Any


def load_param_semantics(path: str | None = None) -> Dict[str, Dict[str, Any]]:
    if path is None:
        path = str(Path(__file__).with_name("nccl_params.yaml"))
    try:
        return _load_simple_yaml(path)
    except Exception:
        return {}


def _load_simple_yaml(path: str) -> Dict[str, Dict[str, Any]]:
    data: Dict[str, Dict[str, Any]] = {}
    current_key: str | None = None
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip("\n")
            if not raw.strip():
                continue
            if not raw.startswith(" ") and raw.endswith(":"):
                current_key = raw[:-1].strip()
                data[current_key] = {}
                continue
            if current_key is None:
                continue
            if raw.startswith("  ") and ":" in raw:
                key, value = raw.strip().split(":", 1)
                value = value.strip().strip('"')
                data[current_key][key] = value
    return data


def get_param_semantics(param: str, path: str | None = None) -> Dict[str, Any]:
    data = load_param_semantics(path)
    return data.get(param, {})
