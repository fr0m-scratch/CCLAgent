from __future__ import annotations

from typing import Dict, Tuple


def classify_bottleneck(derived: Dict[str, float]) -> Tuple[str, float]:
    comm_fraction = derived.get("comm_fraction")
    if comm_fraction is None:
        return "unknown", 0.2
    if comm_fraction > 0.6:
        return "bandwidth_bound", 0.7
    if comm_fraction < 0.3:
        return "compute_bound", 0.6
    return "latency_bound", 0.5
