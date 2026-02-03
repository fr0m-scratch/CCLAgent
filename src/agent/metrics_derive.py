from __future__ import annotations

from typing import Dict

from ..types import Metrics


def derive_metrics(metrics: Metrics) -> Dict[str, float]:
    derived: Dict[str, float] = {}
    if metrics.comm_time_ms is not None and metrics.iteration_time_ms > 0:
        derived["comm_fraction"] = metrics.comm_time_ms / metrics.iteration_time_ms
    if metrics.throughput is not None:
        derived["throughput"] = metrics.throughput
    if metrics.algbw_gbps is not None:
        derived["algbw_gbps"] = metrics.algbw_gbps
    if metrics.busbw_gbps is not None:
        derived["busbw_gbps"] = metrics.busbw_gbps
    if isinstance(metrics.raw, dict):
        if "iter_std_ms" in metrics.raw:
            derived["iter_std_ms"] = float(metrics.raw["iter_std_ms"])
        if "iteration_count" in metrics.raw:
            derived["iteration_count"] = float(metrics.raw["iteration_count"])
        if "simulated_total_ms" in metrics.raw:
            derived["simulated_total_ms"] = float(metrics.raw["simulated_total_ms"])
    return derived
