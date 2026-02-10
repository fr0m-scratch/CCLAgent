from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List


@dataclass
class DebugAction:
    name: str
    description: str
    env_delta: Dict[str, str] = field(default_factory=dict)
    expected_signal: str = ""


class DebugPlaybook:
    """Deterministic debug ladder inspired by uCCL workflow."""

    def plan(self, failure_mode: str, *, bottleneck: str = "unknown") -> List[DebugAction]:
        mode = (failure_mode or "").lower().strip()
        if mode == "hang":
            return [
                DebugAction(
                    name="enable_blocking_wait",
                    description="Force blocking wait to surface stalled ranks deterministically.",
                    env_delta={"NCCL_BLOCKING_WAIT": "1", "NCCL_ASYNC_ERROR_HANDLING": "1"},
                    expected_signal="hang_or_watchdog_logs",
                ),
                DebugAction(
                    name="transport_isolation",
                    description="Disable IB to isolate transport path issues.",
                    env_delta={"NCCL_IB_DISABLE": "1", "NCCL_P2P_DISABLE": "0"},
                    expected_signal="transport_dependency_change",
                ),
            ]
        if mode == "rank_mismatch":
            return [
                DebugAction(
                    name="strict_collective_checks",
                    description="Enable extra debug checks to pinpoint rank divergence.",
                    env_delta={"NCCL_DEBUG": "INFO", "NCCL_DEBUG_SUBSYS": "INIT,COLL"},
                    expected_signal="mismatch_location",
                )
            ]
        if mode == "crash":
            return [
                DebugAction(
                    name="safe_algo_proto",
                    description="Fallback to conservative algorithm/protocol to prevent unsafe fast path.",
                    env_delta={"NCCL_ALGO": "RING", "NCCL_PROTO": "SIMPLE"},
                    expected_signal="crash_recovery",
                )
            ]

        if bottleneck == "bandwidth_bound":
            return [
                DebugAction(
                    name="force_ring",
                    description="A/B test RING for bandwidth-bound collectives.",
                    env_delta={"NCCL_ALGO": "RING"},
                    expected_signal="algbw_improvement",
                )
            ]
        if bottleneck == "latency_bound":
            return [
                DebugAction(
                    name="force_tree_ll",
                    description="A/B test TREE+LL for latency-bound collectives.",
                    env_delta={"NCCL_ALGO": "TREE", "NCCL_PROTO": "LL"},
                    expected_signal="iteration_time_reduction",
                )
            ]
        return []

    def merge_env(self, base_env: Dict[str, str], actions: List[DebugAction]) -> Dict[str, str]:
        merged = dict(base_env)
        for action in actions:
            merged.update(action.env_delta)
        return merged

    def serialize(self, actions: List[DebugAction]) -> List[Dict[str, object]]:
        return [asdict(action) for action in actions]
