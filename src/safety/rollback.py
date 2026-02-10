from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..types import NCCLConfig


@dataclass
class RollbackDecision:
    should_rollback: bool
    reason: str
    config: Optional[NCCLConfig] = None
    mode: str = "none"  # none | soft | hard
    changed_keys: List[str] = field(default_factory=list)


@dataclass
class RollbackPolicy:
    hard_failure_modes: List[str] = field(
        default_factory=lambda: ["hang", "crash", "rank_mismatch", "failure"]
    )
    soft_failure_modes: List[str] = field(default_factory=lambda: ["regression"])
    soft_keys: List[str] = field(
        default_factory=lambda: [
            "NCCL_ALGO",
            "NCCL_PROTO",
            "NCCL_MAX_NCHANNELS",
            "NCCL_NTHREADS",
            "NCCL_BUFFSIZE",
        ]
    )


class RollbackManager:
    def __init__(self, policy: Optional[RollbackPolicy] = None) -> None:
        self.policy = policy or RollbackPolicy()
        self.last_known_good: Optional[NCCLConfig] = None

    def update_success(self, config: NCCLConfig) -> None:
        self.last_known_good = config

    def rollback_hard(self, reason: str) -> RollbackDecision:
        if self.last_known_good is None:
            return RollbackDecision(False, reason, None, mode="hard")
        return RollbackDecision(True, reason, self.last_known_good, mode="hard")

    def rollback_soft(self, *, reason: str, current: NCCLConfig) -> RollbackDecision:
        if self.last_known_good is None:
            return RollbackDecision(False, reason, None, mode="soft")
        merged = dict(current.params)
        changed: List[str] = []
        for key in self.policy.soft_keys:
            if key not in self.last_known_good.params:
                continue
            last_value = self.last_known_good.params.get(key)
            if merged.get(key) != last_value:
                merged[key] = last_value
                changed.append(key)
        if not changed:
            return RollbackDecision(False, reason, None, mode="soft")
        return RollbackDecision(
            True,
            reason,
            NCCLConfig(params=merged, metadata={"source": "soft_rollback"}),
            mode="soft",
            changed_keys=changed,
        )

    def decide(
        self,
        *,
        failure_mode: str | None,
        reason: str,
        current: NCCLConfig,
    ) -> RollbackDecision:
        mode = (failure_mode or "").strip().lower()
        if mode in self.policy.hard_failure_modes:
            return self.rollback_hard(reason)
        if mode in self.policy.soft_failure_modes:
            return self.rollback_soft(reason=reason, current=current)
        return RollbackDecision(False, reason, None)
