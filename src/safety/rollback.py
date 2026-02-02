from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..types import NCCLConfig


@dataclass
class RollbackDecision:
    should_rollback: bool
    reason: str
    config: Optional[NCCLConfig] = None


class RollbackManager:
    def __init__(self) -> None:
        self.last_known_good: Optional[NCCLConfig] = None

    def update_success(self, config: NCCLConfig) -> None:
        self.last_known_good = config

    def rollback(self, reason: str) -> RollbackDecision:
        if self.last_known_good is None:
            return RollbackDecision(False, reason, None)
        return RollbackDecision(True, reason, self.last_known_good)
