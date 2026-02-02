from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from ..memory import MemoryStore
from ..types import Hypothesis, InitialConfigPlan, NCCLConfig
from ..utils import setup_logger


logger = setup_logger("cclagent.hypothesis")


class HypothesisGenerator:
    def __init__(self, memory: MemoryStore, parameter_space: Any):
        self.memory = memory
        self.parameter_space = parameter_space

    def propose(
        self,
        plan: InitialConfigPlan,
        context,
        base_config: NCCLConfig,
        last_metrics: Any,
    ) -> Hypothesis:
        rules = self.memory.retrieve_rules(context, top_k=1)
        if rules:
            rule = rules[0]
            return Hypothesis(
                id=rule.id,
                summary="apply memory rule",
                patch=rule.config_patch,
                expected_effect={"iteration_time_ms": "decrease"},
                risk="low",
                evidence={"rule_id": rule.id, "improvement": rule.improvement},
            )

        important = [ip.param for ip in plan.important_params] or plan.recommended_search_params
        patch = self._mutate_single(base_config, important)
        return Hypothesis(
            id=str(uuid.uuid4()),
            summary="single-parameter mutation",
            patch=patch,
            expected_effect={"iteration_time_ms": "decrease"},
            risk="low",
            evidence={"source": "heuristic"},
        )

    def _mutate_single(self, base_config: NCCLConfig, focus: List[str]) -> Dict[str, Any]:
        if not focus:
            return {}
        param_name = focus[0]
        spec = self.parameter_space.specs.get(param_name)
        if spec is None:
            return {}
        current = base_config.params.get(param_name, spec.default)
        neighbors = spec.neighbors(current)
        if not neighbors:
            return {}
        return {param_name: neighbors[0]}
