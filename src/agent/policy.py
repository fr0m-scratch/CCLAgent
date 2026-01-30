from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

from ..memory import MemoryStore
from ..types import ContextSignature, MicrobenchResult, NCCLConfig, ParameterSpace, SearchResult, TuningAction
from ..utils import setup_logger
from .state import SurrogateModel, TuningState


logger = setup_logger("cclagent.policy")


class DecisionPolicy:
    def __init__(
        self,
        config: Any,
        memory: MemoryStore,
        parameter_space: ParameterSpace,
        surrogate: SurrogateModel,
        search_tool: Any | None = None,
        rng_seed: int = 7,
    ) -> None:
        self.config = config
        self.memory = memory
        self.parameter_space = parameter_space
        self.surrogate = surrogate
        self.search_tool = search_tool
        self._rng = random.Random(rng_seed)

    def decide_next_action(
        self,
        state: TuningState,
        microbench: MicrobenchResult,
        context: ContextSignature,
        step: int,
    ) -> Optional[TuningAction]:
        if step >= self.config.budget.max_steps - 1:
            return None
        use_hypothesis = (step % self.config.budget.hypothesis_every) == 0
        if use_hypothesis:
            config = self._hypothesis_step(state, microbench, context)
            rationale = "hypothesis-driven adjustment"
            return TuningAction(kind="hypothesis", config=config, rationale=rationale)
        config = self._numeric_step(state, microbench)
        return TuningAction(kind="numeric", config=config, rationale="numeric search")

    def _hypothesis_step(
        self,
        state: TuningState,
        microbench: MicrobenchResult,
        context: ContextSignature,
    ) -> NCCLConfig:
        rules = self.memory.get_rules(context)
        if rules:
            patch = rules[0].config_patch
            new_params = dict(state.best_record.action.config.params)
            new_params.update(patch)
            return NCCLConfig(params=new_params, metadata={"source": "rule"})

        focus = microbench.important_params or list(self.parameter_space.specs.keys())
        new_params = self._mutate_best(state, focus)
        return NCCLConfig(params=new_params, metadata={"source": "heuristic"})

    def _numeric_step(self, state: TuningState, microbench: MicrobenchResult) -> NCCLConfig:
        focus = microbench.important_params or list(self.parameter_space.specs.keys())
        best_config = state.best_record.action.config if state.best_record else NCCLConfig()

        suggest = getattr(self.surrogate, "suggest", None)
        if callable(suggest):
            candidate = suggest(
                base_config=best_config,
                focus_params=focus,
                parameter_space=self.parameter_space,
                budget=self.config.budget.max_steps,
            )
            if candidate is not None:
                return candidate

        if self.search_tool:
            try:
                result = self.search_tool.search_sync(
                    base_config=best_config,
                    focus_params=focus,
                    parameter_space=self.parameter_space,
                    scorer=self.surrogate.predict,
                    budget=self.config.budget.max_steps,
                )
                if isinstance(result, SearchResult):
                    return result.best.config
            except RuntimeError:
                logger.warning("Async search requested inside running event loop; falling back.")

        candidates = []
        for _ in range(3):
            mutated = self._mutate_best(state, focus)
            candidate = NCCLConfig(params=mutated)
            score = self.surrogate.predict(candidate, default=1.0)
            candidates.append((score, candidate))
        candidates.sort(key=lambda item: item[0])
        return candidates[0][1]

    def _mutate_best(self, state: TuningState, focus: List[str]) -> Dict[str, Any]:
        best_config = state.best_record.action.config if state.best_record else NCCLConfig()
        chooser = lambda params: self._rng.choice(params)
        return self.parameter_space.mutate(best_config.params, focus_params=focus, chooser=chooser)
