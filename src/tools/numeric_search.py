from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass
from typing import Callable, List, Optional

from ..types import NCCLConfig, ParameterSpace, SearchCandidate, SearchResult
from ..utils import setup_logger


logger = setup_logger("cclagent.numeric_search")


@dataclass
class NumericSearchConfig:
    max_candidates: int = 8
    concurrency: int = 4


class NumericSearchTool:
    def __init__(self, config: Optional[NumericSearchConfig] = None, rng_seed: int = 11) -> None:
        self.config = config or NumericSearchConfig()
        self._rng = random.Random(rng_seed)

    async def search_async(
        self,
        base_config: NCCLConfig,
        focus_params: List[str],
        parameter_space: ParameterSpace,
        scorer: Callable[[NCCLConfig], float],
        budget: Optional[int] = None,
    ) -> SearchResult:
        target = budget or self.config.max_candidates
        semaphore = asyncio.Semaphore(self.config.concurrency)
        tasks = []
        for _ in range(max(1, target)):
            candidate, rationale = self._mutate_candidate(base_config, focus_params, parameter_space)
            tasks.append(
                asyncio.create_task(
                    self._score_candidate(candidate, rationale, scorer, semaphore)
                )
            )

        results = await asyncio.gather(*tasks)
        results.sort(key=lambda item: item.predicted_time_ms)
        best = results[0]
        return SearchResult(best=best, candidates=results)

    def search_sync(
        self,
        base_config: NCCLConfig,
        focus_params: List[str],
        parameter_space: ParameterSpace,
        scorer: Callable[[NCCLConfig], float],
        budget: Optional[int] = None,
    ) -> SearchResult:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            raise RuntimeError("search_sync cannot be called from a running event loop")
        return asyncio.run(self.search_async(base_config, focus_params, parameter_space, scorer, budget))

    async def _score_candidate(
        self,
        candidate: NCCLConfig,
        rationale: str,
        scorer: Callable[[NCCLConfig], float],
        semaphore: asyncio.Semaphore,
    ) -> SearchCandidate:
        async with semaphore:
            await asyncio.sleep(0)
            predicted = scorer(candidate)
            return SearchCandidate(config=candidate, predicted_time_ms=predicted, rationale=rationale)

    def _mutate_candidate(
        self,
        base_config: NCCLConfig,
        focus_params: List[str],
        parameter_space: ParameterSpace,
    ) -> tuple[NCCLConfig, str]:
        chooser = lambda params: self._rng.choice(params)
        mutated_params = parameter_space.mutate(
            base_config.params, focus_params=focus_params, chooser=chooser
        )
        mutated = NCCLConfig(params=mutated_params)
        rationale = self._build_rationale(base_config, mutated)
        return mutated, rationale

    def _build_rationale(self, base: NCCLConfig, mutated: NCCLConfig) -> str:
        changes = []
        for key, value in mutated.params.items():
            if base.params.get(key) != value:
                changes.append(f"{key}: {base.params.get(key)} -> {value}")
        if not changes:
            return "no-op mutation"
        return "mutated " + ", ".join(changes)
