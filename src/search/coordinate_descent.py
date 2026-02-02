from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from ..types import NCCLConfig, ParameterSpace, SearchCandidate, Subspace


@dataclass
class SearchState:
    current_subspace_idx: int = 0
    current_dim_idx: int = 0
    step_size: float = 1.0
    best_in_subspace: Optional[NCCLConfig] = None
    evaluated_hashes: Set[str] = field(default_factory=set)
    history: List[SearchCandidate] = field(default_factory=list)


class CoordinateDescentSearch:
    def __init__(self, parameter_space: ParameterSpace) -> None:
        self.parameter_space = parameter_space

    def propose_candidates(
        self,
        plan_subspaces: List[Subspace],
        state: SearchState,
        base_config: NCCLConfig,
        max_candidates: int = 4,
    ) -> List[NCCLConfig]:
        if not plan_subspaces:
            plan_subspaces = [Subspace(name="default", fixed={}, free=list(self.parameter_space.specs.keys()))]
        subspace = plan_subspaces[state.current_subspace_idx % len(plan_subspaces)]
        focus_params = subspace.free or list(self.parameter_space.specs.keys())
        dim_name = focus_params[state.current_dim_idx % len(focus_params)]

        base_params = dict(base_config.params)
        base_params.update(subspace.fixed)
        base = NCCLConfig(params=base_params)

        spec = self.parameter_space.specs.get(dim_name)
        if spec is None:
            return [base]
        current = base.params.get(dim_name, spec.default)
        neighbors = spec.neighbors(current, max_neighbors=max_candidates)
        candidates: List[NCCLConfig] = []
        for value in neighbors or [current]:
            params = dict(base.params)
            params[dim_name] = value
            candidates.append(NCCLConfig(params=params))
        return candidates

    def update_state(self, plan_subspaces: List[Subspace], state: SearchState, best: SearchCandidate) -> None:
        state.history.append(best)
        state.best_in_subspace = best.config
        if not plan_subspaces:
            state.current_dim_idx += 1
            return
        subspace = plan_subspaces[state.current_subspace_idx % len(plan_subspaces)]
        if subspace.free:
            state.current_dim_idx = (state.current_dim_idx + 1) % len(subspace.free)
            if state.current_dim_idx == 0:
                state.current_subspace_idx = (state.current_subspace_idx + 1) % len(plan_subspaces)
        else:
            state.current_subspace_idx = (state.current_subspace_idx + 1) % len(plan_subspaces)
