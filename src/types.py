from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional


@dataclass
class ParameterSpec:
    name: str
    kind: str  # "int", "float", "enum", "bool"
    description: str = ""
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    step: Optional[float] = None
    choices: Optional[List[Any]] = None
    default: Optional[Any] = None

    def normalize(self, value: Any) -> Any:
        if self.kind == "bool":
            return bool(value)
        if self.kind == "int":
            return int(value)
        if self.kind == "float":
            return float(value)
        if self.kind == "enum":
            return value
        return value

    def is_valid(self, value: Any) -> bool:
        try:
            value = self.normalize(value)
        except (TypeError, ValueError):
            return False
        if self.kind == "enum":
            return self.choices is None or value in self.choices
        if self.kind in ("int", "float"):
            if self.min_value is not None and value < self.min_value:
                return False
            if self.max_value is not None and value > self.max_value:
                return False
        return True

    def neighbors(self, value: Any, max_neighbors: int = 3) -> List[Any]:
        if self.kind == "enum":
            choices = self.choices or []
            return [choice for choice in choices if choice != value][:max_neighbors]
        if self.kind == "bool":
            return [not bool(value)]
        if self.kind in ("int", "float"):
            if self.step is None:
                return []
            neighbors = []
            for delta in (-self.step, self.step):
                candidate = self.normalize(value + delta)
                if self.is_valid(candidate):
                    neighbors.append(candidate)
                if len(neighbors) >= max_neighbors:
                    break
            return neighbors
        return []


@dataclass
class ParameterSpace:
    specs: Dict[str, ParameterSpec] = field(default_factory=dict)

    @classmethod
    def from_list(cls, specs: Iterable[ParameterSpec]) -> "ParameterSpace":
        return cls({spec.name: spec for spec in specs})

    def default_config(self) -> Dict[str, Any]:
        config: Dict[str, Any] = {}
        for name, spec in self.specs.items():
            if spec.default is not None:
                config[name] = spec.default
            elif spec.kind == "enum" and spec.choices:
                config[name] = spec.choices[0]
            elif spec.kind == "bool":
                config[name] = False
            elif spec.kind in ("int", "float"):
                if spec.min_value is not None:
                    config[name] = spec.min_value
                else:
                    config[name] = 0
        return config

    def validate(self, config: Dict[str, Any]) -> List[str]:
        errors = []
        for name, value in config.items():
            spec = self.specs.get(name)
            if spec is None:
                errors.append(f"Unknown parameter: {name}")
                continue
            if not spec.is_valid(value):
                errors.append(f"Invalid value for {name}: {value}")
        return errors

    def mutate(
        self,
        config: Dict[str, Any],
        focus_params: Optional[List[str]] = None,
        chooser: Optional[Callable[[List[str]], str]] = None,
    ) -> Dict[str, Any]:
        if not config:
            config = self.default_config()
        candidates = [name for name in (focus_params or list(self.specs.keys())) if name in self.specs]
        if not candidates:
            return config
        param_name = chooser(candidates) if chooser else candidates[0]
        spec = self.specs.get(param_name)
        if spec is None:
            return config
        current_value = config.get(param_name, spec.default)
        neighbors = spec.neighbors(current_value)
        if not neighbors:
            return config
        mutated = dict(config)
        mutated[param_name] = neighbors[0]
        return mutated


@dataclass
class NCCLConfig:
    params: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MicrobenchResult:
    important_params: List[str] = field(default_factory=list)
    signals: Dict[str, float] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Metrics:
    iteration_time: float
    comm_time: Optional[float] = None
    bandwidth: Optional[float] = None
    errors: int = 0
    timestamp: Optional[float] = None
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkloadSpec:
    name: str
    command: List[str]
    nodes: int = 1
    topology: str = "unknown"
    scale: str = "unknown"
    env: Dict[str, str] = field(default_factory=dict)
    kind: str = "workload"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ContextSignature:
    workload: str
    topology: str
    scale: str
    nodes: int
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TuningAction:
    kind: str
    config: NCCLConfig
    rationale: str


@dataclass
class TuningRecord:
    action: TuningAction
    metrics: Metrics


@dataclass
class SearchCandidate:
    config: NCCLConfig
    predicted_time: float
    rationale: str


@dataclass
class SearchResult:
    best: SearchCandidate
    candidates: List[SearchCandidate]


@dataclass
class TuningBudget:
    max_steps: int = 10
    min_improvement: float = 0.01
    patience: int = 3
    hypothesis_every: int = 2


@dataclass
class AgentConfig:
    parameter_space: ParameterSpace
    budget: TuningBudget
    memory_path: str
    rag_docs_path: Optional[str] = None
    rag_top_k: int = 5
    sla_max_iteration_time: Optional[float] = None


@dataclass
class ToolRegistry:
    microbench: Any
    workload: Any
    metrics: Any
    sla: Any
    compiler: Any
    nccl: Any
    nccltest: Any | None = None
    training: Any | None = None
    autoccl: Any | None = None
    ext_tuner: Any | None = None
    ext_net: Any | None = None
    numeric_search: Any | None = None
