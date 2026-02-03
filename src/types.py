from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence


METRICS_SCHEMA_VERSION = "1.0"


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
            try:
                current = self.normalize(value)
            except (TypeError, ValueError):
                fallback = self.default
                if fallback is None:
                    fallback = self.min_value if self.min_value is not None else 0
                try:
                    current = self.normalize(fallback)
                except (TypeError, ValueError):
                    return []
            neighbors = []
            for delta in (-self.step, self.step):
                try:
                    candidate = self.normalize(current + delta)
                except (TypeError, ValueError):
                    continue
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
class MicrobenchSignal:
    name: str
    value: float | int | str
    unit: Optional[str] = None
    confidence: float = 0.5
    source: str = "unknown"


@dataclass
class ImportantParam:
    param: str
    importance: float
    reason: str = ""
    evidence: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MicrobenchResult:
    important_params: List[ImportantParam] = field(default_factory=list)
    signals: List[MicrobenchSignal] = field(default_factory=list)
    raw_path: str = ""
    command: List[str] = field(default_factory=list)
    runtime_sec: float = 0.0
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Metrics:
    iteration_time_ms: float
    throughput: Optional[float] = None
    comm_time_ms: Optional[float] = None
    busbw_gbps: Optional[float] = None
    algbw_gbps: Optional[float] = None
    loss: Optional[float] = None
    error_budget: Optional[float] = None
    success: bool = True
    failure_reason: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = METRICS_SCHEMA_VERSION


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
    launcher: str = "local"  # local, torchrun, slurm, mpirun
    launcher_args: Dict[str, Any] = field(default_factory=dict)
    gpus_per_node: Optional[int] = None
    eval_mode: str = "full"  # full, short
    eval_steps: Optional[int] = None
    eval_timeout_sec: Optional[int] = None


@dataclass
class ContextSignature:
    workload: str
    workload_kind: str
    topology: str
    scale: str
    nodes: int
    model: Optional[str] = None
    framework: Optional[str] = None
    gpus_per_node: Optional[int] = None
    gpu_type: Optional[str] = None
    network: Optional[str] = None
    nic_count: Optional[int] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RunContext:
    run_id: str
    started_at_iso: str
    artifacts_dir: str
    dry_run: bool
    seed: int
    git_commit: Optional[str]
    host_info: Dict[str, Any]
    config_snapshot_path: Optional[str] = None
    schema_version: str = "1.0"

    @property
    def started_at(self) -> str:
        return self.started_at_iso

@dataclass
class RAGChunk:
    doc_id: str
    chunk_id: str
    text: str
    score: float = 0.0
    meta: Dict[str, Any] = field(default_factory=dict)



@dataclass
class Subspace:
    name: str
    fixed: Dict[str, Any] = field(default_factory=dict)
    free: List[str] = field(default_factory=list)


@dataclass
class InitialConfigPlan:
    baseline_config: NCCLConfig
    constraints: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    important_params: List[ImportantParam] = field(default_factory=list)
    candidate_subspaces: List[Subspace] = field(default_factory=list)
    recommended_search_params: List[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class Hypothesis:
    id: str
    summary: str
    patch: Dict[str, Any]
    mechanism: Optional[str] = None
    expected_effect: Dict[str, str] = field(default_factory=dict)
    risk: str = "low"
    evidence: Dict[str, Any] = field(default_factory=dict)
    test_plan: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CompiledConfig:
    config: NCCLConfig
    env: Dict[str, str]
    warnings: List[str] = field(default_factory=list)
    risk_score: float = 0.0
    risk_reasons: List[str] = field(default_factory=list)


@dataclass
class TuningAction:
    kind: str
    config: NCCLConfig
    rationale: str
    hypothesis: Optional[Hypothesis] = None
    compiled: Optional[CompiledConfig] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HypothesisAction(TuningAction):
    pass


@dataclass
class NumericSearchAction(TuningAction):
    search: Optional['SearchResult'] = None


@dataclass
class StopAction:
    kind: str
    reason: str


@dataclass
class RollbackAction:
    kind: str
    reason: str
    config: NCCLConfig


@dataclass
class TuningRecord:
    step: int
    action: TuningAction
    metrics: Metrics
    decision: Dict[str, Any] = field(default_factory=dict)
    rule_ids: List[str] = field(default_factory=list)
    microbench_snapshot: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchCandidate:
    config: NCCLConfig
    predicted_time_ms: float
    rationale: str
    evaluation_mode: str = "predict_only"
    uncertainty: float = 0.0


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
    plateau_eps: float = 0.003
    plateau_window: int = 5
    target_gain: Optional[float] = None
    stable_steps: int = 2


@dataclass
class MicrobenchSettings:
    mode: str = "dry"  # dry, cclinsight, nccltests
    command_template: List[str] = field(default_factory=list)
    parse_schema: str = "cclinsight_v1"
    timeout_sec: int = 900
    env: Dict[str, str] = field(default_factory=dict)
    collect_topology: bool = False
    repetitions: int = 1
    allow_fallback: bool = True


@dataclass
class RagConfig:
    mode: str = "jaccard"  # jaccard, embeddings
    top_k: int = 5
    rebuild_index: bool = False
    index_path: str = "rag_index"
    docs_paths: List[str] = field(default_factory=list)
    allow_fallback: bool = True


@dataclass
class MemoryConfig:
    path: str = "memory/agent_memory.json"
    top_k: int = 5
    half_life_days: float = 30.0
    allow_negative_rules: bool = True


@dataclass
class MetricsConfig:
    parse_mode: str = "json_stdout_v1"
    allow_missing_metrics: bool = False


@dataclass
class NumericSearchSettings:
    mode: str = "predict_only"  # predict_only, real_eval
    max_candidates: int = 8
    concurrency: int = 4
    short_eval_steps: int = 50
    short_eval_timeout_sec: int = 300
    allow_fallback: bool = True


@dataclass
class SafetyConfig:
    max_channels_safe: int = 32
    min_buffsize_safe: int = 1 << 20
    max_risk_score: float = 0.7
    safe_envelope: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    known_bad_combos_path: Optional[str] = None
    risk_threshold: float = 0.7
    allow_fallback: bool = True


@dataclass
class ExecutionConfig:
    mode: str = "restart_per_step"  # restart_per_step, in_job_ext_tuner
    allow_fallback: bool = True


@dataclass
class SurrogateConfig:
    model_type: str = "rf"
    refit_every_steps: int = 5
    retrain_every: int = 5
    min_records: int = 8
    dataset_path: str = "memory/datasets"
    model_dir: str = "memory/models"


@dataclass
class AgentConfig:
    parameter_space: ParameterSpace
    budget: TuningBudget
    memory: MemoryConfig
    rag: RagConfig
    microbench: MicrobenchSettings
    metrics: MetricsConfig
    numeric_search: NumericSearchSettings
    safety: SafetyConfig
    execution: ExecutionConfig
    surrogate: SurrogateConfig
    artifacts_root: str = "artifacts"
    seed: int = 7
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
    run_context: Optional[RunContext] = None
