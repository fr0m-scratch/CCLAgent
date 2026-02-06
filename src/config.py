from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List

from .types import (
    AgentConfig,
    ExecutionConfig,
    LLMSettings,
    MemoryConfig,
    MetricsConfig,
    MicrobenchSettings,
    NumericSearchSettings,
    ParameterSpace,
    ParameterSpec,
    RagConfig,
    SafetyConfig,
    SurrogateConfig,
    TuningBudget,
    WarmStartSettings,
    WorkloadSpec,
)
from .utils import read_json


DEFAULT_PARAMETER_SPECS: List[ParameterSpec] = [
    ParameterSpec(
        name="NCCL_ALGO",
        kind="enum",
        choices=["TREE", "RING", "COLLNET"],
        default="RING",
        description="Collective algorithm selection.",
    ),
    ParameterSpec(
        name="NCCL_PROTO",
        kind="enum",
        choices=["LL", "LL128", "SIMPLE"],
        default="SIMPLE",
        description="Protocol selection for collectives.",
    ),
    ParameterSpec(
        name="NCCL_NTHREADS",
        kind="int",
        min_value=64,
        max_value=512,
        step=64,
        default=256,
        description="Threads per channel.",
    ),
    ParameterSpec(
        name="NCCL_BUFFSIZE",
        kind="int",
        min_value=1 << 20,
        max_value=1 << 26,
        step=1 << 20,
        default=1 << 22,
        description="Buffer size in bytes.",
    ),
    ParameterSpec(
        name="NCCL_MIN_NCHANNELS",
        kind="int",
        min_value=1,
        max_value=16,
        step=1,
        default=4,
        description="Minimum channels for collectives.",
    ),
    ParameterSpec(
        name="NCCL_MAX_NCHANNELS",
        kind="int",
        min_value=1,
        max_value=32,
        step=1,
        default=8,
        description="Maximum channels for collectives.",
    ),
    ParameterSpec(
        name="NCCL_P2P_LEVEL",
        kind="enum",
        choices=["NVL", "PXB", "SYS"],
        default="SYS",
        description="P2P level selection.",
    ),
    ParameterSpec(
        name="NCCL_NET_GDR_LEVEL",
        kind="int",
        min_value=0,
        max_value=2,
        step=1,
        default=1,
        description="GPUDirect RDMA level.",
    ),
    ParameterSpec(
        name="NCCL_SOCKET_NTHREADS",
        kind="int",
        min_value=1,
        max_value=8,
        step=1,
        default=2,
        description="Socket threads.",
    ),
    ParameterSpec(
        name="NCCL_NSOCKS_PERTHREAD",
        kind="int",
        min_value=1,
        max_value=8,
        step=1,
        default=2,
        description="Sockets per thread.",
    ),
    ParameterSpec(
        name="NCCL_IB_QPS_PER_CONNECTION",
        kind="int",
        min_value=1,
        max_value=8,
        step=1,
        default=1,
        description="QPs per IB connection.",
    ),
    ParameterSpec(
        name="NCCL_SHM_DISABLE",
        kind="bool",
        default=False,
        description="Disable shared memory transport.",
    ),
]


def _default_rag_paths() -> List[str]:
    return ["doc/Design", "doc/Knowledge", "README", "workload"]


def default_agent_config(memory_path: str = "memory/agent_memory.json") -> AgentConfig:
    parameter_space = ParameterSpace.from_list(DEFAULT_PARAMETER_SPECS)
    budget = TuningBudget()
    rag = RagConfig(docs_paths=_default_rag_paths())
    microbench = MicrobenchSettings()
    metrics = MetricsConfig()
    numeric_search = NumericSearchSettings()
    safety = SafetyConfig()
    execution = ExecutionConfig()
    surrogate = SurrogateConfig()
    llm = LLMSettings()
    warm_start = WarmStartSettings()
    memory = MemoryConfig(path=memory_path)
    return AgentConfig(
        parameter_space=parameter_space,
        budget=budget,
        memory=memory,
        rag=rag,
        llm=llm,
        warm_start=warm_start,
        microbench=microbench,
        metrics=metrics,
        numeric_search=numeric_search,
        safety=safety,
        execution=execution,
        surrogate=surrogate,
        artifacts_root="artifacts",
        seed=7,
        sla_max_iteration_time=None,
    )


def _merge_dataclass(default_obj, payload: Dict[str, Any]):
    for key, value in payload.items():
        if hasattr(default_obj, key):
            setattr(default_obj, key, value)
    return default_obj


def load_agent_config(path: str) -> AgentConfig:
    payload = read_json(path)
    param_specs = []
    for spec in payload.get("parameters", []):
        param_specs.append(ParameterSpec(**spec))

    if not param_specs:
        param_specs = DEFAULT_PARAMETER_SPECS

    budget_payload = payload.get("budget", {})
    budget = TuningBudget(**budget_payload) if budget_payload else TuningBudget()

    parameter_space = ParameterSpace.from_list(param_specs)

    rag = _merge_dataclass(RagConfig(docs_paths=_default_rag_paths()), payload.get("rag", {}))
    if not rag.docs_paths:
        rag.docs_paths = _default_rag_paths()
    if payload.get("rag_docs_path") and rag.docs_paths == _default_rag_paths():
        rag.docs_paths = [payload["rag_docs_path"]]
    if payload.get("rag_top_k"):
        rag.top_k = payload.get("rag_top_k")

    microbench = _merge_dataclass(MicrobenchSettings(), payload.get("microbench", {}))
    metrics = _merge_dataclass(MetricsConfig(), payload.get("metrics", {}))
    numeric_search = _merge_dataclass(NumericSearchSettings(), payload.get("numeric_search", {}))
    safety = _merge_dataclass(SafetyConfig(), payload.get("safety", {}))
    execution = _merge_dataclass(ExecutionConfig(), payload.get("execution", {}))
    surrogate = _merge_dataclass(SurrogateConfig(), payload.get("surrogate", {}))
    llm = _merge_dataclass(LLMSettings(), payload.get("llm", {}))
    warm_start = _merge_dataclass(WarmStartSettings(), payload.get("warm_start", {}))

    memory_payload = payload.get("memory", {})
    memory = _merge_dataclass(MemoryConfig(path="memory/agent_memory.json"), memory_payload)
    if payload.get("memory_path"):
        memory.path = payload.get("memory_path")

    return AgentConfig(
        parameter_space=parameter_space,
        budget=budget,
        memory=memory,
        rag=rag,
        llm=llm,
        warm_start=warm_start,
        microbench=microbench,
        metrics=metrics,
        numeric_search=numeric_search,
        safety=safety,
        execution=execution,
        surrogate=surrogate,
        artifacts_root=payload.get("artifacts_root", "artifacts"),
        seed=payload.get("seed", 7),
        sla_max_iteration_time=payload.get("sla_max_iteration_time"),
    )


def load_workload_spec(path: str) -> WorkloadSpec:
    payload = read_json(path)
    return WorkloadSpec(
        name=payload.get("name", "workload"),
        command=payload.get("command", []),
        nodes=payload.get("nodes", 1),
        topology=payload.get("topology", "unknown"),
        scale=payload.get("scale", "unknown"),
        env=payload.get("env", {}),
        kind=payload.get("kind", "workload"),
        metadata=payload.get("metadata", {}),
        launcher=payload.get("launcher", "local"),
        launcher_args=payload.get("launcher_args", {}),
        gpus_per_node=payload.get("gpus_per_node"),
        eval_mode=payload.get("eval_mode", "full"),
        eval_steps=payload.get("eval_steps"),
        eval_timeout_sec=payload.get("eval_timeout_sec"),
    )


def config_to_dict(config: AgentConfig) -> Dict[str, Any]:
    payload = asdict(config)
    payload["parameter_space"] = {
        name: asdict(spec) for name, spec in config.parameter_space.specs.items()
    }
    return payload
