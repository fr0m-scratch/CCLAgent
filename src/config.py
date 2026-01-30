from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List

from .types import AgentConfig, ParameterSpace, ParameterSpec, TuningBudget, WorkloadSpec
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
]


def default_agent_config(memory_path: str = "memory/agent_memory.json") -> AgentConfig:
    parameter_space = ParameterSpace.from_list(DEFAULT_PARAMETER_SPECS)
    budget = TuningBudget()
    return AgentConfig(
        parameter_space=parameter_space,
        budget=budget,
        memory_path=memory_path,
        rag_docs_path="doc/Design",
        rag_top_k=5,
        sla_max_iteration_time=None,
    )


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

    return AgentConfig(
        parameter_space=parameter_space,
        budget=budget,
        memory_path=payload.get("memory_path", "memory/agent_memory.json"),
        rag_docs_path=payload.get("rag_docs_path"),
        rag_top_k=payload.get("rag_top_k", 5),
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
    )


def config_to_dict(config: AgentConfig) -> Dict[str, Any]:
    payload = asdict(config)
    payload["parameter_space"] = {
        name: asdict(spec) for name, spec in config.parameter_space.specs.items()
    }
    return payload
