from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, List, Optional

from ..RAG import RagStore
from ..llm import LLMClient, LLMMessage, NullLLMClient
from ..memory import MemoryStore
from ..types import (
    ContextSignature,
    ImportantParam,
    InitialConfigPlan,
    MicrobenchResult,
    NCCLConfig,
    ParameterSpace,
    Subspace,
    WorkloadSpec,
)
from ..utils import artifact_path, setup_logger, write_json


logger = setup_logger("cclagent.planner")


@dataclass
class PlannerInputs:
    workload: WorkloadSpec
    microbench: MicrobenchResult
    context: ContextSignature
    rules: list[Any]


class OfflinePlanner:
    def __init__(
        self,
        config: Any,
        tools: Any,
        memory: MemoryStore,
        rag: RagStore,
        llm: LLMClient,
        parameter_space: ParameterSpace,
        run_context: Optional[Any] = None,
    ) -> None:
        self.config = config
        self.tools = tools
        self.memory = memory
        self.rag = rag
        self.llm = llm
        self.parameter_space = parameter_space
        self._rag_loaded = False
        self.run_context = run_context

    def build_context(self, workload: WorkloadSpec) -> ContextSignature:
        return ContextSignature(
            workload=workload.name,
            workload_kind=workload.kind,
            topology=workload.topology,
            scale=workload.scale,
            nodes=workload.nodes,
            model=workload.metadata.get("model"),
            framework=workload.metadata.get("framework"),
            gpus_per_node=workload.gpus_per_node,
            gpu_type=workload.metadata.get("gpu_type"),
            network=workload.metadata.get("network"),
            nic_count=workload.metadata.get("nic_count"),
            extra=workload.metadata,
        )

    def offline_plan(self, workload: WorkloadSpec) -> MicrobenchResult:
        logger.info("Running microbench for offline planning.")
        result = self.tools.microbench.run(workload, self.parameter_space)
        if self.run_context:
            write_json(artifact_path(self.run_context, "offline", "microbench_summary.json"), asdict(result))
        return result

    def build_initial_plan(
        self, workload: WorkloadSpec, microbench: MicrobenchResult, context: ContextSignature
    ) -> InitialConfigPlan:
        rules = self.memory.retrieve_rules(context, top_k=3)
        base_params = self.parameter_space.default_config()
        if rules:
            for rule in rules[:2]:
                base_params.update(rule.config_patch)

        if not isinstance(self.llm, NullLLMClient):
            parsed = self._propose_llm_config(workload, microbench, context, rules)
            if parsed:
                base_params.update(parsed)

        important_params = microbench.important_params
        recommended = [ip.param for ip in sorted(important_params, key=lambda x: x.importance, reverse=True)]
        if not recommended:
            recommended = list(self.parameter_space.specs.keys())

        constraints = {}
        for name, spec in self.parameter_space.specs.items():
            constraint = {}
            if spec.min_value is not None:
                constraint["min"] = spec.min_value
            if spec.max_value is not None:
                constraint["max"] = spec.max_value
            if spec.choices:
                constraint["allowed"] = spec.choices
            constraints[name] = constraint

        candidate_subspaces = self._build_subspaces(recommended)

        plan = InitialConfigPlan(
            baseline_config=NCCLConfig(params=base_params, metadata={"source": "plan"}),
            constraints=constraints,
            important_params=important_params,
            candidate_subspaces=candidate_subspaces,
            recommended_search_params=recommended,
            notes="auto-generated from microbench and memory rules",
        )
        if self.run_context:
            write_json(artifact_path(self.run_context, "offline", "initial_plan.json"), asdict(plan))
        return plan

    def propose_initial_config(
        self, workload: WorkloadSpec, microbench: MicrobenchResult, context: ContextSignature
    ) -> NCCLConfig:
        plan = self.build_initial_plan(workload, microbench, context)
        return plan.baseline_config

    def _build_subspaces(self, recommended: List[str]) -> List[Subspace]:
        algo_spec = self.parameter_space.specs.get("NCCL_ALGO")
        proto_spec = self.parameter_space.specs.get("NCCL_PROTO")
        algos = algo_spec.choices if algo_spec and algo_spec.choices else []
        protos = proto_spec.choices if proto_spec and proto_spec.choices else []
        subspaces: List[Subspace] = []
        if algos and protos:
            for algo in algos:
                for proto in protos:
                    subspaces.append(
                        Subspace(
                            name=f"{algo}-{proto}",
                            fixed={"NCCL_ALGO": algo, "NCCL_PROTO": proto},
                            free=[param for param in recommended if param not in ("NCCL_ALGO", "NCCL_PROTO")],
                        )
                    )
        else:
            subspaces.append(Subspace(name="default", fixed={}, free=recommended))
        return subspaces

    def _propose_llm_config(
        self,
        workload: WorkloadSpec,
        microbench: MicrobenchResult,
        context: ContextSignature,
        rules: list[Any],
    ) -> dict[str, Any]:
        prompt = self._build_prompt(PlannerInputs(workload, microbench, context, rules))
        response = self.llm.complete([LLMMessage(role="user", content=prompt)])
        parsed = self._parse_config_response(response.content)
        return parsed

    def _build_prompt(self, inputs: PlannerInputs) -> str:
        doc_snippets = ""
        if self.config.rag:
            if not self._rag_loaded:
                self.rag.load_documents(self.config.rag)
                self._rag_loaded = True
            docs = self.rag.search(
                f"{inputs.workload.name} {inputs.workload.topology} {inputs.workload.scale}",
                top_k=self.config.rag.top_k,
            )
            important = [ip.param for ip in inputs.microbench.important_params]
            extra_docs = []
            for param in important:
                extra_docs.extend(
                    self.rag.search(f"NCCL {param} recommended values and constraints", top_k=max(1, self.config.rag.top_k // 2))
                )
                extra_docs.extend(
                    self.rag.search(f"NCCL {param} interactions with important params", top_k=max(1, self.config.rag.top_k // 2))
                )
            docs = docs + extra_docs
            doc_snippets = self.rag.summarize(docs)

        rule_text = "\n".join(
            f"Rule: {rule.config_patch} (improvement={rule.improvement:.3f})"
            for rule in inputs.rules[:3]
        )

        important_params = [ip.param for ip in inputs.microbench.important_params]
        signals = {signal.name: signal.value for signal in inputs.microbench.signals}

        prompt = (
            "You are a CCL tuning agent. Produce a JSON dict of NCCL env var settings. "
            "Focus on important parameters and keep values valid.\n"
            f"Workload: {inputs.workload}\n"
            f"Context: {inputs.context}\n"
            f"Important params: {important_params}\n"
            f"Signals: {signals}\n"
            f"Rules: {rule_text}\n"
            f"Docs: {doc_snippets}\n"
            "Return JSON only."
        )
        return prompt

    def _parse_config_response(self, text: str) -> dict[str, Any]:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1:
            return {}
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return {}
