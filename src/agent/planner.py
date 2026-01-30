from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, List

from ..RAG import RagStore
from ..llm import LLMClient, LLMMessage, NullLLMClient
from ..memory import MemoryStore
from ..types import ContextSignature, MicrobenchResult, NCCLConfig, ParameterSpace, WorkloadSpec
from ..utils import setup_logger


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
    ) -> None:
        self.config = config
        self.tools = tools
        self.memory = memory
        self.rag = rag
        self.llm = llm
        self.parameter_space = parameter_space
        self._rag_loaded = False

    def build_context(self, workload: WorkloadSpec) -> ContextSignature:
        return ContextSignature(
            workload=workload.name,
            topology=workload.topology,
            scale=workload.scale,
            nodes=workload.nodes,
        )

    def offline_plan(self, workload: WorkloadSpec) -> MicrobenchResult:
        logger.info("Running microbench for offline planning.")
        return self.tools.microbench.run(workload, self.parameter_space)

    def propose_initial_config(
        self, workload: WorkloadSpec, microbench: MicrobenchResult, context: ContextSignature
    ) -> NCCLConfig:
        rules = self.memory.get_rules(context)
        base_params = self.parameter_space.default_config()
        if rules:
            for rule in rules[:2]:
                base_params.update(rule.config_patch)
        if isinstance(self.llm, NullLLMClient):
            return NCCLConfig(params=base_params, metadata={"source": "rules"})

        prompt = self._build_prompt(PlannerInputs(workload, microbench, context, rules))
        response = self.llm.complete([LLMMessage(role="user", content=prompt)])
        parsed = self._parse_config_response(response.content)
        if parsed:
            base_params.update(parsed)
        return NCCLConfig(params=base_params, metadata={"source": "llm"})

    def _build_prompt(self, inputs: PlannerInputs) -> str:
        doc_snippets = ""
        if self.config.rag_docs_path:
            if not self._rag_loaded:
                self.rag.add_documents_from_path(self.config.rag_docs_path)
                self._rag_loaded = True
            docs = self.rag.search(
                f"{inputs.workload.name} {inputs.workload.topology} {inputs.workload.scale}",
                top_k=self.config.rag_top_k,
            )
            doc_snippets = self.rag.summarize(docs)

        rule_text = "\n".join(
            f"Rule: {rule.config_patch} (improvement={rule.improvement:.3f})"
            for rule in inputs.rules[:3]
        )

        prompt = (
            "You are a CCL tuning agent. Produce a JSON dict of NCCL env var settings. "
            "Focus on important parameters and keep values valid.\n"
            f"Workload: {inputs.workload}\n"
            f"Context: {inputs.context}\n"
            f"Important params: {inputs.microbench.important_params}\n"
            f"Signals: {inputs.microbench.signals}\n"
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
