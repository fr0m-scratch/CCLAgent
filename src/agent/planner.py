from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, List, Optional

from ..RAG import RagStore
from ..llm import LLMClient, LLMMessage
from ..llm.context_window import ContextWindowManager, PromptSection, estimate_tokens
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
from ..trace import TraceEmitter, NullTraceEmitter
from .context_pack import build_context_pack, write_context_pack
from .offline_reasoner import OfflineReasoner


logger = setup_logger("cclagent.planner")


@dataclass
class PlannerInputs:
    workload: WorkloadSpec
    microbench: MicrobenchResult
    context: ContextSignature
    rules: list[Any]


@dataclass
class PromptBundle:
    messages: List[LLMMessage]
    context_window: dict[str, Any]
    context_refs: list[str]


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
        trace: TraceEmitter | None = None,
    ) -> None:
        self.config = config
        self.tools = tools
        self.memory = memory
        self.rag = rag
        self.llm = llm
        self.parameter_space = parameter_space
        self._rag_loaded = False
        self.run_context = run_context
        self.trace = trace or NullTraceEmitter()

    def build_context(self, workload: WorkloadSpec) -> ContextSignature:
        context = ContextSignature(
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
        if self.run_context:
            write_json(artifact_path(self.run_context, "offline", "context_snapshot.json"), asdict(context))
            self.trace.event(
                run_id=self.run_context.run_id,
                phase="offline",
                step=None,
                actor="agent",
                type="offline.context.detect",
                payload={"context": asdict(context)},
            )
        return context

    def offline_plan(self, workload: WorkloadSpec) -> MicrobenchResult:
        logger.info("Running microbench for offline planning.")
        if self.run_context:
            self.trace.event(
                run_id=self.run_context.run_id,
                phase="offline",
                step=None,
                actor="agent",
                type="offline.microbench.plan",
                payload={"workload": workload.name, "mode": self.config.microbench.mode},
            )
        result = self.tools.microbench.run(workload, self.parameter_space)
        if self.run_context:
            write_json(artifact_path(self.run_context, "offline", "microbench_summary.json"), asdict(result))
            self.trace.event(
                run_id=self.run_context.run_id,
                phase="offline",
                step=None,
                actor="agent",
                type="offline.microbench.result",
                payload={
                    "important_params": [ip.param for ip in result.important_params],
                    "signals": [signal.name for signal in result.signals],
                },
                refs=[f"microbench:{signal.name}" for signal in result.signals],
            )
        return result

    def build_initial_plan(
        self, workload: WorkloadSpec, microbench: MicrobenchResult, context: ContextSignature
    ) -> InitialConfigPlan:
        rules_with_scores = self.memory.retrieve_rules_with_scores(context, top_k=3)
        rules = [item["rule"] for item in rules_with_scores]
        base_params = self.parameter_space.default_config()
        rule_patches = [rule.config_patch for rule in rules[:2]]

        offline_reasoner = OfflineReasoner(self.config.safety)
        offline_artifacts = offline_reasoner.build_offline_artifacts(
            self.parameter_space,
            microbench,
            rule_patches,
            self.config.microbench.mode,
        )
        if self.run_context:
            self.trace.event(
                run_id=self.run_context.run_id,
                phase="offline",
                step=None,
                actor="agent",
                type="decision.offline_warm_start",
                payload={
                    "selected_id": offline_artifacts.warm_start_decision.selected_id,
                    "reason": offline_artifacts.warm_start_decision.reason,
                },
            )
            if offline_artifacts.pruning:
                self.trace.event(
                    run_id=self.run_context.run_id,
                    phase="offline",
                    step=None,
                    actor="agent",
                    type="search.prune",
                    payload={
                        "count": len(offline_artifacts.pruning),
                        "params": [item.param for item in offline_artifacts.pruning],
                    },
                )
        selected_id = offline_artifacts.warm_start_decision.selected_id
        selected_candidate = next(
            (cand for cand in offline_artifacts.warm_start_candidates if cand.candidate_id == selected_id), None
        )
        if selected_candidate is not None:
            base_params = dict(selected_candidate.config)

        if self.run_context:
            self.trace.event(
                run_id=self.run_context.run_id,
                phase="offline",
                step=None,
                actor="agent",
                type="retrieval.memory",
                payload={
                    "count": len(rules_with_scores),
                    "rules": [
                        {"rule_id": item["rule"].id, "score": item["score"]}
                        for item in rules_with_scores
                    ],
                },
                refs=[f"rule:{item['rule'].id}" for item in rules_with_scores],
            )

        parsed = {}
        try:
            parsed = self._propose_llm_config(workload, microbench, context, rules)
        except Exception as exc:
            logger.warning("LLM proposal failed: %s", exc)
            if not getattr(self.config, "llm", None) or not self.config.llm.allow_fallback:
                raise
        if parsed:
            base_params.update(parsed)

        important_params = microbench.important_params
        recommended = [ip.param for ip in sorted(important_params, key=lambda x: x.importance, reverse=True)]
        if not recommended:
            recommended = list(self.parameter_space.specs.keys())
        # apply pruning: fix low-importance params to default by excluding from recommended
        if offline_artifacts.pruning:
            pruned = {item.param for item in offline_artifacts.pruning}
            recommended = [param for param in recommended if param not in pruned] or recommended

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
            write_json(
                artifact_path(self.run_context, "offline", "microbench_plan.json"),
                offline_artifacts.microbench_plan,
            )
            write_json(
                artifact_path(self.run_context, "offline", "warm_start_candidates.json"),
                [cand.__dict__ for cand in offline_artifacts.warm_start_candidates],
            )
            write_json(
                artifact_path(self.run_context, "offline", "warm_start_decision.json"),
                offline_artifacts.warm_start_decision.__dict__,
            )
            write_json(
                artifact_path(self.run_context, "offline", "search_space_pruning.json"),
                [item.__dict__ for item in offline_artifacts.pruning],
            )
            report = {
                "summary": "offline plan generated",
                "warm_start_selected": offline_artifacts.warm_start_decision.selected_id,
                "pruned_params": [item.param for item in offline_artifacts.pruning],
            }
            write_json(artifact_path(self.run_context, "offline", "offline_report.json"), report)
            with open(artifact_path(self.run_context, "offline", "offline_report.md"), "w", encoding="utf-8") as handle:
                handle.write(
                    f"Offline report\n\nSelected warm start: {offline_artifacts.warm_start_decision.selected_id}\n"
                )
            rag_chunks = []
            if self.config.rag and self._rag_loaded:
                # best-effort: include last retrieval set based on workload signature
                rag_chunks = [
                    {"doc_id": doc.doc_id, "chunk_id": doc.chunk_id, "score": doc.score}
                    for doc in self.rag.search(workload.name, top_k=self.config.rag.top_k)
                ]
            ctx_pack = build_context_pack(
                phase="offline",
                step=None,
                workload=workload,
                context=context,
                observations={
                    "important_params": [ip.param for ip in microbench.important_params],
                    "signals": [signal.name for signal in microbench.signals],
                },
                memory_rules=[
                    {"rule_id": item["rule"].id, "score": item["score"]}
                    for item in rules_with_scores
                ],
                rag_chunks=rag_chunks,
                surrogate={"model_type": self.config.surrogate.model_type},
                constraints={"sla": {"max_iteration_time": self.config.sla_max_iteration_time}},
            )
            write_context_pack(artifact_path(self.run_context, "offline", "context_pack.json"), ctx_pack)
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
        bundle = self._build_prompt_bundle(PlannerInputs(workload, microbench, context, rules))
        response = self.llm.complete(
            bundle.messages,
            trace_phase="offline",
            trace_step=None,
            system_prompt_version=self.config.llm.system_prompt_version,
            context_window=bundle.context_window,
            context_refs=bundle.context_refs,
            injected_context_refs=bundle.context_refs,
            max_tokens=self.config.llm.max_response_tokens or 512,
            temperature=self.config.llm.temperature,
        )
        parsed = self._parse_config_response(response.content)
        return parsed

    def _build_prompt_bundle(self, inputs: PlannerInputs) -> PromptBundle:
        doc_snippets = ""
        rag_refs = []
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
            rag_refs = [f"rag:{doc.doc_id}:{doc.chunk_id}" for doc in docs]
            if self.run_context:
                self.trace.event(
                    run_id=self.run_context.run_id,
                    phase="offline",
                    step=None,
                    actor="agent",
                    type="retrieval.rag",
                    payload={
                        "count": len(docs),
                        "chunks": [
                            {"doc_id": doc.doc_id, "chunk_id": doc.chunk_id, "score": doc.score}
                            for doc in docs
                        ],
                    },
                    refs=[f"rag:{doc.doc_id}:{doc.chunk_id}" for doc in docs],
                )

        rule_payload = [
            {"id": getattr(rule, "id", None), "config_patch": rule.config_patch, "improvement": rule.improvement}
            for rule in inputs.rules[:3]
        ]
        important_params = [ip.param for ip in inputs.microbench.important_params]
        signals = [
            {
                "name": signal.name,
                "value": signal.value,
                "unit": signal.unit,
                "confidence": signal.confidence,
                "source": signal.source,
            }
            for signal in inputs.microbench.signals
        ]

        system_text = (
            "You are a CCL tuning agent. Produce a JSON dict of NCCL env var settings. "
            "Focus on important parameters and keep values valid. Return JSON only."
        )
        sections = [
            PromptSection(name="WORKLOAD", content=json.dumps(asdict(inputs.workload), indent=2), priority=0),
            PromptSection(name="CONTEXT", content=json.dumps(asdict(inputs.context), indent=2), priority=0),
            PromptSection(name="IMPORTANT_PARAMS", content=json.dumps(important_params), priority=0),
            PromptSection(name="SIGNALS", content=json.dumps(signals, indent=2), priority=0),
            PromptSection(name="MEMORY_RULES", content=json.dumps(rule_payload, indent=2), priority=1),
            PromptSection(name="RAG_SNIPPETS", content=doc_snippets, priority=2, max_tokens=800),
        ]
        max_context_tokens = self.config.llm.max_context_tokens or 8000
        max_response_tokens = self.config.llm.max_response_tokens or 512
        reserve_tokens = estimate_tokens(system_text) + int(max_response_tokens)
        manager = ContextWindowManager(max_context_tokens, reserve_tokens=reserve_tokens)
        user_text, ctx_meta = manager.build(sections)
        ctx_meta["system_tokens"] = estimate_tokens(system_text)
        ctx_meta["response_token_budget"] = max_response_tokens
        messages = [
            LLMMessage(role="system", content=system_text),
            LLMMessage(role="user", content=user_text),
        ]
        context_refs = (
            [f"microbench:{signal.name}" for signal in inputs.microbench.signals]
            + [f"rule:{item.get('id')}" for item in rule_payload if item.get("id")]
            + rag_refs
        )
        return PromptBundle(messages=messages, context_window=ctx_meta, context_refs=context_refs)

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
