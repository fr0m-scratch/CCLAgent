from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, List, Optional

from ..RAG import RagStore
from ..llm import LLMClient, LLMMessage
from ..llm.context_window import ContextWindowManager, PromptSection, estimate_tokens
from ..llm.schemas import validate_offline_plan
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
from .offline_reasoner import OfflineReasoner, WarmStartCandidate, SearchSpacePrune


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

    def build_context(self, workload: WorkloadSpec, system_probe: Optional[dict[str, Any]] = None) -> ContextSignature:
        probe = system_probe if isinstance(system_probe, dict) else {}
        extra = dict(workload.metadata or {})
        if probe:
            extra["system_probe"] = probe
        def _int_or_none(value: Any) -> Optional[int]:
            if value is None:
                return None
            try:
                return int(value)
            except (TypeError, ValueError):
                return None
        context = ContextSignature(
            workload=workload.name,
            workload_kind=workload.kind,
            topology=str(probe.get("topology") or workload.topology),
            scale=workload.scale,
            nodes=workload.nodes,
            model=workload.metadata.get("model"),
            framework=workload.metadata.get("framework"),
            gpus_per_node=_int_or_none(probe.get("gpus_per_node")) or _int_or_none(workload.gpus_per_node),
            gpu_type=str(probe.get("gpu_type") or workload.metadata.get("gpu_type")) if (
                probe.get("gpu_type") is not None or workload.metadata.get("gpu_type") is not None
            ) else None,
            network=str(probe.get("network") or workload.metadata.get("network")) if (
                probe.get("network") is not None or workload.metadata.get("network") is not None
            ) else None,
            nic_count=_int_or_none(probe.get("nic_count")) or _int_or_none(workload.metadata.get("nic_count")),
            extra=extra,
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
                refs=[f"candidate:offline:{offline_artifacts.warm_start_decision.selected_id}"],
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
                    refs=["offline/search_space_pruning.json"],
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

        llm_plan: dict[str, Any] = {}
        llm_plan_status: dict[str, Any] = {
            "schema_version": "1.0",
            "attempted": True,
            "accepted": False,
            "validation_errors": [],
            "exception": None,
            "fallback_used": True,
            "fallback_reason": "llm_plan_unavailable",
        }
        try:
            llm_plan, llm_errors = self._propose_llm_plan(workload, microbench, context, rules)
            llm_plan_status["validation_errors"] = llm_errors
            if llm_errors:
                llm_plan_status["fallback_reason"] = "llm_plan_validation_failed"
            if llm_plan and not llm_errors:
                llm_plan_status["accepted"] = True
                llm_plan_status["fallback_used"] = False
                llm_plan_status["fallback_reason"] = ""
        except Exception as exc:
            logger.warning("LLM proposal failed: %s", exc)
            llm_plan_status["exception"] = str(exc)
            llm_plan_status["fallback_reason"] = "llm_plan_exception"
            if not getattr(self.config, "llm", None) or not self.config.llm.allow_fallback:
                raise
        if not llm_plan_status["accepted"]:
            llm_plan = {}
        if llm_plan:
            baseline_patch, _errors = self._clean_patch(llm_plan.get("baseline_patch", {}), max_keys=6)
            if baseline_patch:
                base_params.update(baseline_patch)

        important_params = microbench.important_params
        recommended = [ip.param for ip in sorted(important_params, key=lambda x: x.importance, reverse=True)]
        if not recommended:
            recommended = list(self.parameter_space.specs.keys())
        # apply pruning: fix low-importance params to default by excluding from recommended
        pruned = {item.param for item in offline_artifacts.pruning} if offline_artifacts.pruning else set()
        pruning_guidance = self._normalize_pruning_guidance(
            llm_plan.get("pruning_guidance", []),
            fallback_pruning=offline_artifacts.pruning,
        )
        # map pruning guidance to recommended params and optional fixed values
        frozen_params = set()
        for item in pruning_guidance:
            action = item.get("action")
            param = item.get("param")
            if not param:
                continue
            if action in ("freeze_default", "freeze_value"):
                frozen_params.add(param)
                value = item.get("value")
                if action == "freeze_value" and value is not None:
                    if self._param_is_valid(param, value):
                        base_params[param] = value
        pruned |= frozen_params
        if pruned:
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
        subspace_priors = self._normalize_subspace_priors(llm_plan.get("subspace_priors", []))
        hypothesis_playbook = self._normalize_hypothesis_playbook(llm_plan.get("hypothesis_playbook", []))
        if not hypothesis_playbook:
            hypothesis_playbook = self._fallback_hypothesis_playbook(workload)
        warm_start_program = self._normalize_warm_start_program(
            llm_plan.get("warm_start_program"),
            offline_artifacts.warm_start_candidates,
        )

        plan = InitialConfigPlan(
            baseline_config=NCCLConfig(params=base_params, metadata={"source": "plan"}),
            constraints=constraints,
            important_params=important_params,
            candidate_subspaces=candidate_subspaces,
            recommended_search_params=recommended,
            warm_start_program=warm_start_program,
            hypothesis_playbook=hypothesis_playbook,
            pruning_guidance=pruning_guidance,
            subspace_priors=subspace_priors,
            notes="auto-generated from microbench and memory rules",
        )
        if self.run_context:
            write_json(artifact_path(self.run_context, "offline", "warm_start_program.json"), warm_start_program)
            write_json(artifact_path(self.run_context, "offline", "hypothesis_playbook.json"), hypothesis_playbook)
            write_json(artifact_path(self.run_context, "offline", "pruning_guidance.json"), pruning_guidance)
            write_json(artifact_path(self.run_context, "offline", "subspace_priors.json"), subspace_priors)
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
            if llm_plan:
                write_json(artifact_path(self.run_context, "offline", "llm_strategic_plan.json"), llm_plan)
            write_json(artifact_path(self.run_context, "offline", "llm_plan_status.json"), llm_plan_status)
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
            self.trace.event(
                run_id=self.run_context.run_id,
                phase="offline",
                step=None,
                actor="agent",
                type="decision.warm_start_program",
                payload={
                    "mode": warm_start_program.get("mode"),
                    "candidate_count": len(warm_start_program.get("candidates", [])),
                },
                refs=[f"candidate:offline:{c.get('id')}" for c in warm_start_program.get("candidates", []) if c.get("id")],
            )
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

    def _propose_llm_plan(
        self,
        workload: WorkloadSpec,
        microbench: MicrobenchResult,
        context: ContextSignature,
        rules: list[Any],
    ) -> tuple[dict[str, Any], list[str]]:
        bundle = self._build_prompt_bundle(PlannerInputs(workload, microbench, context, rules))
        response = self.llm.complete(
            bundle.messages,
            trace_phase="offline",
            trace_step=None,
            system_prompt_version=getattr(self.config.llm, "system_prompt_version_offline", None)
            or self.config.llm.system_prompt_version,
            context_window=bundle.context_window,
            context_refs=bundle.context_refs,
            injected_context_refs=bundle.context_refs,
            max_tokens=self.config.llm.max_response_tokens or 512,
            temperature=self.config.llm.temperature,
            timeout_s=getattr(self.config.llm, "online_hard_timeout_s", None),
            **self._json_response_kwargs(),
        )
        parsed = self._parse_llm_json(response.content)
        if not isinstance(parsed, dict):
            parsed = {}
            errors = ["invalid_json"]
        else:
            errors = validate_offline_plan(parsed, self.parameter_space)
        if errors:
            repaired, repair_errors = self._repair_llm_plan(response.content)
            if repaired and not repair_errors:
                return repaired, []
            prefixed = [f"primary:{item}" for item in errors]
            prefixed.extend([f"repair:{item}" for item in repair_errors])
            return parsed, prefixed
        return parsed, []

    def _repair_llm_plan(self, raw_text: str) -> tuple[dict[str, Any], list[str]]:
        system_text = (
            "You are a strict JSON repair assistant.\n"
            "Transform model output into ONE valid JSON object with required keys only:\n"
            "warm_start_program, baseline_patch, pruning_guidance, subspace_priors, "
            "hypothesis_playbook, tool_triggers.\n"
            "Rules:\n"
            "- Use only parameters in PARAM_SPECS.\n"
            "- If uncertain, use safe defaults and empty lists/patches.\n"
            "- Do not output markdown or comments."
        )
        sections = [
            PromptSection(
                name="PARAM_SPECS",
                content=json.dumps(
                    {name: asdict(spec) for name, spec in self.parameter_space.specs.items()},
                    indent=2,
                ),
                priority=0,
            ),
            PromptSection(name="RAW_OUTPUT_TO_REPAIR", content=raw_text, priority=0, max_tokens=1400),
            PromptSection(
                name="SAFE_DEFAULT_TEMPLATE",
                content=json.dumps(
                    {
                        "warm_start_program": {
                            "mode": "single",
                            "candidates": [],
                            "selection_rule": {"objective": "min_iteration_time_ms"},
                        },
                        "baseline_patch": {},
                        "pruning_guidance": [],
                        "subspace_priors": [],
                        "hypothesis_playbook": [],
                        "tool_triggers": [],
                    },
                    indent=2,
                ),
                priority=0,
                max_tokens=700,
            ),
        ]
        max_context_tokens = self.config.llm.max_context_tokens or 8000
        max_response_tokens = min(700, self.config.llm.max_response_tokens or 512)
        reserve_tokens = estimate_tokens(system_text) + int(max_response_tokens)
        manager = ContextWindowManager(max_context_tokens, reserve_tokens=reserve_tokens)
        user_text, ctx_meta = manager.build(sections)
        response = self.llm.complete(
            [LLMMessage(role="system", content=system_text), LLMMessage(role="user", content=user_text)],
            trace_phase="offline",
            trace_step=None,
            system_prompt_version="offline_json_repair_v1",
            context_window=ctx_meta,
            max_tokens=max_response_tokens,
            temperature=min(0.2, self.config.llm.temperature),
            timeout_s=getattr(self.config.llm, "online_hard_timeout_s", None),
            **self._json_response_kwargs(),
        )
        repaired = self._parse_llm_json(response.content)
        if not isinstance(repaired, dict):
            return {}, ["invalid_json"]
        errors = validate_offline_plan(repaired, self.parameter_space)
        return repaired, errors

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
                write_json(
                    artifact_path(self.run_context, "offline", "rag_retrieval.json"),
                    {
                        "schema_version": "1.0",
                        "query": f"{inputs.workload.name} {inputs.workload.topology} {inputs.workload.scale}",
                        "top_k": self.config.rag.top_k,
                        "chunks": [
                            {
                                "doc_id": doc.doc_id,
                                "chunk_id": doc.chunk_id,
                                "score": doc.score,
                                "text": doc.text,
                            }
                            for doc in docs
                        ],
                    },
                )
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
        system_probe = {}
        if isinstance(inputs.context.extra, dict):
            system_probe = inputs.context.extra.get("system_probe", {}) or {}

        system_text = (
            "You are the offline strategic planner for a CCL/NCCL tuning agent.\n\n"
            "You MUST obey:\n"
            "1) Output MUST be a single JSON object, no markdown, no extra keys.\n"
            "2) Only use parameter names that appear in [PARAM_SPECS]. Do not invent knobs.\n"
            "3) Every proposed value must respect constraints in [PARAM_SPECS] (choices, min/max, type).\n"
            "4) Keep baseline_patch small: at most 6 keys. Prefer high-importance parameters.\n"
            "5) No hallucinations: every nontrivial recommendation must cite refs from provided evidence IDs.\n"
            "6) Do NOT include chain-of-thought. Provide brief structured claims with refs instead.\n"
            "7) If evidence is insufficient, set baseline_patch={}, warm_start_program.mode=\"single\", and explain in uncertainty.safe_fallback.\n\n"
            "Primary objective: reduce iteration_time_ms safely.\n"
            "Secondary objective: produce a WarmStartProgram and an online hypothesis playbook."
        )
        sections = [
            PromptSection(name="WORKLOAD", content=json.dumps(asdict(inputs.workload), indent=2), priority=0),
            PromptSection(name="CONTEXT_SIGNATURE", content=json.dumps(asdict(inputs.context), indent=2), priority=0),
            PromptSection(
                name="PARAM_SPECS",
                content=json.dumps(
                    {name: asdict(spec) for name, spec in self.parameter_space.specs.items()}, indent=2
                ),
                priority=0,
            ),
            PromptSection(name="IMPORTANT_PARAMS", content=json.dumps(important_params), priority=0),
            PromptSection(name="MICROBENCH_SIGNALS", content=json.dumps(signals, indent=2), priority=0),
            PromptSection(name="SYSTEM_PROBE", content=json.dumps(system_probe, indent=2), priority=0, max_tokens=600),
            PromptSection(name="MEMORY_RULES", content=json.dumps(rule_payload, indent=2), priority=1),
            PromptSection(
                name="SAFETY_ENVELOPE",
                content=json.dumps(getattr(self.config, "safety", {}).safe_envelope if getattr(self.config, "safety", None) else {}, indent=2),
                priority=1,
                max_tokens=600,
            ),
            PromptSection(
                name="CURRENT_DEFAULTS",
                content=json.dumps(self.parameter_space.default_config(), indent=2),
                priority=1,
                max_tokens=400,
            ),
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

    def _parse_llm_json(self, text: str) -> dict[str, Any]:
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

    def _json_response_kwargs(self) -> dict[str, Any]:
        provider = str(getattr(self.config.llm, "provider", "")).lower()
        if provider in ("openai", "fireworks", "openai-compatible"):
            return {"response_format": {"type": "json_object"}}
        return {}

    def _param_is_valid(self, name: str, value: Any) -> bool:
        spec = self.parameter_space.specs.get(name)
        if spec is None:
            return False
        return spec.is_valid(value)

    def _clean_patch(self, patch: Any, max_keys: int = 6) -> tuple[dict[str, Any], list[str]]:
        errors: list[str] = []
        if not isinstance(patch, dict):
            return {}, ["patch_not_dict"]
        cleaned: dict[str, Any] = {}
        for key, value in patch.items():
            if len(cleaned) >= max_keys:
                break
            if key not in self.parameter_space.specs:
                errors.append(f"unknown_param:{key}")
                continue
            if not self._param_is_valid(key, value):
                errors.append(f"invalid_value:{key}={value}")
                continue
            cleaned[key] = value
        return cleaned, errors

    def _normalize_pruning_guidance(
        self,
        guidance: Any,
        fallback_pruning: list[SearchSpacePrune] | None = None,
    ) -> list[dict[str, Any]]:
        if isinstance(guidance, list) and guidance:
            normalized: list[dict[str, Any]] = []
            for item in guidance:
                if not isinstance(item, dict):
                    continue
                param = item.get("param")
                action = item.get("action")
                if not param or param not in self.parameter_space.specs:
                    continue
                if action not in ("freeze_default", "freeze_value", "keep_free"):
                    continue
                normalized.append(
                    {
                        "param": param,
                        "action": action,
                        "value": item.get("value"),
                        "reason": item.get("reason", ""),
                        "refs": item.get("refs", []),
                    }
                )
            if normalized:
                return normalized
        if fallback_pruning:
            return [
                {
                    "param": item.param,
                    "action": "freeze_default",
                    "reason": item.reason,
                    "refs": [],
                }
                for item in fallback_pruning
            ]
        return []

    def _normalize_subspace_priors(self, priors: Any) -> list[dict[str, Any]]:
        if not isinstance(priors, list):
            return []
        out: list[dict[str, Any]] = []
        for item in priors:
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            if not name:
                continue
            fixed = item.get("fixed") if isinstance(item.get("fixed"), dict) else {}
            free = item.get("free") if isinstance(item.get("free"), list) else []
            out.append(
                {
                    "name": name,
                    "fixed": fixed,
                    "free": free,
                    "reason": item.get("reason", ""),
                    "refs": item.get("refs", []),
                }
            )
        return out

    def _normalize_hypothesis_playbook(self, playbook: Any) -> list[dict[str, Any]]:
        if not isinstance(playbook, list):
            return []
        out: list[dict[str, Any]] = []
        for item in playbook:
            if not isinstance(item, dict):
                continue
            patch = item.get("patch_template") or item.get("patch") or {}
            patch_clean, _ = self._clean_patch(patch, max_keys=4)
            out.append(
                {
                    "id": item.get("id"),
                    "trigger": item.get("trigger", {}),
                    "summary": item.get("summary", ""),
                    "mechanism": item.get("mechanism", ""),
                    "patch_template": patch_clean,
                    "expected_effect": item.get("expected_effect", {}),
                    "risk": item.get("risk", {}),
                    "evidence_refs": item.get("evidence_refs", []),
                    "validation_plan": item.get("validation_plan", {}),
                }
            )
        return out

    def _fallback_hypothesis_playbook(self, workload: WorkloadSpec) -> list[dict[str, Any]]:
        templates = [
            {
                "id": "fb_h1",
                "summary": "Increase channel parallelism for multi-GPU scale to reduce collective serialization",
                "mechanism": "throughput_parallelism",
                "patch_template": {
                    "NCCL_MAX_NCHANNELS": 16,
                    "NCCL_PROTO": "LL128",
                },
            },
            {
                "id": "fb_h2",
                "summary": "Switch to TREE+LL128 for cross-node latency-sensitive synchronization",
                "mechanism": "latency_reduction",
                "patch_template": {
                    "NCCL_ALGO": "TREE",
                    "NCCL_PROTO": "LL128",
                },
            },
            {
                "id": "fb_h3",
                "summary": "Tune thread and buffer settings to improve overlap and effective bandwidth",
                "mechanism": "compute_comm_overlap",
                "patch_template": {
                    "NCCL_NTHREADS": 192,
                    "NCCL_BUFFSIZE": 8 << 20,
                },
            },
        ]
        out: list[dict[str, Any]] = []
        for item in templates:
            patch_clean, _ = self._clean_patch(item.get("patch_template", {}), max_keys=4)
            if not patch_clean:
                continue
            out.append(
                {
                    "id": item.get("id"),
                    "trigger": {"source": "fallback_offline_plan", "workload": workload.name},
                    "summary": item.get("summary", ""),
                    "mechanism": item.get("mechanism", "fallback"),
                    "patch_template": patch_clean,
                    "expected_effect": {"iteration_time_ms": {"direction": "decrease", "magnitude": "medium"}},
                    "risk": {"level": "low", "notes": "fallback template from built-in heuristics"},
                    "evidence_refs": [],
                    "validation_plan": {"metric": "iteration_time_ms", "direction": "decrease"},
                }
            )
        return out

    def _normalize_warm_start_program(
        self,
        program: Any,
        fallback_candidates: list[WarmStartCandidate],
    ) -> dict[str, Any]:
        if isinstance(program, dict) and program.get("candidates"):
            candidates = []
            for item in program.get("candidates", []):
                if not isinstance(item, dict):
                    continue
                patch_clean, _ = self._clean_patch(item.get("patch", {}), max_keys=4)
                candidates.append(
                    {
                        "id": item.get("id") or "WS",
                        "summary": item.get("summary", ""),
                        "patch": patch_clean,
                        "mechanism": item.get("mechanism", ""),
                        "risk_hint": item.get("risk_hint", "unknown"),
                        "evidence_refs": item.get("evidence_refs", []),
                        "eval_plan": item.get("eval_plan", {"mode": "short", "steps": 50, "timeout_sec": 300}),
                    }
                )
            mode = program.get("mode") if program.get("mode") in ("single", "series") else "single"
            return {
                "schema_version": "1.0",
                "mode": mode,
                "candidates": candidates,
                "selection_rule": program.get(
                    "selection_rule",
                    {"objective": "min_iteration_time_ms", "tie_breaker": "lower_risk"},
                ),
                "claims": program.get("claims", []),
                "uncertainty": program.get("uncertainty", {}),
            }
        # fallback: derive from offline reasoner candidates
        defaults = self.parameter_space.default_config()
        candidates: list[dict[str, Any]] = []
        for cand in fallback_candidates:
            patch = {k: v for k, v in cand.config.items() if defaults.get(k) != v}
            patch_clean, _ = self._clean_patch(patch, max_keys=4)
            candidates.append(
                {
                    "id": cand.candidate_id,
                    "summary": cand.reason,
                    "patch": patch_clean,
                    "mechanism": cand.source,
                    "risk_hint": "low" if cand.risk_score <= self.config.safety.max_risk_score else "high",
                    "evidence_refs": [],
                    "eval_plan": {"mode": "short", "steps": 50, "timeout_sec": 300},
                }
            )
        return {
            "schema_version": "1.0",
            "mode": "single",
            "candidates": candidates,
            "selection_rule": {"objective": "min_iteration_time_ms", "tie_breaker": "lower_risk"},
            "claims": [],
            "uncertainty": {"level": "high", "missing_evidence": ["llm_unavailable"], "safe_fallback": "defaults"},
        }
