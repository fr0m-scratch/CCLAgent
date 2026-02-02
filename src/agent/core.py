from __future__ import annotations

from typing import Any, Optional
import hashlib
import threading
from pathlib import Path

from ..RAG import RagStore
from ..llm import LLMClient, NullLLMClient
from ..memory import MemoryStore
from ..types import (
    AgentConfig,
    ContextSignature,
    MicrobenchResult,
    NCCLConfig,
    TuningAction,
    TuningRecord,
    WorkloadSpec,
)
from ..utils import artifact_path, setup_logger, write_json
from ..models.features import ConfigFeaturizer
from ..models.surrogate import SurrogateModel
from .analyzer import TuningAnalyzer
from .executor import WorkloadExecutor
from .hypothesis import HypothesisGenerator
from .numeric import NumericSearchManager
from .post_run import distill_rules, persist_rules, persist_avoid_rules
from .ext_tuner import ExtTunerServer
from ..models.training import export_dataset, train_surrogate_model
from .planner import OfflinePlanner
from .policy import DecisionPolicy
from .state import TuningState


logger = setup_logger("cclagent.agent")


class CCLAgent:
    def __init__(
        self,
        config: AgentConfig,
        tools: Any,
        memory: MemoryStore,
        rag: Optional[RagStore] = None,
        llm: Optional[LLMClient] = None,
        rng_seed: int = 7,
        planner: Optional[OfflinePlanner] = None,
        policy: Optional[DecisionPolicy] = None,
        executor: Optional[WorkloadExecutor] = None,
        run_context=None,
    ) -> None:
        self.config = config
        self.tools = tools
        self.memory = memory
        self.rag = rag or RagStore(config.rag)
        self.llm = llm or NullLLMClient()
        self.parameter_space = config.parameter_space
        self.featurizer = ConfigFeaturizer(self.parameter_space)
        self.surrogate = SurrogateModel(self.featurizer, model_type=config.surrogate.model_type)
        self.run_context = run_context
        self._surrogate_records: list[tuple[NCCLConfig, float]] = []

        self.planner = planner or OfflinePlanner(
            config=config,
            tools=tools,
            memory=memory,
            rag=self.rag,
            llm=self.llm,
            parameter_space=self.parameter_space,
            run_context=run_context,
        )
        self.policy = policy or DecisionPolicy(
            config=config,
            memory=memory,
            parameter_space=self.parameter_space,
            surrogate=self.surrogate,
            search_tool=getattr(tools, "numeric_search", None),
            rng_seed=config.seed,
        )
        self.executor = executor or WorkloadExecutor(tools, run_context=run_context)
        self.hypothesis_generator = HypothesisGenerator(memory=memory, parameter_space=self.parameter_space)
        self.numeric_manager = NumericSearchManager(
            config=config,
            parameter_space=self.parameter_space,
            surrogate=self.surrogate,
            executor=self.executor,
            run_context=run_context,
        )
        self.analyzer = TuningAnalyzer(config=config, hypothesis_generator=self.hypothesis_generator, numeric_manager=self.numeric_manager, compiler=self.tools.compiler, run_context=run_context)

    def tune(self, workload: WorkloadSpec) -> TuningState:
        context = self.planner.build_context(workload)
        self._load_surrogate(context)
        microbench = self.planner.offline_plan(workload)
        plan = self.planner.build_initial_plan(workload, microbench, context)
        initial_config = plan.baseline_config
        state = TuningState(budget=self.config.budget)

        if self.config.execution.mode == "in_job_ext_tuner" and self.run_context is not None:
            session_dir = artifact_path(self.run_context, "ext_tuner_session")
            server = ExtTunerServer(self, workload, session_dir)
            thread = threading.Thread(target=server.serve, daemon=True)
            thread.start()
            metrics = self.executor.run(
                workload,
                plan.baseline_config,
                step=0,
                extra_env={"CCL_TUNER_SESSION_DIR": session_dir},
                execution_mode="in_job_ext_tuner",
            )
            thread.join()
            return server.session.state

        last_metrics = None
        current_config = initial_config
        next_compiled = None

        stop_reason = "budget_exhausted"
        for step in range(self.config.budget.max_steps):
            action = TuningAction(
                kind="initial" if step == 0 else "apply",
                config=current_config,
                rationale="initial plan" if step == 0 else "apply",
            )
            metrics = self.executor.run(workload, action.config, step, compiled=next_compiled)
            next_compiled = None
            record = TuningRecord(
                step=step,
                action=action,
                metrics=metrics,
                microbench_snapshot={
                    "important_params": [ip.param for ip in microbench.important_params],
                    "signals": [signal.name for signal in microbench.signals],
                },
            )
            state.record(record)
            if action.hypothesis is not None:
                self.memory.mark_rule_usage(action.hypothesis.id, metrics.success)
            if not metrics.success:
                self.memory.add_avoid_rule(context, action.config.params, evidence=metrics.raw)
            if metrics.success:
                self._update_surrogate(context, action.config, metrics.iteration_time_ms, step)

            prev_config = state.history[-2].action.config if len(state.history) > 1 else None
            self._persist_step(record, prev_config=prev_config)

            decision = self.analyzer.plan_next_action(
                state=state,
                last_metrics=metrics,
                microbench=microbench,
                context=context,
                step=step + 1,
                plan=plan,
                workload=workload,
                base_config=current_config,
            )
            last_metrics = metrics
            if getattr(decision, "kind", "") == "stop":
                logger.info("Stopping: %s", decision.reason)
                stop_reason = decision.reason
                break
            if getattr(decision, "kind", "") == "rollback":
                logger.info("Rollback requested: %s", decision.reason)
                current_config = decision.config
                next_compiled = None
                continue
            if decision is None:
                stop_reason = "no_decision"
                break
            current_config = decision.config
            next_compiled = getattr(decision, "compiled", None)

        if self.run_context:
            write_json(
                artifact_path(self.run_context, "postrun", "convergence.json"),
                {"reason": stop_reason, "steps": len(state.history)},
            )
        self._post_run(state, context)
        self.memory.save()
        return state

    def _persist_step(self, record: TuningRecord, prev_config: NCCLConfig | None = None) -> None:
        if not self.run_context:
            return
        delta = {}
        if prev_config is not None:
            for key, value in record.action.config.params.items():
                if prev_config.params.get(key) != value:
                    delta[key] = {"from": prev_config.params.get(key), "to": value}
        payload = {
            "step": record.step,
            "action": {
                "kind": record.action.kind,
                "config": record.action.config.params,
                "rationale": record.action.rationale,
                "delta": delta,
            },
            "metrics": record.metrics.__dict__,
        }
        write_json(artifact_path(self.run_context, "steps", f"step_{record.step}.json"), payload)

    def _post_run(self, state: TuningState, context: ContextSignature) -> None:
        if not state.best_record or not state.history:
            return
        baseline = state.history[0].metrics.iteration_time_ms
        best = state.best_record.metrics.iteration_time_ms
        improvement = (baseline - best) / max(1e-9, baseline)
        config_patch = self._diff_configs(state.history[0].action.config, state.best_record.action.config)
        if config_patch:
            self.memory.add_rule(context, config_patch, improvement)
        for record in state.history:
            self.memory.add_surrogate_record(context, record.action.config, record.metrics)
        dataset_records = [
            {"context": context.__dict__, "config": rec.action.config.params, "metrics": rec.metrics.__dict__}
            for rec in state.history
        ]
        dataset_path = f"{self.config.surrogate.dataset_path}/{self._context_hash(context)}.jsonl"
        export_dataset(dataset_records, dataset_path)
        train_surrogate_model(dataset_records, context, self.parameter_space, self.config.surrogate, self._model_path(context))
        rules = distill_rules(state, context)
        for rule in rules:
            self.memory.add_rule(context, rule["config_patch"], rule["improvement"], evidence=rule.get("evidence"))
        persist_rules("memory/rules.jsonl", rules)
        avoids = []
        for record in state.history:
            if not record.metrics.success:
                avoids.append({"context": context.__dict__, "config_patch": record.action.config.params, "evidence": record.metrics.raw})
                self.memory.add_avoid_rule(context, record.action.config.params, evidence=record.metrics.raw)
        persist_avoid_rules("memory/avoid_rules.jsonl", avoids)
        if self.run_context:
            write_json(
                artifact_path(self.run_context, "postrun", "rule_updates.json"),
                {"rules": rules, "avoid_rules": avoids},
            )

    def _load_surrogate(self, context: ContextSignature) -> None:
        latest = self._find_latest_model(context)
        if not latest:
            return
        try:
            self.surrogate.load(latest)
            logger.info("Loaded surrogate model: %s", latest)
        except Exception as exc:
            logger.warning("Failed to load surrogate model %s: %s", latest, exc)

    def _update_surrogate(self, context: ContextSignature, config: NCCLConfig, iteration_time_ms: float, step: int) -> None:
        self._surrogate_records.append((config, iteration_time_ms))
        refit_every = self.config.surrogate.refit_every_steps or self.config.surrogate.retrain_every
        should_refit = len(self._surrogate_records) >= self.config.surrogate.min_records and (
            (step + 1) % refit_every == 0
        )
        self.surrogate.update(
            config,
            iteration_time_ms,
            context=context,
            refit=should_refit,
            min_records=self.config.surrogate.min_records,
        )
        if should_refit and self.run_context:
            model_path = self._model_path(context)
            try:
                Path(self.config.surrogate.model_dir).mkdir(parents=True, exist_ok=True)
                self.surrogate.save(model_path)
            except Exception as exc:
                logger.warning("Failed to save surrogate model: %s", exc)

    def _diff_configs(self, base: NCCLConfig, best: NCCLConfig) -> dict[str, Any]:
        patch = {}
        for key, value in best.params.items():
            if base.params.get(key) != value:
                patch[key] = value
        return patch

    def _context_hash(self, context: ContextSignature) -> str:
        payload = f"{context.workload}|{context.topology}|{context.scale}|{context.nodes}|{context.model}|{context.framework}"
        return hashlib.md5(payload.encode("utf-8")).hexdigest()

    def _model_path(self, context: ContextSignature) -> str:
        timestamp = (self.run_context.started_at_iso if self.run_context else "").replace(":", "-")
        suffix = f"_{timestamp}" if timestamp else ""
        return f"{self.config.surrogate.model_dir}/surrogate_{self._context_hash(context)}{suffix}.pkl"
    def _find_latest_model(self, context: ContextSignature) -> str | None:
        model_dir = Path(self.config.surrogate.model_dir)
        if not model_dir.exists():
            return None
        prefix = f"surrogate_{self._context_hash(context)}_"
        candidates = sorted(model_dir.glob(prefix + "*.pkl"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            return None
        return str(candidates[0])
