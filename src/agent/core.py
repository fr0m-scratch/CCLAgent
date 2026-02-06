from __future__ import annotations

from typing import Any, Optional
from dataclasses import asdict
import hashlib
import threading
import time
import json
import queue
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
from .post_run import persist_avoid_rules
from .distill import distill_semantic_rules, persist_rules, persist_report
from .distill_llm import distill_rules_llm
from .ext_tuner import ExtTunerServer
from ..models.training import export_dataset, train_surrogate_model
from .planner import OfflinePlanner
from .policy import DecisionPolicy
from .state import TuningState
from .metrics_derive import derive_metrics
from .bottleneck import classify_bottleneck
from .context_pack import build_context_pack, write_context_pack
from .warmstart import run_warm_start_program
from .online_advisor import OnlineLLMAdvisor
from .system_probe import SystemProbeCollector
from ..trace import NullTraceEmitter, TraceEmitter


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
        trace: TraceEmitter | None = None,
        mailbox: Any = None,  # Queue for receiving commands from TUI (injected by runner)
    ) -> None:
        self.config = config
        self.mailbox = mailbox
        self.tools = tools
        self.memory = memory
        self.rag = rag or RagStore(config.rag)
        self.llm = llm or NullLLMClient()
        self.parameter_space = config.parameter_space
        self.featurizer = ConfigFeaturizer(self.parameter_space)
        self.surrogate = SurrogateModel(self.featurizer, model_type=config.surrogate.model_type)
        self.run_context = run_context
        self.trace = trace or NullTraceEmitter()
        self._surrogate_records: list[tuple[NCCLConfig, float]] = []
        self.llm_advisor = OnlineLLMAdvisor(
            llm=self.llm,
            parameter_space=self.parameter_space,
            config=self.config,
            run_context=run_context,
            trace=self.trace,
        )
        self.system_probe = SystemProbeCollector(run_context=run_context, trace=self.trace)

        self.planner = planner or OfflinePlanner(
            config=config,
            tools=tools,
            memory=memory,
            rag=self.rag,
            llm=self.llm,
            parameter_space=self.parameter_space,
            run_context=run_context,
            trace=self.trace,
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
            trace=self.trace,
        )
        self.analyzer = TuningAnalyzer(
            config=config,
            hypothesis_generator=self.hypothesis_generator,
            numeric_manager=self.numeric_manager,
            compiler=self.tools.compiler,
            run_context=run_context,
            trace=self.trace,
            memory=self.memory,
            llm_advisor=self.llm_advisor,
        )
        self._mailbox_async_enabled = False
        self._mailbox_worker_stop = threading.Event()
        self._mailbox_thread: Optional[threading.Thread] = None
        self._mailbox_step_hint = 0

    def tune(self, workload: WorkloadSpec) -> TuningState:
        self._trace_event(
            phase="system",
            step=None,
            actor="system",
            type="run.start",
            payload={
                "workload": workload.name,
                "dry_run": getattr(self.run_context, "dry_run", None),
                "seed": self.config.seed,
            },
        )
        probe = self.system_probe.collect(
            workload,
            simulate=bool(getattr(self.run_context, "dry_run", False)),
        )
        context = self.planner.build_context(workload, system_probe=probe.get("summary", {}))
        self._load_surrogate(context)
        microbench = self.planner.offline_plan(workload)
        plan = self.planner.build_initial_plan(workload, microbench, context)
        self._active_context = context
        self._active_workload = workload
        self._active_plan = plan
        initial_config = plan.baseline_config
        warmstart_program = plan.warm_start_program or {}
        warmstart_mode = getattr(self.config, "warm_start", None).mode if getattr(self.config, "warm_start", None) else None
        if warmstart_mode:
            warmstart_program["mode"] = warmstart_mode

        warmstart_probes = []
        if warmstart_program.get("mode") == "series" and warmstart_program.get("candidates"):
            selected_config, warmstart_probes = run_warm_start_program(
                program=warmstart_program,
                defaults=self.parameter_space.default_config(),
                workload=workload,
                executor=self.executor,
                parameter_space=self.parameter_space,
                safety=self.config.safety,
                run_context=self.run_context,
                trace=self.trace,
                max_candidates=self.config.warm_start.max_candidates,
                eval_steps=self.config.warm_start.eval_steps,
                eval_timeout_sec=self.config.warm_start.eval_timeout_sec,
                concurrency=self.config.warm_start.concurrency,
            )
            initial_config = selected_config
            plan.baseline_config = selected_config
            if self.run_context:
                write_json(artifact_path(self.run_context, "offline", "initial_plan.json"), asdict(plan))
        state = TuningState(budget=self.config.budget)
        self._state = state

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
        self._stop_requested = False
        self._pause_requested = False
        self._pending_overrides = getattr(self, "_pending_overrides", {})

        stop_reason = "budget_exhausted"
        max_steps = self.config.budget.max_steps
        if warmstart_probes and self.config.warm_start.counts_toward_budget:
            max_steps = max(0, max_steps - len(warmstart_probes))
        self._start_mailbox_worker()
        try:
            for step in range(max_steps):
                self._mailbox_step_hint = step
                # Poll mailbox before each step so stop/chat config updates apply promptly.
                self._check_mailbox(step)
                if self._stop_requested:
                    stop_reason = "stopped_by_user"
                    break
                while self._pause_requested and not self._stop_requested:
                    time.sleep(0.1)
                    self._check_mailbox(step)
                if self._stop_requested:
                    stop_reason = "stopped_by_user"
                    break

                # Apply any pending config overrides from chat /set commands
                if self._pending_overrides:
                    logger.info("Applying chat overrides: %s", self._pending_overrides)
                    for param, value in self._pending_overrides.items():
                        current_config.params[param] = value
                    self._pending_overrides = {}

                action = TuningAction(
                    kind="initial" if step == 0 else "apply",
                    config=current_config,
                    rationale="initial plan" if step == 0 else "apply",
                )
                metrics = self.executor.run(workload, action.config, step, compiled=next_compiled)
                if self.run_context:
                    derived = derive_metrics(metrics)
                    write_json(
                        artifact_path(self.run_context, "steps", f"step_{step}_metrics_derived.json"),
                        {"schema_version": "1.0", "derived": derived},
                    )
                    bottleneck, confidence = classify_bottleneck(derived)
                    write_json(
                        artifact_path(self.run_context, "steps", f"step_{step}_bottleneck.json"),
                        {"schema_version": "1.0", "class": bottleneck, "confidence": confidence},
                    )
                    self._trace_event(
                        phase="online",
                        step=step,
                        actor="agent",
                        type="analysis.metrics.derive",
                        payload={"derived": derived},
                        refs=[f"metric:{step}:derived:{k}" for k in derived.keys()],
                    )
                    self._trace_event(
                        phase="online",
                        step=step,
                        actor="agent",
                        type="analysis.bottleneck.classify",
                        payload={"class": bottleneck, "confidence": confidence},
                    )
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
        finally:
            self._stop_mailbox_worker()

        if self.run_context:
            write_json(
                artifact_path(self.run_context, "postrun", "convergence.json"),
                {"reason": stop_reason, "steps": len(state.history)},
            )
        self._post_run(state, context)
        self.memory.save()
        try:
            self.llm_advisor.shutdown()
        except Exception:
            pass
        self._trace_event(
            phase="system",
            step=None,
            actor="system",
            type="run.end",
            payload={
                "reason": stop_reason,
                "steps": len(state.history),
            },
        )
        return state

    def _trace_event(
        self,
        *,
        phase: str,
        step: int | None,
        actor: str,
        type: str,
        payload: dict,
        refs: list[str] | None = None,
        status: str = "ok",
    ) -> None:
        if not self.run_context:
            return
        self.trace.event(
            run_id=self.run_context.run_id,
            phase=phase,
            step=step,
            actor=actor,
            type=type,
            payload=payload,
            refs=refs,
            status=status,
        )

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
        if self.run_context:
            ctx_pack = build_context_pack(
                phase="postrun",
                step=None,
                workload=WorkloadSpec(name=context.workload, command=[]),
                context=context,
                observations={
                    "baseline_ms": baseline,
                    "best_ms": best,
                    "improvement": improvement,
                },
                memory_rules=[],
                rag_chunks=[],
                surrogate={"model_type": self.config.surrogate.model_type},
                constraints={"sla": {"max_iteration_time": self.config.sla_max_iteration_time}},
            )
            write_context_pack(artifact_path(self.run_context, "postrun", "context_pack.json"), ctx_pack)
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
        microbench_snapshot = state.history[0].microbench_snapshot if state.history else {}
        rules = distill_rules_llm(
            state=state,
            context=context,
            llm=self.llm,
            parameter_space=self.parameter_space,
            llm_config=self.config,
            microbench_snapshot=microbench_snapshot,
        )
        if not rules:
            rules = distill_semantic_rules(state, context)
        for rule in rules:
            patch = rule.get("action", {}).get("set", {})
            self.memory.add_rule(context, patch, rule.get("effect", {}).get("improvement", 0.0), evidence=rule)
        if self.run_context:
            persist_rules(artifact_path(self.run_context, "postrun", "rules_distilled.jsonl"), rules)
            persist_report(artifact_path(self.run_context, "postrun", "distillation_report.md"), rules)
        else:
            persist_rules("postrun/rules_distilled.jsonl", rules)
            persist_report("postrun/distillation_report.md", rules)
        if self.run_context:
            for rule in rules:
                self._trace_event(
                    phase="postrun",
                    step=None,
                    actor="agent",
                    type="postrun.distill.rule",
                    payload={"rule_id": rule.get("rule_id")},
                    refs=[],
                )
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
            write_json(
                artifact_path(self.run_context, "postrun", "best_config_validation.json"),
                {
                    "schema_version": "1.0",
                    "executed": False,
                    "reason": "validation_not_configured",
                    "best_config": state.best_record.action.config.params,
                },
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

    def _check_mailbox(self, step: int) -> None:
        """Poll mailbox for commands like 'chat'."""
        if self._mailbox_async_enabled:
            return
        if not self.mailbox:
            return
            
        # Drain all pending commands or limiting to one?
        # Let's process until empty to be responsive
        try:
            while True:
                # Non-unblocking get
                cmd = self.mailbox.get_nowait()
                self._handle_command(cmd, step)
        except queue.Empty:
            pass

    def _start_mailbox_worker(self) -> None:
        if not self.mailbox:
            return
        if self._mailbox_async_enabled:
            return
        self._mailbox_worker_stop.clear()
        self._mailbox_async_enabled = True
        self._mailbox_thread = threading.Thread(target=self._mailbox_worker_loop, daemon=True)
        self._mailbox_thread.start()

    def _stop_mailbox_worker(self) -> None:
        if not self._mailbox_async_enabled:
            return
        self._mailbox_worker_stop.set()
        thread = self._mailbox_thread
        if thread is not None:
            thread.join(timeout=0.2)
        self._mailbox_async_enabled = False
        self._mailbox_thread = None

    def _mailbox_worker_loop(self) -> None:
        if not self.mailbox:
            return
        while not self._mailbox_worker_stop.is_set():
            try:
                cmd = self.mailbox.get(timeout=0.15)
            except queue.Empty:
                continue
            try:
                step = int(getattr(self, "_mailbox_step_hint", 0))
            except Exception:
                step = 0
            try:
                self._handle_command(cmd, max(0, step))
            except Exception as exc:
                logger.warning("Mailbox command handling failed: %s", exc)
            
    def _handle_command(self, cmd: Any, step: int) -> None:
        action = getattr(cmd, "action", "")
        payload = getattr(cmd, "payload", {})
        if not isinstance(payload, dict):
            payload = {}
        if action == "stop":
            self._stop_requested = True
            self._pause_requested = False
            if not payload.get("silent"):
                self._emit_chat_response(step, "Stop requested. Exiting after the current check.")
            return
        if action == "pause":
            self._pause_requested = True
            self._emit_chat_response(step, "Paused. Send resume to continue.")
            return
        if action == "resume":
            self._pause_requested = False
            self._emit_chat_response(step, "Resumed.")
            return
        if action == "chat":
            user_msg = payload.get("message", "")

            # Check for slash commands
            if user_msg.startswith("/set "):
                self._handle_set_command(user_msg[5:], step)
            elif user_msg.startswith("/setcfg "):
                self._handle_setcfg_command(user_msg[8:], step)
            elif user_msg.startswith("/setplan "):
                self._handle_setplan_command(user_msg[9:], step)
            elif user_msg.startswith("/state"):
                self._handle_state_query(step)
            elif user_msg.startswith("/context"):
                self._handle_context_dump(step)
            elif user_msg.startswith("/ctxeng"):
                self._handle_context_engineering_query(step)
            elif user_msg.startswith("/best"):
                self._handle_best_query(step)
            elif user_msg.startswith("/help"):
                self._handle_chat_help(step)
            else:
                self._handle_chat(user_msg, step)

    def _emit_chat_response(self, step: int, message: str) -> None:
        """Helper to emit a chat response trace event."""
        self._trace_event(
            phase="online",
            step=step,
            actor="agent",
            type="chat_response",
            payload={"response": message, "original_query": "[command]"},
            status="ok",
        )

    def _handle_set_command(self, args: str, step: int) -> None:
        """Handle /set PARAM=VALUE command to override a config parameter."""
        try:
            if "=" not in args:
                self._emit_chat_response(step, "Usage: /set PARAM_NAME=VALUE")
                return
            param, value = args.strip().split("=", 1)
            param = param.strip()
            value = value.strip()

            spec = self.parameter_space.specs.get(param)
            if spec is None:
                available = list(self.parameter_space.specs.keys())
                self._emit_chat_response(step, f"Unknown parameter: {param}. Available: {available}")
                return

            if spec.kind == "enum":
                typed_value = spec.normalize(value)
            elif spec.kind == "int":
                typed_value = spec.normalize(int(value))
            elif spec.kind == "float":
                typed_value = spec.normalize(float(value))
            else:
                typed_value = spec.normalize(value)

            if not spec.is_valid(typed_value):
                self._emit_chat_response(
                    step,
                    f"Invalid value for {param}: {typed_value}. "
                    f"min={spec.min_value}, max={spec.max_value}, choices={spec.choices}",
                )
                return

            if not hasattr(self, "_pending_overrides"):
                self._pending_overrides = {}
            self._pending_overrides[param] = typed_value
            self._emit_chat_response(step, f"Override queued: {param}={typed_value}. Will apply on next step.")
        except Exception as e:
            self._emit_chat_response(step, f"Error parsing /set command: {e}")

    def _handle_setcfg_command(self, args: str, step: int) -> None:
        """Handle /setcfg PATH=VALUE to adjust runtime tuner configuration."""
        try:
            if "=" not in args:
                self._emit_chat_response(step, "Usage: /setcfg budget.hypothesis_every=1")
                return
            path, raw_value = args.strip().split("=", 1)
            path = path.strip()
            raw_value = raw_value.strip()
            if not path:
                self._emit_chat_response(step, "Usage: /setcfg budget.hypothesis_every=1")
                return
            full_path = path[7:] if path.startswith("config.") else path
            applied, final_value = self._set_runtime_field(self.config, full_path, raw_value)
            if not applied:
                self._emit_chat_response(
                    step,
                    "Invalid /setcfg path. Supported roots: budget, llm, numeric_search, safety, "
                    "execution, warm_start, surrogate, sla_max_iteration_time.",
                )
                return
            self._emit_chat_response(step, f"Runtime config updated: {full_path}={final_value}")
        except Exception as e:
            self._emit_chat_response(step, f"Error parsing /setcfg command: {e}")

    def _handle_setplan_command(self, args: str, step: int) -> None:
        """Handle /setplan PATH=VALUE to adjust online plan fields."""
        try:
            if "=" not in args:
                self._emit_chat_response(step, "Usage: /setplan recommended_search_params='[\"NCCL_ALGO\"]'")
                return
            if not hasattr(self, "_active_plan") or self._active_plan is None:
                self._emit_chat_response(step, "No active plan to modify yet.")
                return
            path, raw_value = args.strip().split("=", 1)
            path = path.strip()
            raw_value = raw_value.strip()
            if not path:
                self._emit_chat_response(step, "Usage: /setplan recommended_search_params='[\"NCCL_ALGO\"]'")
                return
            allowed_roots = {
                "recommended_search_params",
                "pruning_guidance",
                "hypothesis_playbook",
                "subspace_priors",
                "candidate_subspaces",
                "notes",
                "warm_start_program",
            }
            root = path.split(".", 1)[0]
            if root not in allowed_roots:
                self._emit_chat_response(
                    step,
                    "Invalid /setplan root. Supported: recommended_search_params, pruning_guidance, "
                    "hypothesis_playbook, subspace_priors, candidate_subspaces, notes, warm_start_program.",
                )
                return
            applied, final_value = self._set_runtime_field(self._active_plan, path, raw_value)
            if not applied:
                self._emit_chat_response(step, f"Unable to set plan field: {path}")
                return
            if self.run_context:
                write_json(artifact_path(self.run_context, "offline", "initial_plan.json"), asdict(self._active_plan))
            self._emit_chat_response(step, f"Online plan updated: {path}={final_value}")
        except Exception as e:
            self._emit_chat_response(step, f"Error parsing /setplan command: {e}")

    def _handle_state_query(self, step: int) -> None:
        """Emit current TuningState summary as chat response."""
        import json as _json
        state = getattr(self, "_state", None)
        if not state:
            self._emit_chat_response(step, "No tuning state available yet.")
            return
        summary = {
            "steps_completed": len(state.history),
            "best_iteration_time_ms": state.best_record.metrics.iteration_time_ms if state.best_record else None,
            "best_config": state.best_record.action.config.params if state.best_record else None,
            "plateau_count": state.plateau_count,
            "should_stop": state.should_stop,
            "iteration_times": [r.metrics.iteration_time_ms for r in state.history],
        }
        self._emit_chat_response(step, f"Current State:\n```json\n{_json.dumps(summary, indent=2, default=str)}\n```")

    def _handle_best_query(self, step: int) -> None:
        """Emit best config found so far."""
        import json as _json
        state = getattr(self, "_state", None)
        if not state or not state.best_record:
            self._emit_chat_response(step, "No best config found yet.")
            return
        best = {
            "step": state.best_record.step,
            "config": state.best_record.action.config.params,
            "iteration_time_ms": state.best_record.metrics.iteration_time_ms,
            "rationale": state.best_record.action.rationale,
        }
        self._emit_chat_response(step, f"Best config:\n```json\n{_json.dumps(best, indent=2, default=str)}\n```")

    def _handle_context_dump(self, step: int) -> None:
        """Emit full context pack for current step."""
        if not self.run_context:
            self._emit_chat_response(step, "No run context available.")
            return
        ctx_path = artifact_path(self.run_context, "steps", f"step_{step}_context_pack.json")
        try:
            ctx = json.loads(Path(ctx_path).read_text())
            summary = json.dumps(ctx, indent=2, default=str)
            if len(summary) > 3000:
                summary = summary[:3000] + "\n...[truncated]"
            self._emit_chat_response(step, f"Context Pack (step {step}):\n```json\n{summary}\n```")
        except FileNotFoundError:
            self._emit_chat_response(step, f"No context pack found for step {step}.")

    def _handle_context_engineering_query(self, step: int) -> None:
        latest = getattr(self, "_latest_chat_context", None)
        if not latest:
            self._emit_chat_response(step, "No chat context snapshot yet. Ask a normal question first.")
            return
        summary = {
            "step": latest.get("step"),
            "path": latest.get("path"),
            "total_tokens": latest.get("total_tokens"),
            "sections": latest.get("sections", []),
        }
        self._emit_chat_response(
            step,
            "Latest chat context-engineering snapshot:\n```json\n"
            f"{json.dumps(summary, indent=2, default=str)}\n```",
        )

    def _handle_chat_help(self, step: int) -> None:
        self._emit_chat_response(
            step,
            "Commands:\n"
            "- /set NCCL_PARAM=value\n"
            "- /setcfg budget.hypothesis_every=1\n"
            "- /setplan recommended_search_params='[\"NCCL_ALGO\"]'\n"
            "- /state\n"
            "- /best\n"
            "- /context\n"
            "- /ctxeng",
        )

    def _set_runtime_field(self, root: Any, path: str, raw_value: str) -> tuple[bool, Any]:
        segments = [seg for seg in path.split(".") if seg]
        if not segments:
            return False, None
        safe_roots = {
            "budget",
            "llm",
            "numeric_search",
            "safety",
            "execution",
            "warm_start",
            "surrogate",
            "sla_max_iteration_time",
            "recommended_search_params",
            "pruning_guidance",
            "hypothesis_playbook",
            "subspace_priors",
            "candidate_subspaces",
            "notes",
            "warm_start_program",
        }
        if segments[0] not in safe_roots:
            return False, None
        node = root
        for seg in segments[:-1]:
            if isinstance(node, dict):
                if seg not in node:
                    return False, None
                node = node[seg]
            else:
                if not hasattr(node, seg):
                    return False, None
                node = getattr(node, seg)
        leaf = segments[-1]
        current = node.get(leaf) if isinstance(node, dict) else getattr(node, leaf, None)
        value = self._coerce_runtime_value(raw_value, current)
        if isinstance(node, dict):
            node[leaf] = value
        else:
            if not hasattr(node, leaf):
                return False, None
            setattr(node, leaf, value)
        return True, value

    def _coerce_runtime_value(self, raw_value: str, current: Any) -> Any:
        import ast

        text = raw_value.strip()
        if isinstance(current, bool):
            return text.lower() in ("1", "true", "yes", "on")
        if isinstance(current, int) and not isinstance(current, bool):
            return int(float(text))
        if isinstance(current, float):
            return float(text)
        if isinstance(current, (list, dict)):
            try:
                return json.loads(text)
            except Exception:
                return ast.literal_eval(text)
        if current is None:
            lower = text.lower()
            if lower in ("none", "null"):
                return None
            if lower in ("true", "false"):
                return lower == "true"
            try:
                if "." in text:
                    return float(text)
                return int(text)
            except ValueError:
                pass
            try:
                return json.loads(text)
            except Exception:
                try:
                    return ast.literal_eval(text)
                except Exception:
                    return text
        return text

    def _handle_chat(self, message: str, step: int) -> None:
        """Handle a chat message from the user with rich context engineering."""
        from ..llm.base import LLMMessage
        from ..llm.context_window import ContextWindowManager, PromptSection, estimate_tokens

        logger.info("Received chat from user: %s", message)

        state = getattr(self, "_state", None)

        # --- Build rich context sections ---

        # 1. TuningState summary
        state_summary = {"step": step, "run_id": self.run_context.run_id if self.run_context else "N/A"}
        if state:
            state_summary.update({
                "total_steps_completed": len(state.history),
                "plateau_count": state.plateau_count,
                "should_stop": state.should_stop,
                "best_iteration_time_ms": state.best_record.metrics.iteration_time_ms if state.best_record else None,
                "best_config": state.best_record.action.config.params if state.best_record else None,
            })

        # 2. Current config (from latest history record)
        current_config_info = {}
        if state and state.history:
            latest_record = state.history[-1]
            current_config_info = {
                "params": latest_record.action.config.params,
                "action_kind": latest_record.action.kind,
                "rationale": latest_record.action.rationale,
                "iteration_time_ms": latest_record.metrics.iteration_time_ms,
                "success": latest_record.metrics.success,
            }

        # 3. History records (last 10)
        history_records = []
        if state:
            for rec in state.history[-10:]:
                history_records.append({
                    "step": rec.step,
                    "action_kind": rec.action.kind,
                    "config_params": rec.action.config.params,
                    "rationale": rec.action.rationale,
                    "iteration_time_ms": rec.metrics.iteration_time_ms,
                    "success": rec.metrics.success,
                })

        # 4. Observations: derived metrics and bottleneck
        observations = {}
        if state and state.history:
            try:
                from .metrics_derive import derive_metrics
                from .bottleneck import classify_bottleneck
                latest_metrics = state.history[-1].metrics
                derived = derive_metrics(latest_metrics)
                bottleneck, confidence = classify_bottleneck(derived)
                observations = {
                    "derived_metrics": derived,
                    "bottleneck_class": bottleneck,
                    "bottleneck_confidence": confidence,
                }
            except Exception:
                pass

        # 5. Memory rules
        memory_rules_info = []
        for rule in self.memory.rules[:10]:
            memory_rules_info.append({
                "id": rule.id,
                "config_patch": rule.config_patch,
                "improvement": rule.improvement,
                "confidence": rule.confidence,
                "rule_type": rule.rule_type,
            })

        # 6. Numeric search state
        numeric_state = {}
        if hasattr(self, "numeric_manager") and hasattr(self.numeric_manager, "surrogate"):
            surr = self.numeric_manager.surrogate
            numeric_state = {
                "surrogate_model_type": getattr(surr, "model_type", "unknown"),
                "n_training_records": len(getattr(surr, "_records", [])),
            }

        # 7. Agent config summary
        config_summary = {
            "budget": {
                "max_steps": self.config.budget.max_steps,
                "patience": self.config.budget.patience,
                "min_improvement": self.config.budget.min_improvement,
            },
            "llm_model": self.config.llm.model,
            "sla_max_iteration_time": self.config.sla_max_iteration_time,
        }

        # --- Assemble into ContextWindowManager sections ---
        sections = [
            PromptSection(name="TUNING_STATE", content=json.dumps(state_summary, indent=2, default=str), priority=0),
            PromptSection(name="CURRENT_CONFIG", content=json.dumps(current_config_info, indent=2, default=str), priority=0),
            PromptSection(name="OBSERVATIONS", content=json.dumps(observations, indent=2, default=str), priority=0),
            PromptSection(name="HISTORY_RECORDS", content=json.dumps(history_records, indent=2, default=str), priority=1),
            PromptSection(name="MEMORY_RULES", content=json.dumps(memory_rules_info, indent=2, default=str), priority=2),
            PromptSection(name="NUMERIC_SEARCH_STATE", content=json.dumps(numeric_state, indent=2, default=str), priority=2),
            PromptSection(name="AGENT_CONFIG", content=json.dumps(config_summary, indent=2, default=str), priority=2),
        ]

        max_context_tokens = getattr(self.config.llm, "max_context_tokens", 8000)
        max_response_tokens = getattr(self.config.llm, "max_response_tokens", 512)

        system_prompt = (
            "You are the CCL Tuning Agent assistant. The user is interacting with you during a live "
            "NCCL performance tuning session.\n"
            "You have access to the full agent state below. Be helpful, concise, and technical.\n"
            "When answering questions, cite specific data from the context (step numbers, config values, metrics).\n"
            "If the user asks to change the plan or config, explain implications and reference commands: "
            "/set, /setcfg, /setplan.\n\n"
            "Optimization objective: improve end-to-end distributed LLM training by minimizing training "
            "iteration_time_ms through safe NCCL tuning.\n"
        )

        reserve_tokens = estimate_tokens(system_prompt) + max_response_tokens
        manager = ContextWindowManager(max_context_tokens, reserve_tokens=reserve_tokens)
        user_text, ctx_meta = manager.build(sections)

        full_user_message = f"{user_text}\n\n[USER_QUESTION]\n{message}"
        context_snapshot = {
            "schema_version": "1.0",
            "step": step,
            "question": message,
            "system_prompt": system_prompt,
            "assembled_user_prompt": full_user_message,
            "context_window": ctx_meta,
            "sections": [
                {
                    "name": section.name,
                    "priority": section.priority,
                    "content": section.content,
                }
                for section in sections
            ],
            "step_vs_iteration": {
                "agent_step": (
                    "One online control step: choose a NCCL config, execute workload, "
                    "measure metrics, and decide next action."
                ),
                "training_iteration": (
                    "A model training iteration inside the workload loop; each agent step "
                    "usually observes many training iterations."
                ),
                "autoccl_alignment": (
                    "Aligned with AutoCCL Sec. 5.2 (Iterative Online Tuning): "
                    "run one coordinate-descent step across training iterations and "
                    "amortize tuning overhead in early iterations."
                ),
            },
            "optimization_objective": (
                "Minimize end-to-end distributed LLM training iteration_time_ms "
                "safely through NCCL parameter tuning."
            ),
        }
        context_snapshot_path = None
        if self.run_context:
            context_snapshot_path = artifact_path(
                self.run_context,
                "online",
                f"chat_context_step_{step}_{int(time.time() * 1000)}.json",
            )
            write_json(context_snapshot_path, context_snapshot)
        self._latest_chat_context = {
            "step": step,
            "path": context_snapshot_path,
            "total_tokens": ctx_meta.get("total_tokens", 0),
            "sections": [
                {
                    "name": item.get("name"),
                    "tokens_after": item.get("tokens_after"),
                    "truncated": item.get("truncated"),
                }
                for item in ctx_meta.get("sections", [])
            ],
        }
        self._trace_event(
            phase="online",
            step=step,
            actor="agent",
            type="chat.context",
            payload={
                "chat_context_path": context_snapshot_path,
                "context_sections": [s["name"] for s in ctx_meta.get("sections", [])],
                "context_tokens": ctx_meta.get("total_tokens", 0),
            },
            status="ok",
        )

        try:
            response = self.llm.complete(
                [
                    LLMMessage(role="system", content=system_prompt),
                    LLMMessage(role="user", content=full_user_message),
                ],
                max_tokens=max_response_tokens,
                temperature=0.7,
                trace_phase="online",
                trace_step=step,
                system_prompt_version="chat_v1",
                context_window=ctx_meta,
            )

            reply = response.content

            self._trace_event(
                phase="online",
                step=step,
                actor="agent",
                type="chat_response",
                payload={
                    "response": reply,
                    "original_query": message,
                    "context_sections_used": [s["name"] for s in ctx_meta.get("sections", [])],
                    "context_tokens": ctx_meta.get("total_tokens", 0),
                    "chat_context_path": context_snapshot_path,
                },
                status="ok",
            )
        except Exception as e:
            logger.error("Failed to generate chat response: %s", e)
            self._trace_event(
                phase="online",
                step=step,
                actor="agent",
                type="chat_response",
                payload={"response": f"Error generating response: {str(e)}", "error": True},
                status="error",
            )
