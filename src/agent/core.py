from __future__ import annotations

from typing import Any, Optional

from ..RAG import RagStore
from ..llm import LLMClient, NullLLMClient
from ..memory import MemoryStore
from ..types import AgentConfig, ContextSignature, MicrobenchResult, NCCLConfig, TuningAction, TuningRecord, WorkloadSpec
from ..utils import setup_logger
from .executor import WorkloadExecutor
from .planner import OfflinePlanner
from .policy import DecisionPolicy
from .state import SurrogateModel, TuningState


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
    ) -> None:
        self.config = config
        self.tools = tools
        self.memory = memory
        self.rag = rag or RagStore()
        self.llm = llm or NullLLMClient()
        self.parameter_space = config.parameter_space
        self.surrogate = SurrogateModel(self.parameter_space)

        self.planner = planner or OfflinePlanner(
            config=config,
            tools=tools,
            memory=memory,
            rag=self.rag,
            llm=self.llm,
            parameter_space=self.parameter_space,
        )
        self.policy = policy or DecisionPolicy(
            config=config,
            memory=memory,
            parameter_space=self.parameter_space,
            surrogate=self.surrogate,
            search_tool=getattr(tools, "numeric_search", None),
            rng_seed=rng_seed,
        )
        self.executor = executor or WorkloadExecutor(tools)

    def tune(self, workload: WorkloadSpec) -> TuningState:
        context = self.planner.build_context(workload)
        microbench = self.planner.offline_plan(workload)
        initial_config = self.planner.propose_initial_config(workload, microbench, context)
        state = TuningState(budget=self.config.budget)

        for step in range(self.config.budget.max_steps):
            action = TuningAction(
                kind="initial" if step == 0 else "apply",
                config=initial_config,
                rationale="initial plan",
            )
            metrics = self.executor.run(workload, action.config, step)
            state.record(TuningRecord(action=action, metrics=metrics))
            self.surrogate.update(action.config, metrics.iteration_time)

            if metrics.extras.get("sla_ok") is False:
                logger.info("Stopping due to SLA violation.")
                break

            if state.should_stop:
                logger.info("Stopping due to plateau.")
                break

            next_action = self.policy.decide_next_action(state, microbench, context, step)
            if next_action is None:
                break
            initial_config = next_action.config

        self._post_run(state, context)
        self.memory.save()
        return state

    def _post_run(self, state: TuningState, context: ContextSignature) -> None:
        if not state.best_record or not state.history:
            return
        baseline = state.history[0].metrics.iteration_time
        best = state.best_record.metrics.iteration_time
        improvement = (baseline - best) / max(1e-9, baseline)
        config_patch = self._diff_configs(state.history[0].action.config, state.best_record.action.config)
        if config_patch:
            self.memory.add_rule(context, config_patch, improvement)
        for record in state.history:
            self.memory.add_surrogate_record(context, record.action.config, record.metrics)

    def _diff_configs(self, base: NCCLConfig, best: NCCLConfig) -> dict[str, Any]:
        patch = {}
        for key, value in best.params.items():
            if base.params.get(key) != value:
                patch[key] = value
        return patch
