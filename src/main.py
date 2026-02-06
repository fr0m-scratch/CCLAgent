from __future__ import annotations

import argparse
import os

from .agent import CCLAgent
from .config import config_to_dict, default_agent_config, load_agent_config, load_workload_spec
from .llm import create_llm_client, TracedLLMClient
from .memory import MemoryStore
from .RAG import RagStore
from .tools import (
    ConfigCompiler,
    ExtNetBridge,
    ExtTunerBridge,
    MetricsCollector,
    MicrobenchConfig,
    MicrobenchRunner,
    NCCLInterface,
    NcclTestConfig,
    NcclTestRunner,
    NumericSearchConfig,
    NumericSearchTool,
    SLAEnforcer,
    ToolSuite,
    TrainingJobConfig,
    TrainingJobRunner,
    WorkloadRunConfig,
    WorkloadRunner,
    InstrumentedToolSuite,
)
from .types import RunContext, WorkloadSpec
from .utils import artifact_path, create_run_context, load_env_file, setup_logger, write_json
from .trace import TraceEmitterWriter, TraceWriter


logger = setup_logger("cclagent.cli")


def _default_workload() -> WorkloadSpec:
    candidates = [
        "workload/benchmarks/llama3.1-8b-agentic-showcase.json",
        "workload/benchmarks/llama3.1-8b.json",
    ]
    for path in candidates:
        if os.path.exists(path):
            return load_workload_spec(path)
    return WorkloadSpec(name="demo", command=[])


def build_tools(config, run_context: RunContext, dry_run: bool, simulate_workload: bool = False) -> ToolSuite:
    metrics = MetricsCollector(config.metrics, run_context=run_context)
    microbench = MicrobenchRunner(
        MicrobenchConfig.from_settings(config.microbench, dry_run=dry_run),
        run_context=run_context,
    )
    workload_sim = dry_run or simulate_workload
    workload = WorkloadRunner(
        WorkloadRunConfig(dry_run=workload_sim),
        metrics_parser=metrics.parse,
        run_context=run_context,
    )
    sla = SLAEnforcer(max_iteration_time=config.sla_max_iteration_time)
    compiler = ConfigCompiler(config.parameter_space, safety=config.safety)
    nccl = NCCLInterface(compiler)
    nccltest = NcclTestRunner(NcclTestConfig(dry_run=workload_sim), run_context=run_context)
    training = TrainingJobRunner(TrainingJobConfig(dry_run=workload_sim), run_context=run_context)
    ext_tuner = ExtTunerBridge()
    autoccl = ext_tuner
    ext_net = ExtNetBridge()
    numeric_search = NumericSearchTool(
        NumericSearchConfig(
            max_candidates=config.numeric_search.max_candidates,
            concurrency=config.numeric_search.concurrency,
        ),
        rng_seed=config.seed,
    )
    return ToolSuite(
        microbench=microbench,
        workload=workload,
        metrics=metrics,
        sla=sla,
        compiler=compiler,
        nccl=nccl,
        nccltest=nccltest,
        training=training,
        autoccl=autoccl,
        ext_tuner=ext_tuner,
        ext_net=ext_net,
        numeric_search=numeric_search,
        run_context=run_context,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the CCL agent.")
    parser.add_argument("--config", help="Path to agent config JSON.")
    parser.add_argument("--workload", help="Path to workload spec JSON.")
    parser.add_argument(
        "--provider",
        default=None,
        help="LLM provider (ollama, openai, fireworks, claude, gemini, openai-compatible).",
    )
    parser.add_argument("--model", default=None, help="LLM model name.")
    parser.add_argument("--dry-run", action="store_true", help="Use simulated microbench/workload runs.")
    parser.add_argument(
        "--simulate-workload",
        action="store_true",
        help="Simulate workload/training/nccl-tests runs while keeping microbench real.",
    )
    parser.add_argument(
        "--simulate-execution",
        action="store_true",
        help="Alias for --simulate-workload.",
    )
    parser.add_argument("--artifacts-root", default=None, help="Root directory for run artifacts.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed override.")
    parser.add_argument(
        "--env-file",
        default=".env.local",
        help="Optional env file to load API keys from (default: .env.local).",
    )
    args = parser.parse_args()

    load_env_file(args.env_file)

    agent_config = load_agent_config(args.config) if args.config else default_agent_config()
    if args.artifacts_root:
        agent_config.artifacts_root = args.artifacts_root
    if args.seed is not None:
        agent_config.seed = args.seed
    workload = load_workload_spec(args.workload) if args.workload else _default_workload()

    run_context = create_run_context(agent_config.artifacts_root, dry_run=args.dry_run, seed=agent_config.seed)
    if "CCL_LLM_TRACE_DIR" not in os.environ:
        os.environ["CCL_LLM_TRACE_DIR"] = artifact_path(run_context, "llm")
    config_snapshot_path = artifact_path(run_context, "config_snapshot.json")
    write_json(config_snapshot_path, config_to_dict(agent_config))
    run_context.config_snapshot_path = config_snapshot_path
    write_json(artifact_path(run_context, "run_context.json"), run_context.__dict__)
    simulate_workload = args.simulate_workload or args.simulate_execution
    trace_writer = TraceWriter(run_context.run_id, run_context.artifacts_dir)
    trace_emitter = TraceEmitterWriter(trace_writer)
    tools = build_tools(
        agent_config,
        run_context=run_context,
        dry_run=args.dry_run,
        simulate_workload=simulate_workload,
    )
    memory = MemoryStore(agent_config.memory, run_context=run_context)
    rag = RagStore(agent_config.rag)
    provider = args.provider or agent_config.llm.provider
    model = args.model or agent_config.llm.model
    llm = create_llm_client(provider, model)
    llm = TracedLLMClient(llm, trace_emitter, run_context.artifacts_dir, run_context.run_id)
    tools = InstrumentedToolSuite(tools, trace_emitter, run_context.run_id)
    try:
        agent = CCLAgent(
            config=agent_config,
            tools=tools,
            memory=memory,
            rag=rag,
            llm=llm,
            run_context=run_context,
            trace=trace_emitter,
        )
        state = agent.tune(workload)
    finally:
        trace_writer.close()

    if state.best_record:
        logger.info("Best iteration time (ms): %.3f", state.best_record.metrics.iteration_time_ms)
    else:
        logger.info("No tuning records produced.")


if __name__ == "__main__":
    main()
