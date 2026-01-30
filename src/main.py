from __future__ import annotations

import argparse

from .agent import CCLAgent
from .config import default_agent_config, load_agent_config, load_workload_spec
from .llm import create_llm_client
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
    NumericSearchTool,
    SLAEnforcer,
    ToolSuite,
    TrainingJobConfig,
    TrainingJobRunner,
    WorkloadRunConfig,
    WorkloadRunner,
)
from .types import WorkloadSpec
from .utils import setup_logger


logger = setup_logger("cclagent.cli")


def build_tools(config, dry_run: bool) -> ToolSuite:
    metrics = MetricsCollector()
    microbench = MicrobenchRunner(MicrobenchConfig(dry_run=dry_run))
    workload = WorkloadRunner(WorkloadRunConfig(dry_run=dry_run), metrics_parser=metrics.parse)
    sla = SLAEnforcer(max_iteration_time=config.sla_max_iteration_time)
    compiler = ConfigCompiler(config.parameter_space)
    nccl = NCCLInterface(compiler)
    nccltest = NcclTestRunner(NcclTestConfig(dry_run=dry_run))
    training = TrainingJobRunner(TrainingJobConfig(dry_run=dry_run))
    ext_tuner = ExtTunerBridge()
    autoccl = ext_tuner
    ext_net = ExtNetBridge()
    numeric_search = NumericSearchTool()
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
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the CCL agent.")
    parser.add_argument("--config", help="Path to agent config JSON.")
    parser.add_argument("--workload", help="Path to workload spec JSON.")
    parser.add_argument(
        "--provider",
        default="none",
        help="LLM provider (openai, fireworks, claude, gemini, none).",
    )
    parser.add_argument("--model", default="", help="LLM model name.")
    parser.add_argument("--dry-run", action="store_true", help="Use simulated microbench/workload runs.")
    args = parser.parse_args()

    agent_config = load_agent_config(args.config) if args.config else default_agent_config()
    workload = load_workload_spec(args.workload) if args.workload else WorkloadSpec(name="demo", command=[])

    tools = build_tools(agent_config, dry_run=args.dry_run)
    memory = MemoryStore(agent_config.memory_path)
    rag = RagStore()
    llm = create_llm_client(args.provider, args.model) if args.provider else create_llm_client("none", "")

    agent = CCLAgent(config=agent_config, tools=tools, memory=memory, rag=rag, llm=llm)
    state = agent.tune(workload)

    if state.best_record:
        logger.info("Best iteration time: %.4f", state.best_record.metrics.iteration_time)
    else:
        logger.info("No tuning records produced.")


if __name__ == "__main__":
    main()
