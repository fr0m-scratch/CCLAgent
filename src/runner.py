"""
CCL Agent Runner with Three Operation Modes:

1. HEADLESS: Agent runs without TUI (default, existing behavior)
2. LIVE: Agent runs with real-time TUI monitoring and interaction
3. INSPECT: TUI opens to review a completed run

Usage:
    # Headless run (no TUI)
    python -m src.runner --workload workload.json
    
    # Live run with TUI
    python -m src.runner --workload workload.json --mode live
    
    # Inspect completed run
    python -m src.runner --mode inspect --run-dir artifacts/<run_id>
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import threading
import queue
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable

from .agent import CCLAgent
from .config import config_to_dict, default_agent_config, load_agent_config, load_workload_spec
from .llm import create_llm_client, TracedLLMClient
from .memory import MemoryStore
from .RAG import RagStore
from .tools import (
    DebugPlaybook,
    ConfigCompiler,
    ExtNetBridge,
    ExtTunerBridge,
    ExtTunerRuntimeConfig,
    MetricsCollector,
    MicrobenchConfig,
    MicrobenchRunner,
    NCCLInterface,
    NcclDebugTool,
    NcclDebugToolConfig,
    NcclTestConfig,
    NcclTestRunner,
    NumericSearchConfig,
    NumericSearchTool,
    ProfilerCollector,
    ProfilerConfig,
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

logger = setup_logger("cclagent.runner")


def _default_workload() -> WorkloadSpec:
    candidates = [
        Path("workload/benchmarks/llama3.1-8b-agentic-showcase.json"),
        Path("workload/benchmarks/llama3.1-8b.json"),
    ]
    for path in candidates:
        if path.exists():
            return load_workload_spec(str(path))
    return WorkloadSpec(name="demo", command=[])


# -----------------------------------------------------------------------------
# Command Queue for TUI <-> Agent Communication
# -----------------------------------------------------------------------------

@dataclass
class AgentCommand:
    """Command from TUI to agent."""
    action: str  # pause, resume, stop, skip_step, adjust_budget, etc.
    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentEvent:
    """Event from agent to TUI."""
    event_type: str  # step_start, step_complete, decision, llm_call, etc.
    step: Optional[int] = None
    payload: Dict[str, Any] = field(default_factory=dict)


class AgentBridge:
    """
    Bridge for bidirectional communication between agent and TUI.
    Thread-safe queues allow async TUI to interact with sync agent.
    """
    
    def __init__(self) -> None:
        self.command_queue: queue.Queue[AgentCommand] = queue.Queue()
        self.event_queue: queue.Queue[AgentEvent] = queue.Queue()
        self.paused = threading.Event()
        self.stopped = threading.Event()
        self.paused.set()  # Start unpaused
    
    def send_command(self, cmd: AgentCommand) -> None:
        """Called by TUI to send command to agent."""
        self.command_queue.put(cmd)
        if cmd.action == "pause":
            self.paused.clear()
        elif cmd.action == "resume":
            self.paused.set()
        elif cmd.action == "stop":
            self.stopped.set()
            self.paused.set()  # Unblock if paused
    
    def emit_event(self, event: AgentEvent) -> None:
        """Called by agent to notify TUI of events."""
        self.event_queue.put(event)
    
    def check_commands(self) -> List[AgentCommand]:
        """Called by agent to process pending commands."""
        commands = []
        while True:
            try:
                cmd = self.command_queue.get_nowait()
                commands.append(cmd)
            except queue.Empty:
                break
        return commands
    
    def wait_if_paused(self) -> bool:
        """Called by agent to wait if paused. Returns False if stopped."""
        self.paused.wait()
        return not self.stopped.is_set()
    
    def poll_events(self, timeout: float = 0.1, max_events: int = 256) -> List[AgentEvent]:
        """TUI calls this to get events from agent."""
        events: List[AgentEvent] = []
        try:
            events.append(self.event_queue.get(timeout=timeout))
        except queue.Empty:
            return events
        while len(events) < max_events:
            try:
                events.append(self.event_queue.get_nowait())
            except queue.Empty:
                break
        return events


class BridgedTraceEmitter(TraceEmitterWriter):
    """
    Emits events to both disk (via TraceWriter) and the AgentBridge (for TUI).
    """
    def __init__(self, writer: TraceWriter, bridge: AgentBridge) -> None:
        super().__init__(writer)
        self.bridge = bridge

    def emit(self, event: Any) -> None:
        # 1. Write to disk (standard behavior)
        super().emit(event)
        
        # 2. Send to TUI
        payload = {
            "trace_phase": event.phase,
            "trace_actor": event.actor,
            "trace_status": event.status,
            "trace_ts": event.ts,
        }
        if event.payload:
            payload.update(event.payload)
        if event.duration_ms:
            payload["duration_ms"] = event.duration_ms
        if event.error:
            payload["error"] = event.error
        if event.refs:
            payload["refs"] = event.refs
            
        self.bridge.emit_event(AgentEvent(
            event_type=event.type,
            step=event.step,
            payload=payload
        ))


# -----------------------------------------------------------------------------
# Tool Builder
# -----------------------------------------------------------------------------

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
    tuner_session_dir = artifact_path(run_context, config.plugins.tuner_session_dir)
    if config.plugins.enable_tuner_plugin:
        ext_tuner = ExtTunerBridge(
            ExtTunerRuntimeConfig(
                extra_env={
                    "CCL_TUNER_PLUGIN_ENABLED": "1",
                    "CCL_TUNER_SESSION_DIR": tuner_session_dir,
                }
            )
        )
    else:
        ext_tuner = ExtTunerBridge()
    autoccl = ext_tuner
    ext_net = ExtNetBridge()
    nccl_debug = NcclDebugTool(
        NcclDebugToolConfig(
            enabled=bool(config.observability.nccl_debug_enabled),
            level=str(config.observability.nccl_debug_level),
            subsystems=list(config.observability.nccl_debug_subsystems),
            dump_topology=bool(config.observability.nccl_debug_dump_topology),
            dump_graph=bool(config.observability.nccl_debug_dump_graph),
        ),
        run_context=run_context,
    )
    profiler = ProfilerCollector(
        ProfilerConfig(
            enabled=bool(config.observability.profiler_enabled or config.plugins.enable_profiler_plugin),
            session_dir=artifact_path(run_context, config.plugins.profiler_session_dir),
            dry_run=dry_run,
            poll_interval_s=float(config.observability.profiler_poll_interval_s),
            timeout_s=float(config.observability.profiler_timeout_s),
        ),
        run_context=run_context,
    )
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
        nccl_debug=nccl_debug,
        profiler=profiler,
        debug_playbook=DebugPlaybook(),
        run_context=run_context,
    )


# -----------------------------------------------------------------------------
# Run Modes
# -----------------------------------------------------------------------------

def _configure_live_runtime() -> None:
    """
    Keep live mode responsive and quiet.
    - Disable raw LLM trace stdout spam by default.
    - Raise cclagent logger levels to WARNING in-process for TUI sessions.
    """
    os.environ.setdefault("CCL_LLM_TRACE_STDOUT", "0")
    os.environ.setdefault("CCL_AGENT_LOG_LEVEL", "WARNING")

    logging.getLogger("cclagent").setLevel(logging.WARNING)
    manager = logging.Logger.manager.loggerDict
    for name in list(manager.keys()):
        if not name.startswith("cclagent"):
            continue
        target = logging.getLogger(name)
        target.setLevel(logging.WARNING)
        for handler in target.handlers:
            handler.setLevel(logging.WARNING)

def run_headless(args: argparse.Namespace) -> None:
    """Mode 1: Run agent without TUI."""
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
    
    simulate_workload = args.simulate_workload or getattr(args, "simulate_execution", False)
    trace_writer = TraceWriter(run_context.run_id, run_context.artifacts_dir)
    trace_emitter = TraceEmitterWriter(trace_writer)
    
    tools = build_tools(agent_config, run_context=run_context, dry_run=args.dry_run, simulate_workload=simulate_workload)
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
    
    print(f"\nRun completed: {run_context.run_id}")
    print(f"Artifacts: {run_context.artifacts_dir}")


def run_live(args: argparse.Namespace) -> None:
    """Mode 2: Run agent with live TUI monitoring."""
    _configure_live_runtime()

    # Import TUI only when needed
    try:
        from .tui.live_monitor import LiveAgentMonitor
    except ImportError:
        # Fall back to basic monitor if live monitor not available
        from .tui.monitor import AgentMonitor
        print("Note: Live mode not fully implemented yet. Using post-run monitor instead.")
        print("The agent will run in background while TUI monitors artifacts.")
        
        # Start agent in background thread
        bridge = AgentBridge()
        agent_thread = threading.Thread(target=_run_agent_thread, args=(args, bridge), daemon=True)
        agent_thread.start()
        
        # Run monitor TUI (will poll artifacts)
        load_env_file(args.env_file)
        app = AgentMonitor(
            artifacts_root=args.artifacts_root or "artifacts",
            run_dir=None,  # Auto-detect latest
            poll_interval=0.5,  # Faster polling for live mode
            env_file=args.env_file,
        )
        app.run()
        return
    
    # Full live mode with bidirectional communication
    bridge = AgentBridge()
    
    # Create and start agent thread
    # Daemon=True ensures it dies if main thread dies (TUI exit)
    agent_thread = threading.Thread(target=_run_agent_thread, args=(args, bridge), daemon=True)
    agent_thread.start()
    
    # Run live TUI
    app = LiveAgentMonitor(bridge=bridge, env_file=args.env_file)
    try:
        app.run()
    except KeyboardInterrupt:
        pass  # Graceful exit on Ctrl+C if not handled by TUI
    finally:
        # TUI's action_quit already sets bridge.stopped; only send if missed
        if not bridge.stopped.is_set():
            bridge.send_command(AgentCommand(action="stop", payload={"silent": True}))
        # Keep shutdown non-blocking: agent thread is daemonized and receives stop command.
        agent_thread.join(timeout=0.1)


def _run_agent_thread(args: argparse.Namespace, bridge: AgentBridge) -> None:
    """Run agent in background thread for live mode."""
    try:
        load_env_file(args.env_file)
        
        agent_config = load_agent_config(args.config) if args.config else default_agent_config()
        if args.artifacts_root:
            agent_config.artifacts_root = args.artifacts_root
        if args.seed is not None:
            agent_config.seed = args.seed
        workload = load_workload_spec(args.workload) if args.workload else _default_workload()
        
        run_context = create_run_context(agent_config.artifacts_root, dry_run=args.dry_run, seed=agent_config.seed)
        os.environ.setdefault("CCL_LLM_TRACE_STDOUT", "0")
        if "CCL_LLM_TRACE_DIR" not in os.environ:
            os.environ["CCL_LLM_TRACE_DIR"] = artifact_path(run_context, "llm")
        config_snapshot_path = artifact_path(run_context, "config_snapshot.json")
        write_json(config_snapshot_path, config_to_dict(agent_config))
        run_context.config_snapshot_path = config_snapshot_path
        write_json(artifact_path(run_context, "run_context.json"), run_context.__dict__)
        
        simulate_workload = args.simulate_workload or getattr(args, "simulate_execution", False)
        trace_writer = TraceWriter(run_context.run_id, run_context.artifacts_dir)
        
        # KEY FIX: Use BridgedTraceEmitter to send events to TUI in real-time
        trace_emitter = BridgedTraceEmitter(trace_writer, bridge)
        
        tools = build_tools(agent_config, run_context=run_context, dry_run=args.dry_run, simulate_workload=simulate_workload)
        memory = MemoryStore(agent_config.memory, run_context=run_context)
        rag = RagStore(agent_config.rag)
        provider = args.provider or agent_config.llm.provider
        model = args.model or agent_config.llm.model
        llm = create_llm_client(provider, model)
        llm = TracedLLMClient(llm, trace_emitter, run_context.artifacts_dir, run_context.run_id)
        tools = InstrumentedToolSuite(tools, trace_emitter, run_context.run_id)
        
        # Emit run started event
        bridge.emit_event(AgentEvent(
            event_type="run_started",
            payload={"run_id": run_context.run_id, "artifacts_dir": str(run_context.artifacts_dir)}
        ))
        
        try:
            agent = CCLAgent(
                config=agent_config,
                tools=tools,
                memory=memory,
                rag=rag,
                llm=llm,
                run_context=run_context,
                trace=trace_emitter,
                mailbox=bridge.command_queue, 
            )
            state = agent.tune(workload)
            
            # Emit completion
            bridge.emit_event(AgentEvent(
                event_type="run_completed",
                payload={
                    "run_id": run_context.run_id,
                    "best_time_ms": state.best_record.metrics.iteration_time_ms if state.best_record else None,
                }
            ))
        finally:
            trace_writer.close()
    except Exception as e:
        bridge.emit_event(AgentEvent(event_type="run_error", payload={"error": str(e)}))
        raise


def run_inspect(args: argparse.Namespace) -> None:
    """Mode 3: Inspect completed run with TUI."""
    from .tui.monitor import AgentMonitor
    
    load_env_file(args.env_file)
    
    run_dir = args.run_dir
    if not run_dir:
        # Auto-detect latest run
        artifacts_root = Path(args.artifacts_root or "artifacts")
        if not artifacts_root.exists():
            print(f"Error: Artifacts directory not found: {artifacts_root}")
            sys.exit(1)
        runs = sorted(artifacts_root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        runs = [r for r in runs if r.is_dir()]
        if not runs:
            print(f"Error: No runs found in {artifacts_root}")
            sys.exit(1)
        run_dir = str(runs[0])
        print(f"Auto-selected latest run: {run_dir}")
    
    app = AgentMonitor(
        artifacts_root=args.artifacts_root or "artifacts",
        run_dir=run_dir,
        poll_interval=1.0,
        env_file=args.env_file,
    )
    app.run()


# -----------------------------------------------------------------------------
# CLI Entry Point
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="CCL Agent Runner with TUI Support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Headless run (no TUI)
    python -m src.runner --workload workload.json
    
    # Live run with TUI monitoring
    python -m src.runner --workload workload.json --mode live
    
    # Inspect completed run
    python -m src.runner --mode inspect --run-dir artifacts/<run_id>
    python -m src.runner --mode inspect  # Auto-selects latest run
        """,
    )
    
    # Mode selection
    parser.add_argument(
        "--mode",
        choices=["headless", "live", "inspect"],
        default="headless",
        help="Operation mode: headless (no TUI), live (TUI + agent), inspect (TUI for completed run)",
    )
    
    # Agent configuration
    parser.add_argument("--config", help="Path to agent config JSON.")
    parser.add_argument("--workload", help="Path to workload spec JSON.")
    parser.add_argument("--provider", default=None, help="LLM provider.")
    parser.add_argument("--model", default=None, help="LLM model name.")
    parser.add_argument("--dry-run", action="store_true", help="Use simulated runs.")
    parser.add_argument("--simulate-workload", action="store_true", help="Simulate workload runs.")
    parser.add_argument("--artifacts-root", default=None, help="Root directory for artifacts.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--env-file", default=".env.local", help="Environment file.")
    
    # Inspect mode options
    parser.add_argument("--run-dir", help="Specific run directory to inspect (for inspect mode).")
    
    args = parser.parse_args()
    
    # Validate args based on mode
    if args.mode in ("headless", "live") and not args.workload:
        print(
            "Info: No --workload provided; defaulting to "
            "workload/benchmarks/llama3.1-8b-agentic-showcase.json when available."
        )
    
    # Dispatch to mode handler
    if args.mode == "headless":
        run_headless(args)
    elif args.mode == "live":
        run_live(args)
    elif args.mode == "inspect":
        run_inspect(args)


if __name__ == "__main__":
    main()
