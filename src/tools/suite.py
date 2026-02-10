from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..types import RunContext
from .autoccl import AutoCCLBridge
from .config_compiler import ConfigCompiler
from .ext_net import ExtNetBridge
from .ext_tuner import ExtTunerBridge
from .metrics import MetricsCollector
from .microbench import MicrobenchRunner
from .nccl_debug import NcclDebugTool
from .nccl import NCCLInterface
from .nccltest import NcclTestRunner
from .numeric_search import NumericSearchTool
from .profiler import ProfilerCollector
from .debug_playbook import DebugPlaybook
from .sla import SLAEnforcer
from .training import TrainingJobRunner
from .workload import WorkloadRunner


@dataclass
class ToolSuite:
    microbench: MicrobenchRunner
    workload: WorkloadRunner
    metrics: MetricsCollector
    sla: SLAEnforcer
    compiler: ConfigCompiler
    nccl: NCCLInterface
    nccltest: Optional[NcclTestRunner] = None
    training: Optional[TrainingJobRunner] = None
    autoccl: Optional[AutoCCLBridge] = None
    ext_tuner: Optional[ExtTunerBridge] = None
    ext_net: Optional[ExtNetBridge] = None
    numeric_search: Optional[NumericSearchTool] = None
    nccl_debug: Optional[NcclDebugTool] = None
    profiler: Optional[ProfilerCollector] = None
    debug_playbook: Optional[DebugPlaybook] = None
    run_context: Optional[RunContext] = None
