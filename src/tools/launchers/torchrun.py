from __future__ import annotations

import os
from typing import Dict, List

from ...types import WorkloadSpec


def build_torchrun_command(workload: WorkloadSpec) -> List[str]:
    args = workload.launcher_args or {}
    nproc_per_node = str(args.get("nproc_per_node") or workload.gpus_per_node or 1)
    nnodes = str(args.get("nnodes") or workload.nodes or 1)
    node_rank = str(args.get("node_rank") or os.getenv("NODE_RANK") or 0)
    master_addr = str(args.get("master_addr") or os.getenv("MASTER_ADDR") or "127.0.0.1")
    master_port = str(args.get("master_port") or os.getenv("MASTER_PORT") or "29500")
    return [
        "torchrun",
        "--nproc_per_node",
        nproc_per_node,
        "--nnodes",
        nnodes,
        "--node_rank",
        node_rank,
        "--master_addr",
        master_addr,
        "--master_port",
        master_port,
    ] + workload.command
