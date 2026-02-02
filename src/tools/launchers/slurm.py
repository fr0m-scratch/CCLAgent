from __future__ import annotations

import os
from typing import List

from ...types import WorkloadSpec


def build_slurm_command(workload: WorkloadSpec) -> List[str]:
    args = workload.launcher_args or {}
    nodes = str(args.get("nodes") or os.getenv("SLURM_NNODES") or workload.nodes or 1)
    gpus_per_node = str(args.get("gpus_per_node") or os.getenv("SLURM_GPUS_ON_NODE") or workload.gpus_per_node or 1)
    return [
        "srun",
        "--nodes",
        nodes,
        "--gpus-per-node",
        gpus_per_node,
    ] + workload.command
