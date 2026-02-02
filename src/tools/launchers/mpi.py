from __future__ import annotations

import os
from typing import List

from ...types import WorkloadSpec


def build_mpi_command(workload: WorkloadSpec) -> List[str]:
    args = workload.launcher_args or {}
    np = args.get("np") or os.getenv("MPI_RANKS") or (workload.nodes * (workload.gpus_per_node or 1))
    return ["mpirun", "-np", str(np)] + workload.command
