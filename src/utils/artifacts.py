from __future__ import annotations

import os
import platform
import socket
import subprocess
import sys
import uuid
from datetime import datetime
from typing import Any, Dict

from ..types import RunContext
from .json_utils import write_json


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
        return result.stdout.strip()
    except Exception:
        return None


def _host_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split("\n")[0],
    }
    cuda = os.environ.get("CUDA_VERSION") or os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda:
        info["cuda_hint"] = cuda
    return info


def create_run_context(artifacts_root: str, dry_run: bool, seed: int) -> RunContext:
    run_id = str(uuid.uuid4())
    started_at_iso = datetime.utcnow().isoformat() + "Z"
    artifacts_dir = os.path.join(artifacts_root, run_id)
    os.makedirs(os.path.join(artifacts_dir, "steps"), exist_ok=True)
    os.makedirs(os.path.join(artifacts_dir, "offline"), exist_ok=True)
    os.makedirs(os.path.join(artifacts_dir, "online"), exist_ok=True)
    os.makedirs(os.path.join(artifacts_dir, "postrun"), exist_ok=True)
    os.makedirs(os.path.join(artifacts_dir, "logs"), exist_ok=True)
    context = RunContext(
        run_id=run_id,
        started_at_iso=started_at_iso,
        artifacts_dir=artifacts_dir,
        dry_run=dry_run,
        seed=seed,
        git_commit=_git_commit(),
        host_info=_host_info(),
    )
    write_json(os.path.join(artifacts_dir, "run_context.json"), context.__dict__)
    return context


def artifact_path(run_context: RunContext, *parts: str) -> str:
    return os.path.join(run_context.artifacts_dir, *parts)
