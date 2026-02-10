from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from ..observability import NCCLDebugConfig, NCCLDebugParser
from ..types import RunContext
from ..utils import artifact_path, write_json


@dataclass
class NcclDebugToolConfig:
    enabled: bool = False
    level: str = "INFO"
    subsystems: List[str] = field(default_factory=lambda: ["INIT", "GRAPH", "NET", "TUNE"])
    dump_topology: bool = True
    dump_graph: bool = True


class NcclDebugTool:
    def __init__(self, config: NcclDebugToolConfig, run_context: Optional[RunContext] = None) -> None:
        self.config = config
        self.run_context = run_context
        self.parser = NCCLDebugParser()

    def env_overrides(self, *, step: int, artifact_subdir: str = "steps") -> Dict[str, str]:
        if not self.config.enabled:
            return {}
        topo_dump = None
        graph_dump = None
        if self.run_context and self.config.dump_topology:
            topo_dump = artifact_path(self.run_context, artifact_subdir, f"step_{step}_nccl_topo.xml")
        if self.run_context and self.config.dump_graph:
            graph_dump = artifact_path(self.run_context, artifact_subdir, f"step_{step}_nccl_graph.xml")
        cfg = NCCLDebugConfig(
            level=self.config.level,
            subsystems=list(self.config.subsystems),
            topo_dump_file=topo_dump,
            graph_dump_file=graph_dump,
        )
        return cfg.build_env_overrides()

    def collect_step(
        self,
        *,
        step: int,
        stdout_text: str = "",
        stderr_text: str = "",
        artifact_subdir: str = "steps",
    ) -> Dict[str, object]:
        if not self.config.enabled:
            return {
                "schema_version": "1.0",
                "enabled": False,
                "step": int(step),
                "summary": {},
            }
        merged = "\n".join([stdout_text or "", stderr_text or ""]).strip()
        summary = self.parser.parse(merged)
        payload: Dict[str, object] = {
            "schema_version": "1.0",
            "enabled": True,
            "step": int(step),
            "summary": summary,
        }
        if self.run_context:
            write_json(
                artifact_path(self.run_context, artifact_subdir, f"step_{step}_nccl_debug_summary.json"),
                payload,
            )
        return payload

    def collect_from_logs(self, *, step: int, artifact_subdir: str = "steps") -> Dict[str, object]:
        if self.run_context is None:
            return {
                "schema_version": "1.0",
                "enabled": bool(self.config.enabled),
                "step": int(step),
                "summary": {},
            }
        stdout_path = Path(artifact_path(self.run_context, artifact_subdir, f"step_{step}_stdout.log"))
        stderr_path = Path(artifact_path(self.run_context, artifact_subdir, f"step_{step}_stderr.log"))
        stdout_text = stdout_path.read_text(encoding="utf-8") if stdout_path.exists() else ""
        stderr_text = stderr_path.read_text(encoding="utf-8") if stderr_path.exists() else ""
        return self.collect_step(
            step=step,
            stdout_text=stdout_text,
            stderr_text=stderr_text,
            artifact_subdir=artifact_subdir,
        )
