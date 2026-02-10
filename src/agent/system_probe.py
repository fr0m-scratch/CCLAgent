from __future__ import annotations

import re
import shutil
import socket
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..observability import build_topology_signature
from ..trace import NullTraceEmitter, TraceEmitter
from ..types import WorkloadSpec
from ..utils import artifact_path, write_json


@dataclass
class ProbeCommand:
    name: str
    command: List[str]


class SystemProbeCollector:
    """Collect host/GPU/network probe data with simulation fallback."""

    def __init__(
        self,
        run_context: Optional[Any] = None,
        trace: Optional[TraceEmitter] = None,
        timeout_sec: int = 4,
    ) -> None:
        self.run_context = run_context
        self.trace = trace or NullTraceEmitter()
        self.timeout_sec = timeout_sec

    def collect(self, workload: WorkloadSpec, *, simulate: bool = False) -> Dict[str, Any]:
        commands = [
            ProbeCommand("hostname", ["hostname"]),
            ProbeCommand("nvidia_smi_list", ["nvidia-smi", "-L"]),
            ProbeCommand("nvidia_smi_topology", ["nvidia-smi", "topo", "-m"]),
            ProbeCommand("ibstat", ["ibstat"]),
        ]
        results: List[Dict[str, Any]] = []
        for spec in commands:
            result = self._run_probe(spec, workload, simulate=simulate)
            results.append(result)
            self._trace_command(result)
        summary = self._build_summary(workload, results)
        mode = self._derive_mode(results, simulate=simulate)
        payload = {
            "schema_version": "1.0",
            "mode": mode,
            "commands": results,
            "summary": summary,
        }
        if self.run_context:
            write_json(artifact_path(self.run_context, "offline", "system_probe.json"), payload)
            write_json(artifact_path(self.run_context, "offline", "system_probe_summary.json"), summary)
            self.trace.event(
                run_id=self.run_context.run_id,
                phase="offline",
                step=None,
                actor="agent",
                type="offline.system_probe.summary",
                payload={"mode": mode, "summary": summary},
            )
        return payload

    def _run_probe(self, spec: ProbeCommand, workload: WorkloadSpec, *, simulate: bool) -> Dict[str, Any]:
        if simulate:
            return self._simulated_result(spec, workload, reason="dry_run")
        binary = spec.command[0]
        if shutil.which(binary) is None:
            return self._simulated_result(spec, workload, reason=f"missing_binary:{binary}")
        try:
            result = subprocess.run(
                spec.command,
                check=False,
                capture_output=True,
                text=True,
                timeout=self.timeout_sec,
            )
            status = "ok" if result.returncode == 0 else "error"
            return {
                "name": spec.name,
                "command": spec.command,
                "status": status,
                "simulated": False,
                "returncode": result.returncode,
                "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip(),
            }
        except subprocess.TimeoutExpired:
            return self._simulated_result(spec, workload, reason="timeout")
        except Exception as exc:
            return self._simulated_result(spec, workload, reason=f"exception:{exc}")

    def _simulated_result(self, spec: ProbeCommand, workload: WorkloadSpec, *, reason: str) -> Dict[str, Any]:
        stdout = self._simulated_stdout(spec.name, workload)
        return {
            "name": spec.name,
            "command": spec.command,
            "status": "simulated",
            "simulated": True,
            "reason": reason,
            "returncode": 0,
            "stdout": stdout,
            "stderr": "",
        }

    def _simulated_stdout(self, name: str, workload: WorkloadSpec) -> str:
        gpus = workload.gpus_per_node or int(workload.metadata.get("gpus_per_node", 8))
        gpu_type = (
            workload.metadata.get("gpu_type")
            or workload.metadata.get("gpu")
            or self._infer_gpu_from_topology(workload.topology)
            or "NVIDIA A40"
        )
        hostname = socket.gethostname()
        if name == "hostname":
            return hostname
        if name == "nvidia_smi_list":
            return "\n".join(
                [f"GPU {idx}: {gpu_type} (UUID: GPU-SIM-{idx:02d})" for idx in range(gpus)]
            )
        if name == "nvidia_smi_topology":
            head = "GPU0\tGPU1\tGPU2\tGPU3\tGPU4\tGPU5\tGPU6\tGPU7\tCPU Affinity"
            row = "GPU0\tX\tPIX\tPIX\tPIX\tNODE\tNODE\tNODE\tNODE\t0-31"
            return "\n".join([head, row, "Legend:\tPIX=PCIe switch\tNODE=remote NUMA"])
        if name == "ibstat":
            network = workload.metadata.get("network") or "InfiniBand"
            return (
                "CA 'mlx5_0'\n"
                "  Port 1:\n"
                "    State: Active\n"
                "    Physical state: LinkUp\n"
                "    Rate: 100 Gb/sec\n"
                f"    Link layer: {network}\n"
            )
        return ""

    def _build_summary(self, workload: WorkloadSpec, commands: List[Dict[str, Any]]) -> Dict[str, Any]:
        by_name = {item.get("name"): item for item in commands}
        gpu_stdout = str(by_name.get("nvidia_smi_list", {}).get("stdout", ""))
        gpu_lines = [line for line in gpu_stdout.splitlines() if line.strip().startswith("GPU ")]
        discovered_gpu_count = len(gpu_lines) if gpu_lines else None
        gpu_type = None
        if gpu_lines:
            match = re.search(r"GPU \d+:\s*(.+?)\s*\(UUID:", gpu_lines[0])
            if match:
                gpu_type = match.group(1).strip()
        if not gpu_type:
            gpu_type = (
                workload.metadata.get("gpu_type")
                or workload.metadata.get("gpu")
                or self._infer_gpu_from_topology(workload.topology)
            )
        ib_stdout = str(by_name.get("ibstat", {}).get("stdout", ""))
        link_rate = None
        rate_match = re.search(r"Rate:\s*([^\n]+)", ib_stdout)
        if rate_match:
            link_rate = rate_match.group(1).strip()
        gpus_per_node = discovered_gpu_count or workload.gpus_per_node or workload.metadata.get("gpus_per_node")
        if gpus_per_node is None:
            gpus_per_node = 8
        nic_count = workload.metadata.get("nic_count")
        if nic_count is None:
            nic_count = 1
        topo_text = str(by_name.get("nvidia_smi_topology", {}).get("stdout", ""))
        topo_sig = build_topology_signature(
            topo_text=topo_text,
            ib_text=ib_stdout,
            fallback_gpu_count=int(gpus_per_node),
            fallback_nic_count=int(nic_count),
        )
        return {
            "probe_collected": True,
            "nodes": workload.nodes,
            "gpus_per_node": int(gpus_per_node),
            "gpu_count_total": int(gpus_per_node) * int(workload.nodes),
            "gpu_type": gpu_type,
            "topology": workload.topology,
            "network": workload.metadata.get("network") or "ib",
            "network_link_rate": link_rate,
            "nic_count": int(nic_count),
            "topology_signature": topo_sig.to_dict(),
            "commands": [
                {
                    "name": item.get("name"),
                    "status": item.get("status"),
                    "simulated": bool(item.get("simulated")),
                }
                for item in commands
            ],
        }

    def _derive_mode(self, commands: List[Dict[str, Any]], *, simulate: bool) -> str:
        if simulate:
            return "simulated"
        simulated = [item for item in commands if item.get("simulated")]
        if not simulated:
            return "real"
        if len(simulated) == len(commands):
            return "simulated"
        return "mixed"

    def _trace_command(self, result: Dict[str, Any]) -> None:
        if not self.run_context:
            return
        self.trace.event(
            run_id=self.run_context.run_id,
            phase="offline",
            step=None,
            actor="tool",
            type="tool.system_probe.command",
            payload={
                "name": result.get("name"),
                "command": result.get("command"),
                "status": result.get("status"),
                "simulated": result.get("simulated"),
                "reason": result.get("reason"),
            },
            status="ok" if result.get("status") in ("ok", "simulated") else "error",
        )

    def _infer_gpu_from_topology(self, topology: str) -> Optional[str]:
        text = (topology or "").lower()
        if "a40" in text:
            return "NVIDIA A40"
        if "a100" in text:
            return "NVIDIA A100"
        if "h100" in text:
            return "NVIDIA H100"
        if "l40" in text:
            return "NVIDIA L40"
        return None
