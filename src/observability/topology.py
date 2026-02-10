from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
import hashlib
import re


@dataclass
class TopologySignature:
    gpu_count: int
    nic_count: int
    nvlink_matrix_hash: str
    numa_layout_hash: str
    pcie_tree_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gpu_count": self.gpu_count,
            "nic_count": self.nic_count,
            "nvlink_matrix_hash": self.nvlink_matrix_hash,
            "numa_layout_hash": self.numa_layout_hash,
            "pcie_tree_hash": self.pcie_tree_hash,
        }


def build_topology_signature(
    *,
    topo_text: str,
    ib_text: str,
    lspci_text: str = "",
    numactl_text: str = "",
    fallback_gpu_count: int = 0,
    fallback_nic_count: int = 1,
) -> TopologySignature:
    gpu_count = _count_gpus_from_topo(topo_text) or int(fallback_gpu_count or 0)
    nic_count = _count_nics_from_ib(ib_text) or int(fallback_nic_count or 1)
    nvlink_hash = _stable_hash(_extract_nvlink_region(topo_text))
    numa_hash = _stable_hash(numactl_text or topo_text)
    pcie_hash = _stable_hash(lspci_text or topo_text)
    return TopologySignature(
        gpu_count=gpu_count,
        nic_count=nic_count,
        nvlink_matrix_hash=nvlink_hash,
        numa_layout_hash=numa_hash,
        pcie_tree_hash=pcie_hash,
    )


def _count_gpus_from_topo(text: str) -> int:
    if not text:
        return 0
    # header often contains GPU0 GPU1 ...
    header = ""
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("GPU0") or "GPU0" in line:
            header = line
            break
    if not header:
        return 0
    return len(re.findall(r"GPU\d+", header))


def _count_nics_from_ib(text: str) -> int:
    if not text:
        return 0
    # ibstat-style output marks CAs.
    count = len(re.findall(r"\bCA\b\s*'", text))
    if count:
        return count
    # fallback to mlx device mentions
    count = len(re.findall(r"\bmlx\w*", text))
    return count


def _extract_nvlink_region(text: str) -> str:
    if not text:
        return ""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    matrix_lines = [line for line in lines if line.startswith("GPU")]
    return "\n".join(matrix_lines)


def _stable_hash(text: str) -> str:
    value = (text or "").encode("utf-8")
    return hashlib.md5(value).hexdigest()
