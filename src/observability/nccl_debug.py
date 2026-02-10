from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List
import re


_NCCL_LINE = re.compile(r"NCCL\s+(?P<level>INFO|WARN|ERROR)\s+(?P<msg>.*)")
_ALGO_RE = re.compile(r"\b(?:algorithm|algo)\b\s*[:=]?\s*(?P<algo>[A-Za-z0-9_]+)", re.IGNORECASE)
_PROTO_RE = re.compile(r"\b(?:protocol|proto)\b\s*[:=]?\s*(?P<proto>[A-Za-z0-9_]+)", re.IGNORECASE)
_CHANNEL_RE = re.compile(r"\bnChannels\s*[:=]?\s*(?P<ch>\d+)")


@dataclass
class NCCLDebugConfig:
    level: str = "INFO"
    subsystems: List[str] = field(default_factory=lambda: ["INIT", "GRAPH", "TUNE", "NET"])
    topo_dump_file: str | None = None
    graph_dump_file: str | None = None

    def build_env_overrides(self) -> Dict[str, str]:
        env: Dict[str, str] = {
            "NCCL_DEBUG": str(self.level),
            "NCCL_DEBUG_SUBSYS": ",".join(self.subsystems),
        }
        if self.topo_dump_file:
            env["NCCL_TOPO_DUMP_FILE"] = self.topo_dump_file
        if self.graph_dump_file:
            env["NCCL_GRAPH_DUMP_FILE"] = self.graph_dump_file
        return env


class NCCLDebugParser:
    def parse(self, text: str) -> Dict[str, object]:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        counts = {"INFO": 0, "WARN": 0, "ERROR": 0}
        algo_hits: Dict[str, int] = {}
        proto_hits: Dict[str, int] = {}
        channel_values: List[int] = []

        for line in lines:
            line_upper = line.upper()
            for level in counts:
                if f"NCCL {level}" in line_upper:
                    counts[level] += 1
                    break

            algo_match = _ALGO_RE.search(line)
            if algo_match:
                algo = algo_match.group("algo").upper()
                algo_hits[algo] = algo_hits.get(algo, 0) + 1

            proto_match = _PROTO_RE.search(line)
            if proto_match:
                proto = proto_match.group("proto").upper()
                proto_hits[proto] = proto_hits.get(proto, 0) + 1

            channel_match = _CHANNEL_RE.search(line)
            if channel_match:
                try:
                    channel_values.append(int(channel_match.group("ch")))
                except ValueError:
                    pass

        subsys_counts = {
            "INIT": self._count_subsys(lines, "INIT"),
            "GRAPH": self._count_subsys(lines, "GRAPH"),
            "COLL": self._count_subsys(lines, "COLL"),
            "P2P": self._count_subsys(lines, "P2P"),
            "NET": self._count_subsys(lines, "NET"),
            "ENV": self._count_subsys(lines, "ENV"),
            "TUNE": self._count_subsys(lines, "TUNE"),
        }

        return {
            "schema_version": "1.0",
            "line_count": len(lines),
            "levels": counts,
            "subsystems": subsys_counts,
            "algo_hits": algo_hits,
            "proto_hits": proto_hits,
            "channel_observations": {
                "count": len(channel_values),
                "min": min(channel_values) if channel_values else None,
                "max": max(channel_values) if channel_values else None,
            },
        }

    def parse_file(self, path: str | Path) -> Dict[str, object]:
        path = Path(path)
        text = path.read_text(encoding="utf-8") if path.exists() else ""
        summary = self.parse(text)
        summary["path"] = str(path)
        return summary

    def _count_subsys(self, lines: List[str], token: str) -> int:
        token_upper = token.upper()
        return sum(1 for line in lines if token_upper in line.upper())
