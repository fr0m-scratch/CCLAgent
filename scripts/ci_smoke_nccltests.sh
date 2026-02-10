#!/usr/bin/env bash
set -euo pipefail

if ! command -v all_reduce_perf >/dev/null 2>&1; then
  echo "nccl-tests binary not found; skipping"
  exit 0
fi

NCCL_DEBUG=${NCCL_DEBUG:-WARN} all_reduce_perf -b 8 -e 1024 -f 2 >/tmp/nccltests_smoke.out

python3 - <<'PY'
from pathlib import Path
from src.tools.metrics import MetricsCollector
from src.types import MetricsConfig

raw = Path("/tmp/nccltests_smoke.out").read_text(encoding="utf-8")
collector = MetricsCollector(MetricsConfig(parse_mode="nccltests_v1"))
metrics = collector.parse(raw, parse_mode="nccltests_v1")
print({"iteration_time_ms": metrics.iteration_time_ms, "algbw_gbps": metrics.algbw_gbps, "success": metrics.success})
PY

echo "nccl-tests smoke complete"
