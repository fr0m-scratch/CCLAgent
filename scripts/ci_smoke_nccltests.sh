#!/usr/bin/env bash
set -euo pipefail

if ! command -v all_reduce_perf >/dev/null 2>&1; then
  echo "nccl-tests binary not found; skipping"
  exit 0
fi

all_reduce_perf -b 8 -e 1024 -f 2 >/tmp/nccltests_smoke.out

echo "nccl-tests smoke complete"
