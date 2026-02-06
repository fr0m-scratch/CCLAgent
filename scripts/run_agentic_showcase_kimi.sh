#!/usr/bin/env bash
set -euo pipefail

ARTIFACTS_ROOT="${ARTIFACTS_ROOT:-artifacts}"
WORKLOAD="${1:-workload/benchmarks/llama3.1-8b-agentic-showcase.json}"

mkdir -p "${ARTIFACTS_ROOT}"
rm -rf "${ARTIFACTS_ROOT:?}/"*

python3 -m src.runner \
  --mode live \
  --config configs/agentic_showcase_kimi.json \
  --workload "${WORKLOAD}" \
  --provider fireworks \
  --model accounts/fireworks/models/kimi-k2p5 \
  --dry-run \
  --simulate-workload
