#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)

python3 -m src.main \
  --config "${ROOT_DIR}/configs/sim_long.json" \
  --workload "${ROOT_DIR}/workload/benchmarks/torch-demo.json" \
  --dry-run \
  --simulate-workload
