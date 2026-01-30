#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
DEMO_PATH="${ROOT_DIR}/tools/autoccl/ext-tuner/example/example/cuda/pytorch/demo.py"

if [[ ! -f "${DEMO_PATH}" ]]; then
  echo "Missing demo.py at ${DEMO_PATH}" >&2
  exit 1
fi

gpu_count=1
if command -v nvidia-smi >/dev/null 2>&1; then
  gpu_count=$(nvidia-smi -L | wc -l | tr -d ' ')
fi

NPROC_PER_NODE="${NPROC_PER_NODE:-$gpu_count}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"

torchrun \
  --nproc_per_node "${NPROC_PER_NODE}" \
  --nnodes 1 \
  --node_rank 0 \
  --master_addr "${MASTER_ADDR}" \
  --master_port "${MASTER_PORT}" \
  "${DEMO_PATH}"
