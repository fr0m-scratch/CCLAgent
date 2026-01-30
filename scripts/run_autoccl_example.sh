#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
AUTOCCL_ROOT="${ROOT_DIR}/tools/autoccl"
EXAMPLE_DIR="${AUTOCCL_ROOT}/ext-tuner/example/example/cuda/pytorch"
NCCL_HOME="${AUTOCCL_ROOT}/build"
TUNER_HOME="${AUTOCCL_ROOT}/ext-tuner/example/build"

if [[ ! -f "${EXAMPLE_DIR}/demo.py" ]]; then
  echo "Missing demo.py at ${EXAMPLE_DIR}/demo.py" >&2
  exit 1
fi
if [[ ! -f "${NCCL_HOME}/lib/libnccl.so" ]]; then
  echo "Missing AutoCCL NCCL build at ${NCCL_HOME}/lib/libnccl.so" >&2
  exit 1
fi
if [[ ! -f "${TUNER_HOME}/libnccl-plugin.so" ]]; then
  echo "Missing AutoCCL tuner plugin at ${TUNER_HOME}/libnccl-plugin.so" >&2
  exit 1
fi

gpu_count=8
if command -v nvidia-smi >/dev/null 2>&1; then
  gpu_count=$(nvidia-smi -L | wc -l | tr -d ' ')
fi

NPROC_PER_NODE="${NPROC_PER_NODE:-$gpu_count}"

export TUNER_MAXCHANNELS="${TUNER_MAXCHANNELS:-32}"
export TUNER_P2P_NCHANNELS="${TUNER_P2P_NCHANNELS:-2}"
export TUNER_WHITELIST_CASES_FILE="${TUNER_WHITELIST_CASES_FILE:-${EXAMPLE_DIR}/whitelistcases.txt}"
export TUNER_WHITELIST_RULES_FILE="${TUNER_WHITELIST_RULES_FILE:-${EXAMPLE_DIR}/whitelistrules.txt}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-3600}"
export TUNER_PRETRAIN_STEPS="${TUNER_PRETRAIN_STEPS:-360}"
export TUNER_TRAIN_STEPS="${TUNER_TRAIN_STEPS:-240}"
export TUNER_PROFILE_REPEAT="${TUNER_PROFILE_REPEAT:-5}"
export TUNER_COORDINATOR="${TUNER_COORDINATOR:-localhost:12449}"
export TUNER_WORLDSIZE="${TUNER_WORLDSIZE:-$NPROC_PER_NODE}"
export NCCL_TUNER_PLUGIN="${NCCL_TUNER_PLUGIN:-${TUNER_HOME}/libnccl-plugin.so}"
export LD_PRELOAD="${NCCL_HOME}/lib/libnccl.so${LD_PRELOAD:+:${LD_PRELOAD}}"
export LD_LIBRARY_PATH="${NCCL_HOME}/lib:${TUNER_HOME}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

cd "${EXAMPLE_DIR}"
torchrun --nproc_per_node "${NPROC_PER_NODE}" --nnodes 1 --node_rank 0 demo.py
