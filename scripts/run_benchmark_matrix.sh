#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

REPEATS="${REPEATS:-3}"
DRY_RUN="${DRY_RUN:-1}"
SIMULATE_WORKLOAD="${SIMULATE_WORKLOAD:-1}"

IFS=',' read -r -a WORKLOADS <<< "${WORKLOADS:-workload/benchmarks/llama3.1-8b-agentic-showcase.json,workload/benchmarks/llama3.1-8b.json}"
IFS=',' read -r -a CONFIGS <<< "${CONFIGS:-configs/sim_long.json,configs/agentic_showcase_kimi.json}"

TS="$(date +%Y%m%d_%H%M%S)"
MATRIX_DIR="artifacts/matrix_${TS}"
mkdir -p "${MATRIX_DIR}"
MANIFEST="${MATRIX_DIR}/manifest.jsonl"

run_one() {
  local cfg="$1"
  local wk="$2"
  local rep="$3"

  local cmd=(python3 -m src.runner --mode headless --config "$cfg" --workload "$wk")
  if [[ "${DRY_RUN}" == "1" ]]; then
    cmd+=(--dry-run)
  fi
  if [[ "${SIMULATE_WORKLOAD}" == "1" ]]; then
    cmd+=(--simulate-workload)
  fi

  local log="${MATRIX_DIR}/run_$(basename "$cfg" .json)_$(basename "$wk" .json)_r${rep}.log"
  "${cmd[@]}" >"${log}" 2>&1

  local run_id
  run_id="$(grep -Eo 'Run completed: [^[:space:]]+' "${log}" | awk '{print $3}' | tail -n1 || true)"
  if [[ -z "${run_id}" ]]; then
    run_id="unknown"
  fi

  printf '{"config":"%s","workload":"%s","repeat":%s,"run_id":"%s","log":"%s"}\n' \
    "$cfg" "$wk" "$rep" "$run_id" "$log" >> "${MANIFEST}"
}

for cfg in "${CONFIGS[@]}"; do
  for wk in "${WORKLOADS[@]}"; do
    for ((rep=1; rep<=REPEATS; rep++)); do
      echo "[matrix] config=${cfg} workload=${wk} repeat=${rep}/${REPEATS}"
      run_one "$cfg" "$wk" "$rep"
    done
  done
done

echo "Matrix run complete. Manifest: ${MANIFEST}"
