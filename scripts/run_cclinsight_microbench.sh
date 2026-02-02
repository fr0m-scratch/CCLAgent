#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
AF_DIR="${ROOT_DIR}/tools/AF_ICSE26"

OUTPUT_DIR="${CCLINSIGHT_OUT_DIR:-${ROOT_DIR}/artifacts/cclinsight}"
mkdir -p "${OUTPUT_DIR}"

RUN_SCRIPT="${AF_DIR}/Primitive-MicroBenchmark-AllReduce-Run.sh"
if [[ -f "${RUN_SCRIPT}" ]]; then
  bash "${RUN_SCRIPT}" || true
fi

SUMMARY_JSON="${CCLINSIGHT_JSON:-${OUTPUT_DIR}/cclinsight_summary.json}"
if [[ -f "${SUMMARY_JSON}" ]]; then
  cat "${SUMMARY_JSON}"
  exit 0
fi

# Fallback: emit a minimal JSON envelope derived from env hints.
PARAM_LIST=${CCLAGENT_PARAM_LIST:-""}
IFS=',' read -r -a PARAMS <<< "${PARAM_LIST}"

json_params="[]"
if [[ ${#PARAMS[@]} -gt 0 ]]; then
  json_params=$(printf '%s\n' "${PARAMS[@]}" | awk 'NF {printf "{\"name\":\"%s\",\"score\":0.5,\"reason\":\"env_fallback\",\"evidence\":{}}\n", $0}' | paste -sd, -)
  json_params="[${json_params}]"
fi

cat <<EOF
{"important_params": ${json_params}, "signals": [], "note": "fallback_summary"}
EOF
