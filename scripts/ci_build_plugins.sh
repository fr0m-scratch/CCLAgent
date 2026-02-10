#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

"${ROOT_DIR}/scripts/build_plugins.sh"

BUILD_DIR="${ROOT_DIR}/cpp/build"
if [[ ! -d "${BUILD_DIR}" ]]; then
  echo "Missing build directory: ${BUILD_DIR}" >&2
  exit 1
fi

# Basic CI gate: at least one compiled library must exist.
LIB_COUNT="$(find "${BUILD_DIR}" -type f \( -name '*.so' -o -name '*.dylib' -o -name '*.dll' -o -name '*.a' \) | wc -l | tr -d ' ')"
if [[ "${LIB_COUNT}" == "0" ]]; then
  echo "No plugin libraries produced" >&2
  exit 1
fi

echo "Plugin CI build check passed (${LIB_COUNT} library artifact(s))."
