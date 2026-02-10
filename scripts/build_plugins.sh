#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CPP_DIR="${ROOT_DIR}/cpp"
BUILD_DIR="${CPP_DIR}/build"

if [[ ! -d "${CPP_DIR}" ]]; then
  echo "cpp directory not found at ${CPP_DIR}" >&2
  exit 1
fi

if command -v cmake >/dev/null 2>&1; then
  cmake -S "${CPP_DIR}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release
  cmake --build "${BUILD_DIR}" --parallel
else
  echo "cmake not found; using fallback direct compilation"
  mkdir -p "${BUILD_DIR}"
  CXX_BIN="${CXX:-c++}"
  "${CXX_BIN}" -std=c++17 -fPIC -I"${CPP_DIR}" -c "${CPP_DIR}/common/ipc_common.cc" -o "${BUILD_DIR}/ipc_common.o"
  "${CXX_BIN}" -shared -std=c++17 -fPIC -I"${CPP_DIR}" \
    "${CPP_DIR}/nccl_tuner_plugin/tuner_plugin.cc" "${BUILD_DIR}/ipc_common.o" \
    -o "${BUILD_DIR}/libcclagent_tuner_plugin.so"
  "${CXX_BIN}" -shared -std=c++17 -fPIC -I"${CPP_DIR}" \
    "${CPP_DIR}/nccl_profiler_plugin/profiler_plugin.cc" "${BUILD_DIR}/ipc_common.o" \
    -o "${BUILD_DIR}/libcclagent_profiler_plugin.so"
fi

echo "Built plugin artifacts:"
find "${BUILD_DIR}" -type f \( -name '*.so' -o -name '*.dylib' -o -name '*.dll' -o -name '*.a' \) -print || true
