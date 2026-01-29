#!/bin/bash -l

###############################################################################
# NCCL Build Script with Profiling Patch
# This script builds NCCL from source with custom profiling instrumentation.
# Intended for reproducible benchmarking in a portable, double-anonymous setup.
###############################################################################

# === User Configuration ===
export CCL_BASE=    # <-- input this 
export NCCL_COMMIT="v2.27.6-1"

# === CUDA Setup ===
export CUDA_HOME=    # <-- input this to match your CUDA install
export PATH="${CUDA_HOME}/bin:$PATH"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:$LD_LIBRARY_PATH"
export C_INCLUDE_PATH="${CUDA_HOME}/include:$C_INCLUDE_PATH"
export CPLUS_INCLUDE_PATH="${CUDA_HOME}/include:$CPLUS_INCLUDE_PATH"
export CUDACXX="${CUDA_HOME}/bin/nvcc"
export NVCC_GENCODE=   # <-- input this to match your GPU

# === NCCL Source Setup ===
NCCL_SRC_LOCATION="${CCL_BASE}/nccl"
export NCCL_SRC_LOCATION

echo "[INFO] Using NCCL repo at: ${NCCL_SRC_LOCATION}"
echo "[INFO] Target NCCL commit: ${NCCL_COMMIT}"

# === Check and Enter NCCL Directory ===
if [ ! -d "${NCCL_SRC_LOCATION}" ]; then
    echo "[ERROR] NCCL source directory not found at ${NCCL_SRC_LOCATION}."
    echo "Please clone the NCCL repository and apply the patch before building."
    exit 1
fi
pushd "${NCCL_SRC_LOCATION}" || exit 1

# === Build NCCL ===
echo "[INFO] Cleaning and building NCCL..."
make clean
make -j src.build

# === Environment Export ===
NCCL_HOME="${NCCL_SRC_LOCATION}/build"
export NCCL_HOME

echo "[INFO] Setting NCCL environment variables..."
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${NCCL_HOME}/lib"
export PATH="${PATH}:${NCCL_HOME}/include"

echo "[INFO] NCCL build complete. NCCL_HOME=${NCCL_HOME}"
popd || exit 1