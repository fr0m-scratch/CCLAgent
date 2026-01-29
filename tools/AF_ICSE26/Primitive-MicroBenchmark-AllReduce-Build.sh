#!/bin/bash -l

# Set environment variables

export CCL_BASE=

export CUDA_HOME=    # <-- input this to match your CUDA install
export PATH="${CUDA_HOME}/bin:$PATH"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:$LD_LIBRARY_PATH"
export C_INCLUDE_PATH="${CUDA_HOME}/include:$C_INCLUDE_PATH"
export CPLUS_INCLUDE_PATH="${CUDA_HOME}/include:$CPLUS_INCLUDE_PATH"
export CUDACXX="${CUDA_HOME}/bin/nvcc"
export NVCC_GENCODE=   # <-- input this to match your GPU

export MPI_HOME=
# Update to include the correct path for MPI library paths
export LD_LIBRARY_PATH=${MPI_HOME}/lib:$LD_LIBRARY_PATH
export PATH=${MPI_HOME}/bin:$PATH
export C_INCLUDE_PATH=${MPI_HOME}/include:$C_INCLUDE_PATH

export AF_ICSE26="$(pwd)" # <-- Set this to the root of your artifact folder

Primitive_MicroBenchmark="${AF_ICSE26}/Primitive-MicroBenchmark"

# NCCL source location
NCCL_SRC_LOCATION="$CCL_BASE/nccl"

g++ -std=c++11 -fPIC \
  -I"${NCCL_SRC_LOCATION}/build/include" \
  -I"${NCCL_SRC_LOCATION}/src/include" \
  -I"${NCCL_SRC_LOCATION}/src/include/plugin" \
  ${Primitive_MicroBenchmark}/debug_stubs.cpp -c -o ${Primitive_MicroBenchmark}/debug_stubs.o

for mode in AllReduce; do
    # Use proper variable expansion and quoting in the command
    nvcc "$NVCC_GENCODE" -ccbin g++ \
        -I"${NCCL_SRC_LOCATION}/build/include" \
        -I"${MPI_HOME}/include" \
        -I"${NCCL_SRC_LOCATION}/src/include" \
        -I"${NCCL_SRC_LOCATION}/src/include/plugin" \
        -L"${NCCL_SRC_LOCATION}/build/lib" \
        -L"${CUDA_HOME}/lib64" \
        -L"${MPI_HOME}/lib" \
        ${Primitive_MicroBenchmark}/debug_stubs.o \
        -lnccl -lcudart -lmpi \
        "${Primitive_MicroBenchmark}/${mode}_NCCL.cu" \
        -o "${Primitive_MicroBenchmark}/${mode}_NCCL.exe"
    # Verification of the output
    if [ -f "${Primitive_MicroBenchmark}/${mode}_NCCL.exe" ]; then
        echo "Compilation successful. Output file: ${Primitive_MicroBenchmark}/${mode}_NCCL.exe"
    else
        echo "Compilation failed."
    fi
done