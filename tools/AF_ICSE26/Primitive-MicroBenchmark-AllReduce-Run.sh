#!/bin/bash
#SBATCH -N 4
#SBATCH --nodelist=node03,node04,node05,node06
#SBATCH --output=CCLInsight.stdout
#SBATCH -J "CCLInsight"
#SBATCH --gpus-per-node=2

###############################################################################
# NCCL Primitive Benchmark Run Script
# This script executes a sweep of AllReduce micro-benchmarks with profiling.
# It is designed to be used as part of a reproducible, portable artifact setup.
###############################################################################

# === Environment Setup ===
export CCL_BASE=
export AF_ICSE26="$(pwd)" # <-- Set this to the root of your artifact folder

export MPI_HOME=

# CUDA configuration
export CUDA_HOME=    # <-- input this to match your CUDA install
export PATH="${CUDA_HOME}/bin:$PATH"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:$LD_LIBRARY_PATH"
export C_INCLUDE_PATH="${CUDA_HOME}/include:$C_INCLUDE_PATH"
export CPLUS_INCLUDE_PATH="${CUDA_HOME}/include:$CPLUS_INCLUDE_PATH"
export CUDACXX="${CUDA_HOME}/bin/nvcc"
export NVCC_GENCODE=   # <-- input this to match your GPU

# Compiler and linker paths
export LD_LIBRARY_PATH=${MPI_HOME}/lib:$LD_LIBRARY_PATH
export PATH=${MPI_HOME}/bin:$PATH
export C_INCLUDE_PATH=${MPI_HOME}/include:$C_INCLUDE_PATH

# NCCL build output and runtime paths
NCCL_SRC_LOCATION="$CCL_BASE/nccl"
export PATH=${CUDA_HOME}/bin:${MPI_HOME}/bin:$PATH
export LD_LIBRARY_PATH=${NCCL_SRC_LOCATION}/build/lib:${CUDA_HOME}/lib64:$LD_LIBRARY_PATH

# === Benchmark Configuration ===
export NCCL_DEBUG="WARN"
export NCCL_PROTO="Simple"
export NCCL_ALGO="TREE"

# Move into benchmark root directory
cd "$AF_ICSE26" || exit 1

export CCLInsight_OUT_DIR="$AF_ICSE26"
export ITERATION_TIME=10

export GAUGE_MIN_NTHREADS=64
export GAUGE_MAX_NTHREADS=256
export GAUGE_MIN_NCHANNELS=1
export GAUGE_MAX_NCHANNELS=4

export MIN_BUFFSIZE=2097152     # 2MB
export MAX_BUFFSIZE=8388608     # 8MB

NUM_GPUS= # <-- Set this to the total number of GPUs

# === Benchmark Sweep ===
for ((itr = 0; itr < ITERATION_TIME; itr++)); do
  for ((buff_size = MIN_BUFFSIZE; buff_size <= MAX_BUFFSIZE; buff_size *= 2)); do
    for ((nch = GAUGE_MIN_NCHANNELS; nch <= GAUGE_MAX_NCHANNELS; nch *= 2)); do
      for mode in AllReduce; do
        for ((nth = GAUGE_MIN_NTHREADS; nth <= GAUGE_MAX_NTHREADS; nth *= 2)); do

          export GAUGE_NCHANNELS=$nch
          export GAUGE_MODE=$mode
          export NCCL_MIN_NCHANNELS=$nch
          export NCCL_MAX_NCHANNELS=$nch
          export NCCL_NTHREADS=$nth
          export GAUGE_ITERATION=$itr
          export NCCL_BUFFSIZE=$buff_size
          export GAUGE_MESSAGE_SIZE=1  # KB baseline

          # Run initial test at 1KB message size
          $MPI_HOME/bin/mpirun -np ${NUM_GPUS} $AF_ICSE26/Primitive-MicroBenchmark/${mode}_NCCL.exe

          # Sweep over larger message sizes
          for ((msize = 1024; msize <= 1024 * 1024; msize *= 2)); do
            export GAUGE_MESSAGE_SIZE=$msize
            $MPI_HOME/bin/mpirun -np ${NUM_GPUS} $AF_ICSE26/Primitive-MicroBenchmark/${mode}_NCCL.exe
          done

        done
      done
    done
  done
done
