# Primitive-MicroBenchmark: AllReduce with NCCL Profiling

This repository provides a micro-benchmark for NCCL AllReduce, enhanced with a lightweight profiling plugin. Follow the steps below to set up, build, and run the benchmark in a fully self-contained and anonymous way.

---

## Prerequisites

- **MPI** (e.g. MPICH 4.2.3(suggest))  
- **CUDA Toolkit** (compatible with NCCL v2.27.6)  
- **bash** shell  

Ensure your `PATH` and `LD_LIBRARY_PATH` include MPI and CUDA binaries and libraries.

---

## 1. Build NCCL with Profiling Patch

1. In a clean clone of the NCCL repo, check out the `v2.27.6-1` tag. 
   ```bash
   git clone git@github.com:NVIDIA/nccl.git
   cd nccl
   git checkout v2.27.6-1
   ``` 
2. Copy the `CCLInsight-changes-NCCL-v2.27.6-1.patch` file from the `AF-ICSE26` directory (located next to `nccl_build.sh`) into the NCCL repository, then apply the patch:
   ```bash
   git apply CCLInsight-changes-NCCL-v2.27.6-1.patch
   ```
3. Run the NCCL build script:
   Edit `nccl_build.sh` to set `CCL_BASE`, `CUDA_HOME` and `NVCC_GENCODE`. `CCL_BASE` is the base directory of your NCCL repository clone. Then run:
   ```bash
   bash nccl_build.sh
   ```
   By default this installs into `$CCL_BASE/nccl` with profiling hooks enabled.

---

## 2. Build the Primitive Micro-Benchmark

In the `AF_ICSE26` directory:
Edit `Primitive-MicroBenchmark-AllReduce-Build.sh` to set `CCL_BASE`, `CUDA_HOME`, `NVCC_GENCODE`, `MPI_HOME` and `AF_ICSE26`. Then run:
```bash
bash Primitive-MicroBenchmark-AllReduce-Build.sh
```
This compiles:
- `AllReduce_NCCL.cu` – the benchmark kernel  
- `debug_stubs.cpp` – anonymous stubs for NCCL’s internal symbols  

The output executables are named `AllReduce_NCCL.exe`.

---

## 3. Run the Benchmark

Each cluster may require different job submission parameters (e.g., partition name, number of nodes, task layout). Please modify the script header accordingly to suit your environment.

Recommended #nodes allocation: --nodes=4 (as used in the script)

Edit `Primitive-MicroBenchmark-AllReduce-Run.sh` to set `CCL_BASE`, `AF_ICSE26`, `CUDA_HOME`, `NVCC_GENCODE`, `MPI_HOME`.

Edit `NUM_GPUS`: Total number of GPUs across all allocated nodes (used for mpirun -np)

Run the benchmark using your cluster’s job scheduler. For SLURM:
```bash
sbatch Primitive-MicroBenchmark-AllReduce-Run.sh
```

This script sweeps over message sizes, thread, chunk, channel configs, and logs per-primitive profiling data to files in your `CCLInsight_OUT_DIR`.

---

## Logging & Output

- Each MPI rank writes its results to  
  `nccl_allreduce_r-<rank>.out`  
- Files contain timestamp and per-chunk latency entries.

---

> **Note**: All instructions and file names are free of personal or organizational identifiers to support double-blind evaluation.
