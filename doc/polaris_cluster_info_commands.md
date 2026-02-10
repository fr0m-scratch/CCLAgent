# Polaris Cluster Info and Command Guide (Production)

Last updated: **2026-02-10 (UTC)**

This file records:

1. Commands used to collect Polaris facts.
2. Facts observed in this workspace.
3. Official ALCF constraints that must be reflected in production job scripts.

## 1) Local environment snapshot commands

```bash
hostname
date -u +"%Y-%m-%d %H:%M:%S UTC"
uname -a
lscpu | rg "Model name|CPU\(s\)|Socket\(s\)|Core\(s\) per socket|Thread\(s\) per core|NUMA node\(s\)"
free -h
which qsub qstat pbsnodes mpiexec mpirun
qstat -Q
```

Observed in this session:

- Host: `polaris-login-02`
- Login node CPU: `AMD EPYC 7713 64-Core Processor`
- Visible CPUs on login node: `256`
- Login node memory: `~503 GiB`
- PBS tools available: `qsub`, `qstat`, `pbsnodes`

## 2) Required PBS fields on Polaris

Local errors observed when omitted:

- `Account_Name is required to be set.`
- `Resource: filesystems is required to be set.`

Minimum required lines for batch jobs:

```bash
#PBS -A <PROJECT>
#PBS -l filesystems=home:eagle
```

## 3) Queue and policy checks

Local queue listing command:

```bash
qstat -Q
```

Official queue policy reference:

- ALCF running jobs guide: <https://docs.alcf.anl.gov/polaris/running-jobs/>

Official notes currently relevant:

1. `debug` queue is intended for small/short jobs.
2. Production queue policy can change; always verify with `qstat -Qf <queue>` before large submissions.
3. ALCF announced a new `capacity` queue effective **2026-02-09** (see system updates page).

System updates reference:

- <https://docs.alcf.anl.gov/polaris/system-updates/>

## 4) Compute-node GPU probe (must run inside job)

Login nodes may not expose GPUs; run a probe job:

```bash
cat > /tmp/polaris_gpu_probe.pbs <<'EOS'
#!/bin/bash -l
#PBS -N polaris_gpu_probe
#PBS -q debug
#PBS -l select=1:ncpus=256:ngpus=4
#PBS -l walltime=00:05:00
#PBS -l filesystems=home:eagle
#PBS -A <PROJECT>
#PBS -j oe

set -euo pipefail
cd "$PBS_O_WORKDIR"
hostname
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader
EOS

qsub /tmp/polaris_gpu_probe.pbs
```

Observed in this workspace output (`polaris_gpu_probe.o6906112`):

- GPU: `NVIDIA A100-SXM4-40GB` x4
- Driver: `570.124.06`

## 5) MPI + GPU baseline requirements

For GPU MPI jobs, set:

```bash
export MPICH_GPU_SUPPORT_ENABLED=1
```

Reference:

- ALCF GPU guide: <https://docs.alcf.anl.gov/polaris/running-jobs/using-gpus/>

Use compute-node `mpiexec` launch inside PBS job context.

## 6) Internet/proxy note for compute nodes

If a compute-node workflow needs internet access (package install, fetch), ALCF documents proxy usage. Check current values before job runs:

- <https://docs.alcf.anl.gov/polaris/>

## 7) Production preflight block (copy/paste)

```bash
echo "=== Host ==="; hostname; date -u +"%Y-%m-%d %H:%M:%S UTC"
echo "=== PBS queues ==="; qstat -Q
echo "=== Required tools ==="; which qsub qstat pbsnodes mpiexec || true
echo "=== Module list ==="; module list 2>&1 || true
echo "=== Node state summary ==="
pbsnodes -aSj | awk 'NR>3 && $1 !~ /^-+/ {state[$2]++; total++} END{print "total_nodes=" total; for (s in state) print s "=" state[s]}' | sort
```

## 8) Evaluation implications

1. Every benchmark method (native NCCL / AutoCCL / CCLAgent) must use the same queue/resource shape.
2. GPU identity and driver info must be captured per run package.
3. Job scripts must include required `-A` and `filesystems` lines, or submission will fail.

