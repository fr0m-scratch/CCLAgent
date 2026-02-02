AutoCCL Notes (Local)
=====================

- AutoCCL uses an ext-tuner plugin to explore NCCL subspaces at runtime.
- Evaluation uses microbench + real workloads; keep configs valid and bounded.
- Candidate subspaces often fix implementation knobs (ALGO/PROTO) and search
  resource knobs (channels, threads, buffsize) within constraints.
