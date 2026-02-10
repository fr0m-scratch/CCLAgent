AutoCCL Notes (Local)
=====================

- AutoCCL uses an ext-tuner plugin to explore NCCL subspaces at runtime.
- Evaluation uses microbench + real workloads; keep configs valid and bounded.
- Candidate subspaces often fix implementation knobs (ALGO/PROTO) and search
  resource knobs (channels, threads, buffsize) within constraints.
- Plugin control-plane contracts should be request/response versioned with
  `req_id`, timeout/deadline fields, and explicit fallback behavior.
- Nonstop in-job tuning requires policy updates without restarting the workload;
  keep rollback path that disables overrides and restores safe defaults.
