Workload Catalog
================

This folder holds workload specs consumed by `load_workload_spec`. The JSON schema
matches `WorkloadSpec` and supports a `metadata` object for additional context.

Subfolders
----------
- benchmarks: Workloads mirroring the AutoCCL evaluation models.

Usage
-----
- Dry run:
  python3 -m src.main --workload workload/benchmarks/phi2-2b.json --dry-run
- Real run: fill `command` and `env` in the JSON spec with your launcher.
