AutoCCL Evaluation Workloads
============================

This catalog mirrors the AutoCCL paper's evaluation setup:
- Microbenchmarks on 2-node (16x A40, NVLink intra-node, 2x 400Gbps IB) and
  4-node (32x A40, PCIe intra-node, 100Gbps IB) clusters.
- End-to-end training jobs on three LLMs and one vision model.

The JSON specs here describe the four training workloads used in evaluation.
Populate `command` and `env` with your launcher details (torchrun, mpirun,
slurm, etc.) before running for real.

The file `torch-demo.json` is a local torchrun example that drives the same
AutoCCL demo workload through the CCL Agent (offline + online stages).
