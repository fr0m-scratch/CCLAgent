# NCCL Tuning Anti-Patterns and Debugging Guide

## Overview

This document catalogs common mistakes in NCCL tuning and provides debugging strategies.
Anti-patterns are categorized by type: configuration mistakes, methodology errors,
and operational pitfalls. Each entry includes the pattern, why it's wrong, and the fix.

Source: NVIDIA NCCL tuning blog 2025, "Demystifying NCCL" (arXiv 2507.04786),
NCCL 2.29.1 docs, community best practices

---

## Configuration Anti-Patterns

### AP1: "Set and Forget" Environment Variables
**Pattern**: Set NCCL env vars once, never revisit after NCCL version upgrades.
**Why wrong**: NCCL's internal cost model improves across versions. An override that
helped in NCCL 2.18 may prevent a better auto-selected configuration in NCCL 2.29.
**Fix**: Periodically re-benchmark with defaults. Only keep overrides that measurably
improve performance on the current NCCL version.
**Severity**: Medium

### AP2: Copying Env Vars from Different Hardware
**Pattern**: Copy NCCL environment variables from a blog post, paper, or colleague's
cluster that uses different hardware/topology.
**Why wrong**: Optimal NCCL settings are topology-dependent. NVLink settings don't
transfer to PCIe. DGX settings don't transfer to cloud instances. IB settings
don't transfer to Ethernet.
**Fix**: Always benchmark on your specific hardware. Use others' configurations as
starting points, not final answers.
**Severity**: High

### AP3: Over-Allocating Channels for Small Messages
**Pattern**: Set NCCL_MIN_NCHANNELS=16 or higher for all workloads.
**Why wrong**: For small messages, per-channel chunk size becomes less than the 512KB
NIC FIFO buffer. The NIC sends partially filled buffers, wasting PCIe and network
bandwidth. Additionally, each channel consumes one SM.
**Fix**: Let NCCL auto-reduce channels for small messages. If overriding, ensure
minimum per-channel chunk > 512KB.
**Severity**: Medium

### AP4: Setting NCCL_NTHREADS on Modern NCCL
**Pattern**: Manually set NCCL_NTHREADS=512 or NCCL_NTHREADS=128.
**Why wrong**: Since NCCL 2.24, manual NCCL_NTHREADS settings may be ignored and can
lead to incorrect behavior. The CTA-based tuning model supersedes thread-based control.
**Fix**: Use NCCL_MIN_CTAS / NCCL_MAX_CTAS for controlling communication resource
allocation on modern NCCL versions.
**Severity**: Medium-High

### AP5: Ignoring GDR When GPU and NIC Share PCIe Switch
**Pattern**: Running without GPUDirect RDMA when the GPU and NIC are on the same
PCIe switch (PHB-level proximity).
**Why wrong**: Without GDR, every IB transfer makes an extra PCIe copy through host
memory. With GDR, the NIC reads/writes GPU memory directly. The performance
difference can be 30-50% for large messages.
**Fix**: Set NCCL_NET_GDR_LEVEL=PHB or higher. Verify with NCCL_DEBUG=INFO that
GDR is active. On NVLink platforms, GDR read is enabled by default since NCCL 2.4.2.
**Severity**: High

### AP6: Disabling Shared Memory on Cross-Socket PCIe
**Pattern**: NCCL_SHM_DISABLE=1 on a system where GPUs span multiple CPU sockets
without NVLink.
**Why wrong**: Without shared memory, NCCL falls back to direct PCIe P2P which
traverses the CPU interconnect (UPI/IF). This is poorly handled on many platforms
and can be extremely slow.
**Fix**: Keep NCCL_SHM_DISABLE=0 (default). Shared memory transport is the correct
fallback for cross-socket PCIe communication.
**Severity**: High

### AP7: Buffer Too Small with Many Channels
**Pattern**: NCCL_BUFFSIZE=1048576 (1MB) with NCCL_MAX_NCHANNELS=16.
**Why wrong**: Each channel gets BUFFSIZE/slots buffer per slot. With 1MB total and
8 slots, each slot is 128KB. Divided across 16 channels, that's 8KB per channel
per slot — far too small for efficient pipelining.
**Fix**: Either increase BUFFSIZE or decrease NCHANNELS. Keep per-channel buffer
large enough for efficient pipeline operation (at least 64KB per slot per channel).
**Severity**: Medium

### AP8: Forcing LL128 on Unverified Platforms
**Pattern**: NCCL_PROTO=LL128 without verifying hardware 128-byte atomic write support.
**Why wrong**: LL128 requires hardware to guarantee 128-byte atomic writes without
splitting or reordering. On unsupported platforms, this causes SILENT DATA CORRUPTION.
**Fix**: Never force LL128. Let NCCL auto-detect platform support. If disabling LL128
(as a safety measure), use NCCL_PROTO="^LL128".
**Severity**: Critical

---

## Methodology Anti-Patterns

### AP9: Optimizing Microbenchmarks Instead of Workloads
**Pattern**: Tune NCCL settings using nccl-tests allreduce_perf, then apply to training.
**Why wrong**: Microbenchmarks use all GPU SMs for communication (no compute overlap).
More channels and CTAs always improve benchmark results. But in real training, those
SMs are stolen from model computation, potentially slowing training overall.
**NVIDIA quote**: "Benchmark improvement ≠ workload improvement."
**Fix**: Always validate tuning changes with end-to-end training iteration time.
Use nccl-tests for initial exploration only.
**Severity**: High

### AP10: Ignoring the S-Curve Performance Profile
**Pattern**: Tuning for one message size and assuming it works across all sizes.
**Why wrong**: NCCL performance has an S-curve shape. Performance dips at message
size transition boundaries (where algorithm/protocol switches happen). A setting
optimal for 1MB may create a severe dip at 10MB if it forces the wrong protocol.
**Fix**: Profile across the full range of message sizes used by your workload.
Check for dips at powers-of-2 boundaries.
**Severity**: Medium

### AP11: Single-Metric Optimization
**Pattern**: Optimizing only communication bandwidth without monitoring compute impact.
**Why wrong**: Communication bandwidth and compute throughput compete for SM resources.
Maximizing one minimizes the other. The actual target is minimizing total iteration time.
**Fix**: Monitor both communication time AND model iteration time. The optimal
operating point is where their sum is minimized.
**Severity**: High

### AP12: Not Using End-to-End Profiling
**Pattern**: Inferring NCCL performance from model throughput changes alone.
**Why wrong**: Many factors affect model throughput. A 5% throughput drop after
changing NCCL settings might be caused by something else entirely (data loading,
GPU frequency, memory allocation).
**Fix**: Use NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=TUNING to see actual NCCL selections.
Use profiling tools (nsys, NCCL Inspector) for detailed communication timing.
**Severity**: Medium

---

## Operational Anti-Patterns

### AP13: Inconsistent Env Vars Across Nodes
**Pattern**: Different NCCL environment variables on different nodes in the cluster.
**Why wrong**: NCCL communicator initialization negotiates algorithm/protocol/channel
settings collectively. Inconsistent settings can cause deadlocks, incorrect algorithm
selection, or silent performance degradation.
**Fix**: Use a consistent job launcher (e.g., torchrun, mpirun) that propagates
environment variables uniformly to all nodes.
**Severity**: High

### AP14: IB Timeout Too Low at Scale
**Pattern**: Default or low NCCL_IB_TIMEOUT on 1000+ GPU clusters.
**Why wrong**: IB timeout = 4.096μs × 2^TIMEOUT. At large scale, fabric congestion
and adaptive routing can cause legitimate delays. If the timeout expires, the QP
enters error state → connection tears down → NCCL error.
**Fix**: NCCL_IB_TIMEOUT=22 or higher for large clusters. Test with maximum
job concurrency to verify timeout is sufficient.
**Severity**: High

### AP15: Not Checking NCCL Topology Detection
**Pattern**: Assuming NCCL correctly detected the hardware topology without verification.
**Why wrong**: NCCL may misdetect topology on non-standard platforms, virtualized
environments, or when PCIe ACS (Access Control Services) is enabled incorrectly.
**Fix**: Run with NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH at least once.
Compare ring/tree patterns against `nvidia-smi topo -m` output.
**Severity**: Medium

---

## Debugging Decision Tree

### Symptom: Communication is Slow
```
1. Check NCCL_DEBUG=INFO output for selected algorithm/protocol
   → Wrong algorithm for message size? → Adjust NCCL_ALGO
   → Wrong protocol for message size? → Adjust NCCL_PROTO
2. Check channel count in debug output
   → Too few channels for available bandwidth? → Increase NCCL_MIN_NCHANNELS
   → Too many for small messages? → Decrease NCCL_MAX_NCHANNELS
3. Check GDR status (IB only)
   → GDR not active despite local NIC? → Check NCCL_NET_GDR_LEVEL
4. Check NIC-GPU affinity
   → Non-local NIC used? → Check NCCL_NET placement, CROSS_NIC setting
```

### Symptom: Training is Slow Despite Fast Communication
```
1. Check SM utilization during communication
   → High NCCL SM usage? → Reduce NCCL_MAX_NCHANNELS / NCCL_MAX_CTAS
2. Check overlap efficiency
   → Communication on critical path? → Enable overlapping (bucket size, async)
   → Communication fast but compute slow? → Too many SMs for comm
3. Check memory usage
   → High buffer memory? → Reduce BUFFSIZE × NCHANNELS product
```

### Symptom: Communication Hangs or Timeouts
```
1. Check for deadlocks
   → SHM_DISABLE=1 without alternative transport? → Re-enable SHM
   → Mismatched collective calls across ranks? → Check application logic
2. Check IB errors
   → Timeout errors? → Increase NCCL_IB_TIMEOUT
   → Connection reset? → Check fabric health, QP state
3. Check topology consistency
   → All nodes see same topology? → Run NCCL_DEBUG=INFO on all nodes
```

### Symptom: Incorrect Results / NaN in Training
```
1. Check for LL128 data corruption
   → LL128 forced on unsupported platform? → Remove NCCL_PROTO override
   → Intermittent NaN? → Disable LL128 with NCCL_PROTO="^LL128"
2. Check for CUDA errors
   → NCCL kernel failures? → Check GPU health, ECC errors
3. Enable NCCL checks
   → NCCL_CHECKS_DISABLE=0 during debugging (catches argument errors)
```

---

## Debugging Environment Variables

### Information Gathering
```bash
# Full NCCL debug output
NCCL_DEBUG=INFO

# Targeted debug output (recommended over full INFO for large scale)
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=TUNING,ENV    # See algorithm/protocol selections
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH     # Topology detection
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=NET             # Network transport details

# Warning level only (minimal overhead)
NCCL_DEBUG=WARN

# Trace level (very verbose, performance impact)
NCCL_DEBUG=TRACE
```

### Safety Checks
```bash
# Enable argument validation (catches common API misuse)
NCCL_CHECKS_DISABLE=0

# Enable launch logging (see which kernels NCCL launches)
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=COLL
```

---

## Safety Rules Summary

1. **Always benchmark end-to-end** — never trust only nccl-tests results
2. **Keep overrides minimal** — prefer NCCL auto-tuning; override only with evidence
3. **Never force LL128** — let NCCL detect hardware support
4. **Test at scale** — settings that work at 8 GPUs may fail at 128
5. **Verify topology** — run NCCL_DEBUG=INFO at least once on each new platform
6. **Monitor both metrics** — communication bandwidth AND compute overlap
7. **Use current NCCL APIs** — NCCL_MIN_CTAS/MAX_CTAS over NCCL_NTHREADS
8. **Propagate uniformly** — same env vars on all nodes
9. **Profile transitions** — check S-curve dips at message size boundaries
10. **Document overrides** — record why each override was set and when to re-evaluate

Keywords: antipattern, debugging, safety, mistake, performance_dip, timeout, corruption,
hang, deadlock, sm_oversubscription, microbenchmark, s_curve, topology_detection,
gdr, ll128, shared_memory, ib_timeout, nccl_debug, environment_variable
