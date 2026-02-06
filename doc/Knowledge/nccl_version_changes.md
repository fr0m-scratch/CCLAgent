# NCCL Version History and Feature Changes

## Overview

NCCL evolves rapidly. New versions introduce algorithms, protocols, APIs, and tuning
parameters that can change optimal configurations. This document tracks significant
changes across NCCL versions relevant to auto-tuning, highlighting new capabilities,
deprecated parameters, and behavioral changes.

Source: NCCL release notes, NVIDIA blogs, NCCL 2.29.1 docs

---

## NCCL 2.14 - 2.16 (2022-2023)

### Key Changes
- Introduction of basic NVLS (NVLink SHARP) support for NVSwitch systems
- Improved NVLink topology detection for DGX A100
- Better handling of heterogeneous NVLink/PCIe topologies

### Tuning Impact
- NVLS provides hardware-accelerated reductions on NVSwitch
- Auto-tuning begins to prefer NVLS where available
- No new user-facing tuning parameters

---

## NCCL 2.17 (2023)

### Key Changes
- **NVLS enabled by default** on supported NVSwitch systems
- NVLink SHARP algorithm for AllReduce
- Improved cost model for algorithm selection

### Tuning Impact
- Systems with NVSwitch see immediate benefit without configuration changes
- `NCCL_NVLS_ENABLE=0` to disable if memory-constrained
- AllReduce performance on NVSwitch systems improves significantly

---

## NCCL 2.18 (2023)

### Key Changes
- Enhanced NVLS algorithm coverage
- Improved multi-node auto-tuning
- Better CollNet integration

### Tuning Impact
- NVLS now handles more collective types
- Multi-node cost model more accurate
- Fewer manual overrides needed for standard topologies

---

## NCCL 2.19 (2024)

### Key Changes
- **NVLSTree algorithm**: NVLS for intra-node + Tree for inter-node
- Better CollNet support for ReduceScatter and AllGather
- Improved P2P performance on NVLink

### Tuning Impact
- NVLSTree combines benefits of hardware offload (intra) and Tree (inter)
- Available algorithm set: Ring, Tree, CollnetDirect, CollnetChain, NVLS, NVLSTree
- Auto-selection considers NVLSTree for multi-node configurations

---

## NCCL 2.21 (2024)

### Key Changes
- **Per-function algorithm/protocol control**
- Improved cost model accuracy across message size ranges
- Better handling of aggregated collectives

### New Syntax
```bash
# Per-function control (introduced here, refined in 2.24)
NCCL_ALGO="allreduce:tree;broadcast:ring"
NCCL_PROTO="allreduce:ll128;broadcast:simple"
```

### Tuning Impact
- Can now optimize different collective operations independently
- Particularly useful for workloads with mixed TP (Tree) and DP (Ring) patterns
- Cost model improvements reduce need for manual algorithm overrides

---

## NCCL 2.24 (2025)

### Key Changes
- **Full per-function per-protocol control syntax**
- **Dynamic CTA scheduling** — NCCL adapts CTA count per operation
- **Tuner Plugin v4 API** — external tuner plugins get richer context
- Exclude syntax for algorithm/protocol selection

### New Syntax
```bash
# Exclude syntax
NCCL_ALGO="^tree"              # All algorithms except Tree
NCCL_PROTO="^LL128"            # All protocols except LL128

# Full per-function control
NCCL_ALGO="allreduce:tree,collnetdirect;reducescatter:ring"
NCCL_PROTO="allreduce:ll128,simple;broadcast:simple"
```

### CTA Control
```bash
NCCL_MIN_CTAS=1                # Minimum CTAs per operation
NCCL_MAX_CTAS=16               # Maximum CTAs per operation
# NCCL dynamically selects within this range based on operation size and concurrency
```

### Tuner Plugin v4
External tuner plugins (like AutoCCL's ext-tuner) can now:
- Receive message size, collective type, data type, reduction op
- Return algorithm, protocol, channel count, CTA count recommendations
- Integrate with NCCL's internal cost model

### Tuning Impact
- NCCL_NTHREADS may be ignored; prefer CTA-based control
- Exclude syntax enables safe disabling of problematic protocols
- Dynamic CTA scheduling reduces need for manual CTA management
- Per-function control enables topology-aware mixed strategies

---

## NCCL 2.27 (2025)

### Key Changes — Major Release
- **SHARP support for both NVLink and InfiniBand fabrics**
- **AllGather/ReduceScatter SHARP** (preferred over AllReduce for overlap)
- **SM usage reduced from 16+ to 6** with SHARP
- **Symmetric memory support**: 9x latency reduction for small messages
- **Communicator Shrink**: dynamic GPU exclusion for fault tolerance
- **Up to 64 CTAs** simultaneously

### SHARP Details
```
# SHARP-accelerated collectives:
# - AllReduce (primary)
# - ReduceScatter + AllGather (NCCL 2.27 new, preferred for overlap)
#
# SM impact:
# - Without SHARP: 16+ SMs for communication
# - With SHARP: 6 or fewer SMs
# - Net: 10+ SMs freed for compute overlap
```

### Symmetric Memory
For small messages (< few KB), NCCL 2.27 can use symmetric memory allocation:
- All GPUs share a common address space for small control data
- 9x latency reduction for tiny messages
- Most beneficial for MoE All-to-All with small expert sizes

### Communicator Shrink
```python
# Dynamic fault tolerance: exclude failed GPUs without full restart
new_comm = comm.shrink(exclude_ranks=[failed_rank])
# Enables continued training after GPU failure
```

### Tuning Impact
- SHARP is the single biggest SM-saving feature — enables much better overlap
- ReduceScatter/AllGather SHARP preferred over AllReduce SHARP for overlap
  (splitting AllReduce into RS+AG allows pipelining each phase with compute)
- Symmetric memory changes optimal protocol for tiny messages
- CTA headroom increased (up to 64) for multi-operation concurrency
- Communicator Shrink doesn't affect tuning but changes fault model

---

## NCCL 2.29 (2025-2026)

### Key Changes — Latest Stable
- Adaptive routing threshold tied to BUFFSIZE
- Further cost model refinements
- Bug fixes for specific topology/algorithm combinations
- Enhanced CollNet reliability

### Adaptive Routing Interaction
```
# WARNING: Setting BUFFSIZE too high can disable adaptive routing
# The adaptive routing threshold is derived from BUFFSIZE
# Recommendation: keep BUFFSIZE at default (4MiB) unless specifically needed
```

### Tuning Impact
- Cost model now more accurate for edge cases
- Fewer situations where manual overrides outperform defaults
- BUFFSIZE tuning has additional consideration (adaptive routing)
- Most stable release for production deployments

---

## Parameter Deprecation and Migration Guide

### Deprecated Parameters

| Parameter | Status | Replacement | Since |
|-----------|--------|-------------|-------|
| NCCL_NTHREADS | May be ignored | NCCL_MIN_CTAS / NCCL_MAX_CTAS | 2.24 |
| NCCL_IB_DISABLE | Legacy | NCCL_NET (to select network transport) | 2.18 |
| NCCL_LAUNCH_MODE | Removed | Always GROUP mode in modern NCCL | 2.20 |

### Still Supported (current recommendations)

| Parameter | Status | Notes |
|-----------|--------|-------|
| NCCL_ALGO | Active | Syntax expanded in 2.21/2.24 |
| NCCL_PROTO | Active | Exclude syntax added in 2.24 |
| NCCL_MIN_NCHANNELS | Active | Still works, CTA-based control preferred |
| NCCL_MAX_NCHANNELS | Active | Still works, CTA-based control preferred |
| NCCL_BUFFSIZE | Active | New interaction with adaptive routing (2.29) |
| NCCL_NET_GDR_LEVEL | Active | Auto-detected on most platforms |
| NCCL_IB_QPS_PER_CONNECTION | Active | Important for multi-level fabrics |
| NCCL_SOCKET_NTHREADS | Active | Critical for TCP/Ethernet |
| NCCL_NSOCKS_PERTHREAD | Active | Critical for TCP/Ethernet |
| NCCL_SHM_DISABLE | Active | Important for PCIe cross-socket |
| NCCL_CROSS_NIC | Active | Default=2, good for multi-NIC |
| NCCL_NVLS_ENABLE | Active | Default=1 on NVSwitch systems |
| NCCL_MIN_CTAS | Active | Preferred over NTHREADS (2.24+) |
| NCCL_MAX_CTAS | Active | Preferred over NTHREADS (2.24+) |

---

## Version Selection Guidance

### For New Deployments
Use NCCL 2.27+ for:
- SHARP support (massive SM savings)
- ReduceScatter/AllGather SHARP for overlap
- Symmetric memory for small-message workloads
- Communicator Shrink for fault tolerance
- Up to 64 CTAs for multi-operation concurrency

### For Stability
Use NCCL 2.29 (latest stable):
- Most refined cost model
- Best adaptive routing integration
- Fewest known bugs
- Most comprehensive topology support

### For Compatibility
Minimum NCCL 2.24 for:
- Per-function algorithm/protocol control
- CTA-based tuning (replacing NTHREADS)
- Exclude syntax for safe protocol disabling
- Tuner plugin v4 API

---

## Upgrade Considerations

When upgrading NCCL versions:

1. **Remove manual overrides first** — test with defaults on new version
2. **Re-benchmark** — optimal settings may have changed
3. **Check for new features** — new algorithms/protocols may outperform old settings
4. **Verify LL128 support** — some platform changes affect atomic write guarantees
5. **Update tuner plugins** — API changes between versions may require plugin updates
6. **Test at scale** — some changes only manifest at large GPU counts

Keywords: version, changelog, deprecation, new_feature, sharp, nvls, communicator_shrink,
symmetric_memory, cta, tuner_plugin, adaptive_routing, migration, upgrade, compatibility,
nccl_nthreads, nccl_min_ctas, nccl_max_ctas, per_function_control
