# NCCL Protocol Internals

## Overview

NCCL uses three communication protocols that control how data is synchronized and
transferred between GPUs. Protocol selection is the second most impactful tuning
parameter after algorithm selection. Each protocol makes a different tradeoff between
latency and bandwidth.

Source: "Demystifying NCCL" (arXiv 2507.04786), NCCL 2.29.1 docs

---

## Simple Protocol

### Mechanism
Divides data into large chunks and dispatches them across communication channels.
Uses memory fences to enforce correct ordering and visibility of data between sender
and receiver. The fence-based synchronization adds approximately 6 microseconds of
overhead per synchronization point.

### Internal Details
- **Buffer**: 4 MiB total, divided into 8 slots of 512 KiB each
- **Effective data per slot**: 512 KiB (no flag overhead)
- **Synchronization**: Memory fences (coarse-grained)
- **Bandwidth efficiency**: ~95% of peak link bandwidth
- **Supports GPUDirect RDMA**: Yes (data stays in GPU memory)

### Performance Profile
- **Best for**: Messages > 64 KB (bandwidth-dominated)
- **Latency**: Higher startup (~6us per sync point)
- **Throughput**: Highest among all protocols at large messages
- **Scaling**: Amortized sync overhead makes it ideal for bulk transfers

### When to Use
- Large gradient all-reduce in data parallelism
- Any operation where message size >> 64 KB
- When maximum bandwidth utilization is the goal
- Combined with Ring algorithm for peak throughput

---

## LL (Low Latency) Protocol

### Mechanism
Transmits data in 8-byte atomic units: 4 bytes of data followed by a 4-byte flag.
The flag serves as both a data validity marker and synchronization signal. The receiver
polls the flag to know when data is ready, eliminating the need for memory fences.

### Internal Details
- **Buffer**: 256 KiB total, 8 slots of 32 KiB each
- **Effective data per slot**: 16 KiB (50% used for flags)
- **Synchronization**: Flag-based per 8-byte unit (~1us per hop)
- **Bandwidth efficiency**: 25-50% of peak (50% overhead for flags)
- **GPUDirect RDMA**: NO — forces intermediate buffer in host memory

### Critical Constraint
LL protocol forces the intermediate buffer to reside in host memory so the CPU can
poll the flag. GPU memory polling over PCIe is too slow. This restriction eliminates
GPUDirect RDMA capability, limiting bandwidth to 25-50% of peak.

### Performance Profile
- **Best for**: Messages < 64 KB (latency-dominated)
- **Latency**: Lowest (~1us per hop) — optimal for tiny messages
- **Throughput**: Poor for large messages due to 50% flag overhead
- **Trade-off**: Sacrifices bandwidth for minimal latency

### When to Use
- Very small control messages or synchronization barriers
- Latency-critical operations with tiny payloads
- When sub-microsecond per-hop latency is essential

### When NOT to Use
- Large messages (bandwidth waste is extreme)
- When GPUDirect RDMA is needed (LL disables it)

---

## LL128 Protocol

### Mechanism
Extends LL with 128-byte atomic units: 120 bytes of data + 8 bytes of flag. This
achieves 93.75% data efficiency (vs. 50% for LL) while maintaining flag-based
synchronization. Requires hardware guaranteeing 128-byte atomic writes without
splitting or reordering.

### Internal Details
- **Buffer**: ~4800 KiB total, 8 slots of ~600 KiB each
- **Effective data per slot**: ~562.5 KiB (93.75% efficiency)
- **Synchronization**: Flag-based per 128-byte unit (~2us per hop)
- **Bandwidth efficiency**: ~95% of peak (comparable to Simple)
- **GPUDirect RDMA**: Platform-dependent

### Cross-Node Behavior
On network paths, LL128 aggregates a relatively large chunk of data before notifying
the CPU that it is ready to send. This limits pipelining across nodes but still
enables fine-grained pipelining within a node.

### Performance Profile
- **Best for**: Messages 64 KB - 100 MB (balanced regime)
- **Latency**: Low (~2us per hop) — between LL and Simple
- **Throughput**: ~95% of peak — comparable to Simple
- **Intra-node**: Consistent across all message sizes
- **Inter-node**: Can underperform LL at very large scale due to sync overhead

### Critical Warning
**Enabling LL128 on platforms that do not support 128-byte atomic writes can lead to
SILENT DATA CORRUPTION.** NCCL auto-detects hardware support. Do not force LL128 via
NCCL_PROTO unless the platform is known to support it.

### When to Use
- Medium messages where both latency and bandwidth matter
- Intra-node communication on NVLink (excellent performance)
- Tensor parallelism all-reduce (frequent, moderate-sized)
- As default for most workloads on supported hardware

### When NOT to Use
- Platforms without verified 128-byte atomic write support
- Very large messages (>100MB) where Simple protocol is slightly better
- Inter-node at extreme scale where sync overhead per 128-byte unit accumulates

---

## Protocol Comparison Summary

| Aspect | Simple | LL | LL128 |
|--------|--------|-------|-------|
| Data unit | Large chunks | 8 bytes (4B data + 4B flag) | 128 bytes (120B data + 8B flag) |
| Sync mechanism | Memory fences | Flag polling | Flag polling |
| Sync overhead | ~6 us per fence | ~1 us per hop | ~2 us per hop |
| Bandwidth efficiency | ~95% peak | 25-50% peak | ~95% peak |
| GPUDirect RDMA | Yes | No (forces host mem) | Platform-dependent |
| Best message size | > 64 KB | < 64 KB | 64 KB - 100 MB |
| Buffer total | 4 MiB | 256 KiB | ~4800 KiB |
| Data corruption risk | None | None | Yes (if unsupported HW) |

---

## NCCL_PROTO Environment Variable

### Syntax (NCCL 2.24+)
```
NCCL_PROTO="<protocols>"
NCCL_PROTO="<default>;function1:<protocols>"
```

Examples:
- `NCCL_PROTO=SIMPLE` — Force Simple for all operations
- `NCCL_PROTO=LL,LL128` — Allow LL and LL128 only
- `NCCL_PROTO="^LL128"` — Disable LL128 (safe fallback)
- `NCCL_PROTO="allreduce:simple;broadcast:ll"` — Per-function

### Default
All supported protocols enabled: `LL,LL128,Simple` on platforms supporting LL128,
`LL,Simple` otherwise.

### NVIDIA Guidance
"Users are discouraged from setting this variable, with the exception of disabling a
specific protocol in case a bug in NCCL is suspected."

---

## Intra-Node vs Inter-Node Protocol Behavior

### Intra-Node (NVLink)
- LL128: Consistent performance across all sizes, only 5% slower than Simple at large messages
- Simple: Optimal for large messages within node
- LL: Optimal for very small messages within node
- All three work well — differences are modest

### Inter-Node (InfiniBand/RoCE)
- Small messages (<64KB): LL/LL128 outperform Simple (startup latency dominates)
- Large messages (GB range): Simple dominates — LL/LL128 suffer from fine-grained sync overhead
- LL128 can underperform LL at very large scale due to extra cost per 128-byte operation
- Stalls affect larger LL128 data units more under heavy contention

### Inter-Node (Ethernet/TCP)
- Socket transport adds PCIe copy overhead regardless of protocol
- Simple generally preferred (minimize number of small transfers)
- LL/LL128 less beneficial over high-latency networks

Keywords: protocol, simple, ll, ll128, low_latency, bandwidth, atomic, synchronization,
memory_fence, flag_polling, gpudirect, rdma, data_corruption, buffer_size, pipelining
