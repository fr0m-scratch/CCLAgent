# NCCL Algorithm Selection Guide

## Overview

NCCL supports multiple collective communication algorithms, each with distinct performance
characteristics. Algorithm selection is the single most impactful tuning parameter across
all topologies. NCCL's internal cost model auto-selects algorithms based on message size,
GPU count, and topology, but understanding the tradeoffs enables informed override decisions.

Source: NCCL 2.29.1 docs, "Demystifying NCCL" (arXiv 2507.04786), NVIDIA tuning blog 2025

---

## Ring Algorithm

### Mechanism
GPUs arranged in a unidirectional circular topology. AllReduce executes in two phases:
1. **ReduceScatter phase** (k-1 steps): Each GPU sends one segment to its neighbor. At each
   step, the receiving GPU performs element-wise reduction with its local segment and forwards
   the partial result. After k-1 steps, each GPU holds a fully reduced segment.
2. **AllGather phase** (k-1 steps): Each GPU forwards its fully reduced segment around the
   ring until all GPUs hold all segments.

Total: 2(k-1) steps for k GPUs.

### Performance Characteristics
- **Bandwidth**: Optimal. Achieves theoretical peak for large messages.
- **Latency**: O(k) — linear in number of GPUs. This is the major drawback.
- **Message size sweet spot**: >100MB (bandwidth-dominated regime).
- **Pipelining**: Non-pipelined — all steps in a loop iteration must complete before next.

### When to Use Ring
- Large gradient all-reduce in data parallelism (overlapped with backprop)
- Large message sizes where bandwidth matters more than startup latency
- Small GPU counts (latency penalty is manageable)
- When combined with Simple protocol for maximum throughput

### When NOT to Use Ring
- Small/medium messages where startup latency dominates transfer time
- Large GPU counts (100+) where O(k) latency becomes prohibitive
- Latency-sensitive operations like tensor parallelism all-reduce

---

## Tree Algorithm (Double Binary Tree)

### Mechanism
NCCL uses a "double binary tree" where no node is a non-leaf in both trees, and at most
one node appears as a leaf in both. This enables full bandwidth utilization.

AllReduce executes in two concurrent phases:
1. **Reduce phase**: Leaf nodes send data up; middle nodes receive-reduce-send; root receives
   the final reduction. Uses `recvReduceSend` primitives.
2. **Broadcast phase**: Root sends reduced data down; middle nodes receive-copy-send; leaves
   receive final data. Uses `recvCopySend` primitives.

Key detail: Within a single node, NCCL builds a simple chain (not a full tree). The binary
tree structure spans across nodes only.

### Performance Characteristics
- **Bandwidth**: Full bandwidth via double-tree construction.
- **Latency**: O(log k) — logarithmic in number of GPUs. Major advantage over Ring.
- **Message size sweet spot**: Small to medium messages (<12MB).
- **Pipelining**: Can pipeline chunks across loop iterations (unlike Ring AllReduce).
- **Scaling**: 180x latency improvement over Ring at 24,576 GPUs (NCCL 2.4, Summit).

### When to Use Tree
- Small/medium messages where latency matters
- Large GPU counts (32+) where logarithmic latency advantage is significant
- Tensor parallelism all-reduce (frequent, moderate-sized)
- When combined with LL128 protocol for balanced latency/bandwidth

### When NOT to Use Tree
- Very large messages (>100MB) where pure bandwidth matters
- Simple point-to-point patterns (Broadcast/Reduce use chain, not tree)

---

## CollNet (SHARP)

### Mechanism
Offloads reduction operations to network switches via NVIDIA SHARP (Scalable Hierarchical
Aggregation and Reduction Protocol). The switch performs the aggregation in hardware,
reducing data movement and GPU-side compute.

Two variants:
- **CollNet Direct**: All GPUs within a node communicate directly with the switch.
- **CollNet Chain**: GPUs arranged linearly within a node, one gateway to the switch.

### Performance Characteristics
- **SM usage**: Reduces from 16+ SMs to 6 or fewer (NCCL 2.27+)
- **Bandwidth**: Near-wire-speed for supported operations
- **Requirements**: InfiniBand switches with SHARP capability

### When to Use CollNet
- 1000+ GPU scale where SM pressure matters
- AllReduce-heavy workloads (primary supported operation)
- When SHARP-capable switches are available
- To free SMs for better compute-communication overlap

### Supported Operations
- AllReduce (primary)
- ReduceScatter and AllGather (NCCL 2.27+ with SHARP)

---

## NVLS (NVLink SHARP)

### Mechanism
Leverages NVSwitch hardware to perform collective operations with minimal GPU involvement.
The NVSwitch acts as an in-network aggregator similar to InfiniBand SHARP but for
intra-node NVLink communication.

Variants:
- **NVLS**: Uses NVSwitch for intra-node, CollNet for inter-node
- **NVLSTree**: Uses NVSwitch for intra-node, Tree for inter-node

### Performance Characteristics
- **Requires**: NVSwitch (DGX/HGX systems, NVLink Switch v3+)
- **Default**: Enabled on supported systems since NCCL 2.17
- **Toggle**: NCCL_NVLS_ENABLE=0|1
- **Best for**: High bandwidth + low latency, especially AllReduce

### When to Use NVLS
- DGX/HGX systems with NVSwitch
- AllReduce, ReduceScatter, AllGather operations
- When maximizing intra-node bandwidth is critical
- Memory-rich environments (NVLS uses additional memory)

### When to Disable NVLS
- Memory-constrained environments
- When simplified configuration is preferred
- When additional bandwidth isn't necessary for the workload

---

## Algorithm Selection by Message Size

Empirically validated optimal selections (from NVIDIA tuning blog case study):

| Message Size | Algorithm | Protocol | Rationale |
|-------------|-----------|----------|-----------|
| 0 - 98 KB | Tree | LL | Latency-optimized for small messages |
| 98 KB - 12 MB | Tree | LL128 | Balanced latency + bandwidth transition |
| 12 MB - 100 MB | Ring | LL128 | Bandwidth ramping, LL128 maintains low latency |
| 100 MB+ | Ring | Simple | Peak bandwidth, sync overhead amortized |

**Note**: These thresholds are topology-dependent. NCCL's cost model adjusts based on
actual hardware. Override only when benchmarks confirm improvement.

---

## NCCL_ALGO Environment Variable

### Syntax (NCCL 2.24+)
```
NCCL_ALGO="<default_algos>;function1:<algos>;function2:<algos>"
```

Examples:
- `NCCL_ALGO=TREE` — Force Tree for all operations
- `NCCL_ALGO=RING,TREE` — Allow both Ring and Tree (NCCL chooses)
- `NCCL_ALGO="allreduce:tree,collnetdirect;broadcast:ring"` — Per-function
- `NCCL_ALGO="allreduce:^tree"` — All except Tree for allreduce

### Available Algorithms
`Tree`, `Ring`, `CollnetDirect`, `CollnetChain`, `NVLS`, `NVLSTree`

Default (unset): NCCL auto-selects based on topology and message size.

### Related Threshold Variables
- `NCCL_TREE_THRESHOLD`: Size limit (bytes) under which Tree is preferred over Ring.
  Default depends on number of ranks.
- `NCCL_SINGLE_RING_THRESHOLD`: Size limit for using a single ring (limits bandwidth but
  improves latency). Default: 262144 (256KB) on compute capability 7+.

---

## Algorithm-Protocol Compatibility Matrix

| Collective | Ring | Tree | CollNet | NVLS |
|------------|------|------|---------|------|
| AllReduce | Simple, LL, LL128 | Simple, LL, LL128 | Simple only | Simple only |
| Broadcast | Simple, LL, LL128 | N/A (uses Ring chain) | N/A | N/A |
| Reduce | Simple, LL, LL128 | N/A (uses Ring chain) | N/A | N/A |
| ReduceScatter | Simple, LL, LL128 | N/A | N/A | Simple only |
| AllGather | Simple, LL, LL128 | N/A | N/A | Simple only |

---

## Scaling Behavior

| GPU Count | Preferred Algorithm | Rationale |
|-----------|-------------------|-----------|
| 2-8 | Ring or Tree (auto) | Both work well at small scale |
| 8-32 | Ring for large msg, Tree for small | Standard tradeoff |
| 32-128 | Tree increasingly preferred | Logarithmic latency advantage grows |
| 128-1000 | Tree + CollNet if available | Ring latency becomes prohibitive |
| 1000+ | CollNet/SHARP essential | SM pressure + latency both critical |

Keywords: algorithm, ring, tree, collnet, nvls, nvlstree, sharp, allreduce, latency,
bandwidth, message_size, scaling, double_binary_tree, threshold
