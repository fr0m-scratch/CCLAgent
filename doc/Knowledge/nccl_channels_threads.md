# NCCL Channels, CTAs, and Thread Scheduling

## Overview

NCCL uses a multi-level parallelism model: channels provide GPU-side parallelism by
running collective operations across multiple CUDA blocks (CTAs), each on a separate SM.
Understanding the interaction between channels, threads, and buffer pipelining is critical
for balancing communication throughput against compute resource availability.

Source: "Demystifying NCCL" (arXiv 2507.04786), NVIDIA NCCL tuning blog 2025, NCCL 2.29.1 docs

---

## Channels: GPU-Side Parallelism

### What Channels Are
Each communication channel is a separate CUDA block that runs on its own SM (Streaming
Multiprocessor). NCCL partitions the input buffer so channels operate on disjoint chunks
in parallel. This fine-grained parallelism raises aggregate throughput.

### How Channel Count is Determined
1. NCCL detects the physical topology (NVLink, PCIe, IB, etc.)
2. The internal tuning model selects channel count based on:
   - Selected algorithm (Ring, Tree, etc.)
   - Message size
   - Available bandwidth per link
   - Configured threads per channel
3. For smaller messages, NCCL heuristically reduces channel count to avoid under-utilization

### Channel Count Trade-offs

**Too Many Channels**:
- Per-channel chunk becomes smaller than the 512 KB NIC FIFO buffer
- Proxy thread sends partially filled buffers → PCIe/network waste
- More CUDA blocks = more SM resources consumed
- Starves compute kernels (model forward/backward) of SM resources
- "Benchmark improvement != workload improvement"

**Too Few Channels**:
- Single SM bottleneck — cannot saturate high-bandwidth links (NVLink, IB)
- Under-utilized interconnect bandwidth
- Poor load balancing across multiple NICs

### Environment Variables

**NCCL_MIN_NCHANNELS** (default varies by topology):
- Minimum channels NCCL will use
- Useful for aggregated collectives where NCCL defaults to 1 channel
- Increasing ensures enough parallelism for large messages

**NCCL_MAX_NCHANNELS** (default varies by topology):
- Maximum channels NCCL will use
- Reducing frees CUDA compute resources for model computation
- Critical for communication-compute overlap in training

### Topology-Specific Channel Behavior

| Topology | Typical Default | Notes |
|----------|----------------|-------|
| NVLink (intra-node) | 8-16 | High bandwidth supports many channels |
| PCIe (intra-node) | 4-8 | Lower bandwidth limits benefit |
| IB (inter-node) | 2-4 per peer | 2 channels per remote GPU per NIC by default |
| Ethernet (inter-node) | 1-2 | TCP overhead dominates; fewer channels better |

### Multi-NIC Channel Assignment
On NVLink platforms with multiple NICs, channels help balance traffic across NICs.
NCCL_CROSS_NIC=2 (default) allows different rings/trees to use different NICs,
improving aggregate bandwidth.

---

## CTAs (Cooperative Thread Arrays)

### What CTAs Are
CTAs are CUDA blocks executing NCCL communication kernels. Since NCCL 2.27, up to 64
CTAs can run simultaneously. Each CTA operates on a communication channel.

### CTA Count Philosophy
NVIDIA's guidance: "Take just enough CTAs to saturate the line rate of available
transports at large message sizes, but no more."

### Trade-offs

**More CTAs**:
- Higher peak communication throughput
- Better for saturating high-bandwidth links
- Improves isolated communication benchmark results

**Fewer CTAs**:
- More SMs available for model compute
- Better communication-compute overlap
- Preferred for end-to-end training performance

### Environment Variables

**NCCL_MIN_CTAS** / **NCCL_MAX_CTAS**:
- Control the CTA count range
- NCCL auto-selects within this range based on workload
- Preferred over NCCL_NTHREADS for modern NCCL versions

### CTA Starvation
When too many CTAs are requested:
- Chip resource interference between communication and compute blocks
- CTA scheduling delays (waiting for SMs to become available)
- Net result: benchmark shows improvement, but training slows down

---

## Threads (NCCL_NTHREADS)

### What It Controls
Number of CUDA threads per CUDA block (per channel). NCCL launches one block per channel.

### Valid Values
64, 128, 256, 512

### Defaults
- 512 for recent-generation GPUs (Hopper, Ada, Ampere)
- 256 for older generations

### When to Adjust
- **Increase (to 512)**: GPU clocks are low, need more thread-level parallelism
- **Decrease (to 128 or 64)**: Reduce GPU workload for better compute overlap

### Important Warning
In recent NCCL versions (2.24+), manual NCCL_NTHREADS settings may be ignored and can
lead to incorrect behavior. Use NCCL_MIN_CTAS/NCCL_MAX_CTAS instead for controlling
communication resource allocation.

---

## Buffer Pipelining

### Slot-Based Pipeline
Each channel's buffer is divided into 8 slots (controlled by NCCL_STEPS, default 8).
Each slot can independently advance through communication stages, enabling overlap:

```
Slot 0: [Send data]  →  [Wait ack]  →  [Reuse]
Slot 1:               [Send data]  →  [Wait ack]  →  [Reuse]
Slot 2:                             [Send data]  →  [Wait ack]
...
```

### Pipeline Granularity

| Protocol | Slot Size | Pipeline Depth | Pipeline Characteristic |
|----------|-----------|---------------|----------------------|
| Simple | 512 KiB | 8 slots | Coarse-grained, high throughput |
| LL | 32 KiB (16KB effective) | 8 slots | Fine-grained, low latency |
| LL128 | 600 KiB (562.5KB effective) | 8 slots | Moderate granularity |

### Pipelining vs. Non-Pipelining

**Pipelined operations** (chunks from consecutive loop iterations overlap):
- Tree AllReduce (Reduce + Broadcast phases can overlap)
- Ring Broadcast/Reduce (chain propagation allows overlap)

**Non-pipelined operations** (all steps must complete before next iteration):
- Ring AllReduce (2k-1 steps must finish per iteration)
- Ring AllGather
- Ring ReduceScatter

### Group Call Optimization
`ncclGroupStart()` / `ncclGroupEnd()` brackets enable NCCL to schedule multiple
operations concurrently. When many operations run in parallel, NCCL dynamically reduces
CTAs per operation to allow parallel execution. This is critical for workloads with
many small collectives (e.g., expert parallelism in MoE models).

---

## NCCL_BUFFSIZE

### What It Controls
Size of the communication buffer used between pairs of GPUs.

### Default
4,194,304 bytes (4 MiB)

### Tuning Direction
- **Increase**: Improve throughput for large messages; larger pipeline chunks
- **Decrease**: Save GPU memory when many communicators are active

### Constraints
- Recommendation: use powers of 2
- Adaptive routing threshold is tied to BUFFSIZE — setting too high disables adaptive routing
- Memory cost: BUFFSIZE x NCHANNELS x NPEERS per communicator

### Interaction with Channels
Total buffer memory = BUFFSIZE x number_of_channels x number_of_peers.
With many channels and peers, buffer memory can become significant. Balance
BUFFSIZE against channel count to stay within memory budget.

---

## Communication-Compute Overlap

### The Fundamental Tension
More channels/CTAs → higher communication bandwidth → fewer SMs for compute.
Optimal training performance requires finding the balance point where:
- Communication is fast enough to not be on the critical path
- Enough SMs remain for model computation to proceed at full speed

### Practical Guidelines

1. **Start with defaults** — NCCL's auto-tuning handles most cases well
2. **If comm is bottleneck**: Increase NCCL_MIN_NCHANNELS or NCCL_MIN_CTAS
3. **If compute is bottleneck**: Decrease NCCL_MAX_NCHANNELS or NCCL_MAX_CTAS
4. **Always benchmark end-to-end**, not just nccl-tests
5. **SHARP/CollNet** (NCCL 2.27+): Reduces SM usage from 16+ to 6, freeing compute
6. **User Buffer Registration**: Reduces CTA requirements for certain operations

### Monitoring Overlap
Use `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=TUNING` to see NCCL's internal selections.
Monitor both communication time AND model iteration time to detect imbalance.

Keywords: channels, ctas, threads, sm, streaming_multiprocessor, oversubscription,
overlap, pipelining, buffer, buffsize, nchannels, nthreads, compute_overlap, group_call,
nic, multi_nic, slot, pipeline_depth
