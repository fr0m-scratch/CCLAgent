# NCCL Parameter Interaction Matrix

## Overview

NCCL tuning parameters do not operate independently. Changing one parameter often alters
the optimal value of others. This document categorizes parameter interactions as synergistic
(mutually beneficial), antagonistic (conflicting), or dangerous (can cause failures).
Understanding these interactions is essential for proposing multi-parameter changes that
are coherent rather than contradictory.

Source: "Demystifying NCCL" (arXiv 2507.04786), NVIDIA NCCL tuning blog 2025,
NCCL 2.29.1 docs, empirical observations

---

## Synergistic Combinations

These parameter combinations produce compounding benefits when applied together.

### ALGO=Tree + PROTO=LL128
- **Why synergistic**: Tree provides O(log k) latency; LL128 provides ~95% bandwidth
  with low per-hop latency (~2μs). Together they optimize both dimensions.
- **Best for**: Medium messages (98KB-12MB), moderate to large GPU counts
- **Mechanism**: Tree reduces total hops (log k vs k), LL128 reduces per-hop cost
- **Topology**: NVLink intra-node, IB inter-node at moderate scale

### ALGO=Ring + PROTO=Simple + High NCHANNELS
- **Why synergistic**: Ring achieves optimal bandwidth; Simple has ~95% link utilization;
  many channels multiply aggregate throughput across parallel SM resources.
- **Best for**: Large messages (>100MB), bandwidth-dominated workloads
- **Mechanism**: Each channel saturates a portion of link bandwidth independently
- **Constraint**: Per-channel chunk must exceed 512KB NIC FIFO to avoid waste

### High NCHANNELS + High NTHREADS
- **Why synergistic**: More channels × more threads per channel = more GPU resources
  devoted to communication. Saturates multiple high-bandwidth links simultaneously.
- **Best for**: Large-scale NVLink with many links to saturate
- **Constraint**: Total SM usage = NCHANNELS × 1 SM each. Excessive allocation
  starves compute kernels of SM resources.

### BUFFSIZE Increase + ALGO=Ring
- **Why synergistic**: Larger buffers create bigger pipeline chunks for Ring.
  Ring's non-pipelined all-reduce benefits from fewer, larger chunk iterations.
- **Best for**: Large messages over high-bandwidth links
- **Constraint**: Memory cost = BUFFSIZE × NCHANNELS × NPEERS per communicator

### IB_QPS_PER_CONNECTION Increase + CROSS_NIC=2
- **Why synergistic**: Multiple QPs improve ECMP routing entropy on multi-level
  IB fabrics. CROSS_NIC=2 allows different rings/trees to use different NICs.
  Together they maximize path diversity across the fabric.
- **Best for**: Multi-switch InfiniBand, fat-tree topologies, multi-NIC nodes
- **Mechanism**: QP source port randomization + NIC diversity = maximum routing entropy

### NVLS_ENABLE + CollNet/SHARP
- **Why synergistic**: NVLS handles intra-node reduction in NVSwitch hardware;
  CollNet/SHARP handles inter-node reduction in IB switches. Both offload work
  from GPU SMs, compounding SM savings.
- **Best for**: DGX/HGX systems with SHARP-capable IB switches, 1000+ GPU scale
- **SM impact**: From 16+ SMs to potentially 4-6 SMs for communication

### SOCKET_NTHREADS Increase + NSOCKS_PERTHREAD Increase
- **Why synergistic**: More threads × more sockets per thread = more parallel
  TCP streams. Each stream independently utilizes TCP bandwidth.
- **Best for**: Ethernet-only clusters where TCP throughput is the bottleneck
- **Constraint**: Diminishing returns beyond total of ~16-32 streams

---

## Antagonistic Combinations

These parameter combinations conflict or cancel each other's benefits.

### High NCHANNELS + Small Messages
- **Why antagonistic**: With small messages, each channel's chunk becomes tiny
  (message_size / NCHANNELS). If chunk < 512KB NIC FIFO, the NIC sends partially
  filled buffers → wasted PCIe and network bandwidth.
- **Symptom**: Benchmark shows marginal or negative improvement for small messages
  despite more channels
- **Resolution**: Let NCCL auto-reduce channels for small messages, or set
  NCCL_MAX_NCHANNELS lower when workload is dominated by small collectives

### PROTO=LL + NET_GDR_LEVEL=SYS
- **Why antagonistic**: LL protocol forces intermediate buffer to host memory
  (for CPU flag polling). GDR expects data to stay in GPU memory for direct
  NIC access. LL disables the GDR path entirely.
- **Symptom**: GDR setting has no effect; performance matches non-GDR baseline
- **Resolution**: Use LL128 or Simple if GDR is important. LL is only for
  tiny messages where GDR bandwidth advantage is irrelevant.

### High NTHREADS + Compute-Heavy Workload
- **Why antagonistic**: More threads per communication block → more SM resources
  consumed → fewer SMs available for model computation. If communication is
  already fast enough, extra threads don't help but compute gets slower.
- **Symptom**: nccl-tests shows improvement; end-to-end training iteration is slower
- **Resolution**: Use NCCL_MAX_CTAS instead of NTHREADS for modern NCCL.
  Reduce communication resource allocation until it's just enough.

### ALGO=Tree + Very Large Messages (>100MB)
- **Why antagonistic**: Tree's advantage is low latency, which is irrelevant when
  messages are so large that bandwidth dominates. Ring achieves strictly better
  bandwidth for large messages (optimal algorithm).
- **Symptom**: Tree all-reduce takes longer than Ring for the same large message
- **Resolution**: Use Ring for large messages. NCCL's auto-selection handles this
  correctly; only override if benchmarks confirm.

### High BUFFSIZE + High NCHANNELS
- **Why antagonistic**: Total memory = BUFFSIZE × NCHANNELS × NPEERS.
  Both parameters increase memory consumption multiplicatively.
- **Example**: BUFFSIZE=8MiB × NCHANNELS=16 × NPEERS=7 = 896 MiB per communicator
- **Symptom**: Out-of-memory errors or reduced memory available for model/activations
- **Resolution**: Increase one or the other, not both. For large messages, prefer
  increasing BUFFSIZE with moderate channels; for many parallel links, prefer
  more channels with default BUFFSIZE.

### SHM_DISABLE=1 + Cross-Socket PCIe
- **Why antagonistic**: Disabling shared memory forces direct PCIe P2P for
  cross-socket GPU pairs. PCIe P2P across socket boundaries is poorly handled
  on many platforms (traverses CPU interconnect with high latency).
- **Symptom**: Dramatically slower intra-node communication for cross-socket GPU pairs
- **Resolution**: Keep SHM_DISABLE=0 (default) on PCIe systems

### ALGO=RING + Very Small Messages + Large GPU Count
- **Why antagonistic**: Ring has O(k) latency. For small messages, startup latency
  dominates transfer time. With many GPUs, the cumulative startup overhead
  exceeds the message transfer time by orders of magnitude.
- **Symptom**: Small-message all-reduce takes milliseconds instead of microseconds
- **Resolution**: Use Tree for small messages at large GPU counts

---

## Dangerous Combinations

These combinations can cause crashes, hangs, data corruption, or deadlocks.

### PROTO=LL128 on Unsupported Hardware
- **Risk**: SILENT DATA CORRUPTION
- **Mechanism**: LL128 requires hardware guarantee of 128-byte atomic writes.
  On platforms without this guarantee, partial writes can corrupt data.
  NCCL auto-detects support; do not force LL128 via NCCL_PROTO override.
- **Detection**: Results differ between runs; NaN in gradients; model divergence
- **Prevention**: Never force NCCL_PROTO=LL128 unless platform is verified

### SHM_DISABLE=1 + No NVLink + No IB
- **Risk**: DEADLOCK or COMMUNICATION FAILURE
- **Mechanism**: With SHM disabled, no NVLink, and no IB, the only remaining
  transport may be insufficient. Some GPU pairs may have no viable communication path.
- **Detection**: Hang during communicator initialization
- **Prevention**: Always ensure at least one working transport path exists

### Very Low BUFFSIZE (<1MB) + Ring + Large Message
- **Risk**: EXTREME PERFORMANCE DEGRADATION or TIMEOUT
- **Mechanism**: Ring with tiny buffer creates massive fragmentation. Each channel
  processes message_size / (BUFFSIZE / slots) iterations. With 512KB total buffer,
  a 1GB message requires ~16,000 iterations × 2(k-1) steps each.
- **Detection**: Communication timeout, extremely slow collective operations
- **Prevention**: Keep BUFFSIZE at default (4MiB) or higher for large messages

### NCHANNELS > Available SMs
- **Risk**: CUDA RESOURCE EXHAUSTION
- **Mechanism**: Each channel needs one SM. If NCHANNELS exceeds available SMs
  (minus those needed for compute kernels), CUDA block scheduling fails or
  creates extreme contention.
- **Detection**: CUDA launch failure or severe performance degradation
- **Prevention**: NCCL_MAX_NCHANNELS should never exceed ~50% of total SMs

### IB_TIMEOUT Too Low + Large Cluster
- **Risk**: SPURIOUS DISCONNECTION
- **Mechanism**: IB timeout is 4.096μs × 2^TIMEOUT. At large scale, fabric
  congestion can cause legitimate delays exceeding short timeout values.
  Timeout triggers QP error → connection teardown → NCCL error.
- **Detection**: Intermittent "connection reset" or "timeout" errors at scale
- **Prevention**: IB_TIMEOUT=22 or higher for 1000+ GPU clusters

---

## Scaling Laws

How optimal parameter values change with system scale.

### GPU Count Scaling

| GPU Count | ALGO Preference | NCHANNELS | PROTO | Notes |
|-----------|----------------|-----------|-------|-------|
| 2-8 | Ring or Tree | 4-8 | LL128 or Simple | Both algorithms work well |
| 8-32 | Ring (large msg) / Tree (small) | 4-8 | Auto | Standard trade-off |
| 32-128 | Tree increasingly preferred | 4-8 | LL128 for medium | Log latency advantage grows |
| 128-1000 | Tree + CollNet if available | 4-8 | Auto | Ring latency prohibitive |
| 1000+ | CollNet/SHARP essential | Reduced | Simple (SHARP) | SM pressure dominates |

### Message Size Scaling

| Message Size | ALGO | PROTO | NCHANNELS | BUFFSIZE |
|-------------|------|-------|-----------|----------|
| < 1 KB | Tree | LL | 1-2 | Default |
| 1-64 KB | Tree | LL | 2-4 | Default |
| 64 KB - 12 MB | Tree | LL128 | 4-8 | Default |
| 12-100 MB | Ring | LL128 | 4-8 | Default |
| 100 MB - 1 GB | Ring | Simple | 8-16 | 4-8 MiB |
| > 1 GB | Ring | Simple | 8-16 | 8-16 MiB |

### Network Bandwidth Scaling

| Available BW | NCHANNELS | BUFFSIZE | Notes |
|-------------|-----------|----------|-------|
| Low (1-10 Gbps) | 1-2 | Default | Single channel often sufficient |
| Medium (25-100 Gbps) | 2-4 | Default | Moderate parallelism |
| High (200-400 Gbps) | 4-8 | 4-8 MiB | Need channels to saturate |
| Very High (900+ Gbps, NVLink) | 8-16 | Default | NVLink provides per-link BW |

---

## Sensitivity Rankings

Ordered by impact (most to least) for each topology.

### NVLink Intra-Node
1. **NCCL_ALGO** — algorithm choice dominates (Ring vs Tree)
2. **NCCL_PROTO** — protocol choice (LL128 vs Simple vs LL)
3. **NCCL_MAX_NCHANNELS** — channel count for SM management
4. **NCCL_NTHREADS/MAX_CTAS** — fine-tuning SM usage
5. **NCCL_BUFFSIZE** — minimal impact (NVLink latency is low)

### PCIe Intra-Node
1. **NCCL_ALGO** — algorithm choice
2. **NCCL_MAX_NCHANNELS** — PCIe bandwidth limits channel utility
3. **NCCL_PROTO** — protocol choice
4. **NCCL_BUFFSIZE** — buffer size affects PCIe utilization
5. **NCCL_P2P_LEVEL** — transport selection
6. **NCCL_SHM_DISABLE** — critical for cross-socket

### InfiniBand Inter-Node
1. **NCCL_ALGO** — algorithm choice (Tree vs Ring)
2. **NCCL_PROTO** — protocol (Simple vs LL128)
3. **NCCL_IB_QPS_PER_CONNECTION** — routing entropy on multi-level fabrics
4. **NCCL_MAX_NCHANNELS** — parallelism across links
5. **NCCL_NET_GDR_LEVEL** — GDR activation distance
6. **NCCL_IB_TIMEOUT** — reliability at scale

### Ethernet Inter-Node
1. **NCCL_SOCKET_NTHREADS** — TCP thread parallelism (most impactful)
2. **NCCL_NSOCKS_PERTHREAD** — TCP stream count
3. **NCCL_ALGO** — algorithm choice
4. **NCCL_PROTO** — Simple preferred (minimize small sends)
5. **NCCL_BUFFSIZE** — larger for high-latency links

---

## Cross-Reference: Parameter → Interaction Partners

| Parameter | Interacts With | Nature |
|-----------|---------------|--------|
| NCCL_ALGO | PROTO, NCHANNELS, message_size | Algorithm determines protocol/channel optimality |
| NCCL_PROTO | ALGO, NET_GDR_LEVEL, hardware | Protocol affects transport path |
| NCCL_MAX_NCHANNELS | BUFFSIZE, NTHREADS, message_size, ALGO | Channels affect memory, SM, and chunk size |
| NCCL_BUFFSIZE | NCHANNELS, NPEERS, ALGO | Buffer affects memory and pipeline granularity |
| NCCL_NTHREADS | NCHANNELS, compute_workload | Threads affect SM pressure |
| NCCL_NET_GDR_LEVEL | PROTO, PCIe_topology | GDR requires compatible protocol and topology |
| NCCL_IB_QPS_PER_CONNECTION | CROSS_NIC, fabric_topology | QPs affect routing diversity |
| NCCL_SOCKET_NTHREADS | NSOCKS_PERTHREAD, available_cores | Threads × sockets = total TCP parallelism |
| NCCL_SHM_DISABLE | P2P_LEVEL, topology | SHM is critical fallback on PCIe |

Keywords: interaction, synergy, conflict, danger, scaling, sensitivity, combination,
antagonistic, dangerous, compounding, memory_explosion, sm_pressure, ecmp, routing,
chunk_size, fifo, transport_path, deadlock, corruption, timeout
