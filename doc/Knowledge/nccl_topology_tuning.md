# NCCL Topology-Specific Tuning Rules

## Overview

NCCL's performance is highly topology-dependent. The same environment variable settings that
produce optimal results on an 8-GPU NVLink node can severely degrade performance on a
multi-node InfiniBand cluster. This document provides topology-specific tuning guidance
for the four major transport configurations.

Source: "Demystifying NCCL" (arXiv 2507.04786), NVIDIA NCCL tuning blog 2025, GB200 NVL
Multi-Node Tuning Guide, NCCL 2.29.1 docs

---

## Transport Priority and Selection

NCCL selects transports in this priority order:
1. **NVLink** (P2P direct) — highest priority, lowest latency
2. **Shared Memory** (SHM) — cross-socket fallback within node
3. **InfiniBand Verbs** (IB) — inter-node, high bandwidth
4. **Socket/TCP** — inter-node fallback, lowest bandwidth

The selection is based on hardware topology detection at communicator init time.
`NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH` shows the selected transport graph.

---

## Single Node — NVLink (DGX/HGX Systems)

### Transport Characteristics
- Direct GPU-to-GPU memory access via NVLink bridges
- NVLink 4.0 (Hopper): 900 GB/s bidirectional per GPU
- NVLink 3.0 (Ampere): 600 GB/s bidirectional per GPU
- NVSwitch provides all-to-all connectivity (no ring hops needed)

### P2P Optimization
NCCL uses P2P_DIRECT optimization on NVLink: bypasses IPC handle overhead and uses
direct GPU memory pointers. This is the fastest transport path available.

### Recommended Settings
```
# Usually no overrides needed — NCCL defaults are well-tuned for NVLink
# These are reference values, not mandatory overrides:
NCCL_P2P_LEVEL=NVL           # Force NVLink P2P (usually auto-detected)
NCCL_SHM_DISABLE=0           # Keep shared memory as fallback
NCCL_NVLS_ENABLE=1           # Enable NVLink SHARP on NVSwitch systems (default)
```

### Channel Guidance
- NVLink bandwidth is high → can support many channels (8-16)
- SM oversubscription is less critical than on PCIe (NVLink latency is low)
- For compute-heavy workloads, reducing MAX_NCHANNELS still helps overlap

### Protocol Guidance
- LL128 works well intra-node: only ~5% slower than Simple at large messages
- LL128 matches LL performance at small messages
- All three protocols perform well; differences are modest within a node

### NVLS (NVLink SHARP)
On NVSwitch systems (DGX H100, DGX B200, GB200 NVL):
- NVLS offloads reduction to NVSwitch hardware
- Default enabled since NCCL 2.17
- Best for AllReduce, ReduceScatter, AllGather
- Uses additional GPU memory for multicast buffers
- Disable with `NCCL_NVLS_ENABLE=0` only if memory-constrained

### Common Pitfalls
- Setting excessive channels on a compute-bound workload wastes SMs
- Forcing PROTO=SIMPLE when LL128 would give comparable bandwidth with lower latency
- Disabling NVLS unnecessarily on NVSwitch systems

---

## Single Node — PCIe

### Transport Characteristics
- GPU-to-GPU via PCIe switch or root complex
- PCIe 5.0: ~64 GB/s per direction per x16 link
- PCIe 4.0: ~32 GB/s per direction per x16 link
- Cross-socket: PCIe packets may traverse CPU interconnect (UPI/Infinity Fabric)

### Shared Memory Transport
For cross-socket GPU pairs where direct PCIe P2P is poorly handled:
- NCCL uses shared memory (pinned host memory) as intermediary
- GPU → pinned host → other GPU
- `NCCL_SHM_DISABLE=0` keeps this path available (important!)
- Disabling SHM on cross-socket PCIe forces direct P2P which may be slow

### Recommended Settings
```
NCCL_P2P_LEVEL=PHB           # P2P through PCIe bridge (auto-detected)
NCCL_SHM_DISABLE=0           # Essential for cross-socket
NCCL_MAX_NCHANNELS=4         # PCIe bandwidth limits channel benefit
```

### Channel Guidance
- PCIe bandwidth is lower → fewer channels needed (4-8)
- More channels than PCIe can sustain → wasted SM resources
- Buffer size matters more (PCIe latency is higher than NVLink)

### Protocol Guidance
- LL128 may not be supported on all PCIe platforms
- Typically limited to LL + Simple
- Simple preferred for large messages; LL for small

### Common Pitfalls
- Disabling SHM on cross-socket configurations → poor PCIe P2P performance
- Using NVLink-optimized channel counts → SM waste with no bandwidth gain
- Not checking PCIe topology with `nvidia-smi topo -m`

---

## Multi-Node — InfiniBand (IB/RoCE)

### Transport Characteristics
- IB Verbs transport for inter-node communication
- HDR InfiniBand: 200 Gbps (25 GB/s) per port
- NDR InfiniBand: 400 Gbps (50 GB/s) per port
- RoCE v2: RDMA over Converged Ethernet (similar protocol, different fabric)

### GPUDirect RDMA (GDR)
Critical optimization when GPU and NIC share the same PCIe switch:
- **With GDR**: GPU memory → NIC → network (direct, fast)
- **Without GDR**: GPU memory → host memory → NIC → network (extra PCIe hop)
- `NCCL_NET_GDR_READ=1` enabled by default on NVLink platforms (since NCCL 2.4.2)
- `NCCL_NET_GDR_LEVEL` controls the maximum PCIe distance for GDR activation

### GDR Level Settings
```
NCCL_NET_GDR_LEVEL=LOC   # Same PCI device only
NCCL_NET_GDR_LEVEL=PHB   # Same PCIe bridge/switch
NCCL_NET_GDR_LEVEL=SYS   # Across NUMA nodes (most permissive)
```

GDR is most beneficial when GPU and NIC are on the same PCIe switch (PHB level).
Enabling GDR across NUMA boundaries (SYS) may or may not help — benchmark required.

### Queue Pair Architecture
Each NCCL channel creates Queue Pairs (QPs) to remote peers:
- **Forward QP**: carries actual data via RDMA_WRITE (bulk)
- **Reverse QP**: carries CTS (Clear-to-Send) signals (tiny, ~8 bytes)
- Local ordering barrier: dummy RDMA_READ on self-loopback QP

### Multi-QP for ECMP Routing
On multi-level InfiniBand fabrics (fat-tree, dragonfly):
```
NCCL_IB_QPS_PER_CONNECTION=2   # Default: 1
# Increase to 2-4 for multi-level fabrics to improve ECMP hash entropy
# Each QP gets a different source port → different routing path
```

### Channel and NIC Assignment
- Default: 2 channels per remote GPU per NIC (`NCHANNELS_PER_NET_PEER`)
- `NCCL_CROSS_NIC=2` (default): different rings/trees may use different NICs
- On multi-NIC nodes, channels help balance traffic across NICs
- Total channels = NIC count × channels_per_peer for inter-node traffic

### Recommended Settings
```
# InfiniBand multi-node baseline
NCCL_NET_GDR_LEVEL=PHB           # Enable GDR at PCIe switch level
NCCL_NET_GDR_READ=1              # Enable GDR for read operations
NCCL_IB_QPS_PER_CONNECTION=1     # Increase for multi-level fabrics
NCCL_CROSS_NIC=2                 # Balance across NICs (default)
NCCL_IB_TIMEOUT=22               # Default; increase for large clusters
```

### Algorithm Guidance
- Tree increasingly preferred as GPU count grows (O(log k) vs O(k) latency)
- Ring still best for very large messages at moderate scale
- CollNet/SHARP essential at 1000+ GPUs (reduces SM footprint)

### Protocol Guidance
- Small messages (<64KB): LL outperforms Simple (startup latency dominates)
- Large messages (GB range): Simple dominates (sync overhead amortized)
- LL128: good for medium messages, but can underperform at extreme scale
  due to accumulated per-128-byte sync overhead

### Common Pitfalls
- Not enabling GDR when GPU and NIC share PCIe switch → massive performance loss
- Setting IB_TIMEOUT too low on large clusters → spurious disconnections
- Ignoring NIC-GPU affinity (non-local NIC → extra PCIe hops)
- Copying NVLink-optimal settings to IB cluster → wrong trade-off point

---

## Multi-Node — TCP/Ethernet

### Transport Characteristics
- Socket transport: GPU → pinned_host → TCP socket → pinned_host → GPU
- Bidirectional PCIe copy overhead on both ends
- TCP stack overhead: buffering, congestion control, segmentation
- Typical effective bandwidth: 10-25 Gbps per connection

### Critical Tuning: Socket Threads and Connections
These are the most impactful parameters for Ethernet:
```
NCCL_SOCKET_NTHREADS=4          # CPU threads for socket I/O (default: 1)
NCCL_NSOCKS_PERTHREAD=4         # Sockets per thread (default: 1)
# Total connections = NTHREADS × NSOCKS = 16 parallel TCP streams
# More streams → better aggregate TCP bandwidth utilization
```

### Why Multiple Sockets Help
A single TCP connection cannot saturate a high-bandwidth link due to:
- TCP congestion window limits
- Kernel buffer sizes
- CPU scheduling delays
Multiple parallel streams aggregate bandwidth more effectively.

### Recommended Settings
```
# Ethernet/TCP multi-node baseline
NCCL_SOCKET_NTHREADS=4
NCCL_NSOCKS_PERTHREAD=4
NCCL_BUFFSIZE=4194304           # 4MiB (default), may increase for high latency
NCCL_MAX_NCHANNELS=2            # Fewer channels (TCP overhead dominates)
NCCL_ALGO=TREE                  # Tree preferred (lower startup latency per step)
```

### Protocol Guidance
- Simple generally preferred (minimize number of small transfers)
- LL/LL128 less beneficial over high-latency networks
- The fine-grained synchronization of LL/LL128 creates excessive small TCP sends

### Common Pitfalls
- Using default SOCKET_NTHREADS=1 → severely under-utilizing available bandwidth
- Setting too many channels → each channel gets insufficient TCP bandwidth
- Not increasing buffer size for high-latency links
- Using LL protocol → flood of tiny TCP packets → poor throughput

---

## Heterogeneous Topology (NVLink Intra + IB Inter)

### Architecture Pattern
Most production training clusters use:
- **Intra-node**: NVLink/NVSwitch (high bandwidth, low latency)
- **Inter-node**: InfiniBand (moderate bandwidth, higher latency)

### Hierarchical Communication
NCCL automatically constructs hierarchical algorithms:
- **Tree**: Chain topology within each node, binary tree across nodes
- **Ring**: NVLink ring within node, IB ring across nodes
- The intra-node portion uses NVLink transport; inter-node uses IB Verbs

### Parallelism Mapping
Standard practice for large language models:
- **Tensor Parallelism (TP)**: Within node (NVLink) — frequent, moderate-sized all-reduce
- **Data Parallelism (DP)**: Across nodes (IB) — infrequent, large gradient all-reduce
- **Pipeline Parallelism (PP)**: Across nodes (IB) — point-to-point, latency-sensitive

### Tuning Strategy
The key insight: intra-node and inter-node have different optimal settings,
but NCCL environment variables apply globally. Strategies:

1. **Optimize for the bottleneck** — usually inter-node IB
2. **Use per-function algorithm control** (NCCL 2.24+):
   ```
   NCCL_ALGO="allreduce:tree;broadcast:ring"
   ```
3. **Separate communicators**: Create different communicators for TP (intra-node)
   and DP (inter-node), each with different env var settings via tuner plugin
4. **SHARP for inter-node**: Offloads reduction to switches, frees GPU SMs
   for intra-node work

### Multi-NIC Considerations
On nodes with multiple NICs (common in DGX/HGX):
- Each NIC is typically affine to a subset of GPUs
- NCCL auto-detects NIC-GPU affinity via PCIe topology
- `NCCL_CROSS_NIC=2` allows different algorithms to use different NICs
- Ensure NIC firmware and driver versions are consistent across nodes

---

## Topology Detection and Debugging

### Key Diagnostic Commands
```bash
# GPU interconnect topology
nvidia-smi topo -m

# NIC-GPU affinity
nvidia-smi topo -p

# NCCL topology detection output
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH,TUNING <application>

# IB device and port info
ibstat

# PCIe topology
lspci -tv
```

### NCCL Debug Subsystems
- `INIT`: Communicator initialization, transport selection
- `GRAPH`: Topology graph construction, ring/tree selection
- `TUNING`: Algorithm/protocol/channel selection per operation
- `NET`: Network transport details (IB, Socket)
- `ENV`: Environment variable parsing and application

### Topology Graph Interpretation
NCCL logs `Ring` and `Tree` patterns showing which GPUs connect to which:
```
Ring 0: 0→1→2→3→4→5→6→7→0
Tree 0: 0→1, 1→(2,3), 2→(4,5), 3→(6,7)
```
Verify these match expected hardware topology. Mismatched patterns indicate
topology detection errors or suboptimal PCIe/NVLink configuration.

---

## Platform-Specific Quick Reference

### DGX H100 (8x H100, NVSwitch, 8x ConnectX-7)
```
# Defaults are well-tuned. Key overrides for specific workloads:
NCCL_NVLS_ENABLE=1              # NVLink SHARP (default on)
NCCL_NET_GDR_LEVEL=PHB          # GDR for IB (auto-detected)
NCCL_CROSS_NIC=2                # Multi-NIC balancing (default)
# For TP-heavy workloads: NCCL_MAX_NCHANNELS=8 to free SMs
# For DP-heavy workloads: NCCL_MIN_NCHANNELS=8 to saturate IB
```

### DGX A100 (8x A100, NVSwitch, 8x ConnectX-6)
```
NCCL_NVLS_ENABLE=0              # NVLS not supported on A100 NVSwitch
NCCL_NET_GDR_LEVEL=PHB
# LL128 works well intra-node
# Tree preferred for multi-node at moderate GPU counts
```

### PCIe Cluster (no NVLink)
```
NCCL_SHM_DISABLE=0              # Critical for cross-socket
NCCL_P2P_LEVEL=PHB
NCCL_MAX_NCHANNELS=4
# Shared memory transport handles cross-socket communication
# Lower channel count respects PCIe bandwidth limits
```

### Cloud Instances (Ethernet only)
```
NCCL_SOCKET_NTHREADS=4
NCCL_NSOCKS_PERTHREAD=4
NCCL_ALGO=TREE
NCCL_PROTO=SIMPLE
NCCL_MAX_NCHANNELS=2
NCCL_BUFFSIZE=8388608           # 8MiB for high-latency links
# Minimize protocol overhead; maximize TCP stream parallelism
```

Keywords: topology, nvlink, pcie, infiniband, roce, ethernet, gdrdma, transport,
p2p, shared_memory, multi_nic, cross_nic, queue_pair, socket, tcp, dgx, hgx,
nvswitch, hierarchical, tensor_parallelism, data_parallelism, affinity
