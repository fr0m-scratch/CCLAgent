# NCCL Workload-Specific Tuning Patterns

## Overview

Different distributed training workloads produce fundamentally different collective
communication patterns. A configuration optimal for data-parallel gradient all-reduce
may be suboptimal for tensor-parallel activation all-reduce. This document maps common
training paradigms to their communication characteristics and optimal NCCL settings.

Source: Megatron-LM (SC'21), "Demystifying NCCL" (arXiv 2507.04786), NVIDIA NCCL
tuning blog 2025, DeepSpeed documentation

---

## Large Language Models (Transformer-based)

### 3D Parallelism Overview
Modern LLM training uses three orthogonal parallelism dimensions:

| Dimension | Collective Pattern | Message Size | Frequency | Transport |
|-----------|-------------------|-------------|-----------|-----------|
| Tensor Parallelism (TP) | AllReduce | Medium (MB range) | Every layer | NVLink (intra-node) |
| Data Parallelism (DP) | AllReduce | Large (GB range) | Every iteration | IB (inter-node) |
| Pipeline Parallelism (PP) | Send/Recv | Small-medium | Per micro-batch | IB (inter-node) |

### Tensor Parallelism (TP) Tuning
TP partitions each transformer layer across GPUs within a node. Each layer forward/backward
requires an all-reduce of activations/gradients.

**Communication characteristics**:
- Frequency: 2× per transformer layer (forward + backward)
- Message size: proportional to `batch_size × seq_len × hidden_dim / TP_degree`
- Typical sizes: 1-50 MB for common model configurations
- Latency-sensitive: on the critical path of every layer computation

**Optimal NCCL settings**:
```
# TP all-reduce: medium messages, latency-critical, intra-node
NCCL_ALGO=TREE              # O(log k) latency
NCCL_PROTO=LL128            # Low latency + good bandwidth
NCCL_MAX_NCHANNELS=8        # Moderate — preserve SMs for compute
```

**Sequence Parallelism extension**:
Replaces TP all-reduce with reduce-scatter + all-gather (same total communication cost
but reduces activation memory by TP degree). Communication pattern changes:
- ReduceScatter: after attention/MLP forward
- AllGather: before attention/MLP forward
- Same tuning principles apply (latency-sensitive, medium messages)

### Data Parallelism (DP) Tuning
DP replicates the model across GPU groups. Gradients are all-reduced after backpropagation.

**Communication characteristics**:
- Frequency: once per training iteration
- Message size: total gradient size (often GB range for large models)
- Can overlap with backward computation (start all-reduce for earlier layers)
- Bandwidth-dominated: large messages amortize startup latency

**Optimal NCCL settings**:
```
# DP gradient all-reduce: large messages, bandwidth-critical, cross-node
NCCL_ALGO=RING              # Best bandwidth for large messages
NCCL_PROTO=SIMPLE           # Highest throughput, sync overhead amortized
NCCL_MIN_NCHANNELS=8        # Saturate available IB bandwidth
```

**Gradient accumulation note**: With gradient accumulation (micro-batches),
all-reduce happens every N micro-batches, reducing communication frequency.
This makes each all-reduce larger and more bandwidth-sensitive.

### Pipeline Parallelism (PP) Tuning
PP partitions the model vertically (by layers) across nodes. Micro-batches are
pipelined through stages.

**Communication characteristics**:
- Point-to-point Send/Recv between adjacent pipeline stages
- Message size: activation tensor per micro-batch (varies widely)
- Latency-sensitive: pipeline bubble time depends on communication latency
- Often inter-node (IB)

**Optimal approach**:
- PP uses ncclSend/ncclRecv, not collective algorithms
- Minimize inter-node latency: GDR, appropriate IB settings
- `NCCL_BUFFSIZE` affects pipelining of large activations

### Combined 3D Parallelism Example
MT-NLG 530B configuration (NVIDIA, SC'21):
- 8-way TP within each node (NVLink)
- 35-way PP across nodes (IB)
- DP degree = total_GPUs / (TP × PP)
- Key insight: "Sub-optimal combinations of TP+PP can lead to 2× lower throughput"

### Per-Function Algorithm Control (NCCL 2.24+)
```
# Different algorithms for different parallelism dimensions
NCCL_ALGO="allreduce:tree;broadcast:ring"
# TP all-reduce uses Tree; PP broadcast uses Ring
```

---

## Vision Models (CNN-based)

### Communication Pattern
Pure data parallelism is standard for vision models (ResNet, VGG, EfficientNet):
- Single collective: gradient all-reduce after backpropagation
- No TP or PP (vision models fit within single GPU)

### Gradient Size Distribution
- Convolutional layers: large gradient tensors (millions of parameters)
- Batch normalization: small gradient tensors
- Fully connected layers: variable (large for classification heads)

**Total gradient sizes** (approximate):
| Model | Parameters | Gradient Size (FP32) |
|-------|-----------|---------------------|
| ResNet-50 | 25M | 100 MB |
| ResNet-152 | 60M | 240 MB |
| VGG-19 | 144M | 576 MB |
| EfficientNet-B7 | 66M | 264 MB |

### Overlap Strategy
Framework-level gradient bucketing (PyTorch DDP, Horovod):
- Group small gradients into buckets (default 25 MB in PyTorch)
- Start all-reduce for each bucket as soon as all gradients in it are ready
- Earlier layers (closer to input) finish backward last → their all-reduce starts last
- Later layers (closer to output) finish backward first → their all-reduce starts first

**NCCL implications**:
- Multiple concurrent all-reduce operations (one per bucket)
- `ncclGroupStart()`/`ncclGroupEnd()` brackets enable concurrent scheduling
- NCCL dynamically reduces CTAs per operation when many run concurrently

### Optimal NCCL Settings
```
# Vision DP: large gradients, overlap with backward, multi-bucket
NCCL_ALGO=RING              # Best bandwidth for large gradients
NCCL_PROTO=SIMPLE           # Peak throughput
NCCL_MIN_NCHANNELS=4        # Enough to saturate links
NCCL_MAX_NCHANNELS=8        # But not too many (need SMs for compute overlap)
```

---

## Mixture-of-Experts (MoE)

### Communication Pattern
MoE models route tokens to different expert sub-networks. This requires:
- **All-to-All**: redistribute tokens to the GPU hosting the selected expert
- Frequency: every MoE layer (typically every other transformer layer)
- Message size: depends on expert capacity factor and token count

### All-to-All Characteristics
Unlike AllReduce, All-to-All is:
- Asymmetric: each GPU sends different amounts to different destinations
- Latency-sensitive at small expert sizes
- Bandwidth-sensitive at large expert counts
- Not directly controlled by NCCL_ALGO (uses specialized internal algorithm)

### Tuning Approach
```
# MoE All-to-All: many small messages, latency-sensitive
NCCL_PROTO=LL               # Low latency for small expert chunks
NCCL_MIN_NCHANNELS=4        # Enough parallelism for many destinations
# Group calls are critical: many concurrent A2A operations in one MoE layer
```

### Expert Parallelism vs Data Parallelism
- EP: different experts on different GPUs → All-to-All for routing
- DP: same experts replicated → AllReduce for gradients
- Combined EP+DP requires both communication patterns with different optimal settings

---

## Communication-Compute Overlap Patterns

### The Fundamental Principle
Overlap means executing communication and computation simultaneously on different
hardware resources (GPU SMs for compute, NIC/NVLink for communication). Overlap is
only beneficial when:
1. Communication is NOT on the critical path (can run in background)
2. Enough SMs remain for compute after communication claims some
3. The communication kernel doesn't interfere with compute kernel scheduling

### Overlap Opportunities by Training Phase

| Phase | Overlappable Communication | Compute Running |
|-------|--------------------------|-----------------|
| Forward pass | TP all-reduce (between layers) | Next layer computation |
| Backward pass | DP gradient all-reduce (per bucket) | Earlier layer backward |
| Optimizer step | None (synchronization point) | None |
| PP forward | Recv from previous stage | Current stage forward |
| PP backward | Send to previous stage | Current stage backward |

### SM Budget Management
Total GPU SMs (e.g., H100 = 132 SMs) must be shared:
- Communication kernels: NCCL_MAX_NCHANNELS × 1 SM each
- Compute kernels: remaining SMs for model forward/backward

**Rule of thumb**: Communication needs enough SMs to keep links busy, but
the marginal benefit of extra communication SMs decreases rapidly once
link bandwidth is saturated.

```
# SM-conscious overlap settings
NCCL_MAX_NCHANNELS=8        # 8 SMs for comm (6% of H100)
NCCL_MAX_CTAS=8             # Cap communication CTA count
# Leaves 124 SMs for compute (94% of H100)
```

### SHARP for Better Overlap (NCCL 2.27+)
SHARP offloads reduction to switches, reducing GPU SM usage:
- Without SHARP: 16+ SMs for communication
- With SHARP: 6 or fewer SMs
- Frees 10+ SMs for compute → measurably better overlap
- NCCL 2.27+: ReduceScatter/AllGather SHARP preferred over AllReduce SHARP
  for overlap (allows splitting comm into overlappable phases)

### NVSHMEM for Extreme TP Overlap
NVSHMEM provides one-sided communication primitives:
- Up to 36% speedup over NCCL for long-context TP across nodes
- Finer-grained overlap at the warp/thread level
- Complementary to NCCL (different communication patterns)

---

## DeepSpeed ZeRO Integration

### ZeRO Communication Patterns
ZeRO partitions optimizer states, gradients, and/or parameters across GPUs:

| ZeRO Stage | What's Partitioned | Communication Pattern |
|-----------|-------------------|----------------------|
| Stage 1 | Optimizer states | AllReduce for gradients (same as DP) |
| Stage 2 | + Gradients | ReduceScatter for gradients |
| Stage 3 | + Parameters | AllGather before forward/backward + ReduceScatter for gradients |

### ZeRO Stage 2 Tuning
ReduceScatter replaces AllReduce for gradients:
- Each GPU gets only its partition of the reduced gradient
- Message size per GPU = total_gradient / GPU_count
- Same bandwidth requirement as AllReduce, but different collective

```
# ZeRO Stage 2
NCCL_ALGO=RING              # ReduceScatter uses Ring
NCCL_PROTO=SIMPLE           # Large gradients → bandwidth-critical
```

### ZeRO Stage 3 Tuning
AllGather is needed before every forward/backward to reconstruct parameters:
- Frequent, on critical path
- Message size: parameter partition size
- Must be fast to avoid blocking computation

```
# ZeRO Stage 3: frequent AllGather + ReduceScatter
NCCL_MIN_NCHANNELS=4        # Enough parallelism for both collectives
# AllGather and ReduceScatter may run concurrently (group calls)
```

### ZeRO++ Optimizations
- Hierarchical partitioning: reduce cross-node communication
- Quantized gradients: smaller message sizes → different protocol/algorithm optimal
- ZeRO-Offload: moves some compute to CPU → changes SM availability dynamics

---

## FSDP (Fully Sharded Data Parallelism)

### Communication Pattern
PyTorch FSDP is conceptually similar to ZeRO Stage 3:
- AllGather before forward pass (reconstruct parameters)
- ReduceScatter after backward pass (partition gradients)
- Optional: AllGather pre-fetching (overlap with compute)

### Tuning Considerations
- FSDP unit granularity affects message sizes
- Larger FSDP units: fewer, larger collectives → bandwidth-optimal
- Smaller FSDP units: more, smaller collectives → latency-sensitive
- Auto-wrap policy determines unit boundaries

---

## Workload Fingerprinting for Auto-Tuning

### Key Metrics to Extract
When profiling a workload for NCCL tuning, collect:

1. **Collective type distribution**: % AllReduce vs ReduceScatter vs AllGather vs AlltoAll
2. **Message size histogram**: distribution of collective sizes (bimodal is common)
3. **Collective frequency**: calls per second (determines latency sensitivity)
4. **Overlap opportunity**: % of communication overlappable with compute
5. **GPU utilization during communication**: SM utilization when NCCL kernels run
6. **Network utilization**: % of peak link bandwidth consumed

### Workload Categories
| Category | Primary Collective | Size Range | Latency Sensitivity |
|----------|-------------------|-----------|-------------------|
| Pure DP (vision) | AllReduce | 100MB-1GB | Low (overlapped) |
| TP-heavy (LLM) | AllReduce | 1-50MB | High (critical path) |
| PP-heavy (LLM) | Send/Recv | 1-100MB | High (pipeline bubble) |
| ZeRO-3/FSDP | AllGather + ReduceScatter | 10-500MB | Medium |
| MoE | AlltoAll | 1-100MB | High (critical path) |
| Hybrid 3D | Mixed | Mixed | Mixed |

### Microbenchmark-to-Workload Translation
**Critical warning**: Optimizing NCCL microbenchmarks (nccl-tests) does not guarantee
workload improvement. Common reasons:

1. Benchmark uses all SMs for communication → no compute interference
2. Benchmark runs single collective → no concurrent operation scheduling
3. Benchmark uses fixed message size → misses bimodal distribution
4. Benchmark ignores memory pressure from model parameters

Always validate tuning improvements with end-to-end workload benchmarks.

Keywords: llm, training, tensor_parallelism, data_parallelism, pipeline_parallelism,
sequence_parallelism, moe, expert_parallelism, zero, fsdp, overlap, compute_overlap,
gradient_bucketing, allreduce, reducescatter, allgather, alltoall, send_recv,
megatron, deepspeed, vision, cnn, resnet, vgg, workload_fingerprint, microbenchmark
