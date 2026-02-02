NCCL Parameter Notes (Local)
============================

This is a curated scratchpad of commonly tuned NCCL env vars and their intent.
Keep this file short and concrete; add citations to upstream docs as needed.

Core parameters
---------------
- NCCL_ALGO: collective algorithm (RING, TREE, COLLNET)
- NCCL_PROTO: protocol (LL, LL128, SIMPLE)
- NCCL_NTHREADS: threads per channel
- NCCL_BUFFSIZE: buffer size (bytes)
- NCCL_MIN_NCHANNELS / NCCL_MAX_NCHANNELS: channel bounds

Network / transport
-------------------
- NCCL_P2P_LEVEL: peer-to-peer level (e.g., NVL, SYS)
- NCCL_NET_GDR_LEVEL: GPUDirect RDMA enablement level
- NCCL_SOCKET_NTHREADS: socket threads
- NCCL_NSOCKS_PERTHREAD: sockets per thread
- NCCL_IB_QPS_PER_CONNECTION: QPs per connection
- NCCL_SHM_DISABLE: disable shared memory transport

Notes
-----
- Interactions between ALGO/PROTO and channel counts are workload/topology specific.
- Higher channels and threads can improve bandwidth but risk oversubscription.
- Buffsize often trades memory for throughput; avoid too small values.
