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

Debug / observability
---------------------
- NCCL_DEBUG: log verbosity (INFO/WARN/TRACE depending on NCCL build)
- NCCL_DEBUG_SUBSYS: subsystem filters (`INIT,GRAPH,NET,TUNE,...`)
- NCCL_TOPO_DUMP_FILE: topology dump path
- NCCL_GRAPH_DUMP_FILE: graph dump path
- NCCL_BLOCKING_WAIT: blocking wait mode for hang diagnosis
- NCCL_ASYNC_ERROR_HANDLING: async error surfacing

Notes
-----
- Interactions between ALGO/PROTO and channel counts are workload/topology specific.
- Higher channels and threads can improve bandwidth but risk oversubscription.
- Buffsize often trades memory for throughput; avoid too small values.
