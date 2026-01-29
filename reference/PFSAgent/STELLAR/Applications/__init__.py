from .base import Application
from .IO500.IO500 import IO500
from .mdworkbench.mdworkbench import MDWorkbench
from .IOR.IOR import IOR
from .H5Bench.H5Bench import H5Bench
APPLICATIONS = {
    "IO500": IO500,
    "MDWorkbench": MDWorkbench,
    "IOR": IOR,
    "H5Bench": H5Bench
}

TUNING_EXPERIENCE = {
    "IO500": [],
    "MDWorkbench": [],
    "IOR": [],
    "H5Bench": [],
    "COMBINED": [
    {
        "parameter_name": "stripe_settings",
        "rule_description": "Implement workload-specific stripe settings based on file size and access patterns: (1) For bandwidth-intensive large file workloads, use higher stripe counts (all available OSTs) and larger stripe sizes (4MB+); (2) For metadata-intensive workloads with many small files, either use a moderate stripe count (approximately 60% of available OSTs) to balance parallelism with coordination overhead, or maintain a stripe count of 1 while reducing stripe size to match file sizes (e.g., from 1M to 256K) to avoid multi-OST overhead; (3) For mixed workloads, keep default or smaller settings.",
        "tuning_context": "This rule applies to environments with varied I/O patterns. For bandwidth-intensive workloads (large sequential files), using all available OSTs improves performance by 110-400%. For metadata-intensive workloads with small files, the approach depends on file size and access pattern: files significantly smaller than stripe size benefit from a count of 1 with reduced stripe size, while slightly larger files may benefit from moderate stripe counts that balance parallelism with overhead."
    },
    {
        "parameter_name": "stripe_count",
        "rule_description": "For parallel I/O workloads with a shared file access pattern, increase stripe count to utilize all available OSTs to maximize parallelism and throughput.",
        "tuning_context": "Workloads that involve multiple processes accessing a single large shared file (>10GB) with predominantly sequential I/O patterns. This maximizes parallelism across storage targets and prevents bottlenecks from single-OST access."
    },
    {
        "parameter_name": "stripe_size",
        "rule_description": "Adjust stripe size based on I/O operation size: (1) for small files significantly below default stripe size, reduce stripe size to match file sizes (e.g., from 1M to 256K); (2) for medium operations (10K-100K range), use 2-4MB; (3) for larger access patterns, set to 1/2 to 1/4 of the common access size, with a minimum of 1MB for high-performance networks.",
        "tuning_context": "When applications show consistent access patterns, adjusting the stripe size to align with these patterns maximizes performance. For very small files, matching stripe size to file size avoids wasted space. For medium-sized operations, a 2-4MB stripe size reduces stripe crossings while maintaining parallelism. For larger operations, aligning with access patterns reduces the number of OSTs needed per operation."
    },
    {
        "parameter_name": "statahead_max",
        "rule_description": "For metadata-intensive workloads, adjust statahead_max based on the intensity of directory traversal operations: (1) For moderately metadata-intensive workloads, increase to around 64-128; (2) For extremely metadata-intensive workloads with frequent stat operations, consider higher values up to the maximum (512), but monitor for diminishing returns (typically around 128-160).",
        "tuning_context": "This rule applies to workloads that involve frequent directory traversals and stat operations. The benefits increase with the frequency of metadata operations, particularly when traversing directories and accessing file metadata more frequently than reading or writing file contents. However, there can be a point of diminishing returns where additional prefetching creates overhead without benefits, varying by workload characteristics."
    },
    {
        "parameter_name": "mdc-max_rpcs_in_flight",
        "rule_description": "For metadata-intensive workloads, adjust mdc-max_rpcs_in_flight based on metadata operation intensity: (1) For moderate metadata workloads, increase from default (8) to around 32; (2) For extremely metadata-intensive workloads with multiple client processes, consider progressive increases to higher values (48, 64), but monitor for performance plateaus.",
        "tuning_context": "This applies to workloads with high volumes of metadata operations, especially when multiple clients simultaneously access the file system performing operations like file creation, deletion, and stat. While substantial performance gains are observed with increasing values, there is typically a point where additional concurrency introduces coordination overhead that can offset benefits."
    },
    {
        "parameter_name": "max_mod_rpcs_in_flight",
        "rule_description": "For workloads with many file creation or modification operations, scale max_mod_rpcs_in_flight proportionally with mdc-max_rpcs_in_flight, keeping it at approximately 75% of the mdc-max_rpcs_in_flight value to maintain a good balance between modifying and non-modifying operations.",
        "tuning_context": "Workloads with frequent file creation, deletion, or metadata modification operations. The proportional scaling ensures that both modifying operations can proceed efficiently while leaving capacity for non-modifying operations, resulting in better overall balance for metadata-intensive workloads."
    },
    {
        "parameter_name": "max_read_ahead_whole_mb",
        "rule_description": "Adjust max_read_ahead_whole_mb based on predominant file size in the workload: (1) For workloads with predominantly small files, reduce to lower values (4-8MB) to prevent wasteful prefetching; (2) For workloads with medium-to-large files and sequential access patterns, increase from default (64MB) to 128MB.",
        "tuning_context": "For workloads with small files, excessive read-ahead wastes resources when files are smaller than the read-ahead value. For workloads with sequential operations on medium-to-large files, increasing this parameter can improve performance by prefetching entire files more aggressively when access patterns are predictable."
    },
    {
        "parameter_name": "max_read_ahead_per_file_mb",
        "rule_description": "Adjust max_read_ahead_per_file_mb based on file size and access patterns: (1) For workloads with predominantly small files, significantly reduce from default values to prevent wasteful prefetching; (2) For workloads with large sequential file reads, increase to 2-4x the default (256MB to 512-1024MB).",
        "tuning_context": "For workloads with small files, reducing this parameter avoids unnecessary prefetching that wastes resources when the default value is much larger than typical file sizes. For workloads with sequential reads on large files (multi-GB), increasing this parameter substantially improves performance by prefetching more data into memory ahead of actual read requests."
    },
    {
        "parameter_name": "max_read_ahead_mb",
        "rule_description": "Adjust max_read_ahead_mb based on workload characteristics: (1) For metadata-intensive workloads with small files, reduce to free memory for metadata caching; (2) For mixed or bandwidth-intensive workloads, increase to 2-4x the default (1024MB to 2048-4096MB) to support aggressive read-ahead strategies.",
        "tuning_context": "For metadata-intensive workloads where prefetching large amounts of file data provides minimal benefit, reducing this value frees memory for metadata operations. For workloads with sequential read components and sufficient memory, increasing this global read-ahead limit allows bandwidth-intensive components to benefit from aggressive prefetching."
    },
    {
        "parameter_name": "osc-max_rpcs_in_flight",
        "rule_description": "For I/O-intensive workloads using multiple OSTs, increase osc-max_rpcs_in_flight based on the workload characteristics: at least 16 (2x default) for mixed workloads, and up to 32 (4x default) for purely bandwidth-intensive workloads, while avoiding excessive increases that can cause contention with metadata operations.",
        "tuning_context": "This applies to data-intensive workloads with sequential access patterns utilizing multiple OSTs through increased stripe count. Insufficient RPC parallelism can lead to network underutilization and reduced performance. We found that moderate increases improved bandwidth for large file operations, but going too high (24+ for mixed workloads) created resource contention that negatively impacted metadata performance."
    },
    {
        "parameter_name": "osc-max_pages_per_rpc",
        "rule_description": "Adjust osc-max_pages_per_rpc based on I/O operation size: (1) For workloads with small I/O operations or small files, reduce to more moderate values (64-128); (2) For bandwidth-intensive workloads with large I/O operations, increase from default (256) to 512-1024.",
        "tuning_context": "For workloads with small read/write operations where the average I/O size is significantly smaller than the default RPC payload size, smaller RPCs avoid wasted space and improve efficiency. For data-intensive workloads with sequential access patterns and medium to large I/O operations, larger values reduce overhead by packing more data into each RPC."
    },
    {
        "parameter_name": "mdc-max_pages_per_rpc",
        "rule_description": "For metadata-intensive workloads, adjust mdc-max_pages_per_rpc based on operation size: (1) For workloads with very small files or pure metadata operations, reduce to lower values (64-128); (2) For mixed workloads, use moderate values (128-256).",
        "tuning_context": "For workloads with very small files or pure metadata operations, smaller RPCs better match the size of metadata transfers and avoid waste. For mixed workloads with both metadata and data operations, moderate values balance efficiency for both types of operations while avoiding both the overhead of too many small RPCs and the waste of too few large ones."
    },
    {
        "parameter_name": "osc-max_dirty_mb and mdc-max_dirty_mb",
        "rule_description": "Adjust max_dirty_mb parameters based on workload characteristics: (1) For workloads with small files or primarily metadata operations, substantially reduce from large defaults (2000MB) to more moderate values (64-128MB); (2) For large file, write-intensive workloads, ensure values are at least 4x the product of (max_pages_per_rpc * max_rpcs_in_flight).",
        "tuning_context": "For workloads with small files or primarily metadata operations, large dirty buffers waste memory that could be better used for other purposes. Reducing these values frees memory for other operations. For large sequential write workloads, higher dirty cache limits ensure enough data can be aggregated to form full RPCs, improving write efficiency."
    }
]
}

__all__ = ["Application", "IO500", "MDWorkbench", "Elbencho", "QuantumEspresso", "IOR", "H5Bench"]
