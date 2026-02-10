CCL Agent
=========

CCL Agent is a closed-loop tuner for NCCL communication. It combines gray-box measurements, knowledge-grounded reasoning, and bounded search to generate safer and more transferable communication configs.

What It Does
------------
- Runs an offline planning step using microbench/probe signals.
- Runs online tuning with hypothesis steps and numeric search.
- Distills post-run artifacts for reuse in future runs.

Quick Start
-----------
1. Dry-run sanity check:

```bash
python3 -m src.main --workload workload/benchmarks/phi2-2b.json --dry-run
```

2. End-to-end live run (with TUI):

```bash
python3 -m src.runner \
  --mode live \
  --config configs/agentic_showcase_kimi.json \
  --workload workload/benchmarks/llama3.1-8b-agentic-showcase.json \
  --provider fireworks \
  --model accounts/fireworks/models/kimi-k2p5 \
  --dry-run \
  --simulate-workload
```

3. Inspect an existing run:

```bash
python3 scripts/agent_tui.py --run-dir artifacts/<run_id>
```

Useful Scripts
--------------
- `scripts/run_agentic_showcase_kimi.sh`: showcase run wrapper.
- `scripts/run_benchmark_matrix.sh`: matrix runs across workload/config combinations.
- `scripts/build_plugins.sh`: build C++ profiler/tuner plugin artifacts.
- `scripts/run_tests.sh`: compile checks + pytest + dry-run smoke.

RAG Inputs
----------
By default, RAG loads content from:
- `doc/Knowledge`
- `README`
- `workload`

(You can override this through `rag.docs_paths` in config.)

Repository Layout
-----------------
- `src/`: core implementation (agent, tools, llm, observability, trace, RAG).
- `scripts/`: runnable utilities and demos.
- `configs/`: runtime configs.
- `workload/benchmarks/`: workload specs.
- `doc/Knowledge/`: NCCL knowledge references used by RAG.
- `doc/Prompts/`: prompt version documents.
- `tests/`: unit/integration tests.
- `cpp/`: C++ plugin components.
- `tools/`: external/reference tooling (includes AutoCCL submodule).

Notes
-----
- The agent attempts LLM calls even in dry-run mode.
- Select model/provider via CLI (`--provider`, `--model`) or config (`llm.provider`, `llm.model`).
