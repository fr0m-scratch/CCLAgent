# Current Implementation and Design (CCL Agent)

This document describes the current implementation as it exists in this repo, maps it to the intended
three-phase design, and lists the major TODOs/gaps. It also includes a repository structure guide and
end-to-end workflows.

Scope: The details below reflect the code under `src/` and the scripts/workloads in this repo as of the
current checkout. This is an implementation-focused doc, not a new design proposal.

For a call-by-call walkthrough of a **training** run (CLI → agent → tools → artifacts/trace → post-run),
see: `doc/Design/end_to_end_training_workflow.md`.

---

## 1) Design overview (as implemented)

The system follows the three-phase design described in `doc/Design/coreNovelty.md`:

1. Offline planning (microbench + knowledge grounding)
2. Online tuning (hypothesis steps + numeric steps)
3. Post-run updating (rules + surrogate data)

The implementation is a working scaffold with dry-run defaults and a minimal agent loop. The key logic
lives in `src/agent/` and `src/tools/`.

---

## 2) Current implementation mapping

### 2.1 Entry point and orchestration

- CLI entry point: `src/main.py`
  - Loads agent config (or defaults) and workload spec.
  - Builds tool suite via `build_tools`.
  - Instantiates `CCLAgent` and runs `agent.tune(workload)`.

### 2.2 Agent core loop (online tuning)

- `src/agent/core.py` (`CCLAgent`)
  - Phase 1: `planner.build_context`, `planner.offline_plan`, `planner.propose_initial_config`
  - Phase 2: executes actions for `budget.max_steps`:
    - apply config, run workload, collect metrics
    - update surrogate model
    - stop on SLA violation or plateau
    - choose next action via `DecisionPolicy`
  - Phase 3: `post_run` updates memory with rules and surrogate records

### 2.3 Offline planning

- `src/agent/planner.py` (`OfflinePlanner`)
  - `offline_plan` calls `tools.microbench.run`.
  - `propose_initial_config`:
    - starts from parameter-space defaults
    - applies top memory rules (exact context match)
    - **always** queries the LLM (default provider: Ollama) using a sectioned prompt that includes microbench signals, rules, and RAG snippets
  - RAG is simple Jaccard similarity over documents in `doc/Design` by default.

### 2.4 Decision policy (hypothesis + numeric)

- `src/agent/policy.py` (`DecisionPolicy`)
  - Alternates hypothesis steps every `budget.hypothesis_every` iterations.
  - Hypothesis step:
    - applies best rule patch if any
    - otherwise mutates the best config on important parameters
  - Numeric step:
    - uses `SurrogateModel.suggest` if available
    - otherwise uses `NumericSearchTool` (async batch scoring)
    - fallback: mutate candidates and pick best predicted by surrogate

### 2.5 Execution and SLA enforcement

- `src/agent/executor.py` (`WorkloadExecutor`)
  - Validates config through `tools.nccl.apply` (parameter validation).
  - Collects env overrides from `ext_tuner`, `autoccl`, `ext_net`.
  - Uses `TrainingJobRunner` if workload kind is `training`, else `WorkloadRunner`.
  - Applies SLA check (iteration time and error budget).

### 2.6 Post-run memory and surrogate data

- `src/memory.py` (`MemoryStore`)
  - Writes rules and surrogate records to JSON (default: `memory/agent_memory.json`).
  - Rules are indexed only by exact context match (workload/topology/scale/nodes).
  - Surrogate records store full metrics per config for future modeling.

---

## 3) Components and modules

### 3.1 Data models

- `src/types.py`
  - `ParameterSpec`, `ParameterSpace`, `NCCLConfig`, `MicrobenchResult`, `Metrics`
  - `WorkloadSpec`, `ContextSignature`
  - `TuningAction`, `TuningRecord`, `SearchCandidate`, `SearchResult`
  - `TuningBudget`, `AgentConfig`, `ToolRegistry`

### 3.2 Configuration

- `src/config.py`
  - Default parameter space includes:
    - `NCCL_ALGO`, `NCCL_PROTO`, `NCCL_NTHREADS`, `NCCL_BUFFSIZE`,
      `NCCL_MIN_NCHANNELS`, `NCCL_MAX_NCHANNELS`
  - Default RAG docs path: `doc/Design`
  - Default LLM settings:
    - provider: `ollama`
    - model: `deepseek-r1:8b`
    - prompt window: `max_context_tokens=8000`, `max_response_tokens=512`
  - Config file format:
    - `parameters`: list of `ParameterSpec` entries
    - `budget`: `TuningBudget` fields
    - `memory_path`, `rag_docs_path`, `rag_top_k`, `sla_max_iteration_time`

### 3.3 Tool suite (offline + online)

- `src/tools/microbench.py` (`MicrobenchRunner`)
  - Dry-run by default.
  - Real mode expects JSON output with `important_params` and `signals`.
- `src/tools/workload.py` (`WorkloadRunner`)
  - Runs workload command, parsing JSON metrics if a parser is supplied.
  - Dry-run simulates metrics based on config hash.
- `src/tools/training.py` (`TrainingJobRunner`)
  - Same interface as workload runner; used when workload kind is `training`.
- `src/tools/metrics.py` (`MetricsCollector`)
  - Parses JSON from stdout to `Metrics`.
- `src/tools/config_compiler.py` + `src/tools/nccl.py`
  - Validates config values against parameter space; no system-level apply.
- `src/tools/numeric_search.py`
  - Async candidate scoring with configurable concurrency.
- `src/tools/sla.py`
  - Checks iteration time and error budget.
- `src/tools/autoccl.py`, `src/tools/ext_tuner.py`
  - AutoCCL-compatible bridge for env overrides and candidate export.
- `src/tools/ext_net.py`
  - Environment setup for NCCL NET plugins.
- `src/tools/nccltest.py`
  - Minimal nccl-tests runner (dry-run by default).

### 3.4 LLM and RAG

- `src/llm/`
  - OpenAI-compatible client (OpenAI, Fireworks, generic).
  - Claude and Gemini clients (require vendor SDKs).
  - Ollama client for local default runs.
  - Null client for explicit non-LLM runs (if configured).
- `src/RAG/store.py`
  - In-repo document store with Jaccard similarity search.

### 3.5 External references / submodules

- `tools/autoccl/` (git submodule)
  - AutoCCL source and ext-tuner example.
- `tools/AF_ICSE26/`
  - Patch and scripts for CCLInsight microbench and NCCL changes.
- `reference/PFSAgent/`
  - External reference implementation (not wired into the agent).

---

## 4) Repository structure (current)

Top-level:

- `doc/`
  - `Design/` - design notes (`coreNovelty.md`, `techStack.md`, this file)
  - `figures/` - design diagrams referenced by README
  - `references/` - extra docs
- `src/` - core implementation
  - `agent/` - planner, policy, executor, state, ext-tuner session
  - `tools/` - microbench, workload runners, NCCL compiler, numeric search, etc.
  - `llm/` - LLM client interfaces
  - `RAG/` - local document store
  - `memory.py`, `config.py`, `types.py`, `main.py`
- `workload/`
  - `autoccl/` - AutoCCL-style workload specs
  - `README.md` - workload spec guidance
- `scripts/`
  - Agent and AutoCCL demo scripts
- `tools/`
  - AutoCCL submodule + AF_ICSE26 artifacts
- `reference/` - reference agent code (PFSAgent)

---

## 5) End-to-end workflows

### 5.1 Dry-run pipeline (fast sanity check)

This uses simulated microbench + simulated workload metrics.

```bash
python3 -m src.main --workload workload/autoccl/phi2-2b.json --dry-run
```

### 5.2 Agent + torchrun demo (local)

Runs the agent against the AutoCCL torch demo script (requires the AutoCCL submodule checkout).

```bash
bash scripts/run_agent_torch_demo.sh
```

What happens:
1. `src/main.py` loads `workload/autoccl/torch-demo.json`.
2. `CCLAgent` runs offline planning (microbench, RAG, memory rules).
3. Agent applies config and runs `scripts/agent_torchrun_demo.sh` via `TrainingJobRunner`.
4. Metrics are parsed (JSON expected on stdout).
5. Agent iterates until plateau/SLA/budget.
6. Post-run: memory updates are written to `memory/agent_memory.json`.

### 5.3 AutoCCL standalone example (baseline comparison)

This script runs the AutoCCL demo without the CCL Agent (useful for baseline behavior).

```bash
bash scripts/run_autoccl_example.sh
```

### 5.4 Ext-tuner integration mode (programmatic)

For external NCCL ext-tuner workflows, the agent exposes an adapter:

- `src/agent/ext_tuner.py` (`ExtTunerSession`)
  - `initial_config()` and `report_metrics()` allow external drivers to pull configs and report metrics.

---

## 6) TODOs and gaps (current)

Implementation gaps relative to the intended design:

1. Real microbench integration
   - Wire `MicrobenchRunner` to actual NCCL primitive microbench runs (e.g., CCLInsight scripts).
2. Metrics parsing
   - Standardize workload output schema and parsing for iteration time, comm time, bandwidth.
3. NCCL config application
   - Implement concrete config application beyond validation, including env propagation or plugin controls.
4. RAG quality
   - Replace Jaccard similarity with embeddings and a real index if needed.
5. Memory retrieval
   - Add fuzzy/contextual matching beyond exact context keys.
6. Surrogate modeling
   - Replace the in-memory kNN surrogate with persisted or learned models.
7. Numeric search evaluation
   - Integrate real candidate evaluation rather than surrogate-only scoring.
8. Safety + rollback
   - Enforce safer rollback rules and preflight validation for risky configs.
9. Multi-node orchestration
   - Provide launch helpers for Slurm/MPI/torchrun in multi-node settings.
10. Tests and validation
   - Add unit tests for policy logic, config validation, and metric parsing.

---

## 7) Notes on current defaults

- Dry-run is the default for microbench, workload, training, and nccltest tools.
- LLM calls are always attempted (even in dry-run); default provider is **Ollama**.
- Memory persistence is JSON-based and updated after a tuning run.
- Default RAG docs path is `doc/Design` and should be kept updated.
