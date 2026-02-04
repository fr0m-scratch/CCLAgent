# End-to-end training workflow (CCL Agent)

This document is an implementation-accurate walkthrough of what happens when you run the CCL agent on a **`WorkloadSpec.kind == "training"`** workload, starting at the CLI and ending at post-run “training” (dataset/model updates). It is written as a **call trace across source files**: *what calls what*, and *what artifacts/traces are emitted where*.

If you want a higher-level overview first, read:
- `README.md`
- `doc/Design/currentImplementation.md`
- `doc/Design/artifact_schema.md`
- `doc/Design/trace_schema.md`

---

## 0) Quick start commands (training workloads)

Dry-run (simulated microbench + simulated training/workload execution):
```bash
python3 -m src.main --workload workload/autoccl/torch-demo.json --dry-run
```

Notes:
- **LLM calls are always attempted**, even in dry-run. Default is **Ollama** / `deepseek-r1:8b`. Override with `--provider` / `--model`.

Real microbench, simulated training/workload (useful when you don’t want to launch a real job yet):
```bash
python3 -m src.main --workload workload/autoccl/torch-demo.json --simulate-workload
```

Real run (will actually execute `workload.command`):
```bash
python3 -m src.main --workload workload/autoccl/torch-demo.json
```

Run with an LLM provider (offline planner always calls the LLM; see section 6):
```bash
python3 -m src.main --workload workload/autoccl/torch-demo.json --provider openai --model <model>
```

Notes:
- `workload/autoccl/torch-demo.json` has `"kind": "training"`, so the agent will use the **training runner** (`src/tools/training.py`) rather than the generic workload runner (`src/tools/workload.py`).
- `--simulate-execution` is an alias for `--simulate-workload` (both make training/workload/nccl-tests simulated while microbench can remain real).

---

## 1) Big-picture “who talks to who”

At runtime, the call chain is:

1) `src/main.py` (CLI) creates:
   - a `RunContext` (artifacts directory + run_id)
   - a `TraceWriter` and `TraceEmitterWriter` (trace bus)
   - a `ToolSuite` wrapped by `InstrumentedToolSuite` (tool-call tracing)
   - a `MemoryStore` (rules + surrogate records persistence)
   - a `RagStore` (local retrieval over docs)
   - an `LLMClient` wrapped by `TracedLLMClient` (LLM-call tracing)
2) `src/agent/core.py::CCLAgent.tune()` drives a 3-phase loop:
   - offline planning (microbench + retrieval + warm-start selection)
   - online loop (apply config → run training → record metrics → decide next action)
   - post-run updates (rules, dataset export, surrogate model training)

Conceptually:

```
CLI (src/main.py)
  └─ CCLAgent.tune (src/agent/core.py)
      ├─ OfflinePlanner.* (src/agent/planner.py)
      │    ├─ MicrobenchRunner.run (src/tools/microbench.py)
      │    ├─ MemoryStore.retrieve_* (src/memory/__init__.py)
      │    └─ RagStore.search/summarize (src/RAG/store.py)
      ├─ Online loop
      │    ├─ WorkloadExecutor.run (src/agent/executor.py)
      │    │    ├─ NCCLInterface.apply (src/tools/nccl.py)
      │    │    ├─ TrainingJobRunner.run (src/tools/training.py)  <-- training workloads
      │    │    └─ SLAEnforcer.check (src/tools/sla.py)
      │    └─ TuningAnalyzer.plan_next_action (src/agent/analyzer.py)
      │         ├─ HypothesisGenerator.propose_portfolio (src/agent/hypothesis.py)
      │         ├─ ConfigCompiler.compile_hypothesis (src/tools/config_compiler.py)
      │         └─ NumericSearchManager.propose (src/agent/numeric.py)
      │              ├─ CoordinateDescentSearch.propose_candidates (src/search/coordinate_descent.py)
      │              └─ RiskScorer.score (src/safety/risk.py)
      └─ Post-run updates (src/agent/core.py + src/models/training.py + src/agent/distill.py)
```

---

## 2) End-to-end call trace: CLI → agent → training job → post-run

This section is written in the *actual call order* used by `python3 -m src.main ...`.

### 2.1 `src/main.py` — process startup and wiring

Entry point:
- `src/main.py::main()`

What it does, in order:

1) Parse CLI args via `argparse`.
   - Important flags:
     - `--dry-run` (simulate microbench + workload/training)
     - `--simulate-workload` / `--simulate-execution` (simulate workload/training but keep microbench “real” if configured)
     - `--provider` / `--model` (LLM provider/model)
     - `--config` (agent config JSON)
     - `--workload` (workload spec JSON)
     - `--artifacts-root` (root for run artifacts; default `artifacts/`)
2) Load environment variables from an env file:
   - `src/utils/env.py::load_env_file(args.env_file)`
   - Default env file: `.env.local`
3) Load configuration and workload spec:
   - Agent config:
     - `src/config.py::load_agent_config(path)` if `--config` provided
     - otherwise `src/config.py::default_agent_config()`
   - Workload spec:
     - `src/config.py::load_workload_spec(path)` if `--workload` provided
     - otherwise a trivial `WorkloadSpec(name="demo", command=[])`
4) Create a run context and on-disk artifacts directory:
   - `src/utils/artifacts.py::create_run_context(artifacts_root, dry_run, seed)`
   - Creates `artifacts/<run_id>/` and subdirs:
     - `steps/`, `offline/`, `online/`, `postrun/`, `logs/`
   - Writes `artifacts/<run_id>/run_context.json` (then overwritten later with `config_snapshot_path`)
5) Configure LLM tracing directory:
   - If `CCL_LLM_TRACE_DIR` is not already set, set it to:
     - `artifacts/<run_id>/llm/`
   - This is used by:
     - `src/llm/base.py::_trace_llm_event()` (writes `llm_trace.jsonl`)
     - and the wrapper `src/llm/traced_client.py::TracedLLMClient` (writes `call_<id>.json`)
6) Write a config snapshot for reproducibility:
   - `src/config.py::config_to_dict(agent_config)`
   - `src/utils/json_utils.py::write_json(artifacts/<run_id>/config_snapshot.json, ...)`
   - Update and rewrite `run_context.json` to include `config_snapshot_path`.
7) Create the trace bus:
   - `src/trace/writer.py::TraceWriter(run_id, artifacts_dir)`
     - creates/opens `artifacts/<run_id>/trace/events.jsonl` append-only
   - `src/trace/emitter.py::TraceEmitterWriter(trace_writer)`
8) Build tools:
   - `src/main.py::build_tools(config, run_context, dry_run, simulate_workload)`
   - Concrete tools created:
     - `MetricsCollector` (`src/tools/metrics.py`)
     - `MicrobenchRunner` (`src/tools/microbench.py`)
     - `WorkloadRunner` (`src/tools/workload.py`)
     - `TrainingJobRunner` (`src/tools/training.py`)
     - `SLAEnforcer` (`src/tools/sla.py`)
     - `ConfigCompiler` + `NCCLInterface` (`src/tools/config_compiler.py`, `src/tools/nccl.py`)
     - `NcclTestRunner` (`src/tools/nccltest.py`)
     - `ExtTunerBridge`/`AutoCCLBridge` (`src/tools/ext_tuner.py`, `src/tools/autoccl.py`)
     - `ExtNetBridge` (`src/tools/ext_net.py`)
     - `NumericSearchTool` (`src/tools/numeric_search.py`) (note: currently used by `DecisionPolicy`, not by the main analyzer loop)
9) Create storage + retrieval + LLM:
   - Memory:
     - `src/memory/__init__.py::MemoryStore(agent_config.memory, run_context)`
     - loads from `memory/agent_memory.json` by default
   - RAG:
     - `src/RAG/store.py::RagStore(agent_config.rag)`
     - lazily loads docs on first search (or when planner forces load)
   - LLM:
     - `src/llm/__init__.py::create_llm_client(provider, model)`
     - wrap with `src/llm/traced_client.py::TracedLLMClient(...)`
10) Wrap tools for trace instrumentation:
   - `src/tools/instrumented.py::InstrumentedToolSuite(tools, trace_emitter, run_id)`
   - Every tool method call becomes a `tool.call` + `tool.result` event in `trace/events.jsonl`.
11) Construct and run the agent:
   - `src/agent/core.py::CCLAgent(...)`
   - `agent.tune(workload)`
12) Ensure trace writer is closed:
   - `TraceWriter.close()` in a `finally:` block.

Artifacts created at this layer (always):
- `artifacts/<run_id>/config_snapshot.json`
- `artifacts/<run_id>/run_context.json`
- `artifacts/<run_id>/trace/events.jsonl`

---

### 2.2 `src/agent/core.py` — `CCLAgent.tune(workload)` in detail

Entry point:
- `src/agent/core.py::CCLAgent.tune(workload: WorkloadSpec) -> TuningState`

Pseudo-code of the true control flow:
```text
emit trace: run.start

context = planner.build_context(workload)
_load_surrogate(context)                  # load last saved model for this context signature (if any)
microbench = planner.offline_plan(workload)
plan = planner.build_initial_plan(workload, microbench, context)
current_config = plan.baseline_config

state = TuningState(budget)

if execution.mode == "in_job_ext_tuner":
  start ExtTunerServer thread
  executor.run(..., extra_env={"CCL_TUNER_SESSION_DIR": <session_dir>}, execution_mode="in_job_ext_tuner")
  join thread
  return server.session.state

for step in [0..max_steps):
  action = TuningAction(kind="initial" if step==0 else "apply", config=current_config, ...)
  metrics = executor.run(workload, action.config, step, compiled=next_compiled)
  write derived metrics + bottleneck classification
  state.record(TuningRecord(step, action, metrics, microbench_snapshot=...))
  update memory + avoid rules based on success/failure
  update online surrogate model with (config, iteration_time_ms)
  persist step_<k>.json (action+delta+metrics)
  decision = analyzer.plan_next_action(...)
  handle stop/rollback/next-config

post-run: distill rules + export dataset + train surrogate + save memory
emit trace: run.end
return state
```

The next subsections expand each “box” in that pseudo-code.

---

## 3) Offline stage call trace (planning before the first training run)

The offline stage is driven by `src/agent/planner.py::OfflinePlanner`.

### 3.1 Context detection

Call chain:
- `CCLAgent.tune()` → `OfflinePlanner.build_context(workload)`

Where:
- `src/agent/planner.py::build_context()`

What it does:
- Converts `WorkloadSpec` fields into a stable `ContextSignature`:
  - workload name/kind/topology/scale/nodes
  - optional metadata like model/framework/gpu_type/network/nic_count
- Persists (if `run_context` exists):
  - `artifacts/<run_id>/offline/context_snapshot.json`
- Emits trace event:
  - `offline.context.detect`

Why this matters:
- `ContextSignature` is the key used for:
  - memory retrieval scoring (`src/memory/index.py::context_similarity`)
  - naming the exported dataset/model files (`src/agent/core.py::_context_hash`)

### 3.2 Microbench (offline evidence)

Call chain:
- `CCLAgent.tune()` → `OfflinePlanner.offline_plan(workload)` → `tools.microbench.run(workload, parameter_space)`

Where:
- `src/tools/microbench.py::MicrobenchRunner.run()`

Microbench modes:
- If `--dry-run` or `config.microbench.mode == "dry"` or no `command_template` is configured:
  - `MicrobenchRunner._simulate()` returns a synthetic `MicrobenchResult`.
- Otherwise:
  - `_run_real()` builds and runs `command_template` via `subprocess.run(...)` and expects JSON on stdout.

Key real-mode details:
- Adds `CCLAGENT_PARAM_LIST=<comma-separated param names>` to the microbench environment.
- Supports repetitions; merges multiple JSON payloads into a single result.
- Persists stdout/stderr per repetition under `artifacts/<run_id>/offline/` when `run_context` is present.

Planner’s additional persistence:
- Writes `artifacts/<run_id>/offline/microbench_summary.json`
- Emits trace event:
  - `offline.microbench.result` with refs `microbench:<signal.name>`

### 3.3 Initial plan creation (warm start, pruning, LLM)

Call chain:
- `CCLAgent.tune()` → `OfflinePlanner.build_initial_plan(workload, microbench, context)`

Where:
- `src/agent/planner.py::build_initial_plan()`

Inside `build_initial_plan`, in order:

1) Retrieve top memory rules for this context:
   - `MemoryStore.retrieve_rules_with_scores(context, top_k=3)`
   - scoring is:
     - `src/memory/index.py::context_similarity(rule.context, current_context)`
     - × `recency_decay(last_used/created_at, half_life_days)`
     - × `quality = max(0.1, success_rate) * confidence`
   - Emits trace event: `retrieval.memory`
2) Compute warm start + pruning via a small offline “reasoner”:
   - `src/agent/offline_reasoner.py::OfflineReasoner.build_offline_artifacts(...)`
   - Generates:
     - warm-start candidates: defaults + “defaults+top-rule-patch”
     - warm-start decision: chooses min-risk candidate
     - pruning list: fix non-important params to default
   - Risk scoring is done by:
     - `src/safety/risk.py::RiskScorer.score(config)`
   - Emits trace events:
     - `decision.offline_warm_start`
     - `search.prune` (if any pruning happened)
3) Apply the selected warm start:
   - If warm-start candidate selected, `base_params = selected_candidate.config`
4) Call the LLM to refine the baseline config:
   - `OfflinePlanner._propose_llm_config(...)`:
     - builds a **sectioned prompt** via `_build_prompt_bundle(...)`
     - calls `llm.complete([system_message, user_message], ...)`
     - parses JSON from the response
     - merges parsed values into `base_params`
5) Build a recommended parameter set for search:
   - Start from `microbench.important_params` sorted by importance.
   - Apply pruning: drop pruned params from the recommended list.
6) Build candidate subspaces:
   - `OfflinePlanner._build_subspaces(recommended)`
   - If `NCCL_ALGO` and `NCCL_PROTO` are enums, it creates subspaces like:
     - `TREE-LL`, `TREE-LL128`, `RING-SIMPLE`, ...
7) Produce an `InitialConfigPlan`:
   - baseline config + constraints + important params + candidate subspaces

Offline artifacts written (when `run_context` exists):
- `artifacts/<run_id>/offline/initial_plan.json`
- `artifacts/<run_id>/offline/microbench_plan.json`
- `artifacts/<run_id>/offline/warm_start_candidates.json`
- `artifacts/<run_id>/offline/warm_start_decision.json`
- `artifacts/<run_id>/offline/search_space_pruning.json`
- `artifacts/<run_id>/offline/offline_report.json`
- `artifacts/<run_id>/offline/offline_report.md`
- `artifacts/<run_id>/offline/context_pack.json` (see `src/agent/context_pack.py`)

If RAG is enabled and loaded:
- `_build_prompt()` calls:
  - `RagStore.load_documents(...)` (lazy)
  - `RagStore.search(query)` multiple times
  - `RagStore.summarize(chunks)`
- Emits trace event:
  - `retrieval.rag` with refs `rag:<doc_id>:<chunk_id>`

---

## 4) Online stage call trace (the tuning loop that repeatedly runs training)

The online loop is a repeated sequence:
1) run training under a candidate NCCL config
2) record results
3) decide the next config

### 4.1 Running a training step: `WorkloadExecutor.run(...)`

Call chain:
- `CCLAgent.tune()` → `WorkloadExecutor.run(workload, config, step, compiled=...)`

Where:
- `src/agent/executor.py::WorkloadExecutor.run()`

#### 4.1.1 Config validation (“apply”)

First line inside `run()`:
- `apply_result = self.tools.nccl.apply(config)`

Where:
- `src/tools/nccl.py::NCCLInterface.apply()`
  - delegates to `src/tools/config_compiler.py::ConfigCompiler.compile()`
    - calls `src/types.py::ParameterSpace.validate(config.params)`
    - adds safety warnings via `ConfigCompiler._safety_warnings(...)`

Important behavioral notes:
- This is **validation only**. It does *not* mutate system NCCL settings; it returns `env={k: str(v)}` and error/warning lists.
- Validation results are logged but do not automatically stop execution.

#### 4.1.2 Build final environment overrides for the training process

`WorkloadExecutor.run()` builds `env_overrides` from several sources:
1) `compiled.env` (if a compiled hypothesis config was passed in)
2) `tools.ext_tuner.env_overrides()` (AutoCCL-style env vars, possibly including `LD_PRELOAD`, `NCCL_TUNER_PLUGIN`, etc.)
3) `tools.autoccl.env_overrides()` (currently same object as ext_tuner in `build_tools`)
4) `tools.ext_net.env_overrides()` (NCCL net plugin env)
5) `extra_env` (e.g., ext-tuner session dir or eval-mode knobs)

It then persists:
- `artifacts/<run_id>/steps/step_<k>_final_env.json`

#### 4.1.3 Select the runner: training vs workload

`runner = tools.workload` by default, but:
- if `workload.kind` is `"training"` or `"train"`, and `tools.training` exists:
  - `runner = tools.training`

So for training workloads, the next call is:
- `TrainingJobRunner.run(workload, config, step=..., env_overrides=env_overrides)`

#### 4.1.4 Execute the runner and enforce SLA

After runner returns `metrics`:
- `sla_result = tools.sla.check(metrics)`
  - where `src/tools/sla.py::SLAEnforcer.check()` checks:
    - metrics success/failure
    - optional `max_iteration_time`
    - optional error budget
- The executor appends SLA fields into `metrics.raw`:
  - `sla_ok`, `sla_violations`, `sla_severity`, `sla_rollback`

This means downstream decisions can check:
- `metrics.raw["sla_rollback"]` to trigger rollback decisions.

---

### 4.2 What actually runs for training workloads: `TrainingJobRunner.run(...)`

Call chain:
- `WorkloadExecutor.run()` → `TrainingJobRunner.run()`

Where:
- `src/tools/training.py::TrainingJobRunner.run(...)`

Important behaviors:

1) Command selection:
   - Start with:
     - `command` argument, else `TrainingJobConfig.command`, else `workload.command`
   - Then `_select_command(workload, default_cmd)`:
     - if `workload.launcher == "torchrun"`: `src/tools/launchers/torchrun.py::build_torchrun_command(workload)`
     - if `workload.launcher in ("slurm", "srun")`: `src/tools/launchers/slurm.py::build_slurm_command(workload)`
     - if `workload.launcher in ("mpi", "mpirun")`: `src/tools/launchers/mpi.py::build_mpi_command(workload)`
     - else: run the default cmd “as-is”
2) Dry-run path (`self.config.dry_run` or empty cmd):
   - sleeps optionally via `CCL_SIMULATE_SLEEP_SEC`
   - generates synthetic iteration samples via `_simulate_metrics(...)`
   - writes per-step logs and metrics to artifacts
3) Real execution path:
   - merges environment:
     - `os.environ.copy()`
     - + `workload.env`
     - + `config.params` converted to strings (this is where `NCCL_*` settings are injected)
     - + `env_overrides`
   - computes a timeout:
     - default `TrainingJobConfig.timeout_s`
     - overridden if `workload.eval_mode == "short"` and `workload.eval_timeout_sec` is set
     - overridden by `env_overrides["CCL_EVAL_TIMEOUT_SEC"]` if present and parseable
   - runs `subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=timeout, env=merged_env)`
   - returns:
     - `Metrics(iteration_time_ms=<wall clock ms>, success=True, raw={"raw": stdout})`
     - **Note:** for training workloads, we measure wall-clock time for the whole command, not “per-step iteration time”.
4) Failure path:
   - if `subprocess.run` raises and `allow_fallback=True`, it returns:
     - `Metrics(iteration_time_ms=inf, success=False, failure_reason=<exception>, raw={"error": ...})`

Artifacts written by `TrainingJobRunner` when `run_context` exists:
- `artifacts/<run_id>/steps/step_<k>_stdout.log`
- `artifacts/<run_id>/steps/step_<k>_stderr.log`
- `artifacts/<run_id>/steps/training_cmd_step_<k>.json`
- `artifacts/<run_id>/steps/step_<k>_metrics.json`

---

### 4.3 Recording the step and updating online state

After `executor.run()` returns `metrics`, `CCLAgent.tune()` performs:

1) Derived metrics and bottleneck classification:
   - `src/agent/metrics_derive.py::derive_metrics(metrics)`
   - `src/agent/bottleneck.py::classify_bottleneck(derived)`
   - Writes:
     - `artifacts/<run_id>/steps/step_<k>_metrics_derived.json`
     - `artifacts/<run_id>/steps/step_<k>_bottleneck.json`
   - Emits trace events:
     - `analysis.metrics.derive` (refs `metric:<k>:derived:<name>`)
     - `analysis.bottleneck.classify`
2) Build and record a `TuningRecord`:
   - `record = TuningRecord(step=k, action=..., metrics=..., microbench_snapshot=...)`
   - `state.record(record)` (`src/agent/state.py::TuningState.record`)
     - updates `best_record`, `plateau_count`, `last_known_good`
3) Memory updates:
   - If action includes a hypothesis rule: `MemoryStore.mark_rule_usage(rule_id, success)`
   - If metrics failed: `MemoryStore.add_avoid_rule(context, config_patch, evidence=metrics.raw)`
4) Online surrogate update (in-memory model used for decisions):
   - `CCLAgent._update_surrogate(context, config, iteration_time_ms, step)`
   - delegates to `src/models/surrogate.py::SurrogateModel.update(...)`
   - may persist a `.pkl` model snapshot into `memory/models/` every `refit_every_steps`
5) Persist the step record (action + delta + metrics):
   - `CCLAgent._persist_step(record, prev_config=...)`
   - Writes:
     - `artifacts/<run_id>/steps/step_<k>.json`

---

### 4.4 Choosing the next action: `TuningAnalyzer.plan_next_action(...)`

Call chain:
- `CCLAgent.tune()` → `TuningAnalyzer.plan_next_action(...)`

Where:
- `src/agent/analyzer.py::TuningAnalyzer.plan_next_action(...)`

This is the **actual decision engine** used by `CCLAgent.tune()` today (not `DecisionPolicy`).

The analyzer always begins by writing a context pack for the upcoming decision:
- `artifacts/<run_id>/steps/step_<k>_context_pack.json`

Then it chooses among four outcomes:

#### 4.4.1 Rollback (if SLA violation or failure)

Condition:
- `if last_metrics.success is False` OR `last_metrics.raw["sla_rollback"] is True`

Behavior:
- If `state.last_known_good` exists:
  - return a `RollbackAction(config=state.last_known_good)`
  - Writes:
    - `artifacts/<run_id>/steps/step_<k>_rollback_decision.json`
  - Emits trace event:
    - `safety.rollback`
- Otherwise:
  - return `StopAction(reason="failure_without_rollback")`

#### 4.4.2 Stop (budget / plateau / target gain)

Calls:
- `src/agent/stop_policy.py::StopPolicy.evaluate(state, step)`

Common stop reasons:
- `budget_exhausted` (when near max steps)
- `plateau` (when `plateau_count >= patience`)
- `target_gain` (optional: if configured and stable)

Writes:
- `artifacts/<run_id>/steps/step_<k>_stop_decision.json`
Emits trace:
- `stop.decision`

#### 4.4.3 Hypothesis step (every `hypothesis_every` steps)

Condition:
- `use_hypothesis = (step % budget.hypothesis_every) == 0`

Call chain:
1) Generate a portfolio of hypotheses:
   - `HypothesisGenerator.propose_portfolio(plan, context, base_config, last_metrics, max_hypotheses=3)`
   - `src/agent/hypothesis.py`
   - Uses memory rules first; else a single-parameter mutation heuristic.
2) Score hypotheses using the surrogate:
   - `self.numeric_manager.surrogate.predict_one(NCCLConfig(params=merged), context=context)`
   - Writes:
     - `artifacts/<run_id>/steps/step_<k>_hypothesis_portfolio.json`
     - `artifacts/<run_id>/steps/step_<k>_hypothesis_ranked.json`
   - Emits trace:
     - `model.surrogate.predict`
     - `proposal.hypothesis`
3) Compile and risk-score the chosen hypothesis:
   - `compiled = compiler.compile_hypothesis(base_config, hypothesis.patch)`
   - Where:
     - `src/tools/config_compiler.py::ConfigCompiler.compile_hypothesis(...)`
       - merges patch into base config
       - validates values
       - computes `RiskScorer.score(...)`
   - Writes:
     - `artifacts/<run_id>/steps/step_<k>_hypothesis.json`
     - `artifacts/<run_id>/steps/step_<k>_compiled_config.json`
     - `artifacts/<run_id>/steps/step_<k>_risk_report.json`
   - Emits trace:
     - `safety.risk_score`
4) If risk is too high:
   - return a numeric fallback action (`rationale="risk_too_high_fallback"`)
   - (note: this still returns the compiled config today; the intent is “don’t trust the hypothesis rationale”)
5) Otherwise return a `HypothesisAction` containing:
   - `config=compiled.config`
   - `compiled=compiled` (so the next step can reuse compiled env)

Writes:
- `artifacts/<run_id>/steps/step_<k>_decision.json`
- `artifacts/<run_id>/steps/step_<k>_decision_record.json`
Emits trace:
- `decision.select_action`

#### 4.4.4 Numeric step (surrogate-guided neighbor search)

Call chain:
1) `NumericSearchManager.propose(plan, state, workload, base_config, step, context=context)`
   - `src/agent/numeric.py::NumericSearchManager.propose(...)`
2) Candidate generation:
   - `src/search/coordinate_descent.py::CoordinateDescentSearch.propose_candidates(...)`
   - generates “neighbor” configs in the current subspace/dimension
3) Candidate filtering:
   - dedup via `state.search_state.evaluated_hashes`
   - validate via `ParameterSpace.validate(...)`
   - risk score via `RiskScorer.score(...)`
4) Candidate scoring:
   - If `config.numeric_search.mode == "real_eval"`:
     - run a short evaluation batch:
       - `WorkloadExecutor.run_batch(...)` (calls `executor.run()` for each candidate)
     - predicted time is the measured runtime
   - Else (`"predict_only"`, default):
     - score via surrogate:
       - `SurrogateModel.predict(candidates, context=context)`
     - predicted time is the model mean; uncertainty is model std
5) Candidate selection:
   - pick best by predicted time
   - in `"predict_only"` mode, may pick an “uncertain” candidate if it’s within 20% of best predicted time
6) Update the coordinate descent state:
   - `CoordinateDescentSearch.update_state(...)`
7) Persist the search artifacts:
   - `artifacts/<run_id>/steps/step_<k>_candidates.json`
   - `artifacts/<run_id>/steps/step_<k>_candidates_trace.json`
   - `artifacts/<run_id>/steps/step_<k>_pruning_summary.json`
   - `artifacts/<run_id>/online/search_state.json`
   - `artifacts/<run_id>/online/surrogate_predictions_step_<k>.json` (predict-only mode)
8) Emit trace events:
   - `proposal.numeric_candidates`
   - `search.prune`

The analyzer wraps this into a `NumericSearchAction` and writes:
- `artifacts/<run_id>/steps/step_<k>_decision.json`
- `artifacts/<run_id>/steps/step_<k>_decision_record.json`

---

## 5) Post-run stage call trace (“training” after training)

After the online loop stops (stop decision or budget), `CCLAgent.tune()` calls:
- `CCLAgent._post_run(state, context)`

Where:
- `src/agent/core.py::_post_run(...)`

This stage updates **persistent memory** and trains/exports **surrogate artifacts**.

### 5.1 Distill a “best patch” rule and save surrogate records

In order:

1) Compute improvement:
   - baseline = step 0 iteration_time_ms
   - best = best_record iteration_time_ms
   - improvement ratio = (baseline - best) / baseline
2) Persist a post-run context pack (if run_context exists):
   - `artifacts/<run_id>/postrun/context_pack.json`
3) Add a memory rule for the diff between baseline config and best config:
   - `CCLAgent._diff_configs(base, best)` → `{param: value, ...}`
   - `MemoryStore.add_rule(context, config_patch, improvement)`
4) Add every step’s config+metrics to the memory surrogate log:
   - `MemoryStore.add_surrogate_record(context, config, metrics)`

### 5.2 Export a dataset and train a surrogate model

Call chain:
- `src/models/training.py::export_dataset(dataset_records, dataset_path)`
  - writes `memory/datasets/<context_hash>.jsonl`
- `src/models/training.py::train_surrogate_model(dataset_records, context, parameter_space, surrogate_config, model_path)`
  - trains a model (RandomForest if sklearn is installed; else model stays `None`)
  - always saves a `.pkl` (even if model is `None`, the payload is still pickled)
  - writes metadata JSON next to it:
    - `memory/models/surrogate_<context_hash>_<timestamp>.json`

Where model_path comes from:
- `src/agent/core.py::_model_path(context)`
  - default directory: `memory/models/`
  - file name includes the context hash and the run timestamp

### 5.3 Distill semantic rules and persist reports

Call chain:
- `src/agent/distill.py::distill_semantic_rules(state, context)`
  - produces one rule per changed parameter (best vs baseline)
- For each distilled rule:
  - add it to memory via `MemoryStore.add_rule(...)`
- Persist post-run outputs:
  - `src/agent/distill.py::persist_rules(artifacts/<run_id>/postrun/rules_distilled.jsonl, rules)`
  - `src/agent/distill.py::persist_report(artifacts/<run_id>/postrun/distillation_report.md, rules)`
- Emit trace event per distilled rule:
  - `postrun.distill.rule`

### 5.4 Avoid rules persistence

If any steps failed:
- build an avoid-rules list from failed records
- write it to the repo-level file:
  - `memory/avoid_rules.jsonl` (note: not under the run’s artifact dir)

### 5.5 Final post-run artifacts

`src/agent/core.py::_post_run()` also writes:
- `artifacts/<run_id>/postrun/rule_updates.json`
- `artifacts/<run_id>/postrun/best_config_validation.json` (currently a placeholder: `"executed": False`)

Finally, `CCLAgent.tune()`:
- calls `MemoryStore.save()` → writes `memory/agent_memory.json`
- emits `run.end` trace event

---

## 6) LLM layer call flow (what happens when the planner calls an LLM)

The planner **always calls the LLM**, even in dry-run. Default is **Ollama** / `deepseek-r1:8b`, unless overridden via `--provider` or config.

### 6.1 Client selection

Call chain:
- `src/main.py` → `src/llm/__init__.py::create_llm_client(provider, model)`

Providers and their implementations:
- `openai` → `src/llm/openai.py::OpenAIClient` (OpenAI-compatible HTTP)
- `fireworks` → `src/llm/fireworks.py::FireworksClient` (OpenAI-compatible HTTP with default base URL and headers)
- `claude` → `src/llm/claude.py::ClaudeClient` (Anthropic SDK or injected `complete_fn`)
- `gemini` → `src/llm/gemini.py::GeminiClient` (google-generativeai SDK or injected `complete_fn`)
- `openai-compatible` → `src/llm/base.py::OpenAICompatibleClient` (generic)
- `ollama` → `src/llm/ollama.py::OllamaClient` (local Ollama `/api/chat`)
- `none` → `src/llm/base.py::NullLLMClient`

### 6.2 Tracing wrappers

The CLI always wraps whatever client you chose with:
- `src/llm/traced_client.py::TracedLLMClient`

So a planner call looks like:
- `OfflinePlanner._propose_llm_config(...)`
  - → `TracedLLMClient.complete(messages, **kwargs)`
    - → `inner.complete(messages, **kwargs)` (provider-specific)
    - → write `artifacts/<run_id>/llm/call_<uuid>.json`
    - → emit trace event `llm.call` pointing to that file

Additionally, `OpenAICompatibleClient.complete()` calls:
- `src/llm/base.py::_trace_llm_event("request", ...)`
- `src/llm/base.py::_trace_llm_event("response", ...)`

Those events:
- print to stdout if `CCL_LLM_TRACE_STDOUT=1` (default)
- append to `CCL_LLM_TRACE_DIR/llm_trace.jsonl` if `CCL_LLM_TRACE_DIR` is set

So for OpenAI-compatible providers you typically get:
- `artifacts/<run_id>/llm/llm_trace.jsonl` (stream of request/response envelopes)
- `artifacts/<run_id>/llm/call_<id>.json` (structured “prompt pack” per call)
- trace bus events in `artifacts/<run_id>/trace/events.jsonl`

For **all** providers (including Ollama), the prompt pack includes:
- full messages (system + user)
- `context_window` metadata (sections, token budgets, truncation)
- `request_kwargs` (temperature/max_tokens/options)
- response content + raw payload (if available)

### 6.3 Context window management (agentic, Letta-style)

The offline planner builds a **sectioned prompt** and enforces a **token budget** before calling the LLM:
- Implemented in `src/llm/context_window.py`
- Used by `src/agent/planner.py::_build_prompt_bundle`

Behavior:
1) The prompt is split into named sections: workload, context, signals, memory rules, RAG snippets, etc.
2) Each section has a priority and optional per-section max token budget.
3) If the combined prompt is too large, **lower-priority sections are truncated first**.
4) The final prompt + truncation metadata are written to the LLM prompt pack.

This is intentionally “agentic + white-box” (similar to Letta’s layered memory idea): you can see **exactly** what was included in the model context window for each call.

---

## 7) Artifact + trace reference (where each file comes from)

Run directory layout (default):
- `artifacts/<run_id>/`
  - `config_snapshot.json` — from `src/main.py`
  - `run_context.json` — from `src/utils/artifacts.py` then rewritten by `src/main.py`
  - `trace/events.jsonl` — from `src/trace/writer.py`
  - `offline/`
    - `context_snapshot.json` — `OfflinePlanner.build_context`
    - `microbench_summary.json` — `OfflinePlanner.offline_plan`
    - `microbench_stdout_*.log`, `microbench_stderr_*.log` — `MicrobenchRunner._run_real` (real mode)
    - `initial_plan.json`, `warm_start_*.json`, `search_space_pruning.json` — `OfflinePlanner.build_initial_plan`
    - `context_pack.json` — `src/agent/context_pack.py` via planner
  - `steps/`
    - `step_<k>.json` — `CCLAgent._persist_step` (action + delta + metrics)
    - `step_<k>_metrics.json` — runner (`TrainingJobRunner` or `WorkloadRunner`)
    - `step_<k>_stdout.log`, `step_<k>_stderr.log` — runner
    - `step_<k>_final_env.json` — `WorkloadExecutor.run`
    - `step_<k>_metrics_derived.json` — `derive_metrics`
    - `step_<k>_bottleneck.json` — `classify_bottleneck`
    - `step_<k>_context_pack.json` — `TuningAnalyzer.plan_next_action`
    - `step_<k>_decision.json` — `TuningAnalyzer._persist`
    - `step_<k>_decision_record.json` — `TuningAnalyzer._write_decision_record`
    - `step_<k>_stop_decision.json`, `step_<k>_rollback_decision.json` — stop/rollback paths
    - hypothesis artifacts: `step_<k>_hypothesis*.json`, `step_<k>_compiled_config.json`, `step_<k>_risk_report.json`
    - numeric artifacts: `step_<k>_candidates*.json`, `step_<k>_pruning_summary.json`, `step_<k>_batch_results.json`
  - `online/`
    - `search_state.json` — numeric search state (`NumericSearchManager`)
    - `surrogate_predictions_step_<k>.json` — surrogate predictions (predict-only mode)
  - `postrun/`
    - `convergence.json` — stop reason summary
    - `context_pack.json` — postrun context summary
    - `rules_distilled.jsonl`, `distillation_report.md` — semantic rule distillation
    - `rule_updates.json` — combined rules/avoid-rules update
    - `best_config_validation.json` — placeholder
  - `llm/`
    - `call_<uuid>.json` — `TracedLLMClient.complete` (prompt pack)
    - `llm_trace.jsonl` — `OpenAICompatibleClient` tracing (if enabled)

Repo-level persistence (not per-run):
- `memory/agent_memory.json` — `MemoryStore.save()` at end of run
- `memory/avoid_rules.jsonl` — written during post-run if there were failures
- `memory/datasets/<context_hash>.jsonl` — exported dataset for surrogate training
- `memory/models/surrogate_<context_hash>_<timestamp>.pkl` — saved surrogate model snapshots

---

## 8) Appendix: per-source-file “who calls it / what it calls”

This is a compact map of the **entire `src/` tree** (including modules that are present but not yet wired into the main `CCLAgent.tune()` loop).

### Entry points

- `src/main.py`
  - Called by: `python3 -m src.main`
  - Calls into:
    - `src/config.py` (load config + workload)
    - `src/utils/*` (env + artifacts + json)
    - `src/trace/*` (writer/emitter)
    - `src/tools/*` (construct tools; then wrap in instrumented suite)
    - `src/memory/*`, `src/RAG/*`, `src/llm/*` (construct dependencies)
    - `src/agent/core.py::CCLAgent.tune`

- `src/tui/app.py`
  - Called by: `scripts/agent_tui.py`
  - Reads:
    - `artifacts/<run_id>/**` (steps, trace, offline/postrun artifacts, config snapshot)
    - `memory/agent_memory.json`, `memory/avoid_rules.jsonl`
  - Optional calls into:
    - `src/llm/*` (for interactive chat in the UI)

### Agent layer (`src/agent/*`)

- `src/agent/core.py`
  - Called by: `src/main.py`, `src/__init__.py` exports
  - Calls into:
    - planner/analyzer/executor/numeric/hypothesis modules
    - `src/models/*` (surrogate + dataset export/training)
    - `src/memory/*` (rules + surrogate record persistence)
    - `src/utils/*` (artifact writing)
    - `src/trace/*` (events via injected emitter)

- `src/agent/planner.py`
  - Called by: `CCLAgent.tune()` and `ExtTunerSession` setup
  - Calls into:
    - `tools.microbench.run` (offline evidence)
    - `MemoryStore.retrieve_rules_with_scores` (memory retrieval)
    - `OfflineReasoner` (warm start + pruning)
    - `RagStore.search/summarize` (optional)
    - `LLMClient.complete` (wrapped by `TracedLLMClient` in CLI)

- `src/agent/executor.py`
  - Called by: `CCLAgent.tune()`, numeric real-eval path, ext-tuner integration
  - Calls into:
    - `tools.nccl.apply` (validation)
    - `tools.training.run` or `tools.workload.run` (execute command)
    - `tools.sla.check` (SLA enforcement)

- `src/agent/analyzer.py`
  - Called by: `CCLAgent.tune()`, ext-tuner session
  - Calls into:
    - `StopPolicy.evaluate`
    - `HypothesisGenerator.propose_portfolio`
    - `ConfigCompiler.compile_hypothesis` (risk scoring)
    - `NumericSearchManager.propose`
    - `src/agent/context_pack.py` (decision context pack artifacts)

- `src/agent/hypothesis.py`
  - Called by: `TuningAnalyzer.plan_next_action`
  - Calls into:
    - `MemoryStore.retrieve_rules`
    - `ParameterSpace` mutation helpers (single-param neighbor)

- `src/agent/numeric.py`
  - Called by: `TuningAnalyzer.plan_next_action`
  - Calls into:
    - `CoordinateDescentSearch` (candidate generation)
    - `ParameterSpace.validate` (validity filter)
    - `RiskScorer.score` (risk filter)
    - `SurrogateModel.predict` OR `WorkloadExecutor.run_batch` (candidate scoring)
    - `src/utils/*` (artifact writing) + trace emitter

- `src/agent/offline_reasoner.py`
  - Called by: `OfflinePlanner.build_initial_plan`
  - Calls into:
    - `RiskScorer.score` (warm-start ranking)

- `src/agent/state.py`
  - Called by: `CCLAgent.tune()` (TuningState)
  - Contains:
    - `TuningState` (plateau/best bookkeeping)
    - legacy/simple surrogate classes used for older code paths

- `src/agent/policy.py`
  - Present as: an alternative decision policy (`DecisionPolicy`)
  - Note: `CCLAgent.tune()` currently uses `TuningAnalyzer.plan_next_action(...)`, not `DecisionPolicy.decide_next_action(...)`.

- `src/agent/stop_policy.py`
  - Called by: `TuningAnalyzer.plan_next_action`
  - Uses:
    - `TuningState` fields + `AgentConfig.budget`

- `src/agent/metrics_derive.py`, `src/agent/bottleneck.py`
  - Called by: `CCLAgent.tune()` after each step
  - Pure functions:
    - derive computed metrics and classify bottleneck

- `src/agent/context_pack.py`
  - Called by: planner/analyzer/core postrun
  - Builds the structured context artifact used by explainability tooling.

- `src/agent/distill.py`, `src/agent/post_run.py`
  - Called by: `CCLAgent._post_run`
  - Produce “distilled rules” outputs and persist them.

- `src/agent/ext_tuner.py`
  - Called by: `CCLAgent.tune()` when `execution.mode == "in_job_ext_tuner"`
  - Implements a file-based request/response protocol via:
    - `src/tools/tuner_plugin_protocol.py::FileTunerProtocol`

### Tools layer (`src/tools/*`)

- `src/tools/suite.py`
  - Data container created by `src/main.py::build_tools`.

- `src/tools/instrumented.py`
  - Wraps the tool suite and emits `tool.call` / `tool.result` events.

- `src/tools/microbench.py`
  - Called by: offline planner
  - Runs microbench command or simulates, returns `MicrobenchResult`.

- `src/tools/training.py`
  - Called by: `WorkloadExecutor.run()` for training workloads
  - Runs the training command or simulates, returns `Metrics`.

- `src/tools/workload.py`
  - Called by: `WorkloadExecutor.run()` for non-training workloads
  - Runs the workload command or simulates, parses JSON metrics if a parser is supplied.

- `src/tools/metrics.py`
  - Called by: `WorkloadRunner` (when configured in `src/main.py`)
  - Parses `Metrics` from workload stdout (JSON or nccl-tests formats).

- `src/tools/sla.py`
  - Called by: `WorkloadExecutor.run()` for every step
  - Returns `SLAResult`; executor annotates `metrics.raw`.

- `src/tools/config_compiler.py`, `src/tools/nccl.py`
  - Called by: executor/analyzer
  - Validates parameter values and computes safety warnings/risk scores.

- `src/tools/autoccl.py`, `src/tools/ext_tuner.py`, `src/tools/ext_net.py`
  - Called by: executor (env overrides)
  - Provide env overlays for AutoCCL-style tuner plugins and NCCL net plugins.

- `src/tools/numeric_search.py`
  - Currently used by: `src/agent/policy.py` (not the main analyzer loop)
  - Provides an async batch scoring tool for candidate configs.

- `src/tools/nccltest.py`
  - Provides a minimal `nccl-tests` runner (`NcclTestRunner`).
  - Wired into `ToolSuite` by `src/main.py::build_tools`, but not used by the default offline planner today.

- `src/tools/launchers/*`
  - Called by: training/workload runners when `workload.launcher` is set
  - Builds `torchrun`, `srun`, or `mpirun` command prefixes.

- `src/tools/tuner_plugin_protocol.py`
  - Called by: `src/agent/ext_tuner.py`
  - Implements file-based request/response IPC.

### Models (`src/models/*`)

- `src/models/surrogate.py`
  - Called by: `CCLAgent` (online updates + prediction), `train_surrogate_model` (offline fit)
  - Implements:
    - RandomForest predictor (if sklearn available)
    - kNN fallback prediction + uncertainty
    - pickling save/load for persistence

- `src/models/features.py`
  - Called by: surrogate model
  - Encodes configs (+ optional context) into a numeric feature vector.

- `src/models/training.py`
  - Called by: `CCLAgent._post_run()` and `scripts/train_surrogate.py`
  - Exports JSONL datasets and trains/saves surrogate models.

### Storage + retrieval (`src/memory/*`, `src/RAG/*`)

- `src/memory/__init__.py`, `src/memory/schema.py`, `src/memory/index.py`
  - Called by: planner (retrieve rules), agent core (persist rules/surrogates), hypothesis generator
  - Defines:
    - `MemoryStore` (load/save rules + surrogate records)
    - rule scoring functions (context similarity + recency decay)

- `src/RAG/store.py`, `src/RAG/index.py`, `src/RAG/embeddings.py`
  - Called by: planner prompt builder; optional scripts
  - Implements:
    - Jaccard retriever (no deps)
    - embedding retriever (sentence-transformers if present; tf-idf fallback)

### Safety + search (`src/safety/*`, `src/search/*`)

- `src/safety/risk.py`
  - Called by: offline reasoner, numeric manager, compiler
  - Computes a coarse risk score + reasons.

- `src/safety/rollback.py`
  - Currently not wired into `CCLAgent.tune()` (rollback uses `TuningState.last_known_good`).

- `src/search/coordinate_descent.py`
  - Called by: numeric manager
  - Implements a simple coordinate-descent candidate generator + state update.

### Core utilities (`src/types.py`, `src/config.py`, `src/utils/*`, `src/trace/*`)

- `src/types.py`
  - Imported everywhere
  - Defines all dataclasses used across the system.

- `src/config.py`
  - Called by: CLI and scripts
  - Loads JSON config and workload specs; defines defaults.

- `src/utils/*`
  - Called by: CLI/agent/tools/memory/RAG
  - Provides artifact paths, JSON IO, env loading, tokenization, logging.

- `src/trace/*`
  - Called by: CLI (writer/emitter), agent/tools/llm (event emission), TUI/scripts (read)
  - Implements the JSONL trace bus in `artifacts/<run_id>/trace/events.jsonl`.

  Key files:
  - `src/trace/events.py` — `TraceEvent` dataclass and schema version
  - `src/trace/writer.py` — `TraceWriter` append-only JSONL writer
  - `src/trace/emitter.py` — `TraceEmitter` interface + `TraceEmitterWriter` implementation
  - `src/trace/span.py` — `TraceSpan` context manager for timed events
  - `src/trace/reader.py` — helpers to read/filter JSONL traces

### Knowledge (`src/knowledge/*`)

- `src/knowledge/nccl_params.yaml`
  - A small YAML-like file of NCCL parameter descriptions/notes.

- `src/knowledge/param_semantics.py`
  - Provides `load_param_semantics()` / `get_param_semantics()` utilities to load and query `nccl_params.yaml`.
  - Not currently used in the main tuning loop, but useful for building richer prompts or reports.

---

## 9) Scripts (outside `src/`) that matter for “end-to-end training”

- `scripts/run_agent_torch_demo.sh`
  - runs: `python3 -m src.main --workload workload/autoccl/torch-demo.json`

- `scripts/agent_torchrun_demo.sh`
  - invoked by the torch demo workload spec
  - runs a torchrun demo if the AutoCCL submodule exists

- `scripts/agent_tui.py`
  - launches the Textual dashboard reading artifacts and trace

- `scripts/export_sft_dataset.py` / `scripts/export_rl_dataset.py`
  - export supervised/RL-style datasets from `artifacts/<run_id>/steps/step_*.json`
  - implemented in `src/data/export.py`

- `scripts/train_surrogate.py`
  - trains a surrogate model from `memory/agent_memory.json` (surrogate records)

- `scripts/validate_run_dir.py`
  - sanity-checks a run dir contains at least `trace/events.jsonl` and `offline/context_snapshot.json`
