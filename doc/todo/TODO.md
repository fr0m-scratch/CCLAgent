Below is a **Codex-executable, step-by-step modification plan** to bring the current repo implementation to the full **3‑phase CCL Agent design** shown in the diagram and described in your “Current Implementation and Design” mapping doc.

This plan is intentionally **strict, explicit, file-by-file**, and includes **all improvements** needed to “complete” the design: real microbench integration, knowledge grounding, hypothesis compilation, numeric search with real evaluation, safety/rollback, multi-node orchestration, and post-run learning (rules + surrogate/model training).

References (for the design ideas this plan implements):

* CCLInsight (microbench → important params) 
* AutoCCL (subspace split + coordinate descent + online tuning) 
* AFNFA (ML-guided exploration baseline; useful for comparison + data schemas) 
* STELLAR (RAG parameter extraction + rule accumulation + hallucination controls) 

---

# 0) Ground rules for Codex

1. **Do not delete existing dry-run paths.** Add real implementations behind flags/config, defaulting to current behavior.
2. **Every new behavior must be gated** by `AgentConfig` and/or CLI flags.
3. **Every tool invocation must produce structured artifacts** (JSON) saved under an `artifacts/` run directory.
4. **Every tuning decision must be reproducible**: log seeds, candidate sets, and full config deltas.
5. **No “silent failure.”** If a tool cannot run in real mode, it must raise a typed error and fall back only if `allow_fallback=True`.

---

# 1) Target end-state architecture (must match the design diagram)

## Phase 1: Offline Planning

* Run **real primitive-centric microbench** (CCLInsight-style) to generate:

  * key/important parameters (ranked)
  * signals (bandwidth/latency saturation hints, topology hints)
  * recommended subspaces (e.g., algorithm/protocol/transport groupings)
* Ground with **Knowledge Base**:

  * RAG over NCCL docs + your design docs + prior runs
  * Memory rules retrieved by **fuzzy similarity**, not exact key match
* Produce **Initial Config Plan**:

  * a structured plan object: baseline + constraints + candidate subspace list
* Evaluate 1–N initial configs quickly (microbench/workload warmup)
* Apply best initial config and launch workload

## Phase 2: Online Tuning

* Collect online metrics continuously (per step)
* Analyze metrics → choose action type:

  * Hypothesis action (rule-driven patch)
  * Numeric search action (real evaluation or batched scoring)
* Compile hypothesis into a concrete config (“Compile Config From Hypothesis”)
* Apply config (either restart job with env vars OR use runtime tuner plugin / ext-tuner mode)
* Convergence analysis + stop criteria
* Ensure “benefiting iterations with best config”:

  * once best config found, keep running workload with it until job ends

## Phase 3: Post-Run Updating

* Distill rules + insights into memory
* Persist surrogate training data
* Train / update numerical model(s) (lightweight first: GP/RandomForest; optional: NN)
* (Optional advanced) produce datasets for RL/SFT of a policy model

---

# 2) Repository-wide structural changes (do first)

## 2.1 Add run artifacts directory and run context

### Files to modify

* `src/main.py`
* `src/types.py`
* `src/config.py`
* `src/agent/core.py`
* `src/tools/*` (where tools output artifacts)

### Steps

1. **Add a RunContext** dataclass in `src/types.py`:

   * fields:

     * `run_id: str` (uuid4)
     * `started_at: str` (ISO)
     * `artifacts_dir: str` (path)
     * `dry_run: bool`
     * `seed: int`
     * `git_commit: Optional[str]`
     * `host_info: dict` (hostname, cuda version if available, env summary)
2. In `src/main.py`:

   * create `run_id`
   * create directory `artifacts/{run_id}/`
   * write `artifacts/{run_id}/run_context.json`
3. Thread `RunContext` into:

   * `CCLAgent(...)`
   * all tools (microbench/workload/nccltest/metrics)
4. Enforce: every step writes `artifacts/{run_id}/steps/step_{k}.json`

**Definition of done**

* Running dry-run produces an artifacts directory with run context + per-step records.

---

# 3) Phase 1 Offline Planning: implement the missing “real” design

## 3.1 Real microbench integration (CCLInsight-style)

### Current gap

* `MicrobenchRunner` is dry-run by default and assumes JSON output with `important_params` and `signals`, but is not wired to actual CCLInsight/AF_ICSE26 workloads.

### Files to modify

* `src/tools/microbench.py`
* `src/config.py`
* `src/types.py`
* `scripts/` (add a real runner wrapper)
* `tools/AF_ICSE26/` (wire the existing scripts; do not rewrite research code)

### New/updated types

In `src/types.py` add:

* `MicrobenchSignal`:

  * `name: str`
  * `value: float | int | str`
  * `unit: Optional[str]`
  * `confidence: float` (0..1)
  * `source: str` (e.g., "cclinsight", "nccltests", "heuristic")
* `ImportantParam`:

  * `param: str` (canonical internal param name, e.g., `NCCL_MIN_NCHANNELS`)
  * `importance: float` (normalized 0..1)
  * `reason: str` (short)
  * `evidence: dict` (raw stats)
* Update `MicrobenchResult` to include:

  * `important_params: list[ImportantParam]`
  * `signals: list[MicrobenchSignal]`
  * `raw_path: str` (artifact file path)
  * `command: list[str]`
  * `runtime_sec: float`

### Config changes

In `src/config.py` extend `AgentConfig` / tool config:

* `microbench.mode: "dry" | "cclinsight" | "nccltests"`
* `microbench.command_template: list[str]`
* `microbench.parse_schema: "cclinsight_v1" | "nccltests_v1"`
* `microbench.timeout_sec`
* `microbench.env: dict[str,str]`
* `microbench.collect_topology: bool`

### Implementation steps

1. In `src/tools/microbench.py`, implement `MicrobenchRunner.run_real(...)`:

   * Build command from `command_template` + workload context + cluster info
   * Set env (from config)
   * Execute with timeout
   * Save stdout/stderr to artifacts
   * Parse output depending on schema:

     * **CCLInsight schema**: expect JSON with:

       * `important_params`: list of `{name, score, reason, evidence}`
       * `signals`: list of `{name, value, unit, confidence}`
     * **nccl-tests schema**: parse `busbw`, `algbw`, `time` and derive signals
2. Add a new script wrapper:

   * `scripts/run_cclinsight_microbench.sh`
   * It must:

     * run the microbench
     * output a single JSON file to stdout (schema above)
3. Add a fallback parser:

   * If tool output is not JSON, detect and fail with actionable error (no silent parse).
4. Add deterministic microbench re-run option:

   * `microbench.repetitions` and compute mean/stdev → embed into `evidence`.

**Definition of done**

* In real mode, microbench produces ranked params + signals from actual runs.
* Offline planner receives non-empty `important_params`.

---

## 3.2 Offline knowledge grounding: upgrade RAG from Jaccard to embeddings

### Current gap

* RAG uses Jaccard similarity over `doc/Design`. Too weak, no doc ingestion pipeline, no parameter-aware retrieval.

### Files to modify / add

* `src/RAG/store.py` (refactor into interface + impls)
* `src/RAG/embeddings.py` (new)
* `src/RAG/index.py` (new)
* `src/config.py`
* Add `doc/Knowledge/` (new curated docs: NCCL env var docs, AutoCCL notes, etc.)

### Implementation steps

1. Refactor `src/RAG/store.py` into:

   * `BaseRetriever` interface: `search(query, top_k) -> list[RAGChunk]`
   * `JaccardRetriever` (keep existing)
   * `EmbeddingRetriever` (new)
2. Define `RAGChunk` in `src/types.py`:

   * `doc_id, chunk_id, text, score, meta`
3. Implement `EmbeddingRetriever`:

   * Use local embeddings (preferred):

     * Add a pluggable backend:

       * `sentence_transformers` if available, else a simple TF‑IDF fallback
   * Store embeddings in `rag_index/` on disk:

     * `index_meta.json`
     * `embeddings.npy`
     * `chunks.jsonl`
4. Add CLI / config:

   * `rag.mode: "jaccard" | "embeddings"`
   * `rag.rebuild_index: bool`
   * `rag.index_path: str`
5. Add a **parameter-aware query template** in planner:

   * For each candidate param `p`, query:

     * `"NCCL {p} recommended values and constraints"`
     * `"NCCL {p} interactions with {other important params}"`
6. Add a doc ingestion script:

   * `scripts/build_rag_index.py`
   * It scans:

     * `doc/Design/`
     * `doc/Knowledge/`
     * `README.md` + workload READMEs
   * Splits into chunks, builds embeddings.

**Definition of done**

* RAG retrieval returns semantically relevant chunks even when wording differs.
* Planner can retrieve NCCL parameter guidance reliably.

---

## 3.3 Memory: move from exact-match rules → fuzzy contextual matching + decay

### Current gap

* `MemoryStore` indexes rules only by exact context key (workload/topology/scale/nodes). This kills generalization.

### Files to modify / add

* `src/memory.py`
* `src/types.py`
* `src/config.py`
* Add `src/memory/index.py` (new)
* Add `src/memory/schema.py` (new)

### New concepts

* **ContextSignature** becomes a vector-like structure:

  * workload kind + model + parallelism + topology + NICs + GPU type + scale
* **Rule** becomes:

  * condition: context predicate + metric predicate
  * action: param patch
  * metadata: confidence, last_used, success_rate

### Implementation steps

1. Version the memory JSON schema:

   * Add top-level: `"schema_version": "2.0"`
2. Replace exact key lookup with scoring:

   * Implement `MemoryStore.retrieve_rules(context, top_k)`:

     * compute similarity between query context and each stored rule context:

       * categorical overlap (GPU type, topology)
       * numeric proximity (num_gpus, nodes)
       * embedding similarity for free-text workload tags (optional)
     * multiply by rule quality:

       * `success_rate` (wins / tries)
       * `recency_decay = exp(-age_days / half_life)`
3. Add rule lifecycle updates:

   * On each tuning step:

     * if rule applied, mark `tries += 1`
     * if improved, mark `wins += 1`, increase confidence
4. Add conflict handling:

   * If two rules recommend opposite changes for same param in similar context:

     * keep both but reduce confidence
     * require explicit online validation before “promoting” either
5. Store “negative rules”:

   * If a patch consistently harms performance, store as “avoid” rule.

**Definition of done**

* Same rule can apply across “close” contexts (e.g., 8→16 GPUs) with reduced confidence.
* Memory retrieval returns ranked rules with scores.

---

## 3.4 Offline “Initial Config Plan” object (must exist explicitly)

### Current gap

* Offline planner produces an initial config, but no explicit “plan” artifact that ties together: important params, constraints, subspaces, and rationale.

### Files to modify

* `src/agent/planner.py`
* `src/types.py`

### Implementation steps

1. Add `InitialConfigPlan` type:

   * `baseline_config: NCCLConfig`
   * `constraints: dict[param, {min,max,allowed,locked}]`
   * `important_params: list[ImportantParam]`
   * `candidate_subspaces: list[Subspace]`
   * `recommended_search_params: list[str]` (ordered)
   * `notes: str`
2. Add `Subspace` type (AutoCCL-style):

   * `name: str`
   * `fixed: dict[param, value]` (implementation params)
   * `free: list[str]` (resource params)
3. In planner:

   * Use microbench important params to decide:

     * which params are “locked” initially
     * which become numeric search dimensions
   * Create candidate subspaces like:

     * (NCCL_ALGO=Ring, NCCL_PROTO=Simple)
     * (NCCL_ALGO=Tree, NCCL_PROTO=LL128)
4. Save plan to artifacts:

   * `artifacts/{run_id}/offline/initial_plan.json`

**Definition of done**

* Offline planning outputs an explicit JSON plan used by Phase 2.

---

# 4) Phase 2 Online tuning: implement missing actions + real evaluation

## 4.1 Standardize metrics schema and collection (critical)

### Current gap

* Metrics parsing is not standardized; dry-run simulates; SLA enforcement is minimal.

### Files to modify

* `src/tools/metrics.py`
* `src/tools/workload.py`
* `src/tools/training.py`
* `src/types.py`
* `src/tools/sla.py`
* `src/agent/executor.py`

### Metrics schema (must be the single source of truth)

In `src/types.py`, define:

* `Metrics` fields (minimum required):

  * `iteration_time_ms: float`
  * `throughput: Optional[float]`
  * `comm_time_ms: Optional[float]`
  * `busbw_gbps: Optional[float]`
  * `algbw_gbps: Optional[float]`
  * `loss: Optional[float]` (if training)
  * `error_budget: Optional[float]`
  * `success: bool`
  * `failure_reason: Optional[str]`
  * `raw: dict` (for passthrough)
* `MetricsSchemaVersion: str` included in output.

### Implementation steps

1. Update `MetricsCollector`:

   * Accept multiple parsing modes:

     * `json_stdout_v1`
     * `nccltests_v1` (regex parse)
     * `autoccl_demo_v1`
2. Enforce: each workload run must yield either:

   * `Metrics.success=True` with iteration_time_ms set
   * OR `Metrics.success=False` with failure_reason set
3. Update `WorkloadRunner` & `TrainingJobRunner`:

   * Always write:

     * stdout/stderr logs to artifacts
     * parsed metrics JSON to artifacts
4. Add “metrics validity checks”:

   * iteration_time_ms must be > 0
   * if missing, treat as failure (unless config says `allow_missing_metrics`)
5. Update SLA tool:

   * Provide:

     * `check_sla(metrics, sla_config) -> SLAResult`
   * Where `SLAResult` includes:

     * `ok: bool`
     * `violations: list[str]`
     * `severity: "soft" | "hard"`
     * `rollback_recommended: bool`

**Definition of done**

* Every step produces a metrics JSON with schema version and clear success/failure.

---

## 4.2 Implement “Analyze Metrics & Plan Action” as a first-class module

### Current gap

* Analysis is scattered across policy and core loop; design requires a central “analyze & decide”.

### Files to add / modify

* Add `src/agent/analyzer.py` (new)
* Modify `src/agent/core.py`
* Modify `src/agent/policy.py` (policy becomes a strategy used by analyzer)

### Implementation steps

1. Create `TuningAnalyzer` with method:

   * `plan_next_action(state, last_metrics, history, plan) -> TuningAction`
2. It must produce exactly one of:

   * `HypothesisAction`
   * `NumericSearchAction`
   * `StopAction`
   * `RollbackAction`
3. Decision logic order (strict):

   1. If last run failed or SLA hard violation → `RollbackAction`
   2. If budget exhausted → `StopAction`
   3. If convergence plateau detected → `StopAction` (soft) or `NumericSearchAction` if exploration budget remains
   4. Else alternate hypothesis/numeric using `budget.hypothesis_every`
4. Save analyzer decision artifact every step:

   * includes: inputs summary, rule scores used, surrogate predictions, rationale string

**Definition of done**

* Core loop calls analyzer exactly once per iteration and executes its action.

---

## 4.3 Hypothesis path: “Propose Hypothesis → Compile Config From Hypothesis”

### Current gap

* Hypothesis step does mutate config but lacks a structured hypothesis object and explicit compilation stage.

### Files to modify / add

* `src/agent/policy.py` (split into hypothesis generator + numeric generator)
* `src/tools/config_compiler.py` (upgrade)
* `src/types.py` (new types)

### New types

1. `Hypothesis`:

   * `id: str`
   * `summary: str` (human-readable)
   * `patch: dict[param, value]`
   * `expected_effect: dict[metric, "increase"|"decrease"]`
   * `risk: "low"|"med"|"high"`
   * `evidence: dict` (rules used, rag snippets ids, microbench signals)
2. `CompiledConfig`:

   * `config: NCCLConfig`
   * `env: dict[str,str]`
   * `warnings: list[str]`
   * `risk_score: float`

### Implementation steps

1. Implement `HypothesisGenerator`:

   * Input: `InitialConfigPlan`, memory rules, RAG chunks, last metrics
   * Output: 1 hypothesis at a time
   * Must prioritize:

     1. applying high-confidence memory rule patches
     2. microbench-important params first
     3. only one “major” param change per hypothesis unless explicitly allowed
2. Implement `ConfigCompiler.compile(hypothesis.patch, base_config, parameter_space)`:

   * merges patch into base config
   * validates types/ranges
   * produces env var mapping (canonical env names)
   * emits warnings for near-boundary values
3. Add a **risk scoring** hook:

   * see Safety section (below)
4. Persist:

   * hypothesis JSON
   * compiled config JSON

**Definition of done**

* Hypothesis is always a structured object, and compilation happens explicitly.

---

## 4.4 Numeric search path: real evaluation + pruning + subspaces

### Current gap

* Numeric search tool exists but mostly surrogate scoring; need real candidate evaluation and a search algorithm aligned with design.

### Files to modify / add

* `src/tools/numeric_search.py` (upgrade)
* `src/agent/policy.py` (numeric strategy)
* `src/agent/executor.py` (batched evaluation support)
* `src/types.py` (SearchCandidate now includes “evaluation_mode”)
* Add `src/search/coordinate_descent.py` (new)
* Add `src/search/bayesian.py` (optional later)

### Required numeric algorithm: “Subspace-directed coordinate descent”

This matches AutoCCL’s approach  but adapted to your agent.

#### Implementation steps

1. Introduce `NumericSearchManager` that:

   * iterates subspaces from `InitialConfigPlan.candidate_subspaces`
   * within subspace uses coordinate descent over “resource params”

     * e.g., channels, nthreads, buffsize, chunk size
2. Add `SearchState` persisted in agent state:

   * current subspace index
   * current dimension index
   * step size / learning rate
   * best config found in subspace
   * history of evaluated configs in that subspace
3. Update `NumericSearchTool` to support two evaluation modes:

   * `predict_only`: uses surrogate predictions only (fast fallback)
   * `real_eval`: actually runs workload for candidates
4. Implement batched candidate evaluation:

   * Create `WorkloadExecutor.run_batch(candidates, concurrency)`:

     * For each candidate:

       * apply config
       * run workload for a **short evaluation window** (see below)
       * collect metrics
     * Return ranked results
5. Add “short evaluation window” support:

   * Extend `WorkloadSpec` with:

     * `eval_mode: "full" | "short"`
     * `eval_steps: int`
     * `eval_timeout_sec`
   * Training runner must support early-stop runs (e.g., run N iterations then exit)
6. Add pruning:

   * Before evaluating candidates, filter by:

     * `risk_score <= threshold`
     * constraints from offline plan
     * dedupe by config hash
   * After evaluation:

     * keep top K configs to train surrogate
     * update coordinate descent direction based on improvement

**Definition of done**

* Numeric step can run real evaluations for a small batch and advance search state.

---

## 4.5 Surrogate model: replace kNN placeholder with persisted model + uncertainty

### Current gap

* In-memory kNN is too weak; not persisted; no uncertainty; no model versioning.

### Files to modify / add

* Add `src/models/surrogate.py` (new)
* Add `src/models/features.py` (new)
* Modify `src/agent/core.py`
* Modify `src/memory.py` (store training data)
* Modify `src/config.py`

### Implementation steps

1. Define a unified feature vector for a config:

   * One-hot encode categorical params (ALGO/PROTO)
   * Log-scale numeric params (BUFFSIZE, NTHREADS, channels)
   * Add context features (gpu_type, num_gpus, nodes, nic_count if known)
2. Implement `SurrogateModel` interface:

   * `fit(records) -> None`
   * `predict(candidates) -> list[Prediction]`
   * `save(path)`, `load(path)`
3. Implement baseline model:

   * RandomForestRegressor or XGBoost if available
   * Provide:

     * mean prediction
     * uncertainty estimate (via ensemble variance or quantile regression)
4. Persist model to:

   * `memory/models/surrogate_{context_hash}_{timestamp}.pkl`
5. Update agent loop:

   * After each real evaluation:

     * append record to dataset
     * refit surrogate every `n` steps or when dataset size increases by `m`
6. Use uncertainty in policy:

   * In numeric search, include exploration candidates with high uncertainty (within safe bounds)

**Definition of done**

* Surrogate survives across runs, improves with more data, and drives candidate suggestions.

---

## 4.6 Convergence analysis: implement robust plateau detection + budget logic

### Files to modify

* `src/agent/core.py`
* `src/tools/sla.py`
* `src/agent/analyzer.py`

### Implementation steps

1. Maintain a rolling window `W` of best metrics (e.g., last 5 successful steps).
2. Plateau condition:

   * improvement < `plateau_eps` for `plateau_patience` steps
   * AND uncertainty is low OR search space exhausted
3. Early stop condition:

   * if “benefit mode” enabled:

     * stop tuning as soon as config improves by `target_gain` and is stable for `stable_steps`
4. Save convergence report artifact.

**Definition of done**

* Tuning ends when appropriate and does not oscillate forever.

---

# 5) ApplyCCLConfig: make config application real (not just validation)

This is *the* biggest “design vs implementation” gap.

## 5.1 Implement actual application of NCCL config via env propagation

### Files to modify

* `src/tools/nccl.py`
* `src/tools/config_compiler.py`
* `src/agent/executor.py`
* `src/tools/ext_net.py`
* `src/tools/autoccl.py`, `src/tools/ext_tuner.py`

### Implementation steps

1. `ConfigCompiler` must output `env` map:

   * `{"NCCL_ALGO": "...", "NCCL_PROTO": "...", ...}`
2. `WorkloadExecutor` must:

   * merge env overrides from:

     * ext_net
     * ext_tuner
     * autoccl bridge
     * compiled config env
   * precedence order (strict):

     1. safety overrides (forced)
     2. compiled config env
     3. ext_tuner env (if “external controls” mode)
     4. ext_net env
     5. workload default env
3. For every run, record `final_env.json` artifact.

**Definition of done**

* Workload actually runs with NCCL env vars set and recorded.

---

## 5.2 Add “online in-job tuning” path (benefiting iterations)

To match the diagram, you need a mode where the job keeps running while tuning updates the config.

This requires a runtime mechanism. You already have `ExtTunerSession`—finish it.

### Files to modify / add

* `src/agent/ext_tuner.py` (finish protocol)
* `src/tools/ext_tuner.py` (bridge)
* Add `src/tools/tuner_plugin_protocol.py` (new)
* Add example integration script under `scripts/`

### Required behavior

* Training job runs once.
* It queries agent for initial config.
* After each evaluation window, it reports metrics back.
* Agent replies with next config.
* When tuning ends, job continues with best config.

### Implementation steps (strict)

1. Define a transport protocol (simple, robust):

   * file-based JSON RPC OR unix socket
   * must support:

     * `GET_CONFIG(task_id, step_idx, context)`
     * `REPORT_METRICS(task_id, step_idx, metrics)`
2. Implement `ExtTunerSession` server:

   * owns agent instance or forwards to agent
   * persists session state under artifacts
3. Provide a reference “client shim” for workloads:

   * `scripts/ext_tuner_client.py`
   * training demo uses this shim to:

     * fetch config
     * run N iterations
     * report metrics
     * repeat
4. Update `WorkloadExecutor`:

   * add mode `execution_mode: "restart_per_step" | "in_job_ext_tuner"`
   * in `in_job_ext_tuner` mode:

     * do not relaunch job each step
     * instead start session server and launch job once

**Definition of done**

* One job run can complete tuning and then keep running with best config.

---

# 6) Safety + rollback: implement the missing “critical-path protection”

## 6.1 Add risk scoring and safe envelope constraints

### Files to add / modify

* Add `src/safety/risk.py` (new)
* Add `src/safety/rollback.py` (new)
* Modify `src/tools/sla.py`
* Modify `src/agent/analyzer.py`
* Modify `src/config.py`

### Implementation steps

1. Implement `RiskScorer.score(config, context) -> RiskScore`:

   * compute risk based on:

     * extreme channel counts
     * extreme thread counts
     * known-bad combos (from memory)
     * NIC underutilization patterns (from microbench signals)
   * output:

     * `risk_score: float`
     * `risk_level: low/med/high`
     * `reasons: list[str]`
2. Add “safe envelope” in config:

   * hard bounds beyond parameter space:

     * e.g., `max_channels_safe`, `min_buffsize_safe`
3. Add rollback strategy:

   * Maintain `last_known_good_config`
   * On hard SLA violation or failure:

     * rollback immediately
     * mark failed config as “avoid” rule candidate
     * reduce trust in rule/hypothesis that proposed it

**Definition of done**

* Bad configs do not brick runs; agent reliably recovers.

---

# 7) Phase 3 Post-run updating: rule distillation + model training pipeline

## 7.1 Distill rules from tuning records (data-driven rules)

### Current gap

* Memory updates exist but are simplistic and not tied to evidence.

### Files to modify / add

* `src/agent/post_run.py` (new, or expand existing post_run)
* `src/memory.py`
* `src/types.py`

### Implementation steps

1. Define a `TuningRecord` schema that includes:

   * config
   * metrics
   * decision rationale
   * rule ids used
   * microbench signals snapshot
2. Implement rule distillation algorithm:

   * For each parameter:

     * detect consistent direction of improvement under similar contexts
     * create a rule:

       * “If context matches (gpu, nodes, msg_size bucket) and metric is X, then set param to Y”
   * Attach evidence:

     * list of record ids
     * average gain
     * variance
3. Store:

   * `rules.jsonl` (append-only)
   * `avoid_rules.jsonl`

**Definition of done**

* Post-run produces new rules with evidence and confidence updates.

---

## 7.2 TrainAndUpdateNumericalModels()

### Files to add / modify

* `src/models/training.py` (new)
* `src/memory.py`
* `scripts/train_surrogate.py` (new)

### Implementation steps

1. Export dataset:

   * `memory/datasets/{context_hash}.parquet` (or jsonl)
2. Train surrogate:

   * baseline RF model
   * save with version metadata
3. Validate model:

   * cross-val or holdout
   * compute error metrics
4. Update memory index to point to latest model for a context.

**Definition of done**

* After a run, a persisted model exists and is used next run.

---

## 7.3 Optional: RL training / SFT dataset export (Phase 3 advanced)

This matches the diagram’s “TrainAgentBaseModel (RL / SFT)”. Don’t block the core system on this, but implement the pipeline.

### Files to add

* `src/data/export.py`
* `scripts/export_sft_dataset.py`
* `scripts/export_rl_dataset.py`

### Steps

1. Export supervised pairs:

   * input: context + metrics + top rules + rag snippets
   * output: next action (hypothesis patch or numeric move)
2. Export RL transitions:

   * state: feature vector
   * action: config delta
   * reward: improvement in iteration_time / bandwidth
3. Keep datasets versioned per schema.

**Definition of done**

* You can train future learned policies from real traces.

---

# 8) Multi-node orchestration (Slurm/MPI/torchrun) + reproducibility

## 8.1 Add launch helpers and workload spec extensions

### Files to modify

* `src/types.py` (`WorkloadSpec`)
* `src/tools/workload.py`
* `src/tools/training.py`
* Add `src/tools/launchers/slurm.py`, `mpi.py`, `torchrun.py`

### Steps

1. Extend `WorkloadSpec`:

   * `launcher: "local" | "torchrun" | "slurm" | "mpirun"`
   * `launcher_args: dict`
   * `nodes, gpus_per_node, nnodes`
2. Implement launcher wrappers that:

   * build command
   * ensure env propagation
   * collect logs from all ranks (at least rank0 + stderr aggregation)
3. Add “cluster context detection”:

   * parse `SLURM_*` env when present
4. Save launcher plan artifact.

**Definition of done**

* Same tuning flow runs on Slurm without manual hacking.

---

# 9) Expand parameter space to match real NCCL tuning levers

## 9.1 Include missing high-impact NCCL params (config + validation)

### Files to modify

* `src/config.py` (default parameter space)
* `src/types.py` (parameter spec)
* `src/tools/nccl.py` (validation)

### Required additions (minimum)

Add specs for:

* `NCCL_P2P_LEVEL`
* `NCCL_NET_GDR_LEVEL`
* `NCCL_SOCKET_NTHREADS`
* `NCCL_NSOCKS_PERTHREAD`
* `NCCL_IB_QPS_PER_CONNECTION`
* `NCCL_SHM_DISABLE`
* `NCCL_BUFFSIZE` (already)
* `NCCL_NTHREADS` (already)
* channels: map `NCCL_MIN_NCHANNELS` and `NCCL_MAX_NCHANNELS` properly

Also ensure “implementation vs resource” split is supported (AutoCCL-style) .

**Definition of done**

* Agent can tune the meaningful NCCL parameter set, not just a toy subset.

---

# 10) Testing & validation (must be added before declaring “perfect”)

## 10.1 Unit tests (fast)

Add under `tests/`:

* `test_config_validation.py`
* `test_config_compiler_env.py`
* `test_memory_retrieval_scoring.py`
* `test_rag_retrieval.py`
* `test_analyzer_action_selection.py`
* `test_safety_risk_score.py`

Each test must:

* run without GPUs
* use dry-run tool mocks

## 10.2 Integration tests (optional GPU)

Add:

* `scripts/ci_smoke_nccltests.sh`
* Minimal nccl-tests invocation (small size) with known output parse.

## 10.3 Golden artifact tests

For one dry-run workload:

* assert deterministic artifacts (seeded)
* compare JSON schemas

**Definition of done**

* CI can catch regressions in decision logic + schema compatibility.

---

# 11) Step-by-step implementation order (Codex must follow in sequence)

This is the strict “do it in order” list so changes don’t conflict.

## Stage A — Foundations

1. Add `RunContext` + artifacts directory plumbing
2. Standardize `Metrics` schema + enforce success/failure
3. Add analyzer module and refactor core loop to use it

## Stage B — Phase 1 completion

4. Implement real microbench runner wrapper
5. Add InitialConfigPlan output
6. Upgrade RAG to embeddings + index builder
7. Upgrade memory retrieval to fuzzy scoring + decay

## Stage C — Phase 2 completion

8. Hypothesis object + compile stage
9. Numeric search manager with subspaces + coordinate descent
10. Real batch evaluation in executor (short eval runs)
11. Persisted surrogate model with uncertainty

## Stage D — Safety + rollback

12. Risk scoring + safe envelope constraints
13. Rollback actions + avoid-rule writing

## Stage E — Phase 3 completion

14. Distill rules from tuning traces (with evidence)
15. Persist dataset + train/update surrogate models
16. Export optional RL/SFT datasets

## Stage F — Multi-node + online in-job tuning

17. Launcher helpers (slurm/mpirun/torchrun)
18. Ext-tuner “in-job” tuning mode + reference client

## Stage G — Parameter space expansion + hardening

19. Expand NCCL params + validation
20. Add test suite + CI smoke scripts

---

# 12) Acceptance criteria checklist (must all pass)

A run is “design-complete” only if:

### Offline phase

* [ ] Real microbench returns ranked important params + signals
* [ ] Planner outputs `initial_plan.json` with subspaces + constraints
* [ ] Memory retrieval works with fuzzy matching and returns scored rules
* [ ] RAG retrieval uses embeddings and returns relevant chunks

### Online phase

* [ ] Every step has: hypothesis/search decision artifact + compiled env + metrics JSON
* [ ] Numeric step can run **real** short evaluations (not just surrogate scoring)
* [ ] Surrogate model is persisted and reused across runs
* [ ] Convergence logic stops appropriately

### Safety

* [ ] On failure or SLA hard violation, rollback happens automatically
* [ ] Bad configs become avoid-rules with evidence

### Post-run

* [ ] Distilled rules are written with confidence + evidence
* [ ] Dataset is exported and a model is trained/updated

### Orchestration

* [ ] Restart-per-step mode works locally
* [ ] In-job ext-tuner mode works for demo workload
* [ ] Slurm/torchrun launchers can be used via workload spec

---

If you want, I can also output this as a **machine-oriented checklist JSON** (so Codex can literally tick tasks), but the above is already written in a “follow exactly in order” way.
