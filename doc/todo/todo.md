I unpacked and reviewed the full repo inside `CCLAgent.zip` (docs, src, scripts, tests, workload specs, bundled reference tools/papers, and example artifacts). Below is a **deep, “white‑box agent” redesign plan** focused on making *every single action* explainable, traceable, and visualizable in the TUI—offline planning through online tuning through post‑run distillation.

---

## 1) What you already have (quick audit of current design + impl)

### Repo structure (what’s in the zip)

* `doc/Design/*`: core novelty + currentImplementation + techStack
* `doc/Knowledge/*`: NCCL param notes + AutoCCL notes
* `doc/figures/*`: design diagrams (offline→online→postrun pipeline)
* `doc/references/*`: papers/tools PDFs (AutoCCL, AFNFA, CCLInsight draft, STELLAR)
* `src/`:

  * `agent/`: planner, analyzer, numeric search manager, hypothesis generator, executor, post-run distill
  * `tools/`: microbench, workload runner, config compiler, NCCL interface, SLA checks, ext-tuner protocol, launchers
  * `models/`: surrogate model (RF + uncertainty), feature encoding
  * `memory/`: rule store + surrogate snapshots + retrieval scoring
  * `RAG/`: rag store/index (jaccard or embeddings)
  * `tui/`: Textual app reads artifacts and shows steps/tabs
  * `llm/`: OpenAI-compatible clients + trace-to-stdout/jsonl hooks
* `scripts/`: run microbench, train surrogate, export RL/SFT datasets, run tui, etc.
* `workload/`: workload specs (autoccl demos / training scripts)
* `tools/AF_ICSE26/*`: CCLInsight microbench + NCCL profiling patch scaffolding
* `tests/`: unit tests for validation, memory scoring, analyzer selection, etc.
* `artifacts/`: sample run directories already containing `offline/*`, `steps/*`, `online/*`, `postrun/*`

### Current pipeline (as implemented)

* **Offline**

  * `MicrobenchRunner` runs a command or emits dry-run summary
  * `MemoryStore.retrieve()` and `RagStore.retrieve()` exist
  * `OfflinePlanner.build_initial_plan()` produces:

    * `offline/microbench_summary.json`
    * `offline/initial_plan.json` (baseline + candidate_subspaces)
* **Online**

  * `TuningAnalyzer.plan_next_action()` chooses:

    * `hypothesis` every `hypothesis_every`
    * otherwise `numeric` search via `NumericSearchManager`
    * `stop` on plateau / budget
  * Numeric search writes:

    * `steps/step_k_candidates.json`
    * `online/search_state.json`
    * `online/surrogate_predictions_step_k.json`
  * Hypothesis writes:

    * `steps/step_k_hypothesis.json`
    * `steps/step_k_compiled_config.json`
  * Executor writes:

    * stdout/stderr logs
    * `steps/step_k_metrics.json`
    * `steps/step_k_final_env.json`
* **Post-run**

  * `postrun/rule_updates.json` (very simple diff-based rule)
  * `postrun/convergence.json`

### What’s missing for “agentic white‑box”

You have the skeleton, but the **explainability substrate is not real yet**:

* No unified **trace/event log** of *what the agent did*, *what it saw*, *what it retrieved*, *what it believed*, and *why*.
* The current TUI “Tools” tab is basically a **best-effort guess** based on which files exist.
* Hypotheses are simplistic (“apply memory rule” or mutate one param), and **don’t show structured evidence**.
* Numeric search has candidates + surrogate preds, but **does not record pruning stages** (“why was X dropped?”).
* The LLM tracing exists (jsonl), but isn’t integrated as a first-class “agent context window” object with:

  * core prompt + inserted memory + inserted RAG + metrics summaries + tool outputs
* Stopping is plateau-based, but not explained as:

  * “we stopped because … (A/B, noise model, confidence, SLA, budget)”
* Distillation is shallow: you need *semantic* rules (“when topology=… and messagesize regime=… then …”).

---

## 2) North-star requirement: “Agentic Transparency Contract”

To build the best white‑box system, you need a strict contract: **every decision must be reconstructible from artifacts**.

### Core principles

1. **Every step is a mini scientific paper**:

   * Observation → Evidence retrieved → Hypotheses considered → Candidate set → Surrogate predictions → Pruning decisions → Action chosen → Tools executed → Metrics measured → Update beliefs
2. **No “hand-wavy” rationale**:

   * Every explanation must reference *specific evidence objects* (metrics ID, microbench signal ID, rule ID, doc chunk ID, surrogate model version ID, etc.).
3. **Trace first, UI second**:

   * TUI should be a pure viewer of structured trace + artifacts.
4. **Semantic explanation ≠ raw chain-of-thought**:

   * Store *structured reasons*, *evidence links*, *counterfactual “why not”*, *success criteria*.
5. **Reproducibility**:

   * Anyone can replay a run and understand exactly why each action happened.

This contract implies you need two new “products” in the repo:

* **A structured trace format** (JSONL events + indexed)
* **A TUI that renders that trace** (not guessing)

---

## 3) The big architectural upgrade: “Evidence Graph + Trace Bus”

### 3.1 Add a first-class Trace Bus

Create a module: `src/trace/`

#### Key objects

* `TraceEvent` base: `{ts, run_id, phase, step, actor, type, payload, refs, status, duration_ms}`
* Specialized event types:

  * `retrieval.memory`
  * `retrieval.rag`
  * `offline.microbench.plan` / `offline.microbench.run` / `offline.microbench.result`
  * `analysis.metrics.derive`
  * `analysis.bottleneck.classify`
  * `proposal.hypothesis` (multi-hypothesis list)
  * `proposal.numeric_candidates`
  * `model.surrogate.predict`
  * `search.prune` (with stage-by-stage reasoning)
  * `decision.select_action` (includes “why not”)
  * `tool.call` (all tool calls)
  * `tool.result`
  * `safety.risk_score`
  * `safety.rollback`
  * `stop.decision`
  * `postrun.distill.rule`
  * `postrun.train.surrogate`

#### Storage layout

Inside each `artifacts/<run_id>/`:

* `trace/events.jsonl` (append-only)
* `trace/index.json` (optional: step → byte offsets)
* `trace/evidence/` (optional: large blobs, e.g. full LLM messages, long logs)

This is what enables the “ultimate white-box”.

### 3.2 Add an Evidence Registry (stable IDs)

Every referenced item becomes an addressable “evidence object” with an ID:

* `metric:<step>:primary`
* `metric:<step>:derived:<name>`
* `rule:<uuid>`
* `rag:<doc_id>:<chunk_id>`
* `microbench:<signal_id>`
* `surrogate:<model_id>`
* `candidate:<step>:<candidate_id>`
* `tool:<step>:<call_id>`
* `log:<step>:stdout|stderr`

The **trace events must refer to these** (not raw strings).

### 3.3 “Context Pack” becomes a real artifact

Before every decision (offline or online), write:

* `steps/step_k_context_pack.json`

Contents (example shape):

```json
{
  "schema_version": "1.0",
  "phase": "online",
  "step": 7,
  "workload": {...},
  "context_signature": {...},
  "observations": {
    "last_metrics_ref": "metric:6:primary",
    "trend": {"best_ms": 812.3, "baseline_ms": 1000.0, "plateau_count": 2}
  },
  "retrieval": {
    "memory_rules": [{"ref":"rule:...", "score":0.83, "why":"topology+scale match"}],
    "rag_chunks": [{"ref":"rag:docX:chunk12", "score":0.71, "topic":"NCCL_PROTO"}]
  },
  "models": {
    "surrogate_ref": "surrogate:ctxhash:2026-...",
    "n_train": 28,
    "calibration": {"cv_mae_ms": 14.2, "uncertainty_calibrated": true}
  },
  "constraints": {
    "sla": {...},
    "safe_envelope": {...},
    "risk_budget": {...}
  }
}
```

This is the “agent context window” at the system level (separate from LLM).

### 3.4 LLM “Prompt Pack” becomes a first-class artifact

For every LLM call:

* `llm/call_<call_id>.json`

Include:

* `system_prompt_version`
* `messages[]` (role/content)
* `injected_context_refs[]`
* `token_estimates`
* `response`
* `parsed_structured_output` (if using JSON schema)
* `validation_errors` (if any)

Then in trace:

* `llm.call` event refers to `llm:call_<id>`

This makes “what core prompt and memory were used” fully inspectable.

---

## 4) Offline stage: make it a *transparent scientific planning phase*

Your offline stage must visibly answer:

* “What knowledge/rules did we retrieve?”
* “Should we run microbench? Which one?”
* “What warm-start config do we propose? Why?”
* “What search space is pruned? Why?”

### 4.1 Offline = three explicit sub-phases

1. **Context acquisition**

* Detect:

  * NCCL version, CUDA version, driver, GPU model, NVLink/PCIe topology, NIC type
  * Rank topology (nodes, GPUs per node)
  * Workload signature (model, batch, tensor shapes if available)
* Emit:

  * `trace: offline.context.detect`
  * store `offline/context_snapshot.json`

2. **Knowledge retrieval**

* Memory retrieve:

  * top-K rules + scores + explanation of scoring breakdown
* RAG retrieve:

  * doc chunks categorized by topic: params, transport, failure modes, scaling law
* Emit:

  * `retrieval.memory` event with scoring breakdown
  * `retrieval.rag` with chunk IDs and match rationale
* Store:

  * `offline/retrieval_pack.json` (a stable snapshot of what was retrieved)

3. **Experiment design**

* Decide if microbench runs:

  * Full CCLInsight primitive microbench? (expensive)
  * Reduced microbench? (subset sizes, fewer repeats)
  * Skip? (if high-confidence memory + high similarity + low drift)
* Produce:

  * `offline/microbench_plan.json`
* Execute microbench:

  * Write:

    * raw logs
    * summary with parameter importance + signals
  * Emit:

    * `offline.microbench.run` + `offline.microbench.result`
* Produce:

  * **Search pruning report**
  * **Warm-start proposal**
  * **Initial online plan**
* Store:

  * `offline/offline_report.md` (human narrative)
  * `offline/offline_report.json` (structured)

### 4.2 Warm-start selection (state of the art)

Instead of “baseline = defaults”, do:

**Candidate warm starts**

* `C0`: pure defaults
* `C1..Ck`: apply top memory rules (individually + combined, within safe envelope)
* `Cm`: microbench-derived best region (if available)
* `Cs`: surrogate-suggested from historical dataset (transfer)

**Scoring**
For each candidate:

* risk score (safety)
* predicted improvement (surrogate mean)
* uncertainty
* expected cost if wrong (rollback cost + SLA risk)
* compatibility with topology (rules often depend on interconnect)

**Decision output**
Pick warm start with:

* highest expected improvement subject to safety & uncertainty constraints

**Artifacts**

* `offline/warm_start_candidates.json` (full list + scores)
* `offline/warm_start_decision.json` (selected + why + why-not)
* Emit `decision.offline_warm_start`

### 4.3 Search space pruning (must be explainable)

Create an explicit pipeline:

* Start from full `ParameterSpace`
* Apply:

  1. **Hard constraints** (validation)
  2. **Safety constraints** (risk: deadlocks/crashes)
  3. **Context constraints** (transport-specific relevance)
  4. **Importance constraints** (microbench + CCLInsight + learned importance)
  5. **Budget constraints** (max steps/time)

For each removal/prune, record:

* `param`: NCCL_NET_GDR_LEVEL
* `action`: “fix to default” or “exclude”
* `reason`: “hardware-dependent, low gain, failure risk” (plus evidence refs)

Artifact:

* `offline/search_space_pruning.json`

Trace:

* `search.prune` events (phase=offline)

---

## 5) Online stage: every step becomes a fully explainable loop

The online stage must answer:

* “What hypothesis did we form? Why?”
* “How did surrogate shape candidate generation?”
* “Why were candidates pruned?”
* “What tools did we call, in what order, with what inputs/outputs?”
* “When/why did we stop, and why is best config final?”

### 5.1 Replace “single hypothesis” with “Hypothesis Portfolio”

`HypothesisGenerator.propose()` should return **N hypotheses** with structured fields:

* `hypothesis_id`
* `summary`
* `patch` (param changes)
* `mechanism` (latency-bound vs bandwidth-bound, protocol selection, parallelism, transport)
* `evidence_refs[]` (rules, rag chunks, metrics-derived facts)
* `expected_effect` (direction + predicted delta range)
* `risk_assessment`
* `test_plan` (what metrics validate it)

Then you do:

1. score hypotheses with surrogate
2. prune by risk
3. optionally diversify (don’t test 5 variants of the same idea)

Artifacts:

* `steps/step_k_hypothesis_portfolio.json`
* `steps/step_k_hypothesis_ranked.json`

Trace:

* `proposal.hypothesis` event
* `model.surrogate.predict` for each hypothesis candidate
* `decision.select_action` referencing chosen hypothesis

### 5.2 Numeric search: must show generation + pruning stages

Right now you record candidates; you need **a full pruning audit**.

Implement numeric search as a multi-stage pipeline:

1. generate raw neighbors from subspace (coordinate descent / subspace coord descent)
2. canonicalize + deduplicate
3. validate constraints (ParameterSpace.validate)
4. compute risk score → drop over threshold
5. surrogate predict → compute acquisition
6. prune by acquisition threshold (budget-aware)
7. select top-N to evaluate (or 1-step if sequential)

For each candidate, record a lifecycle:

* `candidate_id`
* `config`
* `stages`:

  * `generated` (source: neighbor rule)
  * `deduped` (kept/dropped)
  * `validated` (kept/dropped)
  * `risk_scored` (risk=…, kept/dropped)
  * `predicted` (mean/std)
  * `ranked` (rank)
  * `selected` (true/false)
  * `evaluated` (true/false)

Artifact:

* `steps/step_k_candidates_trace.json` (table-like)
* `steps/step_k_pruning_report.md` (human-readable)
  Trace:
* `proposal.numeric_candidates`
* `search.prune` (stage-by-stage stats)
* `model.surrogate.predict`

### 5.3 “Tools called” must be literal, not inferred

Wrap every tool call through a single instrumentation layer:

* `ToolRunner.call(tool_name, args) -> result`
* this wrapper:

  * emits `tool.call` with args
  * measures time
  * captures outputs summary + artifact refs
  * emits `tool.result`

Tools to instrument:

* microbench runner (offline)
* config compiler (hypothesis + numeric)
* NCCL apply (env patch)
* workload runner (training / benchmark)
* metrics collector
* SLA enforcer
* rag retrieval
* memory retrieval
* surrogate predict / train

Then the TUI shows **exact call sequence** per step.

### 5.4 Metrics ingestion must produce “Derived Metrics + Interpretation”

Do not feed raw metrics only. Add:

* `analysis.metrics.derive` event
* `steps/step_k_metrics_derived.json`

Derived metrics examples (design-level; depends on what you can measure):

* `comm_time_ms`, `comp_time_ms`, `comm_fraction`
* `algbw`, `busbw` (from NCCL tests or profiling)
* `variance`, `confidence_interval`
* `sla_status` and “why”
* `regression_detected` (thresholded)

Then run a **bottleneck classifier** (even if heuristic first):

* output:

  * class: `latency_bound` / `bandwidth_bound` / `launch_overhead` / `network_saturation` / `gpu_memcpy_bound` / `unknown`
  * confidence
  * evidence refs

Artifacts:

* `steps/step_k_bottleneck.json`
  Trace:
* `analysis.bottleneck.classify`

This becomes the semantic “why” backbone.

### 5.5 Stopping criteria must be a structured argument

Replace “plateau_count >= patience” with a richer stop policy that is still explainable:

* Budget stop:

  * max steps
  * max wall time
* Plateau stop:

  * no statistically significant improvement in last W steps
  * improvement < min_improvement AND uncertainty overlaps
* Safety stop:

  * failure rate
  * SLA violations
* Diminishing returns stop:

  * predicted remaining improvement (surrogate) < threshold

Artifacts:

* `steps/step_k_stop_candidate.json` (if considered)
* `steps/step_k_stop_decision.json` (if taken)
  Trace:
* `stop.decision` with “argument” fields:

  * `claims[]` each with evidence refs

Also add **final confirmation run**:

* re-run best config X times to verify
* store `postrun/best_config_validation.json`

---

## 6) Post-run: distill *semantic knowledge*, not just diffs

This is where you become “best in the world”:

* you’re not just tuning, you’re building **portable, explainable expertise**

### 6.1 Distill rules from *causal-ish* evidence

Current: “best config differs from baseline, improvement=…”
Upgrade: rule distillation should answer:

* which parameter(s) actually caused improvement?
* under what conditions does it generalize?
* how reliable is it (variance)?
* what are the side effects / risks?

#### Distillation pipeline

1. **Normalize history**:

   * group identical configs
   * compute mean/variance per config
2. **Estimate marginal effects**:

   * for each parameter, compare configs differing primarily in that param
   * compute effect size + confidence
3. **Identify interactions**:

   * e.g., `NCCL_ALGO` × `NCCL_PROTO` × channels
4. **Induce conditions**:

   * context features: topology, GPU model, nodes, scale, message size regime (from microbench/profiling)
   * derive “when this helps”
5. **Write rule object**:

   * condition, action, effect distribution, evidence refs, risk

Artifacts:

* `postrun/rules_distilled.jsonl` (one per rule)
* `postrun/distillation_report.md` (human narrative)
  Trace:
* `postrun.distill.rule` events (one per rule)

### 6.2 Surrogate training should be transparent

When training:

* log dataset stats
* CV metrics
* feature importance
* drift checks vs previous model

Artifacts:

* `postrun/surrogate_training_report.json`
* `memory/models/surrogate_<id>.json` (already exists; expand fields)
  Trace:
* `postrun.train.surrogate`

---

## 7) The TUI redesign: panes that actually make sense

Your current TUI is a good start, but for “ultimate white-box” you want a **trace-driven, evidence-first** UI.

### 7.1 New TUI navigation model

#### Left sidebar: Run + Step Timeline

* **Run picker** (like you have)
* For selected run:

  * offline phase node (expandable)
  * online steps list
  * postrun node
* Each step row shows:

  * icon: `M` microbench, `H` hypothesis, `N` numeric, `R` rollback, `S` stop
  * delta vs best/baseline
  * status: ok/fail/SLA
  * quick tag: “latency-bound”, “bandwidth-bound”, etc.

#### Center main: “Narrative” panel (Markdown)

For the selected node (offline/step/postrun), show a **structured narrative**:

* **What we observed**
* **What evidence we retrieved**
* **What we believe**
* **What we tried (and why)**
* **What we expected**
* **What happened**
* **What we learned**

This is generated deterministically from trace + artifacts (optionally LLM-polished, but grounded).

#### Right panel: Tabbed deep dive

Tabs (these are the ones that “actually make sense” for your requirements):

1. **Trace**

* chronological list of trace events for this node
* selecting an event shows payload + refs

2. **Tools**

* literal tool call list:

  * tool name
  * args (collapsed)
  * runtime
  * output summary
  * links to artifacts (stdout/stderr/metrics json)

3. **Candidates**

* (numeric steps) candidate table with lifecycle stages
* filter: show pruned / kept / evaluated

4. **Surrogate**

* model id/version
* training set size
* last fit stats
* candidate predictions table (mean/std, acquisition)
* feature importance snapshot (even a simple top-10 list)

5. **Hypotheses**

* hypothesis portfolio
* ranked list with:

  * evidence refs
  * expected effect
  * risk
  * why selected / why rejected

6. **Memory + RAG**

* retrieved rules list with scoring breakdown
* doc chunks with snippet preview
* “applied rule” markers

7. **Context Window**

* shows the **Context Pack** JSON (system-level)
* shows **LLM Prompt Pack(s)** if any occurred in that step
* token estimates, redactions if needed

8. **Metrics**

* raw metrics
* derived metrics
* trend view (sparkline / ascii plot)
* variance/confidence info

9. **Config**

* baseline vs current vs best diff
* include:

  * “semantic label” per param (from nccl_params knowledge)
  * risk highlights

10. **Stop / Convergence**

* plateau analysis
* confidence intervals
* stop decision argument graph
* final validation runs

### 7.2 “Agent Context Window Visualizer” (your explicit request)

In TUI:

* show exactly what the agent saw *as structured objects*:

  * context signature
  * latest metrics summary + derived interpretation
  * retrieved rules (IDs + content)
  * retrieved docs (IDs + snippet)
  * surrogate summary
  * constraints
* and what the LLM saw (prompt pack), clearly segmented:

  * system prompt
  * injected memory section
  * injected rag section
  * metrics summary section
  * “requested output schema” section

### 7.3 Hotkeys / UX polish (professional operator UX)

* `j/k` next/prev step
* `g/G` top/bottom
* `t` jump to Trace tab
* `o` open stdout/stderr selector
* `c` copy current config patch (printable export)
* `b` copy best config export
* `/` search within current panel
* `f` filter candidates (pruned/kept/evaluated)
* `?` help overlay
* `r` refresh (live mode)

### 7.4 Live mode (must-have for monitoring)

Your artifacts are written during runs; implement a “tail trace” mode:

* watch `trace/events.jsonl`
* update UI live:

  * tool calls appear as they happen
  * current running tool highlighted
  * last metrics update causes narrative refresh

This is the killer “monitoring & understanding” feature.

---

## 8) Extremely detailed TODO plan (engineering roadmap)

Below is a **very detailed, implementation-oriented** TODO list (with file-level guidance). I’m optimizing for “best system in the world” criteria: auditable, reproducible, operator-friendly, and scientifically grounded.

---

### TODO 0 — Define the Trace + Evidence Schemas (the contract)

**Goal:** one stable spec that everything must write to.

**Tasks**

* Create `doc/Design/trace_schema.md`:

  * `TraceEvent` common fields
  * event types + payload schemas
  * evidence ID format rules
* Create `doc/Design/artifact_schema.md`:

  * `ContextPack`, `PromptPack`, `CandidateTrace`, `StopDecision`, etc.
* Add `schema_version` fields everywhere.

**Acceptance criteria**

* You can validate a run directory against the schema without running the agent.

---

### TODO 1 — Implement `src/trace/` (Trace Bus)

**Files**

* `src/trace/events.py` (dataclasses)
* `src/trace/emitter.py` (TraceEmitter interface)
* `src/trace/writer.py` (JSONL writer + flush + index)
* `src/trace/reader.py` (load + query by step/type)
* `src/trace/span.py` (context manager for timing, exceptions)

**Key design choices**

* JSONL append-only for robustness
* optional index file for fast random access
* each event has `refs[]` to evidence objects

**Acceptance criteria**

* A run produces `trace/events.jsonl` even in dry-run.
* TUI can show events without reading other artifacts.

---

### TODO 2 — Instrument ALL tools via a single wrapper

**Goal:** “Tools called” is always correct.

**Files**

* Add `src/tools/instrumented.py`:

  * `InstrumentedToolSuite` wrapping `ToolSuite`
  * or a generic `tool_call()` wrapper used everywhere

**Implementation details**

* For each tool call:

  * generate `call_id`
  * emit `tool.call` with args + refs
  * run tool
  * emit `tool.result` with outputs + artifact refs

**Acceptance criteria**

* For any step, the tool sequence is visible from trace alone.

---

### TODO 3 — Make LLM calls first-class (Prompt Packs + Trace)

**Goal:** “core prompt, memory, context window” visible.

**Files**

* `src/llm/traced_client.py` wrapper around existing clients
* Modify `create_llm_client()` or `main.py` to wrap client with tracer if run_context exists

**What to store**

* system prompt version ID
* messages (or chunked if huge)
* injected refs (memory/rag/metrics)
* response
* parse status (if JSON)
* token estimate

**Acceptance criteria**

* You can open a step and see every LLM request/response with full context.

---

### TODO 4 — Build `ContextPack` at every decision point

**Files**

* `src/agent/context_pack.py`
* Integrate in:

  * offline planner before plan generation
  * online analyzer before selecting action
  * postrun distiller before rule creation

**Acceptance criteria**

* `steps/step_k_context_pack.json` exists for every online step.
* offline has `offline/context_pack.json`.

---

### TODO 5 — Offline stage redesign into explicit “Offline Reasoner”

**Files**

* `src/agent/offline_reasoner.py` (new)
* Update `planner.py` to use it

**Capabilities**

* decide microbench type + scope (full vs reduced vs skip)
* retrieve memory + rag with scoring breakdown
* warm-start candidate generation + surrogate scoring
* prune search space with full justification
* write offline_report.md/json
* emit trace events for each decision

**Acceptance criteria**

* Offline phase produces a complete, readable narrative + structured report.

---

### TODO 6 — Hypothesis Generator → Hypothesis Portfolio + scoring

**Files**

* Replace/extend `src/agent/hypothesis.py`

**Implementation**

* Generate N hypotheses from:

  * memory rules (top-k)
  * rag heuristics (parameter semantics)
  * bottleneck class mapping (latency-bound vs bandwidth-bound)
  * surrogate feature importance (what matters most right now)
* Score each hypothesis:

  * surrogate predicted improvement + uncertainty
  * risk score
  * novelty/diversity penalty

**Artifacts**

* `step_k_hypothesis_portfolio.json`
* `step_k_hypothesis_ranked.json`

**Acceptance criteria**

* Hypothesis step shows “why this hypothesis now” + “why not the others”.

---

### TODO 7 — Numeric Search → candidate lifecycle + pruning transparency

**Files**

* Extend `src/agent/numeric.py`
* Extend `src/search/coordinate_descent.py` (or replace with subspace coord descent)

**Implementation**

* Candidate lifecycle tracking
* Stage-by-stage prune logs with reason codes:

  * `duplicate`
  * `invalid_config`
  * `risk_too_high`
  * `predicted_regression`
  * `uncertainty_too_high` (if you want)
  * `budget_prune`

**Artifacts**

* `step_k_candidates_trace.json` (full)
* `step_k_pruning_summary.json` (stage stats)

**Acceptance criteria**

* For any candidate, you can answer “why was it pruned?”

---

### TODO 8 — Metrics pipeline: derived metrics + bottleneck classifier

**Files**

* `src/agent/metrics_derive.py`
* `src/agent/bottleneck.py`

**Implementation**

* derive stable metrics fields
* compute noise estimates over repeats
* classify bottleneck with explanation

**Artifacts**

* `step_k_metrics_derived.json`
* `step_k_bottleneck.json`

**Acceptance criteria**

* “why we tried NCCL_PROTO next” can cite “latency_bound@0.78”.

---

### TODO 9 — Decision engine: structured “why/why-not” output

**Files**

* Refactor `src/agent/analyzer.py`

**Implementation**

* Produce `DecisionRecord` with:

  * `chosen_action`
  * `candidate_considered[]`
  * `why_selected[]` (claims + evidence refs)
  * `why_rejected[]` (for top alternatives)
  * `expected_outcome` (surrogate-based)
  * `success_criteria` (what metric validates it)
  * `rollback_plan`

**Artifacts**

* `step_k_decision.json` becomes richer (or split into `decision_record.json`)

**Acceptance criteria**

* Every step has a structured argument, not just “action=numeric”.

---

### TODO 10 — Safety + rollback becomes first-class and visible

**Files**

* Integrate `src/safety/rollback.py` into core loop
* Expand `src/safety/risk.py` to output factor contributions

**Implementation**

* risk score should include:

  * per-parameter risk factors
  * interaction risk
  * “unknown risk” penalty for untested regions
* rollback should trigger on:

  * SLA violation
  * hard failure
  * severe regression beyond threshold

**Artifacts**

* `step_k_risk_report.json`
* `step_k_rollback_decision.json` (if used)

**Acceptance criteria**

* UI can show exactly why rollback happened.

---

### TODO 11 — Stop policy becomes explicit + final validation run

**Files**

* `src/agent/stop_policy.py`
* integrate into analyzer

**Implementation**

* multi-cause stop reasoning:

  * plateau w/ stats
  * budget
  * predicted remaining gain
  * safety
* final validation:

  * run best config repeats
  * store stats

**Artifacts**

* `postrun/best_config_validation.json`
* `step_k_stop_decision.json`

**Acceptance criteria**

* Stopping is justified with concrete evidence and confidence.

---

### TODO 12 — Post-run distillation: semantic rules + evidence links

**Files**

* Replace `src/agent/post_run.py` with a real distiller:

  * `src/agent/distill.py`

**Implementation**

* effect estimation (mean/var, not just delta)
* conditions induction
* rule JSONL with evidence refs

**Artifacts**

* `postrun/rules_distilled.jsonl`
* `postrun/distillation_report.md`

**Acceptance criteria**

* Rules are usable, searchable, and explainable.

---

### TODO 13 — TUI overhaul: trace-driven viewer (no inference)

**Files**

* Replace/major-refactor `src/tui/app.py`
* Add `src/tui/views/*` (split views)

**Implementation**

* Left: Run + step timeline
* Center: Narrative panel
* Right: tabbed deep dive (Trace/Tools/Candidates/Surrogate/Hypotheses/Memory+RAG/Context/Metrics/Config/Stop)
* Live tail mode from `trace/events.jsonl`

**Acceptance criteria**

* You can run the agent and watch the trace stream live with full explainability.

---

### TODO 14 — “Semantic labels” for NCCL parameters everywhere

**Goal:** no raw env var appears without meaning.

**Files**

* Expand `doc/Knowledge/nccl_params.md` into machine-readable:

  * `src/knowledge/nccl_params.yaml`
* Add `src/knowledge/param_semantics.py`

**Usage**

* TUI config diff tab shows:

  * param meaning
  * expected mechanism
  * risk notes
  * links to rules/docs

**Acceptance criteria**

* Operator sees “what NCCL_BUFFSIZE does” inline.

---

### TODO 15 — Export / replay / audit tooling (professional ops)

**Files**

* `scripts/export_run_report.py` → produces Markdown/PDF report from artifacts
* `scripts/replay_trace.py` → replays events and reconstructs decisions (debug)
* `scripts/validate_run_dir.py` → schema validation

**Acceptance criteria**

* Every run can be exported as a “tuning report” for humans.

---

## 9) “Best-in-the-world” extras (optional but high impact)

### A) Per-collective tuning mode (AutoCCL-style)

* Instead of only global env vars:

  * tune per collective call signature (op type, msg size, topology)
* Requires:

  * profiler tool producing per collective metrics
  * config scoping mechanism (ext-tuner plugin)
* TUI: add “Collectives” view (grouped by signature)

### B) Profiling integration (CCLInsight-style primitive-centric)

* integrate your `tools/AF_ICSE26` pipeline outputs as:

  * evidence objects
  * parameter importance
  * scaling-law hints
* show primitive breakdown in TUI.

### C) Multi-objective tuning

* Optimize iteration time but also:

  * stability
  * GPU util
  * network congestion
  * memory overhead
* Decision engine becomes Pareto-based; UI shows tradeoffs.

---

## 10) The single most important change: make explanations *structural*

If you do only one thing: implement the **Trace Bus + Evidence Registry + ContextPack**.

Everything else (better hypotheses, better search, better UI) becomes easy once:

* every action has a stable ID
* every reason points to evidence objects
* the UI just renders the recorded truth

That’s how you get “white-box ultimate system”.

---

If you want, next I can produce:

* a concrete **`trace/events.jsonl` event type catalog** (every event + exact payload fields)
* a **new Textual layout sketch** (grid layout + widget choices + hotkeys)
* a **migration plan** from current artifacts to the new schema (backward compatible so existing runs still render)

But the plan above is already end-to-end and implementable as a professional roadmap.
