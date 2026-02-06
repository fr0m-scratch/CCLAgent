# True-Agentic LLM TODO (Design + Implementation Checklist)

This TODO is the “LLM is truly agentic” upgrade plan. It is **implementation-oriented** and intended for another coding agent to execute with minimal ambiguity.

Scope:
1. Make the **warm start** an explicit **hypothesis program** (single config or short probe series).
2. Make the **online loop** **LLM-involved by default**.
3. Keep the system **white-box**: every decision is reconstructible from artifacts + trace, with strict schemas and evidence refs.
4. Make online LLM usage **async and time-boxed** so tuning never hard-blocks on model latency.

Non-negotiables:
- The LLM is an adviser. Final actions are selected via deterministic validation + risk + SLA + budget checks.
- No chain-of-thought storage. Persist **structured claims with evidence refs**.
- Every LLM call produces a `Prompt Pack` (`llm/call_<id>.json`) and a trace event, and is linkable to a step.

Related docs/code:
- Design novelty: `doc/Design/coreNovelty.md`
- Existing white-box TODO: `doc/todo/TODO.md`
- Offline planner: `src/agent/planner.py`
- Online decision: `src/agent/analyzer.py`
- LLM tracing: `src/llm/traced_client.py`
- Context window: `src/llm/context_window.py`
- Artifact schemas: `doc/Design/artifact_schema.md`
- Trace schema: `doc/Design/trace_schema.md`

---

## 0) Terminology

- **WarmStartProgram**: A small plan that is itself a hypothesis (or a sequence of hypotheses) evaluated quickly to pick a baseline for the full online budget.
- **Online Decision Support**: An LLM call (async) that produces hypotheses + numeric guidance + optional tool requests, tied to a specific online step.
- **Soft wait**: The time the agent is willing to pause to incorporate online LLM output; after that it proceeds with deterministic fallbacks.
- **Hard timeout**: The request timeout applied to the LLM provider client (network timeout).

---

## 1) Target Behavior (What “Done” Looks Like)

1. Offline planning produces:
   - `offline/warm_start_program.json`
   - `offline/hypothesis_playbook.json`
   - `offline/pruning_guidance.json`
   - `offline/subspace_priors.json`
   - plus existing offline artifacts (`initial_plan.json`, etc.)
2. At run start, the agent optionally runs a warm-start probe series:
   - runs `N` short evaluations, selects a winner deterministically
   - records artifacts + trace explaining why that baseline was selected
3. Online loop is **LLM-involved by default**:
   - every step schedules `online_decision_support_v1` (async)
   - the agent proceeds even if LLM is slow/unavailable
   - when available, LLM output influences hypothesis portfolio and numeric focus for the next action
4. Post-run distillation uses an LLM to generate semantic rules:
   - rules include conditions, actions, expected effects, limitations, risk notes
   - every rule references evidence IDs (metrics/steps)
5. TUI can display per-step:
   - the selected action, candidates, pruning, and the LLM’s decision-support output (if available)
   - the `Prompt Pack` content for each LLM call

---

## 2) Schema & Artifacts (Update First)

### Task 2.1 — Extend Artifact Schema Docs

Files:
- Update `doc/Design/artifact_schema.md`

Add schemas:
- WarmStartProgram
- OnlineDecisionSupportOutput
- OnlineLLMAdvice (optional “late advice” format)

Acceptance criteria:
- Docs specify required fields, optional fields, and evidence ref requirements for each artifact.

### Task 2.2 — Extend Trace Schema Docs

Files:
- Update `doc/Design/trace_schema.md`

Add event types:
- `decision.warm_start_program`
- `warmstart.probe.run` (one per probe evaluation)
- `warmstart.probe.select` (final selection + why)
- `llm.advice` (parsed decision-support output stored, with refs)

Acceptance criteria:
- Each new event type has a canonical payload shape and a list of expected `refs`.

---

## 3) Config Changes (Make Latency + Frequency Explicit)

### Task 3.1 — Add LLM Online Settings

Files:
- Update `src/types.py` (`LLMSettings`)
- Update `src/config.py` default config + config load/merge
- Update `doc/Design/currentImplementation.md` (defaults)

Recommended new fields in `LLMSettings`:
- `system_prompt_version_offline: str` (default `offline_strategic_plan_v2`)
- `system_prompt_version_online: str` (default `online_decision_support_v1`)
- `system_prompt_version_postrun: str` (default `postrun_distill_rules_v2`)
- `online_enabled: bool` (default `true`)
- `online_call_every_steps: int` (default `1`)
- `online_soft_wait_s: float` (default `2.0`)
- `online_hard_timeout_s: int` (default `15`)
- `online_triggers: list[str]` (default includes `always`, `plateau`, `bottleneck_flip`, `high_uncertainty`, `failure`)

Notes:
- “soft wait” is about *decision integration*.
- “hard timeout” must propagate to provider clients (see Task 6.3).

Acceptance criteria:
- New settings appear in config snapshot.
- CLI can override via JSON config (no need for new CLI flags yet).

### Task 3.2 — Add Warm Start Settings

Files:
- Update `src/types.py` (either `TuningBudget` or a new dataclass like `WarmStartSettings`)
- Update `src/config.py`
- Update `doc/Design/currentImplementation.md`

Recommended fields:
- `warm_start_mode: str` (`single|series`, default `series`)
- `warm_start_max_candidates: int` (default `3`)
- `warm_start_eval_steps: int` (default `50`)
- `warm_start_eval_timeout_sec: int` (default `300`)
- `warm_start_concurrency: int` (default `1`)
- `warm_start_counts_toward_budget: bool` (default `false`)

Acceptance criteria:
- Warm start behavior can be enabled/disabled and bounded by config.

---

## 4) Prompts Package (Put Prompt Text Under Version Control)

### Task 4.1 — Create Prompt Library Files

Files to add:
- `doc/Prompts/offline_strategic_plan_v2.md`
- `doc/Prompts/online_decision_support_v1.md`
- `doc/Prompts/online_playbook_refresh_v1.md` (optional trigger-based)
- `doc/Prompts/postrun_distill_rules_v2.md`

Each file must include:
- Prompt version name
- Output JSON schema
- SYSTEM prompt text
- USER template section list (what ContextWindow sections must exist)
- Validation rules (keys must exist in param specs, max patch size, evidence refs required)

Acceptance criteria:
- `LLMSettings.*system_prompt_version*` refers to one of these docs by name.

---

## 5) Offline: LLM Produces a Strategic Plan (Not Just a Patch)

### Task 5.1 — Replace Offline LLM Output Contract

Current:
- `OfflinePlanner._propose_llm_config()` expects a JSON dict of env var settings.

New:
- Offline LLM returns a single JSON object with:
  - `warm_start_program`
  - `baseline_patch` (<= 6 keys)
  - `pruning_guidance`
  - `subspace_priors`
  - `hypothesis_playbook`
  - `tool_triggers`
  - `claims` + `uncertainty`

Files:
- Update `src/agent/planner.py`
- Update `src/agent/offline_reasoner.py` (merge logic, fallback)
- Update `doc/Design/end_to_end_training_workflow.md` section “Initial plan creation”

Implementation notes:
- Keep the existing warm-start candidates + pruning artifacts from `OfflineReasoner`, but allow the LLM plan to:
  - choose `mode=single|series`
  - propose a short list of probe candidates with eval plan
- Validate everything:
  - proposed keys exist in `ParameterSpace`
  - values pass `ParameterSpace.validate`
  - compile + risk score with existing safety logic
- Persist structured plan artifacts:
  - `offline/warm_start_program.json`
  - `offline/hypothesis_playbook.json`
  - `offline/pruning_guidance.json`
  - `offline/subspace_priors.json`

Acceptance criteria:
- Offline phase produces the plan artifacts even in `--dry-run` (LLM may be `NullLLMClient`; still write empty plan with uncertainty).

### Task 5.2 — Offline Strategic Plan Prompt

Prompt version:
- `offline_strategic_plan_v2`

SYSTEM prompt (canonical, paste into prompt library file):
```text
You are the offline strategic planner for a CCL/NCCL tuning agent.

You MUST obey:
1) Output MUST be a single JSON object, no markdown, no extra keys.
2) Only use parameter names that appear in [PARAM_SPECS]. Do not invent knobs.
3) Every proposed value must respect constraints in [PARAM_SPECS] (choices, min/max, type).
4) Keep baseline_patch small: at most 6 keys. Prefer high-importance parameters.
5) No hallucinations: every nontrivial recommendation must cite refs from provided evidence IDs.
6) Do NOT include chain-of-thought. Provide brief structured claims with refs instead.
7) If evidence is insufficient, set baseline_patch={}, warm_start_program.mode="single", and explain in uncertainty.safe_fallback.

Primary objective: reduce iteration_time_ms safely.
Secondary objective: produce a WarmStartProgram and an online hypothesis playbook.
```

USER sections (must be in the ContextWindow):
- `WORKLOAD`
- `CONTEXT_SIGNATURE`
- `PARAM_SPECS`
- `MICROBENCH_SIGNALS`
- `IMPORTANT_PARAMS`
- `MEMORY_RULES`
- `RAG_SNIPPETS`
- `SAFETY_ENVELOPE`
- `CURRENT_DEFAULTS`

Output schema (summarized; full version should be in `doc/Prompts/offline_strategic_plan_v2.md`):
- `warm_start_program.mode` is `single|series`
- `warm_start_program.candidates[].patch` is <= 4 keys
- `hypothesis_playbook[]` entries are <= 4 keys each
- `pruning_guidance[]` actions are `freeze_default|freeze_value|keep_free`

---

## 6) Warm Start Execution: Probe Series Before Full Budget

### Task 6.1 — Implement WarmStartProgram Runner

Goal:
- If `warm_start_program.mode == "series"`, run short probes and select baseline config deterministically.

Files:
- Update `src/agent/core.py` (between offline plan and entering the main step loop)
- Possibly add `src/agent/warmstart.py` (recommended to keep core clean)

Proposed algorithm:
1. Build `N` candidate configs by applying each candidate patch onto defaults (or onto selected warm-start base).
2. Validate + compile + risk score each candidate.
3. Drop invalid or over-risk candidates and record why.
4. Evaluate remaining candidates using `WorkloadExecutor.run_batch` with:
   - `CCL_EVAL_MODE=short`
   - `CCL_EVAL_STEPS` and `CCL_EVAL_TIMEOUT_SEC` from settings
5. Select winner:
   - primary: min `iteration_time_ms` among successful
   - tie-break: lower risk_score, then lower surrogate uncertainty (if available)
6. Persist:
   - `offline/warm_start_program.json`
   - `offline/warm_start_probe_results.json` (metrics per candidate)
   - `offline/warm_start_decision.json` (winner + why + why-not)
7. Emit trace:
   - `warmstart.probe.run` per candidate (refs include `candidate:*` and `metric:*`)
   - `warmstart.probe.select`

Acceptance criteria:
- Warm-start probes can be disabled via config (`warm_start_mode=single` or max_candidates=1).
- Warm-start probes do not count against `budget.max_steps` by default.

### Task 6.2 — Make Trace Writer Thread-Safe

Reason:
- Online LLM decision support is async, and warmstart probing may be concurrent later.

Files:
- Update `src/trace/writer.py`
- Update `src/trace/emitter.py` if needed

Implementation:
- Add a `threading.Lock` inside `TraceWriter` and guard writes + flushes.

Acceptance criteria:
- Concurrent trace writes do not produce interleaved/corrupt JSONL lines in stress tests.

### Task 6.3 — Add Provider Hard Timeout Support

Goal:
- Respect `LLMSettings.online_hard_timeout_s`.

Files:
- Update `src/llm/base.py` (`OpenAICompatibleClient` should accept per-call timeout or be constructed with timeout)
- Update `src/llm/__init__.py::create_llm_client` to pass timeout from settings
- Update `src/llm/ollama.py` (already supports per-call `timeout_s`)

Acceptance criteria:
- Online calls do not hang indefinitely on network issues.

---

## 7) Online: LLM-Involved by Default (Async Decision Support)

### Task 7.1 — Add Async Online LLM Advisor

Files to add:
- `src/agent/online_advisor.py`

Responsibilities:
- Schedule one LLM decision-support call for a step.
- Provide `poll(step)` and `get_latest_ready()` APIs.
- Enforce:
  - hard timeout (provider request timeout)
  - soft wait budget at decision time
  - call frequency (`online_call_every_steps`)
- Persist parsed output:
  - `steps/step_<k>_llm_decision_support.json`

Suggested API:
```python
class OnlineLLMAdvisor:
    def request(self, *, step: int, context_pack: dict, plan: dict, param_specs: dict, recent_history: dict) -> None: ...
    def try_get(self, *, step: int) -> dict | None: ...
    def shutdown(self) -> None: ...
```

Acceptance criteria:
- In runs with `online_enabled=true`, every step schedules a call (subject to call frequency config).
- If the LLM is slow, the agent still produces an action using deterministic fallbacks.
- If the LLM finishes late, its output is still persisted and can influence later steps.

### Task 7.2 — Integrate Advisor into `TuningAnalyzer.plan_next_action`

Files:
- Update `src/agent/analyzer.py`

Integration design:
1. Build `step_k_context_pack.json` (already exists).
2. Call `advisor.request(step=k, ...)` at the start of `plan_next_action`.
3. Build deterministic hypothesis portfolio and numeric candidates as today.
4. `soft_wait` for LLM output:
   - `advice = advisor.try_get(step=k)` with timeout semantics.
5. If `advice` exists:
   - merge `advice.hypotheses` into hypothesis portfolio
   - apply `advice.numeric_guidance` to numeric search manager (focus_params/freeze/subspace_bias)
   - optionally schedule `tool_request` (gated, see Task 7.4)
6. Persist:
   - `steps/step_<k>_decision_record.json` includes:
     - refs to `llm:call_<id>` if advice used
     - why-selected and why-rejected claims with evidence refs

Acceptance criteria:
- Decision records explicitly reference whether LLM advice was used or not.
- Trace has:
  - `llm.call` events with `phase=online`, `step=k`
  - `decision.select_action` referencing the call ID when applicable

### Task 7.3 — Update Numeric Search to Accept Guidance

Files:
- Update `src/agent/numeric.py`
- Update `src/search/coordinate_descent.py` if needed

Add inputs:
- `focus_params`: restrict mutations to a subset
- `freeze_params`: fixed values or “don’t touch”
- `subspace_bias`: weights to rank subspaces first

Acceptance criteria:
- Candidate traces show the effect of guidance (why a candidate was generated, and what focus params were active).
- Guidance does not break determinism and is fully recorded in artifacts.

### Task 7.4 — Tool Request Gating (“Measure” Actions)

Goal:
- LLM may request at most one measurement when uncertainty is high.
- The system decides deterministically whether to comply.

Files:
- Update `src/agent/analyzer.py`
- Potentially add `src/agent/measurements.py`

Rules:
- Only allow tool requests from a small whitelist:
  - `nccltest.short`
  - `workload.short`
  - `microbench.reduced`
- Require:
  - reason claims with evidence refs
  - max cost budget (time) check
  - safety check

Acceptance criteria:
- Tool request decisions are explicit in trace and decision records.

---

## 8) Online Prompt: `online_decision_support_v1` (Default Every Step)

SYSTEM prompt (canonical):
```text
You are the online decision-support reasoner for a white-box CCL/NCCL tuning agent.

You MUST:
- Output a single JSON object exactly matching the requested schema.
- Only propose parameter keys that exist in [PARAM_SPECS].
- Keep each hypothesis.patch to <= 4 keys and prefer low-risk changes.
- Cite evidence using the provided evidence IDs in refs/evidence_refs.
- Do NOT output chain-of-thought. Use short claims with refs.

Objective: reduce iteration_time_ms safely under the provided risk/SLA budgets.
```

Required USER sections:
- `CONTEXT_PACK`
- `RECENT_HISTORY_SUMMARY`
- `CURRENT_PLAYBOOK` (from offline)
- `PARAM_SPECS`
- `RECENT_PRUNING`

Output schema summary:
- `interpretation` includes bottleneck class + confidence + claims
- `hypotheses[]` is a portfolio (<= 3 by default; allow config)
- `numeric_guidance` includes focus/freeze/subspace bias
- `tool_request` is whitelisted or `none`
- `recommended_action` is advisory only

Persistence:
- Always write parsed output to `steps/step_<k>_llm_decision_support.json` with:
  - `call_id`
  - `used_in_decision: bool`
  - `parse_errors: []`

---

## 9) Optional Online Prompt: `online_playbook_refresh_v1` (Trigger-Based)

Use only on triggers:
- plateau
- repeated rollbacks/failures
- bottleneck flip with low confidence
- high surrogate uncertainty

This prompt is similar to decision_support but focused on updating:
- `hypothesis_playbook`
- `pruning_guidance`
- `subspace_priors`

If implemented:
- Persist to `online/llm_playbook_refresh_step_<k>.json`
- Merge into playbook used for subsequent hypothesis generation

---

## 10) Post-Run: LLM Semantic Rule Distillation

### Task 10.1 — Add Post-Run Distill Call

Files:
- Update `src/agent/core.py` post-run path
- Update `src/agent/distill.py` or replace with a new `src/agent/distill_llm.py`

Pipeline:
1. Deterministically compute distillation inputs:
   - baseline/best, per-step deltas, variance if available
   - parameter diffs and simple marginal comparisons
2. Call LLM prompt `postrun_distill_rules_v2` to produce semantic rules.
3. Validate rule object schema + ensure evidence refs point to real step metrics.
4. Persist:
   - `postrun/rules_distilled.jsonl`
   - `postrun/distillation_report.md`
   - trace events `postrun.distill.rule`

Acceptance criteria:
- Rules include “when NOT to use” and risk notes.
- Each rule cites evidence refs like `metric:<step>:primary`.

### Task 10.2 — Post-Run Prompt: `postrun_distill_rules_v2`

SYSTEM prompt (canonical):
```text
You are the post-run distiller for a white-box CCL/NCCL tuning agent.

Task: produce portable, evidence-backed tuning rules.

Rules:
1) Output MUST be a single JSON object with a "rules" array; no markdown.
2) Every rule must cite evidence_refs using the provided evidence IDs.
3) Do not claim causality beyond the provided effect estimates. Use probabilistic language and confidence.
4) Keep rules minimal: prefer rules that change <= 4 parameters.
5) Include explicit limitations ("when NOT to use") and risk notes.
6) Do NOT output chain-of-thought. Provide brief structured claims with refs.
```

---

## 11) TUI Updates (Make “Agentic LLM” Visible)

### Task 11.1 — Step View Should Show LLM Advice

Files:
- Update `src/tui/app.py`

Requirements:
- For each step, show:
  - whether an online decision-support call was scheduled
  - whether it was used in selection
  - a summary of interpretation + recommended_action + top hypothesis IDs
- Link to the full `Prompt Pack` JSON (`llm/call_<id>.json`)

Acceptance criteria:
- A user can select step `k` and see:
  - `step_k_context_pack.json`
  - `step_k_llm_decision_support.json`
  - `step_k_decision_record.json`

---

## 12) Tests & Validation

### Task 12.1 — Add Schema Validators

Files to add:
- `src/llm/schemas.py` (JSON schema-like validation helpers, lightweight)
- `tests/test_llm_schemas.py`

Validate:
- offline strategic plan output
- warm start program
- online decision support output
- postrun rules output

Acceptance criteria:
- Tests fail if:
  - unknown param keys are suggested
  - patch size exceeds limits
  - required fields missing

### Task 12.2 — Add Deterministic Merge Tests

Files:
- `tests/test_online_advisor_merge.py`

Cases:
- LLM advice arrives before soft wait: used
- advice arrives after soft wait: persisted but not used for that step
- invalid JSON: ignored with parse_errors recorded
- unsafe patch: dropped with reason recorded

### Task 12.3 — Concurrency Stress Test (Trace Writer)

Files:
- `tests/test_trace_thread_safety.py`

Goal:
- multiple threads emitting trace events should not corrupt JSONL.

---

## 13) Implementation Pseudocode (Core Loops)

### Warmstart runner
```text
offline_plan = planner.build_initial_plan(...)
strategic = planner.llm_offline_strategic_plan(...)
write offline artifacts (warm_start_program, playbook, pruning, priors)

if warm_start_program.mode == "series":
  candidates = build_candidate_configs(warm_start_program)
  candidates = validate_and_risk_filter(candidates)
  results = executor.run_batch(candidates, eval_mode="short", concurrency=warmstart_concurrency)
  winner = select_min_time(results, tie_break=[risk, uncertainty])
  baseline = winner.config
else:
  baseline = apply_patch(defaults, strategic.baseline_patch)
```

### Online step with async LLM
```text
for step in range(max_steps):
  metrics = executor.run(workload, current_config, step)
  derive metrics + bottleneck
  write step context pack

  advisor.request(step, context_pack, playbook, param_specs, recent_history)
  deterministic_portfolio = hypothesis_generator.propose_portfolio(...)
  deterministic_numeric = numeric_manager.prepare(...)

  advice = advisor.try_get(step, soft_wait_s)
  if advice:
    merge advice hypotheses into portfolio (validate + risk-filter)
    apply numeric_guidance to numeric_manager

  action = choose(hypothesis vs numeric vs tool_request) deterministically
  persist decision record with refs (include llm:call_id if used)
```

---

## 14) Open Questions (Decide Before Coding)

1. Do warmstart probe evaluations get their own step numbering, or use a separate `warmstart/` artifact folder?
2. Should online decision-support call run:
   - before candidate generation (to guide it), or
   - in parallel with deterministic candidate gen and merge late?
3. Do we want online LLM to be “every step forever,” or “always schedule but only soft-wait for early steps + triggers”?

Recommended defaults:
- Always schedule every step.
- Soft-wait only for steps `< 5`, and on triggers thereafter.

