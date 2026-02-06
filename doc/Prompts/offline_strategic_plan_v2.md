# Prompt: offline_strategic_plan_v2

## Purpose
Produce a safe, evidence-grounded **offline strategic plan** that includes:
- WarmStartProgram (single or short probe series)
- Baseline patch
- Pruning guidance
- Subspace priors
- Hypothesis playbook
- Tool triggers

## Output Schema (JSON)
Required top-level keys:
- `schema_version`: "1.0"
- `warm_start_program`
- `baseline_patch`
- `pruning_guidance`
- `subspace_priors`
- `hypothesis_playbook`
- `tool_triggers`
- `claims`
- `uncertainty`

### `warm_start_program`
```json
{
  "mode": "single|series",
  "candidates": [
    {
      "id": "WS0",
      "summary": "string",
      "patch": { "NCCL_*": "..." },
      "mechanism": "string",
      "risk_hint": "low|medium|high",
      "evidence_refs": ["microbench:...", "rule:...", "rag:..."],
      "eval_plan": { "mode": "short", "steps": 50, "timeout_sec": 300 }
    }
  ],
  "selection_rule": {
    "objective": "min_iteration_time_ms",
    "tie_breaker": "lower_risk|lower_uncertainty"
  }
}
```

### `baseline_patch`
- Object of <= 6 params (param->value).

### `pruning_guidance`
Array of:
```json
{ "param": "NCCL_*", "action": "freeze_default|freeze_value|keep_free", "value": "optional", "reason": "string", "refs": ["..."] }
```

### `subspace_priors`
Array of:
```json
{ "name": "string", "fixed": { "NCCL_*": "..." }, "free": ["NCCL_*"], "reason": "string", "refs": ["..."] }
```

### `hypothesis_playbook`
Array of:
```json
{
  "id": "H_*",
  "trigger": { "bottleneck_class": "latency_bound|bandwidth_bound|unknown", "min_confidence": 0.0 },
  "summary": "string",
  "mechanism": "string",
  "patch_template": { "NCCL_*": "..." },
  "expected_effect": { "iteration_time_ms": { "direction": "decrease", "magnitude": "small|medium|large" } },
  "risk": { "level": "low|medium|high", "notes": "string" },
  "evidence_refs": ["rule:...", "rag:...", "microbench:..."],
  "validation_plan": { "primary_metric": "iteration_time_ms", "success_criteria": "string" }
}
```

### `tool_triggers`
Array of:
```json
{ "when": "string", "tool_request": "nccltest.short|microbench.reduced|enable_profile|none", "reason": "string", "refs": ["..."] }
```

### `claims`
Array of `{ "claim": "string", "refs": ["..."] }`

### `uncertainty`
```json
{ "level": "low|medium|high", "missing_evidence": ["string"], "safe_fallback": "string" }
```

## SYSTEM Prompt (canonical)
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

## USER Template Sections (ContextWindow)
- WORKLOAD
- CONTEXT_SIGNATURE
- PARAM_SPECS
- MICROBENCH_SIGNALS
- IMPORTANT_PARAMS
- MEMORY_RULES
- RAG_SNIPPETS
- SAFETY_ENVELOPE
- CURRENT_DEFAULTS

## Validation Rules
- All param keys must appear in PARAM_SPECS.
- No more than 6 keys in baseline_patch.
- Each hypothesis patch_template <= 4 keys.
- Each warm_start_program candidate patch <= 4 keys.
- All nontrivial claims must have evidence refs.
