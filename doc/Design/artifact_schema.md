Artifact Schema (v1)
====================

This document defines structured artifacts emitted by the agent. All artifacts
include a schema_version field and are JSON unless noted.

Context Pack
------------
Path: offline/context_pack.json and steps/step_<k>_context_pack.json

Required fields:
- schema_version: "1.0"
- phase: offline | online | postrun
- step: integer | null
- workload: object (WorkloadSpec snapshot)
- context_signature: object
- observations: object
- retrieval: object (memory + rag)
- models: object (surrogate status)
- constraints: object (sla + risk budget)

Prompt Pack
-----------
Path: llm/call_<id>.json

Required fields:
- schema_version: "1.0"
- call_id: string
- model: string
- system_prompt_version: string
- messages: array (role/content)
- injected_context_refs: array of evidence IDs
- token_estimates: object
- response: object (raw + parsed)
- validation_errors: array

Optional fields (implementation adds these for full observability):
- context_refs: array of evidence IDs (memory/microbench/RAG refs used to build context)
- context_window: object (sectioned prompt window, token budgets, truncation metadata)
- request_kwargs: object (temperature/max_tokens/options/etc passed to LLM)
- error: string (present if the LLM call failed)

WarmStart Program
-----------------
Path: offline/warm_start_program.json

Required fields:
- schema_version: "1.0"
- mode: "single" | "series"
- candidates: array of objects with:
  - id: string
  - summary: string
  - patch: object (param->value)
  - mechanism: string
  - risk_hint: string (low|medium|high)
  - evidence_refs: array of evidence IDs
  - eval_plan: object (mode/steps/timeout)
- selection_rule: object (objective + tie_breaker)

Optional fields:
- claims: array of objects (claim + refs)
- uncertainty: object (level + missing_evidence + safe_fallback)

Online Decision Support Output
------------------------------
Path: steps/step_<k>_llm_decision_support.json

Required fields:
- schema_version: "1.0"
- step: integer
- call_id: string
- used_in_decision: bool
- output: object (the parsed LLM output)
- parse_errors: array

Online LLM Advice (Late)
------------------------
Path: online/llm_advice_step_<k>.json

Required fields:
- schema_version: "1.0"
- step: integer
- call_id: string
- output: object (the parsed LLM output)
- parse_errors: array

Candidate Trace
---------------
Path: steps/step_<k>_candidates_trace.json

Required fields:
- schema_version: "1.0"
- step: integer
- candidates: array of objects with:
  - candidate_id
  - config
  - stages: object with stage status + reason

Decision Record
---------------
Path: steps/step_<k>_decision_record.json

Required fields:
- schema_version: "1.0"
- step: integer
- chosen_action: object
- candidates_considered: array
- why_selected: array of claims (with evidence refs)
- why_rejected: array of claims (with evidence refs)
- expected_outcome: object
- success_criteria: object
- rollback_plan: object

Stop Decision
-------------
Path: steps/step_<k>_stop_decision.json

Required fields:
- schema_version: "1.0"
- step: integer
- reason: string
- claims: array of objects (claim + refs)

Distillation Rule
-----------------
Path: postrun/rules_distilled.jsonl

Required fields per rule:
- schema_version: "1.0"
- rule_id: string
- context: object
- condition: object
- action: object
- effect: object (mean/var)
- evidence_refs: array
- risk: object

Surrogate Training Report
-------------------------
Path: postrun/surrogate_training_report.json

Required fields:
- schema_version: "1.0"
- model_id: string
- model_type: string
- dataset_stats: object
- validation: object
- drift: object
