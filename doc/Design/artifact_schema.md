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
