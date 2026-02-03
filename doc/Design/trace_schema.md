Trace Schema (v1)
=================

This document defines the Trace Bus JSONL event schema used by CCL Agent.
The trace is the source of truth for all explainability views in the TUI.

Schema version
--------------
- schema_version: "1.0"

Event storage
-------------
- Path: artifacts/<run_id>/trace/events.jsonl
- Format: one JSON object per line (append-only)

TraceEvent (common fields)
--------------------------
Required fields:
- schema_version: string
- ts: float (unix seconds)
- run_id: string
- phase: string (offline | online | postrun | system)
- step: integer | null
- actor: string (agent | tool | llm | system)
- type: string (event type)
- payload: object
- refs: array of evidence IDs (may be empty)
- status: string (ok | error | warn)
- duration_ms: number | null

Optional fields:
- error: string
- tags: array of strings

Evidence IDs
------------
Evidence IDs are stable references to artifacts and objects.
Format (prefix:value):
- metric:<step>:primary
- metric:<step>:derived:<name>
- rule:<uuid>
- rag:<doc_id>:<chunk_id>
- microbench:<signal_id>
- surrogate:<model_id>
- candidate:<step>:<candidate_id>
- tool:<step>:<call_id>
- log:<step>:stdout|stderr
- llm:call_<id>

Event types
-----------
Core (system):
- run.start
- run.end

Retrieval:
- retrieval.memory
- retrieval.rag

Offline:
- offline.context.detect
- offline.microbench.plan
- offline.microbench.run
- offline.microbench.result
- decision.offline_warm_start
- search.prune

Online:
- proposal.hypothesis
- proposal.numeric_candidates
- model.surrogate.predict
- analysis.metrics.derive
- analysis.bottleneck.classify
- decision.select_action
- safety.risk_score
- safety.rollback
- stop.decision

Tools:
- tool.call
- tool.result

Post-run:
- postrun.distill.rule
- postrun.train.surrogate

Example event
-------------
{
  "schema_version": "1.0",
  "ts": 1738500000.123,
  "run_id": "<uuid>",
  "phase": "online",
  "step": 4,
  "actor": "agent",
  "type": "decision.select_action",
  "payload": {
    "action": "hypothesis",
    "summary": "apply memory rule",
    "risk_score": 0.22
  },
  "refs": ["rule:abcd", "metric:3:primary"],
  "status": "ok",
  "duration_ms": 3.2
}
