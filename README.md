CCL Agent
=========

CCL Agent is an agentic, closed-loop tuner for NCCL communication. It couples gray-box profiling with knowledge-grounded reasoning and bounded search to produce safe, transferable configurations for real workloads.

Key idea
--------
Tuning is not just search. A practical tuner must:
- ground actions in domain knowledge and primitive-level evidence,
- run experiments and interpret metrics during the real workload,
- and distill post-run rules/models for reuse.

This repo implements that three-phase design.

Core novelty (one-paragraph version)
------------------------------------
CCL Agent is a three-phase, budget-aware agentic tuner that grounds actions in primitive-level evidence (gray-box profiling), uses knowledge-guided hypotheses plus constrained numeric search for online adaptation, and distills post-run rules/models into persistent memory, enabling safer, cheaper, and more transferable NCCL tuning than offline sampling or purely algorithmic online tuning.

System overview
---------------
The system is a closed-loop pipeline with explicit offline, online, and post-run stages.

Figures:
- `doc/figures/Design.png` - end-to-end agent loop and control flow
- `doc/figures/TuningSys.png` - tuning system components and interfaces

Phases
------
1) Offline planning (cheap thinking)
- Run microbenchmarks / primitive profiling to prune the search space.
- Retrieve rules, constraints, and heuristics (RAG) to propose valid configs.
- Produce an initial configuration plan before running the real workload.

2) Online tuning (evidence-driven adaptation)
- Collect iteration time + communication metrics.
- Plan and apply next actions using two complementary modes:
  - hypothesis steps: knowledge-guided, explainable changes
  - numeric steps: budgeted search in the pruned subspace (surrogate-guided search)
- Stop when improvements plateau and lock the best config.

3) Post-run updating (make today cheaper tomorrow)
- Distill traces into reusable rules and insights.
- Train/update surrogate or importance models.
- Store context-indexed knowledge for future retrieval.

What makes it different
-----------------------
- Gray-box + agentic tuning: decisions are constrained by measured primitive signals, not blind knob flipping.
- Offline-to-online continuity: microbench seeds online tuning; online evidence corrects microbench bias.
- Two-lane decision engine: hypothesis steps + numeric steps in a pruned subspace.
- Safety and validity first: config compilation/validation, bounded exploration, rollback criteria.
- Persistent, CCL-specific memory: rules tied to topology/scale/workload signatures.

Tech stack (design goals)
-------------------------
- LLM interface layer under `src/llm` with unified completion API (Ollama / OpenAI / Claude / Gemini / Fireworks).
- Agent-centric architecture (not a workflow with embedded LLM calls).
- Tooling interface for: memory/knowledge pool, microbench, workload launch, metric collection, SLA enforcement, config compilation/validation.
- Modular agent that can interact with NCCL and be used as an ext-net tuner (AutoCCL-style integration).

Default behavior
----------------
- The agent **always attempts an LLM call**, even in dry-run.
- Default provider/model: **Ollama** / `deepseek-r1:8b` (override with `--provider` / `--model` or config).

TUI dashboard
-------------
To browse artifacts/trace and inspect **exact LLM prompts + responses**:

```bash
python3 scripts/agent_tui.py --run-dir artifacts/<run_id>
```

Notes:
- If you don’t have the UI deps yet: `pip install textual rich`
- The dashboard’s interactive chat defaults to **Ollama** / `deepseek-r1:8b` (override with `--provider` / `--model`).

Repository layout
-----------------
- `doc/Design/coreNovelty.md` - novelty and phase design
- `doc/Design/techStack.md` - architecture constraints and goals
- `doc/Design/end_to_end_training_workflow.md` - call-by-call end-to-end training run walkthrough
- `doc/figures/` - design diagrams
- `src/` - implementation (LLM interfaces, agent, tools)
- `workload/` - workload specs (JSON) for tuning runs
- `tools/` - build and integration scripts

Workloads
---------
The `workload/` catalog contains JSON specs aligned with `WorkloadSpec`. The
`workload/autoccl` folder mirrors the AutoCCL evaluation workloads (Phi-2-2B,
Llama-3.1-8B, Yi-1.5-34B, VGG-19). Populate the `command` field to run for real.

Example:
```
python3 -m src.main --workload workload/autoccl/phi2-2b.json --dry-run
```

Status
------
This repository captures the design and scaffolding for the NCCL-focused CCL Agent. Implementation should follow the phase structure above and align with the interfaces described in the design docs.
