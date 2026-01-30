According to the ICSE 2026 **CCLInsight** design and the SC’25 **STELLAR** agentic tuning workflow, the “core” of an effective agentic tuner is not just *search*, but a **closed-loop system** that (1) grounds itself in domain knowledge, (2) runs tools/experiments, (3) interprets evidence, and (4) distills reusable rules. 

Below is the core design + what is genuinely novel about *our* CCL Agent.

---

## Core design of our CCL Agent (what the system *is*)

### Phase 1 — Offline planning (cheap “thinking” to save expensive GPU-hours)

**Goal:** pick a strong *initial* configuration plan before the real job runs.

1. **Run microbench / primitive-level profiling** to reduce the effective search space and identify which parameters matter most (instead of treating everything as a flat black box). CCLInsight does this by isolating parameters first, then doing deeper exploration only on the most important ones to control interaction explosion. 

2. **RAG knowledge (rules & insights)**: retrieve parameter definitions, constraints, and known heuristics so the agent proposes *valid* and *plausible* settings (mirrors why STELLAR uses RAG to avoid hallucinations and ground tuning decisions in manuals/specs). 

3. **Decide + evaluate initial config plan → apply initial config → launch workload** (your diagram’s “Decide/Evaluate Initial Config” loop).

**Why this matters:** you avoid spending the first N expensive online iterations “wandering” in bad regions.

---

### Phase 2 — Online tuning (evidence-driven adaptation during the real workload)

**Goal:** adapt to interference, real execution paths, and the true objective (iteration time), not just microbench bandwidth.

Core loop:

* **Collect online metrics** (training iteration timing + comm metrics).
* **Analyze metrics & plan next action**.
* Two complementary action modes:

  1. **Hypothesis-driven step (knowledge-guided):** propose a causal hypothesis (“bottleneck is channel count / chunking / algo path”), compile a config from that hypothesis, then apply it. For every hypothesis, the agent should use surrogated model trained if available (from Phase 3) to predict performance and carefully provide semantic knowledge guided hypotheses with justifications.
   
  2. **Numeric search step (budgeted):** do constrained, small-step exploration in the pruned subspace (only the parameters judged important), then apply the best candidate.

Then:

* **Analyze convergence** (stop when improvements plateau / risk increases).
* **End tuning → run remaining iterations with best config**.

This is distinct from AutoCCL’s design, which is an online tuner built around a Leader “Optimizer” and “Coordinator” broadcasting configurations, but still fundamentally an algorithmic search loop without semantic hypotheses, rule distillation, or multi-phase learning. 

---

### Phase 3 — Post-run updating (make today’s tuning make tomorrow cheaper)

**Goal:** turn one tuning episode into reusable knowledge.

* **Distill data → rules & insights** (human-readable “if topology/workload looks like X, try Y”). Prevent excessive raw trace storage by summarizing into rules. 
* **Train/update numeric/surrogate models** (predictor or importance model).
* Potentially **train the agent** (SFT/RL on traces + decisions) so the base agent improves over time (your diagram’s “RL Training / Supervised FT” box).
* Carefully engineered **persistent memory & knowledge base**, by deciding what to store (rules, models) and how to index it (by topology/scale/workload signatures) for future retrieval. The design of this memory is crucial: generic “tips” are less useful than *contextualized, evidence-backed* rules.

This “experience → rules → faster future tuning” is exactly the compounding advantage STELLAR highlights (accumulating rule sets to tune new apps faster), but we apply it to the *much noisier, more failure-prone* CCL setting. 

---

## Core novelty (what we can claim that isn’t “just use an LLM agent”)

### 1) **Hybrid “gray-box + agent” tuning (structured action space)**

Instead of letting an agent blindly flip dozens of knobs, we **bind the agent to gray-box signals** (primitive-level metrics / path awareness) to *justify pruning and action selection*. This mirrors CCLInsight’s key insight: isolate parameters, find importance, then only do deeper exploration where it matters. 

**Novelty:** agentic reasoning is constrained by *measured* primitive evidence, not vibes.

---

### 2) **Offline-to-online continuity (microbench informs live tuning, live tuning corrects microbench bias)**

AFNFA-style offline sampling trains a regressor from a small random sample of a huge space and then searches the surrogate. 
AutoCCL-style online tuning adapts in-situ, but without explicit knowledge grounding and without turning outcomes into reusable semantic rules. 

**Our novelty:** a *single* agent spans both worlds:

* offline: learn/prune/seed,
* online: adapt under interference and real objectives,
* post-run: distill + learn for transfer.

---

### 3) **Two-lane decision engine: “hypothesis steps” + “numeric steps”**

Most tuners are either:

* expert/white-box reasoning, or
* black-box numeric search.

We explicitly combine:

* **knowledge-guided hypothesis proposals** (fast, low-risk, interpretable), and
* **budgeted numeric search** in a pruned subspace (systematic, convergent).

That separation is a real architectural novelty because it makes the agent:

* safer (fewer reckless trials),
* cheaper (search only where it matters),
* more explainable (hypotheses produce reasons and rules).

---

### 4) **Operational safety + validity as first-class constraints**

CCL tuning isn’t “just optimize”; it’s “optimize without breaking jobs.”
Your system design bakes in:

* config compilation / validation steps,
* rollback/stop criteria via convergence analysis,
* bounded exploration.

This is crucial because CCL configurations can come from multiple sources (environment variables, source-code changes, or IR-level modifications), each with different validity constraints and costs. 

---

### 5) **Compounding knowledge (persistent memory) that is *CCL-specific***

STELLAR shows the value of accumulating tuning rules. 
Our novelty is tailoring that idea to CCL tuning where:

* scale/topology/message regime matter a lot,
* “best settings” can be non-transferable,
* compile-time and runtime knobs coexist.

So our memory isn’t generic “tips”—it’s indexed by context (topology/scale/workload signatures) and tied to evidence.

---

## A crisp “novelty statement” you can reuse in writing

**CCL Agent is a three-phase, budget-aware agentic tuner that grounds actions in primitive-level evidence (gray-box profiling), uses knowledge-guided hypotheses plus constrained numeric search for online adaptation, and distills post-run rules/models into persistent memory—enabling safer, cheaper, and more transferable CCL tuning than offline sampling (e.g., AFNFA) or purely algorithmic online tuning (e.g., AutoCCL).** 

If you want, I can turn this into: (1) an “Architecture” subsection (components + interfaces), and (2) a “Novelty vs. Prior Work” paragraph that directly contrasts AFNFA / AutoCCL / CCLInsight / STELLAR in 6–8 sentences.
