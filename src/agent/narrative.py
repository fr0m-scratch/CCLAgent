"""WP10a: Run narrative generation.

Auto-generates a human-readable run summary from structured artifacts:
context, hypothesis scorecard, attribution report, convergence evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .hypothesis_tracker import HypothesisScorecard
    from .attribution import AttributionReport
    from .convergence_argument import ConvergenceEvidence
    from .llm_influence import LLMInfluenceSummary
    from ..types import ContextSignature


@dataclass
class RunNarrative:
    """Structured narrative for a completed tuning run."""

    title: str = ""
    context_summary: str = ""
    performance_summary: str = ""
    hypothesis_summary: str = ""
    attribution_summary: str = ""
    convergence_summary: str = ""
    influence_summary: str = ""
    key_findings: List[str] = field(default_factory=list)
    full_text: str = ""

    def to_markdown(self) -> str:
        sections = [f"# {self.title}", ""]
        if self.context_summary:
            sections += ["## Context", self.context_summary, ""]
        if self.performance_summary:
            sections += ["## Performance", self.performance_summary, ""]
        if self.hypothesis_summary:
            sections += ["## Hypothesis Evaluation", self.hypothesis_summary, ""]
        if self.attribution_summary:
            sections += ["## Attribution", self.attribution_summary, ""]
        if self.influence_summary:
            sections += ["## LLM Influence", self.influence_summary, ""]
        if self.convergence_summary:
            sections += ["## Convergence", self.convergence_summary, ""]
        if self.key_findings:
            sections += ["## Key Findings"]
            for i, f in enumerate(self.key_findings, 1):
                sections.append(f"{i}. {f}")
            sections.append("")
        return "\n".join(sections)


def generate_narrative(
    context: Optional["ContextSignature"] = None,
    baseline_ms: float = 0.0,
    best_ms: float = 0.0,
    total_steps: int = 0,
    scorecard: Optional["HypothesisScorecard"] = None,
    attribution: Optional["AttributionReport"] = None,
    convergence: Optional["ConvergenceEvidence"] = None,
    influence: Optional["LLMInfluenceSummary"] = None,
) -> RunNarrative:
    """Generate a structured narrative from run artifacts."""
    n = RunNarrative()

    # Title
    improvement_pct = ((baseline_ms - best_ms) / baseline_ms * 100) if baseline_ms > 0 else 0.0
    workload = getattr(context, "workload", "unknown") if context else "unknown"
    n.title = f"CCLAgent Tuning Report: {workload}"

    # Context
    if context:
        parts = []
        if context.topology:
            parts.append(f"topology={context.topology}")
        if context.scale:
            parts.append(f"scale={context.scale}")
        if getattr(context, "network", None):
            parts.append(f"network={context.network}")
        if context.nodes:
            parts.append(f"nodes={context.nodes}")
        n.context_summary = f"Workload: {workload}. Configuration: {', '.join(parts)}."

    # Performance
    n.performance_summary = (
        f"Baseline: {baseline_ms:.2f}ms. Best: {best_ms:.2f}ms. "
        f"Improvement: {improvement_pct:.1f}% over {total_steps} steps."
    )

    # Hypotheses
    if scorecard and scorecard.total > 0:
        n.hypothesis_summary = (
            f"{scorecard.total} hypotheses tested: "
            f"{scorecard.confirmed} confirmed, "
            f"{scorecard.refuted} refuted, "
            f"{scorecard.inconclusive} inconclusive. "
            f"Confirmation rate: {scorecard.confirmation_rate:.0%}. "
            f"Average margin: {scorecard.avg_margin:+.2f}pp."
        )
        if scorecard.per_mechanism:
            mech_parts = []
            for mech, counts in scorecard.per_mechanism.items():
                mech_parts.append(
                    f"{mech}: {counts.get('confirmed',0)}/{sum(counts.values())} confirmed"
                )
            n.hypothesis_summary += " By mechanism: " + "; ".join(mech_parts) + "."

    # Attribution
    if attribution and attribution.param_attributions:
        top = attribution.param_attributions[:3]
        parts = [
            f"{a.param} ({a.attributed_delta_pct:+.1f}%)" for a in top
        ]
        n.attribution_summary = (
            f"Top contributing parameters: {', '.join(parts)}. "
            f"Consistency check: {attribution.consistency_check:.2f} "
            f"(1.0 = perfect additivity)."
        )
        if attribution.lane_attributions:
            lane_parts = [
                f"{la.lane}: {la.total_delta_ms:+.2f}ms ({la.step_count} steps)"
                for la in attribution.lane_attributions
            ]
            n.attribution_summary += f" By lane: {'; '.join(lane_parts)}."

    # Influence
    if influence and influence.total_steps > 0:
        n.influence_summary = (
            f"LLM advice available in {influence.advice_available_count}/{influence.total_steps} steps. "
            f"Influence rate: {influence.influence_rate:.0%}. "
            f"Override rate: {influence.override_rate:.0%}. "
            f"LLM-attributed gain: {influence.llm_attributed_gain_pct:.1f}% of total improvement."
        )

    # Convergence
    if convergence:
        reasons = [c["claim"] for c in convergence.claims]
        n.convergence_summary = (
            f"Converged: {convergence.converged}. "
            f"Effect size: {convergence.effect_size:.3f}. "
            f"Evidence: {'; '.join(reasons[:3])}."
        )

    # Key findings
    findings = []
    if improvement_pct > 0:
        findings.append(f"Achieved {improvement_pct:.1f}% improvement in {total_steps} steps.")
    if scorecard and scorecard.confirmation_rate > 0.5:
        findings.append(f"Hypothesis confirmation rate of {scorecard.confirmation_rate:.0%} indicates effective mechanism-based reasoning.")
    if influence and influence.llm_attributed_gain_pct > 20:
        findings.append(f"LLM contributed {influence.llm_attributed_gain_pct:.0f}% of total gain, validating advisor integration.")
    if attribution and attribution.param_attributions:
        top_param = attribution.param_attributions[0]
        findings.append(f"Most impactful parameter: {top_param.param} ({top_param.attributed_delta_pct:+.1f}%).")
    n.key_findings = findings

    n.full_text = n.to_markdown()
    return n
