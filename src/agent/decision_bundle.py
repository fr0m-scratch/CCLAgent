from __future__ import annotations

from typing import Any, Dict, List


SCHEMA_VERSION = "2.0"


def build_decision_bundle(
    *,
    step: int,
    chosen_action: Dict[str, Any],
    why_selected: List[Any],
    why_rejected: List[Any],
    context_ref: str,
    constraints_snapshot: Dict[str, Any],
    rollback_plan: Dict[str, Any],
    refs_fallback: List[str],
    candidates_considered: List[Dict[str, Any]] | None = None,
    counterfactuals: List[Dict[str, Any]] | None = None,
    quality_flags: List[str] | None = None,
    call_chain: List[str] | None = None,
) -> Dict[str, Any]:
    normalized_refs = _normalize_refs(refs_fallback)
    chosen = {
        "kind": chosen_action.get("kind"),
        "rationale": chosen_action.get("rationale"),
        "call_chain": _normalize_refs(call_chain or normalized_refs),
    }
    candidates = _normalize_candidates(step, candidates_considered or [], normalized_refs)
    selected_claims = _normalize_claims(why_selected, normalized_refs)
    rejected_claims = _normalize_claims(why_rejected, normalized_refs)
    cfacts = _normalize_counterfactuals(
        step=step,
        counterfactuals=counterfactuals or [],
        candidates_considered=candidates,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "step": step,
        "context_ref": context_ref,
        "chosen_action": chosen,
        "candidates_considered": candidates,
        "why_selected": selected_claims,
        "why_rejected": rejected_claims,
        "counterfactuals": cfacts,
        "constraints_snapshot": constraints_snapshot or {},
        "rollback_plan": rollback_plan or {},
        "quality_flags": list(quality_flags or []),
    }


def validate_decision_bundle(payload: Any) -> List[str]:
    errors: List[str] = []
    if not isinstance(payload, dict):
        return ["bundle_not_dict"]
    required = (
        "schema_version",
        "step",
        "context_ref",
        "chosen_action",
        "candidates_considered",
        "why_selected",
        "why_rejected",
        "counterfactuals",
        "constraints_snapshot",
        "rollback_plan",
        "quality_flags",
    )
    for key in required:
        if key not in payload:
            errors.append(f"missing:{key}")
    if payload.get("schema_version") != SCHEMA_VERSION:
        errors.append("invalid_schema_version")

    chosen_action = payload.get("chosen_action")
    if not isinstance(chosen_action, dict):
        errors.append("invalid_type:chosen_action")
    else:
        call_chain = chosen_action.get("call_chain")
        if not isinstance(call_chain, list) or not call_chain:
            errors.append("invalid_call_chain")

    why_selected = payload.get("why_selected")
    if not isinstance(why_selected, list) or not why_selected:
        errors.append("invalid_why_selected")
    else:
        for idx, item in enumerate(why_selected):
            errors.extend(_validate_claim(item, key=f"why_selected[{idx}]"))

    why_rejected = payload.get("why_rejected")
    if not isinstance(why_rejected, list):
        errors.append("invalid_why_rejected")
    else:
        for idx, item in enumerate(why_rejected):
            errors.extend(_validate_claim(item, key=f"why_rejected[{idx}]"))

    candidates = payload.get("candidates_considered")
    if not isinstance(candidates, list):
        errors.append("invalid_candidates_considered")
    else:
        for idx, item in enumerate(candidates):
            if not isinstance(item, dict):
                errors.append(f"invalid_candidate:{idx}")
                continue
            if not isinstance(item.get("candidate_ref"), str) or not item.get("candidate_ref"):
                errors.append(f"missing_candidate_ref:{idx}")
            refs = item.get("refs")
            if not isinstance(refs, list) or not refs:
                errors.append(f"missing_candidate_refs:{idx}")

    counterfactuals = payload.get("counterfactuals")
    if not isinstance(counterfactuals, list):
        errors.append("invalid_counterfactuals")
    elif isinstance(candidates, list) and len(candidates) >= 3 and len(counterfactuals) < 2:
        errors.append("insufficient_counterfactuals")

    quality_flags = payload.get("quality_flags")
    if not isinstance(quality_flags, list):
        errors.append("invalid_quality_flags")

    return errors


def _normalize_refs(refs: Any) -> List[str]:
    if isinstance(refs, list):
        out = [str(item) for item in refs if str(item)]
        if out:
            return out
    return ["metric:unknown:primary"]


def _normalize_claims(claims: List[Any], refs_fallback: List[str]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for item in claims:
        if isinstance(item, dict):
            claim = str(item.get("claim", "")).strip()
            refs = _normalize_refs(item.get("refs") if item.get("refs") else refs_fallback)
            confidence = _safe_float(item.get("confidence"), default=0.7)
        else:
            claim = str(item).strip()
            refs = list(refs_fallback)
            confidence = 0.7
        if not claim:
            continue
        out.append(
            {
                "claim": claim,
                "refs": refs,
                "confidence": confidence,
            }
        )
    if out:
        return out
    return [
        {
            "claim": "no_explicit_claim_recorded",
            "refs": list(refs_fallback),
            "confidence": 0.5,
        }
    ]


def _normalize_candidates(step: int, candidates: List[Dict[str, Any]], refs_fallback: List[str]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for idx, item in enumerate(candidates):
        if not isinstance(item, dict):
            continue
        score = item.get("score_breakdown", {}) if isinstance(item.get("score_breakdown"), dict) else {}
        out.append(
            {
                "candidate_ref": str(item.get("candidate_ref") or f"candidate:{step}:{idx}"),
                "score_breakdown": {
                    "pred_time_ms": _safe_float(score.get("pred_time_ms")),
                    "uncertainty": _safe_float(score.get("uncertainty")),
                    "risk_score": _safe_float(score.get("risk_score")),
                    "feasibility": _safe_float(score.get("feasibility"), default=1.0),
                    "final_rank_score": _safe_float(score.get("final_rank_score")),
                },
                "status": str(item.get("status") or "rejected"),
                "reject_reason": str(item.get("reject_reason") or ""),
                "refs": _normalize_refs(item.get("refs") if item.get("refs") else refs_fallback),
            }
        )
    return out


def _normalize_counterfactuals(
    *,
    step: int,
    counterfactuals: List[Dict[str, Any]],
    candidates_considered: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for item in counterfactuals:
        if not isinstance(item, dict):
            continue
        out.append(
            {
                "candidate_ref": str(item.get("candidate_ref") or f"candidate:{step}:unknown"),
                "expected_delta_ms": _safe_float(item.get("expected_delta_ms")),
                "risk_delta": _safe_float(item.get("risk_delta")),
                "why_not": str(item.get("why_not") or "dominated"),
            }
        )
    if out:
        return out

    selected_pred = None
    selected_risk = None
    rejected: List[Dict[str, Any]] = []
    for item in candidates_considered:
        if item.get("status") == "selected":
            score = item.get("score_breakdown", {}) if isinstance(item.get("score_breakdown"), dict) else {}
            selected_pred = _safe_float(score.get("pred_time_ms"))
            selected_risk = _safe_float(score.get("risk_score"), default=0.0)
        else:
            rejected.append(item)
    for item in rejected[:2]:
        score = item.get("score_breakdown", {}) if isinstance(item.get("score_breakdown"), dict) else {}
        pred = _safe_float(score.get("pred_time_ms"))
        risk = _safe_float(score.get("risk_score"), default=0.0)
        expected_delta = None
        if pred is not None and selected_pred is not None:
            expected_delta = pred - selected_pred
        risk_delta = None
        if selected_risk is not None and risk is not None:
            risk_delta = risk - selected_risk
        out.append(
            {
                "candidate_ref": str(item.get("candidate_ref") or f"candidate:{step}:unknown"),
                "expected_delta_ms": expected_delta,
                "risk_delta": risk_delta,
                "why_not": str(item.get("reject_reason") or "dominated"),
            }
        )
    return out


def _validate_claim(item: Any, *, key: str) -> List[str]:
    errors: List[str] = []
    if not isinstance(item, dict):
        return [f"invalid_claim:{key}"]
    if not isinstance(item.get("claim"), str) or not item.get("claim"):
        errors.append(f"missing_claim:{key}")
    refs = item.get("refs")
    if not isinstance(refs, list) or not refs:
        errors.append(f"missing_refs:{key}")
    confidence = item.get("confidence")
    if confidence is not None:
        try:
            value = float(confidence)
            if value < 0.0 or value > 1.0:
                errors.append(f"invalid_confidence:{key}")
        except (TypeError, ValueError):
            errors.append(f"invalid_confidence:{key}")
    return errors


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default
