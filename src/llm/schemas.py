from __future__ import annotations

from typing import Any, Dict, List

from ..types import ParameterSpace


ALLOWED_TOOL_REQUESTS = {"none", "nccltest.short", "workload.short", "microbench.reduced"}
ALLOWED_ACTION_PREFERENCES = {"auto", "hypothesis", "numeric"}
ALLOWED_CONVERGENCE_DECISIONS = {"continue", "stop"}
REQUIRED_ONLINE_KEYS = ("hypotheses", "numeric_guidance", "tool_request", "action_preference", "convergence")


def validate_patch(patch: Any, parameter_space: ParameterSpace, max_keys: int = 6) -> List[str]:
    errors: List[str] = []
    if not isinstance(patch, dict):
        return ["patch_not_dict"]
    if len(patch) > max_keys:
        errors.append("patch_too_large")
    for key, value in patch.items():
        spec = parameter_space.specs.get(key)
        if spec is None:
            errors.append(f"unknown_param:{key}")
            continue
        if not spec.is_valid(value):
            errors.append(f"invalid_value:{key}={value}")
    return errors


def validate_offline_plan(plan: Any, parameter_space: ParameterSpace) -> List[str]:
    errors: List[str] = []
    if not isinstance(plan, dict):
        return ["plan_not_dict"]
    required = [
        "warm_start_program",
        "baseline_patch",
        "pruning_guidance",
        "subspace_priors",
        "hypothesis_playbook",
        "tool_triggers",
    ]
    for key in required:
        if key not in plan:
            errors.append(f"missing:{key}")
    errors.extend(validate_patch(plan.get("baseline_patch", {}), parameter_space, max_keys=6))

    warm = plan.get("warm_start_program", {})
    if isinstance(warm, dict):
        for cand in warm.get("candidates", []) if isinstance(warm.get("candidates", []), list) else []:
            errors.extend(validate_patch(cand.get("patch", {}), parameter_space, max_keys=4))
    for hyp in plan.get("hypothesis_playbook", []) if isinstance(plan.get("hypothesis_playbook", []), list) else []:
        errors.extend(validate_patch(hyp.get("patch_template", {}), parameter_space, max_keys=4))
    return errors


def validate_online_decision_support(output: Any, parameter_space: ParameterSpace) -> List[str]:
    errors: List[str] = []
    if not isinstance(output, dict):
        return ["output_not_dict"]
    for key in REQUIRED_ONLINE_KEYS:
        if key not in output:
            errors.append(f"missing:{key}")

    hypotheses = output.get("hypotheses")
    if hypotheses is not None and not isinstance(hypotheses, list):
        errors.append("invalid_type:hypotheses")
        hypotheses = []
    for hyp in hypotheses or []:
        errors.extend(validate_patch(hyp.get("patch", {}), parameter_space, max_keys=4))

    numeric_guidance = output.get("numeric_guidance")
    if numeric_guidance is not None and not isinstance(numeric_guidance, dict):
        errors.append("invalid_type:numeric_guidance")

    tool_request = output.get("tool_request")
    if tool_request is None:
        tool_request = {}
    elif not isinstance(tool_request, dict):
        errors.append("invalid_type:tool_request")
        tool_request = {}
    name = tool_request.get("name", "none")
    if name not in ALLOWED_TOOL_REQUESTS:
        errors.append("invalid_tool_request")

    action_preference = output.get("action_preference")
    if action_preference is not None and action_preference not in ALLOWED_ACTION_PREFERENCES:
        errors.append("invalid_action_preference")

    convergence = output.get("convergence")
    if convergence is None:
        convergence = {}
    elif not isinstance(convergence, dict):
        errors.append("invalid_type:convergence")
        convergence = {}
    if convergence:
        decision = convergence.get("decision")
        if decision not in ALLOWED_CONVERGENCE_DECISIONS:
            errors.append("invalid_convergence_decision")
        confidence = convergence.get("confidence")
        if confidence is not None:
            try:
                value = float(confidence)
                if value < 0.0 or value > 1.0:
                    errors.append("invalid_convergence_confidence")
            except (TypeError, ValueError):
                errors.append("invalid_convergence_confidence")
    return errors


def validate_postrun_rules(rules: Any, parameter_space: ParameterSpace) -> List[str]:
    errors: List[str] = []
    if not isinstance(rules, list):
        return ["rules_not_list"]
    for rule in rules:
        if not isinstance(rule, dict):
            errors.append("rule_not_dict")
            continue
        action = rule.get("action", {}) if isinstance(rule.get("action", {}), dict) else {}
        patch = action.get("set", {})
        errors.extend(validate_patch(patch, parameter_space, max_keys=4))
    return errors
