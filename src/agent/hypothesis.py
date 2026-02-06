from __future__ import annotations

import re
import uuid
from typing import Any, Dict, List, Optional

from ..memory import MemoryStore
from ..types import Hypothesis, InitialConfigPlan, NCCLConfig
from ..utils import setup_logger


logger = setup_logger("cclagent.hypothesis")


class HypothesisGenerator:
    def __init__(self, memory: MemoryStore, parameter_space: Any):
        self.memory = memory
        self.parameter_space = parameter_space

    def propose(
        self,
        plan: InitialConfigPlan,
        context,
        base_config: NCCLConfig,
        last_metrics: Any,
    ) -> Hypothesis:
        portfolio = self.propose_portfolio(plan, context, base_config, last_metrics, max_hypotheses=1)
        return portfolio[0]

    def propose_portfolio(
        self,
        plan: InitialConfigPlan,
        context,
        base_config: NCCLConfig,
        last_metrics: Any,
        max_hypotheses: int = 3,
    ) -> List[Hypothesis]:
        hypotheses: List[Hypothesis] = []
        if plan and getattr(plan, "hypothesis_playbook", None):
            for entry in plan.hypothesis_playbook:
                patch = entry.get("patch_template", {}) if isinstance(entry, dict) else {}
                if not patch:
                    continue
                summary = str(entry.get("summary", "")).strip()
                if not summary:
                    summary = self._patch_mechanism_summary(
                        patch=patch,
                        context=context,
                        last_metrics=last_metrics,
                        prefix="playbook hypothesis",
                    )
                hypotheses.append(
                    Hypothesis(
                        id=entry.get("id", str(uuid.uuid4())),
                        summary=summary,
                        patch=patch,
                        mechanism=entry.get("mechanism", "playbook"),
                        expected_effect=entry.get("expected_effect", {"iteration_time_ms": "decrease"}),
                        risk=entry.get("risk", {}).get("level", "low") if isinstance(entry.get("risk", {}), dict) else "low",
                        evidence={"refs": entry.get("evidence_refs", [])},
                        test_plan=entry.get("validation_plan", {}),
                    )
                )
        rules = self.memory.retrieve_rules(context, top_k=max_hypotheses)
        for rule in rules:
            claims = self._memory_rule_claims(rule=rule, context=context, last_metrics=last_metrics)
            if claims:
                summary = claims[0]
            else:
                summary = self._patch_mechanism_summary(
                    patch=rule.config_patch,
                    context=context,
                    last_metrics=last_metrics,
                    prefix="memory-guided hypothesis",
                )
            hypotheses.append(
                Hypothesis(
                    id=rule.id,
                    summary=summary,
                    patch=rule.config_patch,
                    mechanism="memory_rule",
                    expected_effect={"iteration_time_ms": "decrease"},
                    risk="low",
                    evidence={
                        "rule_id": rule.id,
                        "improvement": rule.improvement,
                        "confidence": getattr(rule, "confidence", None),
                        "success_rate": getattr(rule, "success_rate", None),
                        "claims": claims,
                        "refs": [f"rule:{rule.id}"],
                    },
                    test_plan={"metric": "iteration_time_ms", "direction": "decrease"},
                )
            )
        if len(hypotheses) < max_hypotheses:
            important = []
            if plan is not None:
                important = [ip.param for ip in plan.important_params] or list(plan.recommended_search_params or [])
            patch = self._mutate_single(base_config, important)
            summary = self._patch_mechanism_summary(
                patch=patch,
                context=context,
                last_metrics=last_metrics,
                prefix="single-parameter mutation",
            )
            hypotheses.append(
                Hypothesis(
                    id=str(uuid.uuid4()),
                    summary=summary,
                    patch=patch,
                    mechanism="mutation",
                    expected_effect={"iteration_time_ms": "decrease"},
                    risk="low",
                    evidence={"source": "heuristic"},
                    test_plan={"metric": "iteration_time_ms", "direction": "decrease"},
                )
            )
        return hypotheses[:max_hypotheses]

    def _mutate_single(self, base_config: NCCLConfig, focus: List[str]) -> Dict[str, Any]:
        if not focus:
            return {}
        param_name = focus[0]
        spec = self.parameter_space.specs.get(param_name)
        if spec is None:
            return {}
        current = base_config.params.get(param_name, spec.default)
        neighbors = spec.neighbors(current)
        if not neighbors:
            return {}
        return {param_name: neighbors[0]}

    def _patch_mechanism_summary(
        self,
        *,
        patch: Dict[str, Any],
        context: Any,
        last_metrics: Any,
        prefix: str,
    ) -> str:
        if not isinstance(patch, dict) or not patch:
            return f"{prefix}: no-op fallback due to missing valid patch"
        reasons: List[str] = []
        for key, value in patch.items():
            reasons.append(self._reason_for_param_change(key, value, context=context, last_metrics=last_metrics))
            if len(reasons) >= 2:
                break
        if not reasons:
            body = ", ".join(f"{k}={v}" for k, v in list(patch.items())[:3])
            return f"{prefix}: evaluate patch {{{body}}} for measurable iteration-time impact"
        return f"{prefix}: " + " | ".join(reasons)

    def _memory_rule_claims(self, *, rule: Any, context: Any, last_metrics: Any) -> List[str]:
        patch = rule.config_patch if isinstance(getattr(rule, "config_patch", {}), dict) else {}
        claims: List[str] = []
        for key, value in patch.items():
            claims.append(self._reason_for_param_change(key, value, context=context, last_metrics=last_metrics))
            if len(claims) >= 2:
                break
        improvement = getattr(rule, "improvement", None)
        if isinstance(improvement, (int, float)):
            claims.append(f"historical gain in similar context: {improvement * 100.0:.2f}% (rule score prior)")
        return claims

    def _reason_for_param_change(
        self,
        key: str,
        value: Any,
        *,
        context: Any,
        last_metrics: Any,
    ) -> str:
        total_gpus = self._infer_total_gpus(context)
        metric_hint = self._metric_hint_for_param(key, last_metrics)

        if key == "NCCL_MAX_NCHANNELS":
            scale_hint = f" at {total_gpus}-GPU scale" if total_gpus is not None else ""
            base = f"set NCCL_MAX_NCHANNELS={value} to increase channel parallelism{scale_hint} and avoid channel under-utilization"
            return f"{base}; {metric_hint}" if metric_hint else base
        if key == "NCCL_MIN_NCHANNELS":
            base = f"set NCCL_MIN_NCHANNELS={value} to enforce minimum parallel channels and reduce collective serialization"
            return f"{base}; {metric_hint}" if metric_hint else base
        if key == "NCCL_BUFFSIZE":
            mib = self._safe_float(value)
            if mib is not None:
                mib = mib / float(1 << 20)
            if mib is not None and mib < 4:
                base = f"increase NCCL_BUFFSIZE to {int(mib)} MiB-equivalent to reduce per-chunk launch overhead"
            elif mib is not None and mib > 24:
                base = f"cap NCCL_BUFFSIZE near {int(mib)} MiB to avoid oversized chunks hurting overlap"
            else:
                base = f"set NCCL_BUFFSIZE={value} to balance chunking overhead and bandwidth utilization"
            return f"{base}; {metric_hint}" if metric_hint else base
        if key == "NCCL_ALGO":
            if str(value).upper() == "TREE":
                base = "switch NCCL_ALGO=TREE to reduce collective latency on multi-node synchronization"
            elif str(value).upper() == "RING":
                base = "switch NCCL_ALGO=RING to maximize steady-state bandwidth on large payload collectives"
            else:
                base = f"switch NCCL_ALGO={value} to test topology-aware collective path efficiency"
            return f"{base}; {metric_hint}" if metric_hint else base
        if key == "NCCL_PROTO":
            proto = str(value).upper()
            if proto == "LL":
                base = "set NCCL_PROTO=LL to lower latency for small/medium message collectives"
            elif proto == "LL128":
                base = "set NCCL_PROTO=LL128 as a latency/bandwidth compromise for mixed message sizes"
            else:
                base = "set NCCL_PROTO=SIMPLE to favor throughput for large payload collectives"
            return f"{base}; {metric_hint}" if metric_hint else base
        if key == "NCCL_NTHREADS":
            base = f"set NCCL_NTHREADS={value} to tune per-channel compute occupancy without over-threading"
            return f"{base}; {metric_hint}" if metric_hint else base
        if key == "NCCL_SOCKET_NTHREADS":
            base = f"set NCCL_SOCKET_NTHREADS={value} to increase network progress parallelism per socket"
            return f"{base}; {metric_hint}" if metric_hint else base
        if key == "NCCL_NSOCKS_PERTHREAD":
            base = f"set NCCL_NSOCKS_PERTHREAD={value} to rebalance socket fan-out and host networking pressure"
            return f"{base}; {metric_hint}" if metric_hint else base
        if key == "NCCL_IB_QPS_PER_CONNECTION":
            base = f"set NCCL_IB_QPS_PER_CONNECTION={value} to tune IB queue-pair parallelism under cross-node traffic"
            return f"{base}; {metric_hint}" if metric_hint else base
        base = f"set {key}={value} to test targeted impact on communication bottlenecks"
        return f"{base}; {metric_hint}" if metric_hint else base

    def _metric_hint_for_param(self, key: str, last_metrics: Any) -> str:
        raw = getattr(last_metrics, "raw", None)
        if not isinstance(raw, dict):
            return ""
        details = raw.get("simulated_effects", {})
        if not isinstance(details, dict):
            return ""
        if key == "NCCL_MAX_NCHANNELS":
            observed = details.get("max_channels")
            if observed is not None:
                return f"last run observed max_channels={observed}"
        if key == "NCCL_BUFFSIZE":
            observed = details.get("buffsize")
            if observed is not None:
                return f"last run observed buffsize={observed}"
        if key == "NCCL_NTHREADS":
            observed = details.get("nthreads")
            if observed is not None:
                return f"last run observed nthreads={observed}"
        return ""

    def _infer_total_gpus(self, context: Any) -> Optional[int]:
        nodes = self._safe_int(self._ctx_value(context, "nodes"))
        gpus_per_node = self._safe_int(self._ctx_value(context, "gpus_per_node"))
        if isinstance(nodes, int) and isinstance(gpus_per_node, int) and nodes > 0 and gpus_per_node > 0:
            return nodes * gpus_per_node
        scale = self._ctx_value(context, "scale")
        if not isinstance(scale, str):
            return None
        match = re.search(r"(\d+)\s*-\s*gpu|(\d+)\s*gpu", scale.lower())
        if not match:
            return None
        value = match.group(1) or match.group(2)
        return self._safe_int(value)

    def _ctx_value(self, context: Any, key: str) -> Any:
        if isinstance(context, dict):
            return context.get(key)
        return getattr(context, key, None)

    def _safe_int(self, value: Any) -> Optional[int]:
        try:
            if value is None:
                return None
            return int(value)
        except (TypeError, ValueError):
            return None

    def _safe_float(self, value: Any) -> Optional[float]:
        try:
            if value is None:
                return None
            return float(value)
        except (TypeError, ValueError):
            return None
