from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from ..agent.decision_engine import DecisionEngine, TunerContext, TunerResponse
from ..tools.tuner_plugin_protocol import FileTunerProtocol, ProtocolError


@dataclass
class TunerServerStats:
    handled: int = 0
    errors: int = 0


class TunerServer:
    def __init__(self, *, protocol: FileTunerProtocol, decision_engine: Optional[DecisionEngine] = None) -> None:
        self.protocol = protocol
        self.decision_engine = decision_engine or DecisionEngine()
        self.stats = TunerServerStats()

    def process_once(self, timeout_s: float | None = None) -> bool:
        try:
            request = self.protocol.wait_for_request(timeout_s=timeout_s)
        except ProtocolError as exc:
            if exc.code == "timeout":
                return False
            self.stats.errors += 1
            return False

        req_id = str(request.get("req_id") or "")
        response = self._handle_request(request)
        self.protocol.send_response(response.to_dict(), req_id=req_id or response.req_id)
        self.stats.handled += 1
        if response.status != "ok":
            self.stats.errors += 1
        return True

    def _handle_request(self, request: Dict[str, Any]) -> TunerResponse:
        req_id = str(request.get("req_id") or "")
        req_type = str(request.get("type") or "GET_TUNING_DECISION")
        if req_type not in ("GET_TUNING_DECISION", "GET_CONFIG"):
            return TunerResponse(
                req_id=req_id,
                status="error",
                override={},
                source="server",
                reasons=[f"bad_request_type:{req_type}"],
            )

        try:
            coll_type = str(request["coll_type"])
            size_bytes = int(request["bytes"])
            nranks = int(request["nranks"])
        except Exception:
            return TunerResponse(
                req_id=req_id,
                status="error",
                override={},
                source="server",
                reasons=["bad_request:missing_required_fields"],
            )

        ctx = TunerContext(
            req_id=req_id,
            coll_type=coll_type,
            bytes=size_bytes,
            nranks=nranks,
            topo_sig=str(request.get("topo_sig") or "unknown"),
            comm_hash=str(request.get("comm_hash") or ""),
            constraints=request.get("constraints") if isinstance(request.get("constraints"), dict) else {},
        )
        deadline_ms = request.get("deadline_ms")
        if deadline_ms is not None:
            try:
                ctx.constraints["deadline_ms"] = int(deadline_ms)
            except (TypeError, ValueError):
                pass
        return self.decision_engine.decide_for_collective(ctx)
