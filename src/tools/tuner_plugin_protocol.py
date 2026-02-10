from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


PROTOCOL_VERSION = "2.0"


class ProtocolError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass
class ProtocolConfig:
    poll_interval_s: float = 0.2
    timeout_s: float | None = None


class FileTunerProtocol:
    def __init__(self, session_dir: str, config: Optional[ProtocolConfig] = None) -> None:
        self.session_dir = session_dir
        self.config = config or ProtocolConfig()
        Path(session_dir).mkdir(parents=True, exist_ok=True)

        # Legacy single-file paths (kept for backward compatibility).
        self.request_path = os.path.join(session_dir, "request.json")
        self.response_path = os.path.join(session_dir, "response.json")
        self.metrics_path = os.path.join(session_dir, "metrics.json")

        # Versioned protocol directories.
        self.requests_dir = os.path.join(session_dir, "requests")
        self.responses_dir = os.path.join(session_dir, "responses")
        self.metrics_dir = os.path.join(session_dir, "metrics")
        Path(self.requests_dir).mkdir(parents=True, exist_ok=True)
        Path(self.responses_dir).mkdir(parents=True, exist_ok=True)
        Path(self.metrics_dir).mkdir(parents=True, exist_ok=True)

    def wait_for_request(self, timeout_s: float | None = None) -> Dict[str, Any]:
        timeout = self.config.timeout_s if timeout_s is None else timeout_s
        deadline = None if timeout is None else time.time() + timeout
        while True:
            payload = self._try_read_and_delete_json(self.request_path)
            if payload is not None:
                return self._normalize_incoming(payload)

            payload = self._try_read_from_dir(self.requests_dir)
            if payload is not None:
                return self._normalize_incoming(payload)

            if deadline is not None and time.time() > deadline:
                raise ProtocolError("timeout", "timed out waiting for request")
            time.sleep(self.config.poll_interval_s)

    def wait_for_metrics(self, timeout_s: float | None = None) -> Dict[str, Any]:
        timeout = self.config.timeout_s if timeout_s is None else timeout_s
        deadline = None if timeout is None else time.time() + timeout
        while True:
            payload = self._try_read_and_delete_json(self.metrics_path)
            if payload is not None:
                return self._normalize_incoming(payload)

            payload = self._try_read_from_dir(self.metrics_dir)
            if payload is not None:
                return self._normalize_incoming(payload)

            if deadline is not None and time.time() > deadline:
                raise ProtocolError("timeout", "timed out waiting for metrics")
            time.sleep(self.config.poll_interval_s)

    def send_response(self, payload: Dict[str, Any], req_id: str | None = None) -> None:
        normalized = self._normalize_outgoing(payload, req_id=req_id)
        req_id_value = str(normalized.get("req_id"))
        response_path = os.path.join(self.responses_dir, f"{req_id_value}.json")
        self._atomic_write_json(response_path, normalized)
        # Legacy response path for old readers.
        self._atomic_write_json(self.response_path, normalized)

    def read_response(self, req_id: str | None = None, timeout_s: float | None = None) -> Dict[str, Any]:
        timeout = self.config.timeout_s if timeout_s is None else timeout_s
        deadline = None if timeout is None else time.time() + timeout
        target_req_id = str(req_id) if req_id else None
        while True:
            if target_req_id is not None:
                response_path = os.path.join(self.responses_dir, f"{target_req_id}.json")
                payload = self._try_read_and_delete_json(response_path)
                if payload is not None:
                    return self._normalize_incoming(payload)
            else:
                payload = self._try_read_and_delete_json(self.response_path)
                if payload is not None:
                    return self._normalize_incoming(payload)
                payload = self._try_read_from_dir(self.responses_dir)
                if payload is not None:
                    return self._normalize_incoming(payload)

            if deadline is not None and time.time() > deadline:
                raise ProtocolError("timeout", "timed out waiting for response")
            time.sleep(self.config.poll_interval_s)

    def send_request(self, payload: Dict[str, Any], req_id: str | None = None) -> str:
        normalized = self._normalize_outgoing(payload, req_id=req_id)
        req_id_value = str(normalized["req_id"])
        request_path = os.path.join(self.requests_dir, f"{req_id_value}.json")
        self._atomic_write_json(request_path, normalized)
        return req_id_value

    def send_metrics(self, payload: Dict[str, Any], req_id: str | None = None) -> str:
        normalized = self._normalize_outgoing(payload, req_id=req_id)
        req_id_value = str(normalized["req_id"])
        metrics_path = os.path.join(self.metrics_dir, f"{req_id_value}.json")
        self._atomic_write_json(metrics_path, normalized)
        return req_id_value

    def _normalize_outgoing(self, payload: Dict[str, Any], req_id: str | None = None) -> Dict[str, Any]:
        out = dict(payload or {})
        out.setdefault("protocol_version", PROTOCOL_VERSION)
        if req_id:
            out["req_id"] = str(req_id)
        out.setdefault("req_id", str(uuid.uuid4()))
        return out

    def _normalize_incoming(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(payload, dict):
            raise ProtocolError("bad_json", "payload is not an object")
        out = dict(payload)
        out.setdefault("protocol_version", "1.0")
        if "req_id" not in out:
            out["req_id"] = str(uuid.uuid4())
        return out

    def _atomic_write_json(self, path: str, payload: Dict[str, Any]) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temp_path = target.with_suffix(target.suffix + ".tmp")
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        os.replace(str(temp_path), str(target))

    def _try_read_and_delete_json(self, path: str) -> Dict[str, Any] | None:
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except json.JSONDecodeError as exc:
            try:
                os.remove(path)
            except OSError:
                pass
            raise ProtocolError("bad_json", f"invalid json: {exc}") from exc
        try:
            os.remove(path)
        except OSError:
            pass
        if not isinstance(payload, dict):
            raise ProtocolError("bad_json", "payload must be object")
        return payload

    def _try_read_from_dir(self, folder: str) -> Dict[str, Any] | None:
        try:
            entries = sorted(Path(folder).glob("*.json"), key=lambda p: p.stat().st_mtime)
        except OSError:
            entries = []
        for path in entries:
            payload = self._try_read_and_delete_json(str(path))
            if payload is not None:
                return payload
        return None
