from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class ProtocolConfig:
    poll_interval_s: float = 0.5


class FileTunerProtocol:
    def __init__(self, session_dir: str, config: Optional[ProtocolConfig] = None) -> None:
        self.session_dir = session_dir
        self.config = config or ProtocolConfig()
        os.makedirs(session_dir, exist_ok=True)
        self.request_path = os.path.join(session_dir, "request.json")
        self.response_path = os.path.join(session_dir, "response.json")
        self.metrics_path = os.path.join(session_dir, "metrics.json")

    def wait_for_request(self) -> Dict[str, Any]:
        while True:
            if os.path.exists(self.request_path):
                with open(self.request_path, "r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                os.remove(self.request_path)
                return payload
            time.sleep(self.config.poll_interval_s)

    def wait_for_metrics(self) -> Dict[str, Any]:
        while True:
            if os.path.exists(self.metrics_path):
                with open(self.metrics_path, "r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                os.remove(self.metrics_path)
                return payload
            time.sleep(self.config.poll_interval_s)

    def send_response(self, payload: Dict[str, Any]) -> None:
        with open(self.response_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)

    def read_response(self) -> Dict[str, Any]:
        while True:
            if os.path.exists(self.response_path):
                with open(self.response_path, "r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                os.remove(self.response_path)
                return payload
            time.sleep(self.config.poll_interval_s)
