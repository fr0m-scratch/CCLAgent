#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config import default_agent_config
from src.models.training import export_dataset, train_surrogate_model
from src.types import ContextSignature


def main() -> None:
    parser = argparse.ArgumentParser(description="Train surrogate model from memory")
    parser.add_argument("--memory", default="memory/agent_memory.json")
    parser.add_argument("--context", default="{}", help="JSON for context signature")
    parser.add_argument("--out", default="memory/models/surrogate_manual.pkl")
    args = parser.parse_args()

    payload = json.loads(Path(args.memory).read_text(encoding="utf-8"))
    records = payload.get("surrogates", [])
    context_payload = json.loads(args.context)
    context = ContextSignature(**context_payload) if context_payload else ContextSignature(
        workload="unknown", workload_kind="workload", topology="unknown", scale="unknown", nodes=1
    )
    cfg = default_agent_config()
    export_dataset(records, "memory/datasets/manual.jsonl")
    train_surrogate_model(records, context, cfg.parameter_space, cfg.surrogate, args.out)
    print(f"Saved model to {args.out}")


if __name__ == "__main__":
    main()
