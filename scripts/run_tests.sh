#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

PYTHON=${PYTHON:-python3}

$PYTHON -m py_compile \
  src/utils/env.py \
  src/main.py \
  src/types.py \
  src/agent/core.py \
  src/agent/analyzer.py \
  src/agent/numeric.py \
  src/models/surrogate.py \
  src/models/training.py \
  src/data/export.py \
  src/safety/risk.py

echo "py_compile OK"

if command -v pytest >/dev/null 2>&1; then
  pytest -q
else
  echo "pytest not found; skipping unit tests"
fi

# Lightweight dry-run smoke (no real workloads)
OLLAMA_MODEL=${OLLAMA_MODEL:-deepseek-r1:8b}
$PYTHON -m src.main --dry-run --provider ollama --model "$OLLAMA_MODEL" >/dev/null

echo "dry-run OK"
