#!/usr/bin/env bash
set -euo pipefail

PORT=${PORT:-7681}
HOST=${HOST:-127.0.0.1}
MODEL=${MODEL:-deepseek-r1:8b}
PROVIDER=${PROVIDER:-ollama}

# Web TUI via ttyd
exec ttyd -W -p "$PORT" -i "$HOST" -t titleFixed="CCLAgent TUI" -t cursorBlink=true \
  python3 -m scripts.agent_tui --provider "$PROVIDER" --model "$MODEL"
