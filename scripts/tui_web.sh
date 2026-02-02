#!/usr/bin/env bash
set -euo pipefail

PORT=${PORT:-7681}
HOST=${HOST:-127.0.0.1}
MODEL=${MODEL:-qwen3:0.6b}
PROVIDER=${PROVIDER:-openai}

# Web TUI via ttyd
exec ttyd -W -p "$PORT" -i "$HOST" -t titleFixed="CCLAgent TUI" -t cursorBlink=true \
  python3 -m scripts.agent_tui --provider "$PROVIDER" --model "$MODEL"
