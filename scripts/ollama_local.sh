#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
OLLAMA_BIN="$ROOT/.tools/ollama/Ollama.app/Contents/Resources/ollama"
export OLLAMA_MODELS="$ROOT/.tools/ollama-models"
export OLLAMA_HOST="${OLLAMA_HOST:-127.0.0.1:11434}"

if [ ! -x "$OLLAMA_BIN" ]; then
  echo "Project-local Ollama binary not found: $OLLAMA_BIN" >&2
  exit 1
fi

exec "$OLLAMA_BIN" "$@"
