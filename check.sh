#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${ROOT_DIR}/editor/.venv/bin/python"

if [[ ! -x "$PYTHON_BIN" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  else
    PYTHON_BIN="python"
  fi
fi

echo "== TypeScript =="
cd "$ROOT_DIR/extensions/dual-memory"
npm run check

echo
echo "== Python compile =="
cd "$ROOT_DIR/editor"
"$PYTHON_BIN" -m compileall editor

echo
echo "== Python tests =="
"$PYTHON_BIN" -m pytest

echo
echo "== Python lint =="
"$PYTHON_BIN" -m ruff check editor tests
