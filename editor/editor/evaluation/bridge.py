"""Runs extensions/dual-memory/eval-bridge.ts so the harness exercises the real
TypeScript Writer write path and retrieval rather than a Python copy."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

from . import REPO_ROOT

PLUGIN_DIR = REPO_ROOT / "extensions" / "dual-memory"


def run_bridge(command: str, payload: dict, neo4j_env: dict[str, str]) -> dict:
    tsx = PLUGIN_DIR / "node_modules" / ".bin" / "tsx"
    if not tsx.exists():
        raise RuntimeError(f"{tsx} not found; run `npm ci` in {PLUGIN_DIR}")
    with tempfile.TemporaryDirectory() as tmp:
        in_path = Path(tmp) / "in.json"
        out_path = Path(tmp) / "out.json"
        in_path.write_text(json.dumps(payload))
        proc = subprocess.run(
            [str(tsx), "eval-bridge.ts", command, str(in_path), str(out_path)],
            cwd=PLUGIN_DIR,
            env={**os.environ, **neo4j_env},
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"eval-bridge {command} failed:\n{proc.stderr[-2000:]}")
        return json.loads(out_path.read_text())
