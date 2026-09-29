"""Evaluation harness for the Writer → Editor → retrieval loop.

Loads a gold-labelled fixture (eval/fixture.json) through the real Writer write
path, runs the Editor pipeline, queries the real retrieval, and scores dedup,
contradiction handling, entity resolution, and retrieval against the labels.

LLM replies and embeddings are recorded to eval/cache/ so CI can replay a run
deterministically without API keys. See eval/README.md.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
EVAL_DIR = REPO_ROOT / "eval"
