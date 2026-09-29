"""Shared setup for the Neo4j integration tests (idempotency, rollback).

They run against the eval harness's scratch Neo4j and are skipped unless
EVAL_NEO4J_URI is set:

    docker compose -f eval/docker-compose.yml up -d
    EVAL_NEO4J_URI=bolt://localhost:7688 python -m pytest tests/test_idempotency.py tests/test_rollback.py

No API keys: fact embeddings replay from eval/cache, and the LLM and entity
embeddings are deterministic fakes.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import unittest
from collections import Counter

import numpy as np

from editor.db import EditorDB
from editor.embeddings import EmbeddingsClient
from editor.evaluation import EVAL_DIR
from editor.evaluation.bridge import run_bridge
from editor.evaluation.cache import CachedEmbeddingsClient, RecordingCache
from editor.evaluation.fixture import load_fixture
from editor.evaluation.graph import EvalGraph
from editor.evaluation.loading import embed_all, load_payload
from editor.llm import LLMClient, LLMReply

EVAL_URI = os.environ.get("EVAL_NEO4J_URI")
NEO4J_ENV = {
    "NEO4J_URI": EVAL_URI or "",
    "NEO4J_USER": os.environ.get("EVAL_NEO4J_USER", "neo4j"),
    "NEO4J_PASSWORD": os.environ.get("EVAL_NEO4J_PASSWORD", "dualmemory-eval"),
}

requires_neo4j = unittest.skipUnless(EVAL_URI, "set EVAL_NEO4J_URI to run Neo4j integration tests")


def _hash(text: str) -> int:
    return int(hashlib.sha256(text.encode()).hexdigest(), 16)


class FakeLLM(LLMClient):
    """Deterministic stand-in for the Editor's LLM.

    Each pair's verdict depends only on that pair's text, so asking about the
    same pair again gets the same answer, and a mix of verdicts exercises every
    mutation path.
    """

    def __init__(self) -> None:
        super().__init__("", "fake-llm", "low")

    def _complete(self, system: str, user: str) -> LLMReply:
        parts = re.split(r"\nPair (\d+):\n", user)
        if len(parts) > 1:
            items = [self._verdict(system, int(n), _hash(body)) for n, body in zip(parts[1::2], parts[2::2])]
            return LLMReply(json.dumps(items), 0, 0, 0.0)
        if "category name" in system:
            return LLMReply(json.dumps({"name": f"Cluster {_hash(user) % 1000}", "description": "test"}), 0, 0, 0.0)
        count = len(re.findall(r"^\d+\. ", user, re.M))
        return LLMReply(json.dumps(["fact"] * count), 0, 0, 0.0)

    @staticmethod
    def _verdict(system: str, n: int, h: int) -> dict:
        if "deduplication" in system:
            return {"pair": n, "decision": ["duplicate", "related", "update", "distinct"][h % 4],
                    "canonical": "AB"[h // 4 % 2], "relation": "supports", "weight": 0.6}
        if "both can be true" in system:
            return {"pair": n, "verdict": ["update", "compatible", "compatible"][h % 3],
                    "current": "AB"[h // 3 % 2], "attribute": "test"}
        if "same real-world entity" in system:
            return {"pair": n, "decision": ["same", "different", "unsure"][h % 3], "reason": "test"}
        if "entity pair" in system:
            return {"pair": n, "relation": ["friend", "none"][h % 2], "detail": "", "sentiment": "neutral",
                    "strength": 0.5, "direction": ["mutual", "a_to_b"][h // 2 % 2]}
        return {"pair": n, "type": ["supports", "none"][h % 2], "weight": 0.7}


class FakeEmbeddings(EmbeddingsClient):
    """Hash-seeded unit vectors: deterministic, and no recording needed."""

    def __init__(self) -> None:
        super().__init__("", "fake-embeddings")

    def _embed(self, texts: list[str]) -> tuple[list[list[float]], int]:
        vectors = []
        for t in texts:
            v = np.random.default_rng(_hash(t) % 2**32).standard_normal(1024)
            vectors.append((v / np.linalg.norm(v)).tolist())
        return vectors, 0


def reset_and_load(sessions: int) -> tuple[EvalGraph, dict]:
    """Wipe the scratch graph and write the first `sessions` fixture sessions
    through the real Writer path. Returns the graph and the load payload."""
    fx = load_fixture(EVAL_DIR / "fixture.json")
    fx.sessions = fx.sessions[:sessions]
    graph = EvalGraph(NEO4J_ENV["NEO4J_URI"], NEO4J_ENV["NEO4J_USER"], NEO4J_ENV["NEO4J_PASSWORD"])
    graph.reset()
    emb = CachedEmbeddingsClient(RecordingCache(EVAL_DIR / "cache" / "embeddings.jsonl", False), "", "voyage-4-lite")
    vectors = dict(zip([f.id for f in fx.facts], embed_all(emb, [f.content for f in fx.facts])))
    payload = load_payload(fx, vectors)
    run_bridge("load", payload, NEO4J_ENV)
    return graph, payload


def editor_db() -> EditorDB:
    return EditorDB(NEO4J_ENV["NEO4J_URI"], NEO4J_ENV["NEO4J_USER"], NEO4J_ENV["NEO4J_PASSWORD"])


# Derived or bookkeeping data that rollback deliberately leaves alone.
_SKIP_PROPS = {"embedding", "embeddingText"}
_SKIP_LABELS = {"EditAction", "EditorRun", "EvalSentinel"}


def _plain(value: object) -> object:
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, list):
        return tuple(_plain(v) for v in value)
    if value is None or isinstance(value, (str, int, bool)):
        return value
    return str(value)  # Neo4j temporals


def _props(props: dict) -> tuple:
    return tuple(sorted((k, _plain(v)) for k, v in props.items() if k not in _SKIP_PROPS))


def fingerprint(graph: EvalGraph) -> tuple[Counter, Counter]:
    """Every node and relationship (with properties) outside the audit trail,
    as multisets, so duplicates show up."""
    nodes = Counter()
    for r in graph._rows("MATCH (n) RETURN labels(n) AS labels, properties(n) AS props"):
        if _SKIP_LABELS & set(r["labels"]):
            continue
        nodes[(tuple(sorted(r["labels"])), _props(r["props"]))] += 1
    rels = Counter()
    for r in graph._rows("""
        MATCH (a)-[r]->(b)
        RETURN type(r) AS type, coalesce(a.id, a.name) AS a, coalesce(b.id, b.name) AS b,
               properties(r) AS props
    """):
        rels[(r["type"], r["a"], r["b"], _props(r["props"]))] += 1
    return nodes, rels


def duplicate_relationships(graph: EvalGraph) -> list[dict]:
    """Relationships repeated between the same two nodes (RELATES_TO counts
    per type)."""
    return graph._rows("""
        MATCH (a)-[r]->(b)
        WITH a, b, type(r) AS type, r.type AS subtype, count(r) AS n
        WHERE n > 1
        RETURN coalesce(a.id, a.name) AS a, coalesce(b.id, b.name) AS b, type, subtype, n
    """)
