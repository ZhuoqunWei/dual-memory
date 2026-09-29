"""Turning the fixture into Writer input: embeddings and the bridge's load payload."""

from __future__ import annotations

from ..embeddings import EmbeddingsClient
from .fixture import Fixture

VECTOR_DIMS = 1024


def embed_all(client: EmbeddingsClient, texts: list[str], batch: int = 64) -> list[list[float]]:
    vectors: list[list[float]] = []
    for i in range(0, len(texts), batch):
        vectors.extend(client.embed_batch(texts[i : i + batch]))
    return vectors


def load_payload(fx: Fixture, fact_vec: dict[str, list[float]]) -> dict:
    return {
        "vectorDims": VECTOR_DIMS,
        "sessions": [
            {
                "key": s.id,
                "date": s.date,
                "summary": s.summary,
                "channel": s.channel,
                "facts": [
                    {
                        "content": f.content,
                        "kind": f.kind,
                        "confidence": f.confidence,
                        "salience": f.salience,
                        "sourceRef": f.id,
                        "sourceChannel": s.channel,
                        "entities": [
                            {
                                "name": name,
                                "type": fx.entities[name].type,
                                "aliases": fx.entities[name].aliases,
                                "role": role,
                            }
                            for name, role in f.entities
                        ],
                    }
                    for f in s.facts
                ],
                "embeddings": [fact_vec[f.id] for f in s.facts],
            }
            for s in fx.sessions
        ],
    }
