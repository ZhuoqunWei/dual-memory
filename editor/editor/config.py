"""Editor configuration from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

from .constants import (
    BATCH_CAP,
    CATEGORY_COSINE,
    CATEGORY_QUORUM,
    CONFIDENCE_DECAY,
    DEFAULT_LLM_EFFORT,
    DEFAULT_LLM_MODEL,
    EXACT_DEDUP_COSINE,
    SEMANTIC_DEDUP_COSINE,
)


@dataclass(frozen=True)
class EditorConfig:
    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "dualmemory2026"

    # LLM (Editor reasoning)
    anthropic_api_key: str = ""
    llm_model: str = DEFAULT_LLM_MODEL
    llm_effort: str = DEFAULT_LLM_EFFORT

    # Embeddings
    voyage_api_key: str = ""
    embedding_model: str = "voyage-4-lite"
    vector_dims: int = 1024

    # Scheduling
    nightly_time: str = "02:00"
    check_interval_minutes: int = 5
    raw_threshold: int = 40
    batch_cap: int = BATCH_CAP

    # Thresholds
    exact_dedup_cosine: float = EXACT_DEDUP_COSINE
    semantic_dedup_cosine: float = SEMANTIC_DEDUP_COSINE
    category_cosine: float = CATEGORY_COSINE
    category_quorum: int = CATEGORY_QUORUM
    confidence_decay: float = CONFIDENCE_DECAY

    @classmethod
    def from_env(cls) -> EditorConfig:
        load_dotenv()
        return cls(
            neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
            neo4j_password=os.getenv("NEO4J_PASSWORD", "dualmemory2026"),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
            llm_model=os.getenv("EDITOR_LLM_MODEL", DEFAULT_LLM_MODEL),
            llm_effort=os.getenv("EDITOR_LLM_EFFORT", DEFAULT_LLM_EFFORT),
            voyage_api_key=os.getenv("VOYAGE_API_KEY", ""),
            embedding_model=os.getenv("EMBEDDING_MODEL", "voyage-4-lite"),
            batch_cap=int(os.getenv("EDITOR_BATCH_CAP", str(BATCH_CAP))),
            raw_threshold=int(os.getenv("EDITOR_RAW_THRESHOLD", "40")),
        )
