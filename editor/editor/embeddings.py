"""Voyage AI embeddings client for the Editor.

Used for category centroid computation and entity embedding generation.
Calls are metered like LLMClient so the pipeline can report them per step.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, replace

import voyageai

log = logging.getLogger("editor.embeddings")


@dataclass
class EmbeddingUsage:
    calls: int = 0
    tokens: int = 0
    seconds: float = 0.0

    def snapshot(self) -> EmbeddingUsage:
        return replace(self)

    def since(self, earlier: EmbeddingUsage) -> EmbeddingUsage:
        return EmbeddingUsage(
            calls=self.calls - earlier.calls,
            tokens=self.tokens - earlier.tokens,
            seconds=self.seconds - earlier.seconds,
        )


class EmbeddingsClient:
    def __init__(self, api_key: str, model: str = "voyage-4-lite") -> None:
        # The SDK defaults to no retries and no timeout; a dropped connection
        # would otherwise hang a nightly run for minutes and then fail it.
        self._client = (
            voyageai.Client(api_key=api_key, max_retries=3, timeout=60) if api_key else None
        )
        self.model = model
        self.usage = EmbeddingUsage()

    def embed(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        start = time.monotonic()
        vectors, tokens = self._embed(texts)
        self.usage.calls += 1
        self.usage.tokens += tokens
        self.usage.seconds += time.monotonic() - start
        return vectors

    def _embed(self, texts: list[str]) -> tuple[list[list[float]], int]:
        """One Voyage call. Returns (vectors, tokens billed)."""
        if self._client is None:
            raise RuntimeError("EmbeddingsClient has no API key")
        result = self._client.embed(texts, model=self.model, input_type="document")
        return result.embeddings, result.total_tokens
