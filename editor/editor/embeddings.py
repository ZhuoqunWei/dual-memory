"""Voyage AI embeddings client for the Editor.

Used for category centroid computation and entity embedding generation.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import voyageai

if TYPE_CHECKING:
    from .config import EditorConfig

log = logging.getLogger("editor.embeddings")


class EmbeddingsClient:
    def __init__(self, api_key: str, model: str = "voyage-4-lite") -> None:
        self._client = voyageai.Client(api_key=api_key)
        self._model = model

    def embed(self, text: str) -> list[float]:
        result = self._client.embed([text], model=self._model, input_type="document")
        return result.embeddings[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        result = self._client.embed(texts, model=self._model, input_type="document")
        return result.embeddings
