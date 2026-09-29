"""Record/replay caches for LLM replies and embeddings.

Each cache is an append-only JSONL file keyed by a hash of the full request, so
a replayed run sends byte-identical prompts or fails loudly. Recording needs API
keys; replay (CI) needs none.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import threading
from pathlib import Path

import numpy as np

from ..embeddings import EmbeddingsClient
from ..llm import LLMClient, LLMReply

log = logging.getLogger("editor.evaluation.cache")


class CacheMiss(BaseException):
    """A replayed request has no recording.

    Derives from BaseException on purpose: Editor steps catch Exception and
    carry on, which would turn a stale cache into a silently skipped step.
    """


class RecordingFailed(BaseException):
    """An API call failed while recording; aborting keeps the cache complete."""


def request_key(*parts: str) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(part.encode())
        h.update(b"\0")
    return h.hexdigest()


class RecordingCache:
    """Append-only JSONL store: one {"key": ..., ...} object per line."""

    def __init__(self, path: Path, record: bool) -> None:
        self.path = path
        self.record = record
        self._entries: dict[str, dict] = {}
        self._used: list[str] = []
        self._lock = threading.Lock()  # safe to share across threads
        if path.exists():
            for line in path.read_text().splitlines():
                if line.strip():
                    entry = json.loads(line)
                    self._entries[entry["key"]] = entry

    def get(self, key: str) -> dict | None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None and key not in self._used:
                self._used.append(key)
            return entry

    def put(self, entry: dict) -> None:
        with self._lock:
            self._entries[entry["key"]] = entry
            self._used.append(entry["key"])
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def prune(self) -> int:
        """Rewrite the file with only the entries used this run. Returns removed count."""
        kept = [self._entries[k] for k in self._used]
        removed = len(self._entries) - len(kept)
        self.path.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in kept))
        self._entries = {e["key"]: e for e in kept}
        return removed

    @property
    def size(self) -> int:
        return len(self._entries)


class CachedLLMClient(LLMClient):
    def __init__(self, cache: RecordingCache, api_key: str, model: str, effort: str) -> None:
        super().__init__(api_key if cache.record else "", model, effort)
        self._cache = cache

    def _complete(self, system: str, user: str) -> LLMReply:
        key = request_key("llm", self.model, self.effort, system, user)
        entry = self._cache.get(key)
        if entry is None:
            if not self._cache.record:
                raise CacheMiss(
                    f"no recorded LLM reply for prompt {user[:80]!r}. The Editor's "
                    "prompts changed; re-record with `python -m editor.evaluation --record`."
                )
            try:
                reply = super()._complete(system, user)
            except Exception as e:
                raise RecordingFailed(f"LLM call failed while recording: {e}") from e
            entry = {
                "key": key,
                "model": self.model,
                "prompt": user[:120],
                "text": reply.text,
                "input_tokens": reply.input_tokens,
                "output_tokens": reply.output_tokens,
                "seconds": round(reply.seconds, 3),
            }
            self._cache.put(entry)
        # Replays report the recorded API latency, so LLM time stays meaningful.
        return LLMReply(entry["text"], entry["input_tokens"], entry["output_tokens"], entry["seconds"])


def encode_vector(vec: list[float]) -> str:
    return base64.b64encode(np.asarray(vec, dtype="<f4").tobytes()).decode()


def decode_vector(data: str) -> list[float]:
    return np.frombuffer(base64.b64decode(data), dtype="<f4").astype(float).tolist()


class CachedEmbeddingsClient(EmbeddingsClient):
    """Per-text cache, so batches can be regrouped without re-recording."""

    def __init__(self, cache: RecordingCache, api_key: str, model: str) -> None:
        super().__init__(api_key if cache.record else "", model)
        self._cache = cache

    def _embed(self, texts: list[str]) -> tuple[list[list[float]], int]:
        keys = [request_key("embed", self.model, "document", t) for t in texts]
        missing = [(k, t) for k, t in zip(keys, texts) if self._cache.get(k) is None]
        if missing:
            if not self._cache.record:
                raise CacheMiss(
                    f"no recorded embedding for {missing[0][1][:80]!r}; re-record with "
                    "`python -m editor.evaluation --record`."
                )
            try:
                vectors, tokens = super()._embed([t for _, t in missing])
            except Exception as e:
                raise RecordingFailed(f"embedding call failed while recording: {e}") from e
            total_chars = sum(len(t) for _, t in missing) or 1
            for (key, text), vec in zip(missing, vectors):
                self._cache.put({
                    "key": key,
                    "model": self.model,
                    "text": text[:120],
                    "tokens": round(tokens * len(text) / total_chars),
                    "vector": encode_vector(vec),
                })
        entries = [self._cache.get(k) for k in keys]
        return (
            [decode_vector(e["vector"]) for e in entries],
            sum(e["tokens"] for e in entries),
        )
