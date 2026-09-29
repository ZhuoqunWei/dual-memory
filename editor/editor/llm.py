"""Anthropic Claude client for Editor reasoning tasks.

All LLM interactions go through ask_json() which returns parsed JSON.
Handles markdown fence stripping (same pattern as Writer in writer.ts).
Every call is metered (calls, tokens, wall time) so the pipeline can report
cost per step.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, replace
from typing import NamedTuple

import anthropic

from .constants import DEFAULT_LLM_EFFORT, DEFAULT_LLM_MODEL, LLM_PRICES_PER_MTOK

log = logging.getLogger("editor.llm")

JSON_ONLY = "\n\nIMPORTANT: Respond with ONLY valid JSON, no markdown fences."


class LLMReply(NamedTuple):
    text: str
    input_tokens: int
    output_tokens: int
    seconds: float


@dataclass
class LLMUsage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0

    def snapshot(self) -> LLMUsage:
        return replace(self)

    def since(self, earlier: LLMUsage) -> LLMUsage:
        return LLMUsage(
            calls=self.calls - earlier.calls,
            input_tokens=self.input_tokens - earlier.input_tokens,
            output_tokens=self.output_tokens - earlier.output_tokens,
            seconds=self.seconds - earlier.seconds,
        )


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float | None:
    """List-price cost of a token count, or None when the model isn't priced."""
    prices = LLM_PRICES_PER_MTOK.get(model)
    if prices is None:
        return None
    return (input_tokens * prices[0] + output_tokens * prices[1]) / 1_000_000


def request_options(model: str, effort: str) -> dict:
    """Per-model request knobs.

    Haiku 4.5 predates adaptive thinking and effort (both return 400 there) but
    still accepts temperature. Sonnet 5 and later reject non-default temperature
    and think adaptively; effort bounds how much.
    """
    if model.startswith("claude-haiku-4"):
        return {"temperature": 0.1}
    return {"thinking": {"type": "adaptive"}, "output_config": {"effort": effort}}


def parse_json(text: str) -> dict | list:
    """Parse a JSON reply, tolerating a surrounding markdown fence and prose
    after the JSON value (models sometimes append an explanation)."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        # Remove opening fence (```json or ```)
        first_newline = cleaned.find("\n")
        cleaned = cleaned[first_newline + 1 :] if first_newline != -1 else cleaned[3:]
    cleaned = cleaned.removesuffix("```").strip()
    value, _ = json.JSONDecoder().raw_decode(cleaned)
    return value


def verdicts_by_pair(reply: object, count: int, key: str = "pair") -> list[dict]:
    """Per-pair verdicts from a JSON reply, in pair order ({} where missing).

    Matches on the 1-based pair number rather than list position, and accepts
    the list wrapped in a one-key object ({"results": [...]}), which models
    sometimes return.
    """
    if isinstance(reply, dict) and len(reply) == 1:
        reply = next(iter(reply.values()))
    if not isinstance(reply, list):
        return [{} for _ in range(count)]
    by_pair = {v.get(key): v for v in reply if isinstance(v, dict)}
    return [by_pair.get(n, {}) for n in range(1, count + 1)]


class LLMClient:
    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_LLM_MODEL,
        effort: str = DEFAULT_LLM_EFFORT,
    ) -> None:
        self._client = anthropic.Anthropic(api_key=api_key) if api_key else None
        self.model = model
        self.effort = effort
        self.usage = LLMUsage()

    def ask_json(self, system: str, user: str) -> dict | list:
        """Send prompt, get parsed JSON response."""
        reply = self._complete(system + JSON_ONLY, user)
        self.usage.calls += 1
        self.usage.input_tokens += reply.input_tokens
        self.usage.output_tokens += reply.output_tokens
        self.usage.seconds += reply.seconds
        return parse_json(reply.text)

    def _complete(self, system: str, user: str) -> LLMReply:
        """One Messages API call."""
        if self._client is None:
            raise RuntimeError("LLMClient has no API key")
        start = time.monotonic()
        response = self._client.messages.create(
            model=self.model,
            max_tokens=16000,
            system=system,
            messages=[{"role": "user", "content": user}],
            **request_options(self.model, self.effort),
        )
        # With adaptive thinking the first block may be a thinking block.
        text = next((b.text for b in response.content if b.type == "text"), "")
        return LLMReply(
            text,
            response.usage.input_tokens,
            response.usage.output_tokens,
            time.monotonic() - start,
        )
