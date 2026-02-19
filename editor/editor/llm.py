"""Anthropic Claude client for Editor reasoning tasks.

All LLM interactions go through ask_json() which returns parsed JSON.
Handles markdown fence stripping (same pattern as Writer in writer.ts).
"""

from __future__ import annotations

import json
import logging

import anthropic

log = logging.getLogger("editor.llm")


class LLMClient:
    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514") -> None:
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model

    def ask_json(self, system: str, user: str) -> dict | list:
        """Send prompt, get parsed JSON response."""
        response = self._client.messages.create(
            model=self._model,
            max_tokens=4096,
            temperature=0.1,
            system=system + "\n\nIMPORTANT: Respond with ONLY valid JSON, no markdown fences.",
            messages=[{"role": "user", "content": user}],
        )
        text = response.content[0].text

        # Strip markdown fences if present
        cleaned = text.strip()
        if cleaned.startswith("```"):
            # Remove opening fence (```json or ```)
            first_newline = cleaned.find("\n")
            if first_newline != -1:
                cleaned = cleaned[first_newline + 1 :]
            else:
                cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        return json.loads(cleaned)
