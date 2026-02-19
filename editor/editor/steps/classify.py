"""Step 2: Classification normalization — set normalizedKind and normalizedType.

Mostly rule-based (canonical set + synonym mapping). Falls back to LLM for unknowns.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..constants import (
    CANONICAL_ENTITY_TYPES,
    CANONICAL_KINDS,
    ENTITY_TYPE_SYNONYMS,
    KIND_SYNONYMS,
    LLM_CLASSIFY_BATCH,
)

if TYPE_CHECKING:
    from ..audit import AuditLog
    from ..db import EditorDB
    from ..llm import LLMClient

log = logging.getLogger("editor.steps.classify")


@dataclass
class ClassifyResult:
    memories_classified: int = 0
    entities_classified: int = 0
    llm_calls: int = 0


def run_classify(
    batch: list[dict],
    db: EditorDB,
    audit: AuditLog,
    llm: LLMClient | None = None,
) -> ClassifyResult:
    result = ClassifyResult()

    # --- Memory kind normalization ---
    needs_llm: list[dict] = []

    for mem in batch:
        if mem.get("normalizedKind"):
            continue  # Already classified

        kind = (mem.get("kind") or "").strip().lower()
        normalized = _resolve_kind(kind)

        if normalized:
            db.set_normalized_kind(mem["id"], normalized)
            result.memories_classified += 1
        else:
            needs_llm.append(mem)

    # LLM fallback for unknown kinds
    if needs_llm and llm:
        for chunk_start in range(0, len(needs_llm), LLM_CLASSIFY_BATCH):
            chunk = needs_llm[chunk_start : chunk_start + LLM_CLASSIFY_BATCH]
            decisions = _llm_classify_kinds(llm, chunk)
            result.llm_calls += 1

            for mem, nk in zip(chunk, decisions):
                if nk in CANONICAL_KINDS:
                    db.set_normalized_kind(mem["id"], nk)
                    result.memories_classified += 1

    # --- Entity type normalization ---
    entities = db.fetch_all_entities()
    for ent in entities:
        if ent.get("normalizedType"):
            continue

        etype = (ent.get("type") or "").strip()
        normalized = _resolve_entity_type(etype)

        if normalized:
            db.set_entity_normalized_type(ent["id"], normalized)
            result.entities_classified += 1

    if result.memories_classified or result.entities_classified:
        audit.log(
            "reclassify",
            [m["id"] for m in batch if not m.get("normalizedKind")],
            f"Classified {result.memories_classified} memories, {result.entities_classified} entities",
        )

    log.info(
        "Classify: %d memories, %d entities, %d LLM calls",
        result.memories_classified, result.entities_classified, result.llm_calls,
    )
    return result


def _resolve_kind(kind: str) -> str | None:
    """Try to map a kind string to a canonical kind. Returns None if unknown."""
    if kind in CANONICAL_KINDS:
        return kind
    return KIND_SYNONYMS.get(kind)


def _resolve_entity_type(etype: str) -> str | None:
    """Try to map an entity type to a canonical type."""
    if etype in CANONICAL_ENTITY_TYPES:
        return etype
    return ENTITY_TYPE_SYNONYMS.get(etype.lower())


def _llm_classify_kinds(llm: LLMClient, memories: list[dict]) -> list[str]:
    """Ask LLM to classify unknown memory kinds."""
    items = ""
    for i, m in enumerate(memories, 1):
        items += f"\n{i}. kind=\"{m.get('kind', '?')}\" content=\"{m['content'][:120]}\""

    system = """Classify each memory into exactly one of these kinds:
fact | decision | preference | goal | emotion | observation | event

Return a JSON array of strings, one per memory:
["fact", "decision", ...]"""

    user = f"Classify these {len(memories)} memories:{items}"

    try:
        result = llm.ask_json(system, user)
        if isinstance(result, list) and len(result) == len(memories):
            return [str(k).lower() for k in result]
    except Exception as e:
        log.warning("LLM classify failed: %s", e)

    # Fallback: use "fact" for everything
    return ["fact"] * len(memories)
