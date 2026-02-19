"""Step 5: Entity resolution — merge duplicate entities via MERGED_INTO.

Tier 1: Alias overlap (algorithmic).
Tier 2: Embedding similarity + LLM confirmation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..audit import AuditLog
    from ..db import EditorDB
    from ..llm import LLMClient

log = logging.getLogger("editor.steps.entity_resolution")


@dataclass
class EntityResolutionResult:
    merged: int = 0
    llm_calls: int = 0


def run_entity_resolution(
    db: EditorDB,
    audit: AuditLog,
    llm: LLMClient | None = None,
) -> EntityResolutionResult:
    result = EntityResolutionResult()

    entities = db.fetch_all_entities()
    if len(entities) < 2:
        return result

    merged_ids: set[str] = set()

    # --- Tier 1: Alias overlap ---
    for i, e1 in enumerate(entities):
        if e1["id"] in merged_ids:
            continue
        aliases1 = {a.lower() for a in (e1.get("aliases") or [])} | {e1["name"].lower()}

        for j in range(i + 1, len(entities)):
            e2 = entities[j]
            if e2["id"] in merged_ids:
                continue
            aliases2 = {a.lower() for a in (e2.get("aliases") or [])} | {e2["name"].lower()}

            overlap = aliases1 & aliases2
            if overlap:
                # Merge: entity with more aliases becomes canonical
                if len(aliases1) >= len(aliases2):
                    source, target = e2, e1
                else:
                    source, target = e1, e2

                db.merge_entities(source["id"], target["id"])
                merged_ids.add(source["id"])
                result.merged += 1

                audit.log(
                    "merge",
                    [source["id"], target["id"]],
                    f"Entity alias overlap: {overlap}. Merged '{source['name']}' into '{target['name']}'",
                )

    # --- Tier 2: Name similarity + LLM (future enhancement) ---
    # For now, alias overlap catches the obvious cases.
    # Embedding-based entity similarity can be added when entity embeddings are generated.

    log.info("Entity resolution: %d merged", result.merged)
    return result
