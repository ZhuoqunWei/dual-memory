"""Step 6: RELATES_TO creation between memories.

For memory pairs sharing entities (no existing edge), ask LLM for relationship type.

Pairs come only from non-hub entities: an entity mentioned by more than
RELATIONSHIP_HUB_MENTIONS memories (the user themself, typically) links nearly
every pair, which made this step 95% of the Editor's LLM spend on the eval
fixture while adding edges nothing reads. Each pair is considered once, when
the later-reviewed of its memories is in the batch, and at most
RELATIONSHIP_MAX_PAIRS pairs go to the LLM per run.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..constants import RELATIONSHIP_HUB_MENTIONS, RELATIONSHIP_MAX_PAIRS

if TYPE_CHECKING:
    from ..audit import AuditLog
    from ..db import EditorDB
    from ..llm import LLMClient

log = logging.getLogger("editor.steps.relationships")


@dataclass
class RelationshipResult:
    created: int = 0
    llm_calls: int = 0


def run_relationships(
    batch: list[dict],
    db: EditorDB,
    audit: AuditLog,
    llm: LLMClient,
) -> RelationshipResult:
    result = RelationshipResult()
    batch_ids = {m["id"] for m in batch}

    # Group memories by shared entities, leaving out hubs
    entity_groups = db.fetch_memories_by_shared_entity()

    # Collect candidate pairs (share entity, no existing RELATES_TO)
    candidates: list[tuple[dict, dict]] = []
    seen_pairs: set[tuple[str, str]] = set()

    for memories in entity_groups.values():
        if len(memories) > RELATIONSHIP_HUB_MENTIONS:
            continue
        group_with_batch = [m for m in memories if m["id"] in batch_ids]
        if not group_with_batch:
            continue

        for i, m1 in enumerate(memories):
            for j in range(i + 1, len(memories)):
                m2 = memories[j]
                # At least one must be from the batch, and the other reviewed
                # already or in the batch too (so each pair is asked once)
                if m1["id"] not in batch_ids and m2["id"] not in batch_ids:
                    continue
                if (m1["status"] == "raw" and m1["id"] not in batch_ids) or (
                    m2["status"] == "raw" and m2["id"] not in batch_ids
                ):
                    continue

                pair_key = tuple(sorted((m1["id"], m2["id"])))
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)

                if not db.has_relates_to(m1["id"], m2["id"]):
                    candidates.append((m1, m2))

    if not candidates:
        return result
    candidates = candidates[:RELATIONSHIP_MAX_PAIRS]

    # Batch LLM calls (12 pairs per call)
    for chunk_start in range(0, len(candidates), 12):
        chunk = candidates[chunk_start : chunk_start + 12]

        items = ""
        for i, (m1, m2) in enumerate(chunk, 1):
            items += f"\nPair {i}:\n  A: [{m1.get('kind', '?')}] {m1['content'][:120]}\n  B: [{m2.get('kind', '?')}] {m2['content'][:120]}\n"

        system = """Determine the relationship between each memory pair.

For each pair, return:
- type: "caused_by" | "follows" | "supports" | "elaborates" | "none"
  Direction: A -[type]-> B (e.g., "caused_by" means A was caused by B)
- weight: 0.0-1.0 strength of connection

NEVER use types starting with "editor:" — those are reserved.

Return JSON array:
[{"pair": 1, "type": "supports", "weight": 0.7}, ...]
Use "none" if the pair has no meaningful relationship."""

        user = f"Classify relationships for {len(chunk)} memory pairs:{items}"

        try:
            decisions = llm.ask_json(system, user)
            result.llm_calls += 1
        except Exception as e:
            log.warning("LLM relationship call failed: %s", e)
            continue

        if not isinstance(decisions, list):
            continue

        valid_types = {"caused_by", "follows", "supports", "elaborates", "contradicts"}

        for d in decisions:
            idx = d.get("pair", 0) - 1
            rel_type = d.get("type", "none")
            weight = d.get("weight", 0.5)

            if 0 <= idx < len(chunk) and rel_type in valid_types:
                m1, m2 = chunk[idx]
                db.create_relates_to(m1["id"], m2["id"], rel_type, weight)
                result.created += 1

    if result.created:
        audit.log("reclassify", [], f"Created {result.created} RELATES_TO edges")

    log.info("Relationships: %d created, %d LLM calls", result.created, result.llm_calls)
    return result
