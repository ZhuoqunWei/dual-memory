"""Step 8: Entity-to-Entity LINKED_TO creation.

Mines co-mentioned entity pairs from memories, asks LLM for relationship type,
creates LINKED_TO edges between entities (e.g., Alice -[colleague]-> Bob).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..audit import AuditLog
    from ..db import EditorDB
    from ..llm import LLMClient

log = logging.getLogger("editor.steps.entity_links")


@dataclass
class EntityLinksResult:
    created: int = 0
    llm_calls: int = 0


def run_entity_links(
    db: EditorDB,
    audit: AuditLog,
    llm: LLMClient,
) -> EntityLinksResult:
    result = EntityLinksResult()

    # Find co-mentioned Person entity pairs with supporting memory content
    pairs = db.fetch_entity_co_mentions(entity_type="Person")

    if not pairs:
        log.info("Entity links: no co-mentioned entity pairs found")
        return result

    # Filter out pairs that already have LINKED_TO
    candidates = [p for p in pairs if not db.has_linked_to(p["id_a"], p["id_b"])]

    if not candidates:
        log.info("Entity links: all pairs already linked")
        return result

    # Batch LLM calls (8 pairs per call)
    for chunk_start in range(0, len(candidates), 8):
        chunk = candidates[chunk_start : chunk_start + 8]

        items = ""
        for i, p in enumerate(chunk, 1):
            # Include sample memories as evidence
            evidence = "; ".join(p["sample_contents"][:3])
            items += (
                f"\nPair {i}:\n"
                f"  A: {p['name_a']} [{p['type_a']}]\n"
                f"  B: {p['name_b']} [{p['type_b']}]\n"
                f"  Co-mentions: {p['co_count']}\n"
                f"  Evidence: {evidence[:300]}\n"
            )

        system = """Determine the relationship between each entity pair based on evidence from memory content.

For each pair, return:
- relation: the relationship type (e.g., "friend", "sibling", "colleague", "teacher", "student", "creator", "uses", "member_of", "knows")
- detail: a short human-readable description (e.g., "recommended Rust", "teaches the user")
- sentiment: "positive" | "negative" | "neutral" | "mixed"
- strength: 0.0-1.0 how strong/close the relationship is
- direction: "a_to_b" | "b_to_a" | "mutual" — who relates to whom

Return JSON array:
[{"pair": 1, "relation": "friend", "detail": "...", "sentiment": "positive", "strength": 0.8, "direction": "mutual"}, ...]
Use "none" as relation if no meaningful relationship exists."""

        user = f"Classify relationships for {len(chunk)} entity pairs:{items}"

        try:
            decisions = llm.ask_json(system, user)
            result.llm_calls += 1
        except Exception as e:
            log.warning("LLM entity links call failed: %s", e)
            continue

        if not isinstance(decisions, list):
            continue

        for d in decisions:
            idx = d.get("pair", 0) - 1
            relation = d.get("relation", "none")

            if 0 <= idx < len(chunk) and relation != "none":
                p = chunk[idx]
                detail = d.get("detail", "")
                sentiment = d.get("sentiment", "neutral")
                strength = d.get("strength", 0.5)
                direction = d.get("direction", "mutual")

                # Create LINKED_TO edge(s) based on direction
                if direction == "b_to_a":
                    db.create_linked_to(
                        p["id_b"], p["id_a"], relation, detail, sentiment, strength,
                    )
                elif direction == "mutual":
                    # Create bidirectional
                    db.create_linked_to(
                        p["id_a"], p["id_b"], relation, detail, sentiment, strength,
                    )
                    db.create_linked_to(
                        p["id_b"], p["id_a"], relation, detail, sentiment, strength,
                    )
                else:  # a_to_b
                    db.create_linked_to(
                        p["id_a"], p["id_b"], relation, detail, sentiment, strength,
                    )

                result.created += 1

                audit.log(
                    "link",
                    [p["id_a"], p["id_b"]],
                    f"LINKED_TO: {p['name_a']} -[{relation}]-> {p['name_b']}: {detail}",
                )

    log.info("Entity links: %d created, %d LLM calls", result.created, result.llm_calls)
    return result
