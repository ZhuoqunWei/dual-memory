"""Step 4: Contradiction detection — find conflicting facts via LLM.

Groups memories by shared entities, sends each group to Claude for review.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..constants import CONFIDENCE_CONTRADICTION_PENALTY, LLM_CONTRADICTION_BATCH

if TYPE_CHECKING:
    from ..audit import AuditLog
    from ..db import EditorDB
    from ..llm import LLMClient

log = logging.getLogger("editor.steps.contradictions")


@dataclass
class ContradictionResult:
    contradictions_found: int = 0
    confidence_adjusted: int = 0
    llm_calls: int = 0


def run_contradictions(
    batch: list[dict],
    db: EditorDB,
    audit: AuditLog,
    llm: LLMClient,
) -> ContradictionResult:
    result = ContradictionResult()

    batch_ids = {m["id"] for m in batch}

    # Group memories by shared entities
    entity_groups = db.fetch_memories_by_shared_entity()

    for entity_name, memories in entity_groups.items():
        # Only process groups that include at least one batch memory
        if not any(m["id"] in batch_ids for m in memories):
            continue

        # Skip small groups
        if len(memories) < 2:
            continue

        # Batch for LLM
        items = ""
        mem_list = memories[:LLM_CONTRADICTION_BATCH]
        for i, m in enumerate(mem_list, 1):
            items += f"\n{i}. [{m.get('kind', '?')}] {m['content'][:150]}"

        system = f"""Review these memories about "{entity_name}" for contradictions.

A contradiction means two memories state opposite or incompatible facts.
Time-based changes are NOT contradictions (e.g., "liked X in Jan" + "stopped liking X in Mar" = evolution).
Duplicates or near-duplicates are NOT contradictions.

Return a JSON array of contradictions found:
[{{"memory_a": 1, "memory_b": 3, "reason": "..."}}]
Return empty array [] if no contradictions."""

        user = f"Check these {len(mem_list)} memories about '{entity_name}':{items}"

        try:
            decisions = llm.ask_json(system, user)
            result.llm_calls += 1
        except Exception as e:
            log.warning("LLM contradiction check failed for '%s': %s", entity_name, e)
            continue

        if not isinstance(decisions, list):
            continue

        for d in decisions:
            idx_a = d.get("memory_a", 0) - 1
            idx_b = d.get("memory_b", 0) - 1
            reason = d.get("reason", "LLM-detected contradiction")

            if 0 <= idx_a < len(mem_list) and 0 <= idx_b < len(mem_list):
                mem_a = mem_list[idx_a]
                mem_b = mem_list[idx_b]

                # Create RELATES_TO {type: "contradicts"}
                if not db.has_relates_to(mem_a["id"], mem_b["id"]):
                    db.create_relates_to(mem_a["id"], mem_b["id"], "contradicts", 0.8)
                    result.contradictions_found += 1

                    # Lower confidence on both
                    for m in (mem_a, mem_b):
                        old_conf = m.get("confidence", 0.5)
                        new_conf = max(0.1, old_conf - CONFIDENCE_CONTRADICTION_PENALTY)
                        db.set_confidence(m["id"], new_conf)
                        result.confidence_adjusted += 1

                    audit.log(
                        "reweight",
                        [mem_a["id"], mem_b["id"]],
                        f"Contradiction: {reason}",
                    )

    log.info("Contradictions: %d found, %d confidence adjusted, %d LLM calls",
             result.contradictions_found, result.confidence_adjusted, result.llm_calls)
    return result
