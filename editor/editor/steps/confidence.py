"""Step 8: Confidence management — multi-session boost + decay.

Purely algorithmic, no LLM calls.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..constants import CONFIDENCE_DECAY, CONFIDENCE_FLOOR, CONFIDENCE_MULTI_SESSION_BOOST

if TYPE_CHECKING:
    from ..audit import AuditLog
    from ..db import EditorDB

log = logging.getLogger("editor.steps.confidence")


@dataclass
class ConfidenceResult:
    boosted: int = 0
    decayed: int = 0


def run_confidence(
    batch: list[dict],
    db: EditorDB,
    audit: AuditLog,
) -> ConfidenceResult:
    result = ConfidenceResult()

    # --- Boost: memories appearing across multiple sessions ---
    # Group batch memories by content similarity to detect cross-session reinforcement
    session_counts: dict[str, set[str]] = {}  # content_prefix -> set of session IDs
    content_map: dict[str, list[dict]] = {}  # content_prefix -> memories

    for mem in batch:
        # Use first 80 chars as a rough content key
        key = mem["content"][:80].lower().strip()
        session_id = mem.get("sessionId")
        if session_id:
            session_counts.setdefault(key, set()).add(session_id)
            content_map.setdefault(key, []).append(mem)

    for key, sessions in session_counts.items():
        if len(sessions) > 1:
            # This content appeared in multiple sessions — boost confidence (once)
            for mem in content_map[key]:
                if db.boost_confidence_once(mem["id"], CONFIDENCE_MULTI_SESSION_BOOST):
                    result.boosted += 1

    if result.boosted:
        audit.log("reweight", [], f"Boosted {result.boosted} memories (multi-session reinforcement)")

    # --- Decay: all reviewed memories lose confidence over time ---
    decayed = db.decay_reviewed_confidence(CONFIDENCE_DECAY, CONFIDENCE_FLOOR)
    result.decayed = decayed

    if decayed:
        audit.log("reweight", [], f"Decayed confidence on {decayed} reviewed memories (-{CONFIDENCE_DECAY})")

    log.info("Confidence: %d boosted, %d decayed", result.boosted, result.decayed)
    return result
