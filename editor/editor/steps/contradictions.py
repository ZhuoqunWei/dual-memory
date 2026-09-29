"""Step 4: Contradictions — resolve conflicting memories by time.

Two memories conflict when they cannot both be true now: the same attribute
with different values ("lives in Seattle" / "now lives in Portland"). The newer
memory SUPERSEDES the older one, whose validTo becomes the newer one's
validFrom. Nothing is deleted and no confidence is penalized; retrieval ranks
superseded memories lower but can still answer questions about the past.

Candidates: each batch memory's nearest still-valid memories (cosine >=
CONFLICT_CANDIDATE_COSINE) that mention a shared entity. A pair is considered
once, when the later-reviewed of its two memories is in the batch. The LLM
labels each pair an update or compatible.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ..constants import (
    CONFLICT_CANDIDATE_COSINE,
    CONFLICT_MAX_PAIRS,
    CONFLICT_NEIGHBORS,
    LLM_CONFLICT_BATCH,
)
from ..llm import verdicts_by_pair

if TYPE_CHECKING:
    from ..audit import AuditLog
    from ..db import EditorDB
    from ..llm import LLMClient

log = logging.getLogger("editor.steps.contradictions")


@dataclass
class ContradictionResult:
    candidates: int = 0
    superseded: int = 0
    llm_calls: int = 0


def run_contradictions(
    batch: list[dict],
    db: EditorDB,
    audit: AuditLog,
    llm: LLMClient,
) -> ContradictionResult:
    result = ContradictionResult()

    db.backfill_valid_from()
    pairs = candidate_pairs(batch, db.fetch_current_memories(), db.fetch_conflict_links())
    result.candidates = len(pairs)

    for chunk_start in range(0, len(pairs), LLM_CONFLICT_BATCH):
        chunk = pairs[chunk_start : chunk_start + LLM_CONFLICT_BATCH]
        verdicts = _llm_verdicts(llm, chunk)
        result.llm_calls += 1

        for (a, b, _), verdict in zip(chunk, verdicts):
            if verdict.get("verdict") != "update":
                continue
            ordered = order_by_time(a, b, verdict.get("current"))
            if ordered is None:
                continue
            newer, older = ordered
            reason = verdict.get("attribute") or "LLM-detected update"
            db.supersede(newer["id"], older["id"], reason)
            audit.log("supersede", [newer["id"], older["id"]], f"Superseded: {reason}")
            result.superseded += 1

    log.info("Contradictions: %d candidates, %d superseded, %d LLM calls",
             result.candidates, result.superseded, result.llm_calls)
    return result


def candidate_pairs(
    batch: list[dict],
    memories: list[dict],
    already_linked: set[frozenset[str]],
) -> list[tuple[dict, dict, float]]:
    """Nearest-neighbour pairs worth asking about, most similar first."""
    if len(memories) < 2:
        return []
    batch_ids = {m["id"] for m in batch}
    rows = [i for i, m in enumerate(memories) if m["id"] in batch_ids]
    if not rows:
        return []

    matrix = np.array([m["embedding"] for m in memories], dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / np.where(norms == 0, 1, norms)
    sims = matrix[rows] @ matrix.T
    entities = [set(m["entityIds"]) for m in memories]

    seen: set[frozenset[str]] = set()
    pairs: list[tuple[dict, dict, float]] = []
    for r, i in enumerate(rows):
        taken = 0
        for j in np.argsort(-sims[r], kind="stable"):
            j = int(j)
            cosine = float(sims[r, j])
            if cosine < CONFLICT_CANDIDATE_COSINE or taken == CONFLICT_NEIGHBORS:
                break
            other = memories[j]
            # Raw memories outside this batch get their turn when their batch runs.
            if j == i or (other["status"] == "raw" and other["id"] not in batch_ids):
                continue
            if not entities[i] & entities[j]:
                continue
            taken += 1
            key = frozenset((memories[i]["id"], other["id"]))
            if key in seen or key in already_linked:
                continue
            seen.add(key)
            pairs.append((memories[i], other, cosine))

    pairs.sort(key=lambda p: -p[2])
    return pairs[:CONFLICT_MAX_PAIRS]


def order_by_time(a: dict, b: dict, current: str | None) -> tuple[dict, dict] | None:
    """(newer, older) by validFrom; on a tie fall back to the LLM's pick of the
    current one, and give up if it has none."""
    if a["validFrom"] != b["validFrom"]:
        return (a, b) if a["validFrom"] > b["validFrom"] else (b, a)
    if current == "A":
        return a, b
    if current == "B":
        return b, a
    return None


def _day(value: object) -> str:
    return str(value)[:10] if value is not None else "unknown date"


def _llm_verdicts(llm: LLMClient, pairs: list[tuple[dict, dict, float]]) -> list[dict]:
    items = ""
    for i, (a, b, _) in enumerate(pairs, 1):
        items += (
            f"\nPair {i}:\n"
            f"  A ({_day(a['validFrom'])}): {a['content']}\n"
            f"  B ({_day(b['validFrom'])}): {b['content']}\n"
        )

    system = """You maintain a personal memory graph. For each pair of memories, decide whether both can be true right now.

- "update": both describe the same attribute (where someone lives or works, who manages them, what they use, a count, a plan, a habit) with different values, so one replaces the other. Changes over time count: "lives in Boston" then "now lives in Denver" is an update.
- "compatible": both can be true at once. This includes history versus the present ("grew up in Ohio" and "lives in Denver"), facts about different people or things, and one memory adding detail about the same thing (a schedule, a visit, a side project) without changing its value.

Each memory shows the date it was recorded. First name the attribute each memory is about; only memories about the same attribute can be an update. For an update, also say which memory is current ("A" or "B"), normally the later one.

Return a JSON array with one object per pair:
[{"pair": 1, "attribute": "where User lives", "verdict": "update", "current": "B"},
 {"pair": 2, "attribute": "User's diet vs. User's pet", "verdict": "compatible"}]"""

    user = f"Review these {len(pairs)} memory pairs:{items}"

    try:
        reply = llm.ask_json(system, user)
    except Exception as e:
        log.warning("LLM conflict check failed: %s", e)
        reply = None
    return verdicts_by_pair(reply, len(pairs))
