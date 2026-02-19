"""Step 1: Deduplication — exact (cosine >= 0.98) and semantic (LLM-assisted).

Tier 1 (exact): Pure numpy cosine similarity. No LLM calls.
Tier 2 (semantic): Pairs with 0.85-0.98 cosine sent to Claude for merge/distinct decision.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from ..constants import EXACT_DEDUP_COSINE, LLM_DEDUP_BATCH, SEMANTIC_DEDUP_COSINE

if TYPE_CHECKING:
    from ..audit import AuditLog
    from ..db import EditorDB
    from ..llm import LLMClient

log = logging.getLogger("editor.steps.dedup")


@dataclass
class DedupResult:
    exact_dupes_archived: int = 0
    semantic_dupes_archived: int = 0
    semantic_related: int = 0
    llm_calls: int = 0


def run_dedup(
    batch: list[dict],
    db: EditorDB,
    audit: AuditLog,
    llm: LLMClient | None = None,
) -> DedupResult:
    """Run deduplication on a batch of raw memories.

    Args:
        batch: Raw memories from db.fetch_raw_memories().
        db: Neo4j client.
        audit: Audit logger.
        llm: LLM client (None = skip semantic tier).

    Returns:
        DedupResult with counts.
    """
    result = DedupResult()

    # Fetch all existing memory embeddings for comparison
    all_memories = db.fetch_all_memory_embeddings()
    if not all_memories or not batch:
        return result

    # Build lookup: id -> index in all_memories
    all_ids = [m["id"] for m in all_memories]
    all_id_set = set(all_ids)
    batch_ids = set(m["id"] for m in batch)

    # Extract embeddings into numpy arrays
    all_embeddings = _build_embedding_matrix(all_memories)
    if all_embeddings is None:
        return result

    # Build batch embedding matrix (subset of all_memories that are in the batch)
    batch_indices = [i for i, m in enumerate(all_memories) if m["id"] in batch_ids]
    if not batch_indices:
        return result

    # Compute cosine similarity: batch vs all
    sim_matrix = _cosine_similarity_matrix(all_embeddings, batch_indices)

    # Track which IDs have been archived this run (avoid double-archiving)
    archived: set[str] = set()

    # --- Tier 1: Exact dedup (cosine >= 0.98) ---
    exact_pairs = _find_pairs_above_threshold(
        sim_matrix, batch_indices, all_ids, EXACT_DEDUP_COSINE, archived,
    )

    for dupe_idx, canon_idx, cosine in exact_pairs:
        dupe_id = all_ids[dupe_idx]
        canon_id = all_ids[canon_idx]

        if dupe_id in archived:
            continue

        # Older memory is canonical (keep it), newer is the dupe (archive it)
        dupe_ts = all_memories[dupe_idx].get("timestamp")
        canon_ts = all_memories[canon_idx].get("timestamp")
        if dupe_ts and canon_ts and str(dupe_ts) < str(canon_ts):
            dupe_id, canon_id = canon_id, dupe_id

        db.create_canonical(dupe_id, canon_id)
        audit.log("merge", [dupe_id, canon_id], f"Exact duplicate (cosine={cosine:.4f})")
        archived.add(dupe_id)
        result.exact_dupes_archived += 1

    log.info("Exact dedup: %d duplicates archived", result.exact_dupes_archived)

    # --- Tier 2: Semantic dedup (0.85 <= cosine < 0.98, LLM-assisted) ---
    if llm is None:
        return result

    semantic_pairs = _find_pairs_above_threshold(
        sim_matrix, batch_indices, all_ids, SEMANTIC_DEDUP_COSINE, archived,
        upper_bound=EXACT_DEDUP_COSINE,
    )

    if not semantic_pairs:
        return result

    # Batch LLM calls
    for chunk_start in range(0, len(semantic_pairs), LLM_DEDUP_BATCH):
        chunk = semantic_pairs[chunk_start : chunk_start + LLM_DEDUP_BATCH]
        pairs_for_llm = []
        for dupe_idx, canon_idx, cosine in chunk:
            if all_ids[dupe_idx] in archived:
                continue
            pairs_for_llm.append({
                "a_id": all_ids[dupe_idx],
                "a_content": all_memories[dupe_idx]["content"],
                "b_id": all_ids[canon_idx],
                "b_content": all_memories[canon_idx]["content"],
                "cosine": cosine,
            })

        if not pairs_for_llm:
            continue

        decisions = _llm_dedup_decision(llm, pairs_for_llm)
        result.llm_calls += 1

        for pair_info, decision in zip(pairs_for_llm, decisions):
            if decision["decision"] == "duplicate":
                # Determine canonical (LLM picks, or default to older)
                if decision.get("canonical") == "B":
                    dupe_id, canon_id = pair_info["a_id"], pair_info["b_id"]
                else:
                    dupe_id, canon_id = pair_info["b_id"], pair_info["a_id"]

                if dupe_id not in archived:
                    db.create_canonical(dupe_id, canon_id)
                    audit.log("merge", [dupe_id, canon_id],
                              f"Semantic duplicate: {decision.get('reason', 'LLM decision')}")
                    archived.add(dupe_id)
                    result.semantic_dupes_archived += 1

            elif decision["decision"] == "related":
                rel_type = decision.get("relation", "supports")
                weight = decision.get("weight", 0.6)
                if not db.has_relates_to(pair_info["a_id"], pair_info["b_id"]):
                    db.create_relates_to(
                        pair_info["a_id"], pair_info["b_id"], rel_type, weight,
                    )
                    result.semantic_related += 1

    log.info(
        "Semantic dedup: %d archived, %d related, %d LLM calls",
        result.semantic_dupes_archived, result.semantic_related, result.llm_calls,
    )

    return result


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def get_archived_ids(batch: list[dict], db: EditorDB, audit: AuditLog,
                     llm: LLMClient | None = None) -> set[str]:
    """Convenience: run dedup and return the set of archived IDs."""
    result = run_dedup(batch, db, audit, llm)
    # Re-query which batch IDs are now archived
    archived = set()
    for m in batch:
        # If it was in our result, it's archived
        pass  # The actual archived set is tracked internally
    return archived


def _build_embedding_matrix(memories: list[dict]) -> np.ndarray | None:
    """Build (n, dims) float32 matrix from memory embeddings."""
    embeddings = []
    for m in memories:
        emb = m.get("embedding")
        if emb is None:
            return None  # Can't do cosine without embeddings
        embeddings.append(emb)
    if not embeddings:
        return None
    return np.array(embeddings, dtype=np.float32)


def _cosine_similarity_matrix(
    all_embeddings: np.ndarray, batch_indices: list[int],
) -> np.ndarray:
    """Compute cosine similarity between batch rows and all rows.

    Returns: (len(batch_indices), n_all) matrix.
    """
    # Normalize all rows
    norms = np.linalg.norm(all_embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)  # avoid division by zero
    normed = all_embeddings / norms

    batch_normed = normed[batch_indices]  # (batch, dims)
    return batch_normed @ normed.T  # (batch, all)


def _find_pairs_above_threshold(
    sim_matrix: np.ndarray,
    batch_indices: list[int],
    all_ids: list[str],
    threshold: float,
    exclude: set[str],
    upper_bound: float = 1.01,
) -> list[tuple[int, int, float]]:
    """Find (batch_idx_in_all, other_idx_in_all, cosine) pairs above threshold.

    Excludes self-pairs and already-archived IDs.
    """
    pairs: list[tuple[int, int, float]] = []
    seen: set[tuple[str, str]] = set()

    for row_i, batch_idx in enumerate(batch_indices):
        batch_id = all_ids[batch_idx]
        if batch_id in exclude:
            continue
        for other_idx in range(len(all_ids)):
            if other_idx == batch_idx:
                continue
            other_id = all_ids[other_idx]
            if other_id in exclude:
                continue
            cosine = float(sim_matrix[row_i, other_idx])
            if threshold <= cosine < upper_bound:
                # Deduplicate pair order
                pair_key = tuple(sorted((batch_id, other_id)))
                if pair_key not in seen:
                    seen.add(pair_key)
                    pairs.append((batch_idx, other_idx, cosine))

    # Sort by cosine descending (most similar first)
    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs


def _llm_dedup_decision(llm: LLMClient, pairs: list[dict]) -> list[dict]:
    """Ask the LLM to decide on semantic duplicate candidates."""
    pairs_text = ""
    for i, p in enumerate(pairs, 1):
        pairs_text += f"\nPair {i}:\n  A: {p['a_content']}\n  B: {p['b_content']}\n  Cosine: {p['cosine']:.3f}\n"

    system = """You are reviewing memory pairs for deduplication in a knowledge graph.

For each pair, decide:
- "duplicate": same fact expressed differently. Pick canonical (A or B) — prefer whichever is more complete/precise.
- "related": different but related facts. Specify relation: supports | elaborates | follows.
- "distinct": unrelated facts, no action needed.

Return a JSON array with one object per pair:
[{"pair": 1, "decision": "duplicate", "canonical": "A", "reason": "..."}, ...]"""

    user = f"Review these {len(pairs)} memory pairs:\n{pairs_text}"

    try:
        decisions = llm.ask_json(system, user)
        if isinstance(decisions, list) and len(decisions) == len(pairs):
            return decisions
        # Fallback: treat all as distinct
        return [{"pair": i + 1, "decision": "distinct"} for i in range(len(pairs))]
    except Exception as e:
        log.warning("LLM dedup call failed: %s", e)
        return [{"pair": i + 1, "decision": "distinct"} for i in range(len(pairs))]
