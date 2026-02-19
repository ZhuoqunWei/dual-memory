"""Pipeline orchestrator — runs all Editor steps in order.

Steps (priority order):
1. Dedup (exact + semantic)
2. Classify (normalizedKind/normalizedType)
3. Categories (create + assign)
4. Contradictions (LLM-detected)
5. Entity resolution (alias overlap)
6. Relationships (RELATES_TO between memories)
7. Entity links (LINKED_TO between entities)
8. Confidence (multi-session boost + decay)
9. Mark survivors as reviewed
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .audit import AuditLog
from .steps.categories import CategoryResult, run_categories
from .steps.classify import ClassifyResult, run_classify
from .steps.confidence import ConfidenceResult, run_confidence
from .steps.contradictions import ContradictionResult, run_contradictions
from .steps.dedup import DedupResult, run_dedup
from .steps.entity_resolution import EntityResolutionResult, run_entity_resolution
from .steps.entity_links import EntityLinksResult, run_entity_links
from .steps.relationships import RelationshipResult, run_relationships

if TYPE_CHECKING:
    from .config import EditorConfig
    from .db import EditorDB
    from .embeddings import EmbeddingsClient
    from .llm import LLMClient

log = logging.getLogger("editor.pipeline")

VALID_STEPS = {
    "dedup", "classify", "categories", "contradictions",
    "entity_resolution", "relationships", "entity_links", "confidence",
}


@dataclass
class PipelineResult:
    batch_size: int = 0
    duration_seconds: float = 0.0
    edit_actions: int = 0
    dedup: DedupResult | None = None
    classify: ClassifyResult | None = None
    categories: CategoryResult | None = None
    contradictions: ContradictionResult | None = None
    entity_resolution: EntityResolutionResult | None = None
    relationships: RelationshipResult | None = None
    entity_links: EntityLinksResult | None = None
    confidence: ConfidenceResult | None = None
    reviewed_count: int = 0


def run_pipeline(
    db: EditorDB,
    llm: LLMClient | None = None,
    embeddings: EmbeddingsClient | None = None,
    config: EditorConfig | None = None,
    only_step: str | None = None,
    batch_cap: int = 50,
) -> PipelineResult:
    """Run the full Editor pipeline (or a single step).

    Args:
        db: Neo4j client.
        llm: LLM client (None = skip LLM-dependent steps).
        embeddings: Voyage AI client (None = skip embedding-dependent steps).
        config: Editor config (optional, for threshold overrides).
        only_step: If set, run only this step (e.g., "dedup").
        batch_cap: Max memories to process in one run.

    Returns:
        PipelineResult with per-step results.
    """
    start = time.monotonic()
    result = PipelineResult()
    audit = AuditLog(db)

    # Validate step name
    if only_step and only_step not in VALID_STEPS:
        raise ValueError(f"Unknown step: {only_step!r}. Valid: {sorted(VALID_STEPS)}")

    # Fetch batch
    batch = db.fetch_raw_memories(limit=batch_cap)
    result.batch_size = len(batch)

    if not batch:
        log.info("No raw memories to process")
        result.duration_seconds = time.monotonic() - start
        return result

    log.info("Processing batch of %d raw memories", len(batch))

    def should_run(step: str) -> bool:
        return only_step is None or only_step == step

    # Step 1: Dedup
    if should_run("dedup"):
        log.info("=== Step 1: Dedup ===")
        result.dedup = run_dedup(batch, db, audit, llm)
        # Remove archived memories from batch for subsequent steps
        if result.dedup.exact_dupes_archived or result.dedup.semantic_dupes_archived:
            archived_count = result.dedup.exact_dupes_archived + result.dedup.semantic_dupes_archived
            # Re-fetch to get updated statuses
            surviving_ids = _get_surviving_ids(batch, db)
            batch = [m for m in batch if m["id"] in surviving_ids]
            log.info("Batch reduced to %d after dedup (%d archived)", len(batch), archived_count)

    # Step 2: Classify
    if should_run("classify"):
        log.info("=== Step 2: Classify ===")
        result.classify = run_classify(batch, db, audit, llm)

    # Step 3: Categories
    if should_run("categories") and llm and embeddings:
        log.info("=== Step 3: Categories ===")
        result.categories = run_categories(batch, db, audit, llm, embeddings)

    # Step 4: Contradictions
    if should_run("contradictions") and llm:
        log.info("=== Step 4: Contradictions ===")
        result.contradictions = run_contradictions(batch, db, audit, llm)

    # Step 5: Entity resolution
    if should_run("entity_resolution"):
        log.info("=== Step 5: Entity Resolution ===")
        result.entity_resolution = run_entity_resolution(db, audit, llm)

    # Step 6: Relationships
    if should_run("relationships") and llm:
        log.info("=== Step 6: Relationships ===")
        result.relationships = run_relationships(batch, db, audit, llm)

    # Step 7: Entity links (LINKED_TO between entities)
    if should_run("entity_links") and llm:
        log.info("=== Step 7: Entity Links ===")
        result.entity_links = run_entity_links(db, audit, llm)

    # Step 8: Confidence
    if should_run("confidence"):
        log.info("=== Step 7: Confidence ===")
        result.confidence = run_confidence(batch, db, audit)

    # Mark all surviving batch memories as reviewed
    if only_step is None:
        surviving_ids = [m["id"] for m in batch]
        if surviving_ids:
            db.mark_reviewed_batch(surviving_ids)
            result.reviewed_count = len(surviving_ids)
            log.info("Marked %d memories as reviewed", result.reviewed_count)

    result.edit_actions = audit.count
    result.duration_seconds = time.monotonic() - start

    log.info(
        "Pipeline complete: %d memories, %.1fs, %d edit actions",
        result.batch_size, result.duration_seconds, result.edit_actions,
    )

    return result


def _get_surviving_ids(batch: list[dict], db: EditorDB) -> set[str]:
    """Check which batch memories are still active (not archived)."""
    batch_ids = [m["id"] for m in batch]
    query = """
    UNWIND $ids AS mid
    MATCH (m:Memory {id: mid})
    WHERE m.status <> 'archived'
    RETURN m.id AS id
    """
    with db._driver.session() as s:
        return {r["id"] for r in s.run(query, ids=batch_ids)}
