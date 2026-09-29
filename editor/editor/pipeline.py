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

Every step is metered: wall time, LLM calls/tokens/cost, embedding calls/tokens.
Every run is an EditorRun whose changes can be undone (see rollback.py).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Collection
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, TypeVar

from .audit import AuditLog
from .llm import cost_usd
from .steps.categories import CategoryResult, run_categories
from .steps.classify import ClassifyResult, run_classify
from .steps.confidence import ConfidenceResult, run_confidence
from .steps.contradictions import ContradictionResult, run_contradictions
from .steps.dedup import DedupResult, run_dedup
from .steps.entity_links import EntityLinksResult, run_entity_links
from .steps.entity_resolution import EntityResolutionResult, run_entity_resolution
from .steps.relationships import RelationshipResult, run_relationships

if TYPE_CHECKING:
    from .config import EditorConfig
    from .db import EditorDB
    from .embeddings import EmbeddingsClient
    from .llm import LLMClient

log = logging.getLogger("editor.pipeline")

STEP_ORDER = (
    "dedup", "classify", "categories", "contradictions",
    "entity_resolution", "relationships", "entity_links", "confidence",
)
VALID_STEPS = set(STEP_ORDER)

T = TypeVar("T")


@dataclass
class StepMetrics:
    step: str
    seconds: float = 0.0  # wall time for the whole step
    llm_seconds: float = 0.0  # time spent waiting on the LLM API
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = 0.0  # None when the model has no known price
    embed_calls: int = 0
    embed_tokens: int = 0

    def add(self, other: StepMetrics) -> None:
        self.seconds += other.seconds
        self.llm_seconds += other.llm_seconds
        self.llm_calls += other.llm_calls
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        if self.cost_usd is None or other.cost_usd is None:
            self.cost_usd = None
        else:
            self.cost_usd += other.cost_usd
        self.embed_calls += other.embed_calls
        self.embed_tokens += other.embed_tokens


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
    run_id: str | None = None
    step_metrics: list[StepMetrics] = field(default_factory=list)


def run_pipeline(
    db: EditorDB,
    llm: LLMClient | None = None,
    embeddings: EmbeddingsClient | None = None,
    config: EditorConfig | None = None,
    steps: Collection[str] | None = None,
    batch_cap: int = 50,
    mark_reviewed: bool | None = None,
) -> PipelineResult:
    """Run the full Editor pipeline (or a subset of steps).

    Args:
        db: Neo4j client.
        llm: LLM client (None = skip LLM-dependent steps).
        embeddings: Voyage AI client (None = skip embedding-dependent steps).
        config: Editor config (optional, for threshold overrides).
        steps: If set, run only these steps (e.g., {"dedup"}).
        batch_cap: Max memories to process in one run.
        mark_reviewed: Mark surviving batch memories reviewed at the end.
            Defaults to True for a full run and False for a subset, so a
            single-step debug run doesn't consume the batch.

    Returns:
        PipelineResult with per-step results and metrics.
    """
    start = time.monotonic()
    result = PipelineResult()

    unknown = set(steps or ()) - VALID_STEPS
    if unknown:
        raise ValueError(f"Unknown step(s): {sorted(unknown)}. Valid: {list(STEP_ORDER)}")
    if mark_reviewed is None:
        mark_reviewed = steps is None

    # Fetch batch
    batch = db.fetch_raw_memories(limit=batch_cap)
    result.batch_size = len(batch)

    if not batch:
        log.info("No raw memories to process")
        result.duration_seconds = time.monotonic() - start
        return result

    log.info("Processing batch of %d raw memories", len(batch))
    result.run_id = db.start_run(
        llm.model if llm else None, llm.effort if llm else None, len(batch),
    )
    audit = AuditLog(db)
    status = "failed"
    try:
        batch = _run_steps(batch, db, llm, embeddings, steps, audit, result)
        if mark_reviewed:
            surviving_ids = [m["id"] for m in batch]
            if surviving_ids:
                db.mark_reviewed_batch(surviving_ids)
                result.reviewed_count = len(surviving_ids)
                log.info("Marked %d memories as reviewed", result.reviewed_count)
            audit.flush("mark_reviewed")
        status = "completed"
    finally:
        result.edit_actions = audit.count
        result.duration_seconds = time.monotonic() - start
        total = summarize_step_metrics(result.step_metrics)[-1]
        db.finish_run(status, {
            "batch": result.batch_size,
            "reviewed": result.reviewed_count,
            "editActions": result.edit_actions,
            "llmCalls": total.llm_calls,
            "costUsd": total.cost_usd,
            "seconds": round(result.duration_seconds, 2),
        })

    log.info(
        "Pipeline complete: %d memories, %.1fs, %d edit actions (run %s)",
        result.batch_size, result.duration_seconds, result.edit_actions, result.run_id,
    )
    return result


def _run_steps(
    batch: list[dict],
    db: EditorDB,
    llm: LLMClient | None,
    embeddings: EmbeddingsClient | None,
    steps: Collection[str] | None,
    audit: AuditLog,
    result: PipelineResult,
) -> list[dict]:
    """Run the selected steps in order; returns the batch minus archived dupes."""

    def should_run(step: str) -> bool:
        return steps is None or step in steps

    def metered(step: str, fn: Callable[[], T]) -> T:
        llm_before = llm.usage.snapshot() if llm else None
        emb_before = embeddings.usage.snapshot() if embeddings else None
        step_start = time.monotonic()
        value = fn()
        m = StepMetrics(step=step, seconds=time.monotonic() - step_start)
        if llm and llm_before:
            used = llm.usage.since(llm_before)
            m.llm_seconds = used.seconds
            m.llm_calls = used.calls
            m.input_tokens = used.input_tokens
            m.output_tokens = used.output_tokens
            m.cost_usd = cost_usd(llm.model, used.input_tokens, used.output_tokens)
        if embeddings and emb_before:
            used_emb = embeddings.usage.since(emb_before)
            m.embed_calls = used_emb.calls
            m.embed_tokens = used_emb.tokens
        result.step_metrics.append(m)
        audit.flush(step)
        return value

    # Step 1: Dedup
    if should_run("dedup"):
        log.info("=== Step 1: Dedup ===")
        result.dedup = metered("dedup", lambda: run_dedup(batch, db, audit, llm))
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
        result.classify = metered("classify", lambda: run_classify(batch, db, audit, llm))

    # Step 3: Categories
    if should_run("categories") and llm and embeddings:
        log.info("=== Step 3: Categories ===")
        result.categories = metered(
            "categories", lambda: run_categories(batch, db, audit, llm, embeddings),
        )

    # Step 4: Contradictions
    if should_run("contradictions") and llm:
        log.info("=== Step 4: Contradictions ===")
        result.contradictions = metered(
            "contradictions", lambda: run_contradictions(batch, db, audit, llm),
        )

    # Step 5: Entity resolution
    if should_run("entity_resolution"):
        log.info("=== Step 5: Entity Resolution ===")
        result.entity_resolution = metered(
            "entity_resolution", lambda: run_entity_resolution(db, audit, llm, embeddings),
        )

    # Step 6: Relationships
    if should_run("relationships") and llm:
        log.info("=== Step 6: Relationships ===")
        result.relationships = metered(
            "relationships", lambda: run_relationships(batch, db, audit, llm),
        )

    # Step 7: Entity links (LINKED_TO between entities)
    if should_run("entity_links") and llm:
        log.info("=== Step 7: Entity Links ===")
        result.entity_links = metered("entity_links", lambda: run_entity_links(db, audit, llm))

    # Step 8: Confidence
    if should_run("confidence"):
        log.info("=== Step 8: Confidence ===")
        result.confidence = metered("confidence", lambda: run_confidence(batch, db, audit))

    return batch


def summarize_step_metrics(metrics: list[StepMetrics]) -> list[StepMetrics]:
    """Sum metrics per step (steps repeat across cycles), in pipeline order,
    followed by a TOTAL row."""
    totals: dict[str, StepMetrics] = {}
    for m in metrics:
        totals.setdefault(m.step, StepMetrics(step=m.step)).add(m)
    rows = [totals[s] for s in STEP_ORDER if s in totals]
    total = StepMetrics(step="TOTAL")
    for r in rows:
        total.add(r)
    return [*rows, total]


def format_step_metrics(metrics: list[StepMetrics]) -> str:
    """Render per-step metrics as a fixed-width table."""

    def cost(c: float | None) -> str:
        return "n/a" if c is None else f"${c:.4f}"

    lines = [
        (
            f"  {'step':<18} {'wall s':>7} {'llm s':>7} {'llm':>5} {'in tok':>8} "
            f"{'out tok':>8} {'cost':>9} {'emb':>4} {'emb tok':>8}"
        ),
    ]
    for r in summarize_step_metrics(metrics):
        lines.append(
            f"  {r.step:<18} {r.seconds:>7.2f} {r.llm_seconds:>7.2f} {r.llm_calls:>5} "
            f"{r.input_tokens:>8} {r.output_tokens:>8} {cost(r.cost_usd):>9} "
            f"{r.embed_calls:>4} {r.embed_tokens:>8}"
        )
    return "\n".join(lines)


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
