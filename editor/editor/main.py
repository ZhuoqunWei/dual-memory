"""Editor agent entry point.

Usage:
    python -m editor.main --once          # Run one cycle (all steps)
    python -m editor.main --once --all    # Process ALL raw memories (no batch cap)
    python -m editor.main --step dedup    # Run only dedup step
    python -m editor.main --stats         # Show graph stats
    python -m editor.main --runs          # List recent Editor runs
    python -m editor.main --rollback ID   # Undo an Editor run
    python -m editor.main                 # Start scheduler (nightly + threshold)
"""

from __future__ import annotations

import argparse
import json
import logging
import time

import schedule

from .config import EditorConfig
from .db import EditorDB
from .embeddings import EmbeddingsClient
from .llm import LLMClient
from .pipeline import PipelineResult, StepMetrics, format_step_metrics, run_pipeline
from .rollback import RollbackRefused, rollback_run

log = logging.getLogger("editor")


def main() -> None:
    parser = argparse.ArgumentParser(description="Dual-Memory Editor Agent")
    parser.add_argument("--once", action="store_true", help="Run one cycle and exit")
    parser.add_argument("--all", action="store_true", help="Process ALL raw memories (loop until none remain)")
    parser.add_argument("--step", type=str, help="Run only this step (dedup, classify, categories, ...)")
    parser.add_argument("--stats", action="store_true", help="Show graph stats and exit")
    parser.add_argument("--runs", action="store_true", help="List recent Editor runs and exit")
    parser.add_argument("--rollback", metavar="RUN_ID", help="Undo an Editor run and exit")
    parser.add_argument("--force", action="store_true", help="With --rollback: allow undoing a run later runs built on")
    parser.add_argument("--verbose", "-v", action="store_true", help="Debug logging")
    args = parser.parse_args()
    if args.all and args.step:
        # A single step never marks memories reviewed, so the --all loop would
        # keep fetching the same raw batch forever.
        parser.error("--all cannot be combined with --step")

    # Logging setup
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    config = EditorConfig.from_env()
    db = EditorDB(config.neo4j_uri, config.neo4j_user, config.neo4j_password)

    try:
        if args.stats:
            _print_stats(db)
            return
        if args.runs:
            _print_runs(db)
            return
        if args.rollback:
            try:
                r = rollback_run(db, args.rollback, force=args.force)
            except RollbackRefused as e:
                parser.exit(1, f"Rollback refused: {e}\n")
            print(
                f"Rolled back run {r.run_id}: {r.undo_ops} changes restored, "
                f"{r.relationships_deleted} relationships and {r.nodes_deleted} nodes removed"
            )
            return

        # Build clients
        llm = None
        embeddings = None

        if config.anthropic_api_key:
            llm = LLMClient(config.anthropic_api_key, config.llm_model, config.llm_effort)
            log.info("LLM client ready (%s)", config.llm_model)
        else:
            log.warning("No ANTHROPIC_API_KEY — LLM-dependent steps will be skipped")

        if config.voyage_api_key:
            embeddings = EmbeddingsClient(config.voyage_api_key, config.embedding_model)
            log.info("Embeddings client ready (%s)", config.embedding_model)
        else:
            log.warning("No VOYAGE_API_KEY — embedding-dependent steps will be skipped")

        if args.once or args.step:
            if args.all:
                _run_all(db, llm, embeddings, config)
            else:
                result = run_pipeline(
                    db, llm, embeddings, config,
                    steps={args.step} if args.step else None,
                    batch_cap=config.batch_cap,
                )
                _print_result(result)
                _print_cost_table(result.step_metrics)
        else:
            _run_scheduler(db, llm, embeddings, config)
    finally:
        db.close()


def _run_all(
    db: EditorDB,
    llm: LLMClient | None,
    embeddings: EmbeddingsClient | None,
    config: EditorConfig,
) -> None:
    """Loop until no raw memories remain."""
    cycle = 0
    total_processed = 0
    metrics: list[StepMetrics] = []

    while True:
        cycle += 1
        raw_count = db.get_raw_count()
        if raw_count == 0:
            log.info("No more raw memories. Total processed: %d across %d cycles", total_processed, cycle - 1)
            break

        log.info("=== Cycle %d (%d raw remaining) ===", cycle, raw_count)
        result = run_pipeline(db, llm, embeddings, config, batch_cap=config.batch_cap)
        _print_result(result)
        metrics.extend(result.step_metrics)
        total_processed += result.batch_size

        if result.batch_size == 0:
            break

    if metrics:
        _print_cost_table(metrics)


def _run_scheduler(
    db: EditorDB,
    llm: LLMClient | None,
    embeddings: EmbeddingsClient | None,
    config: EditorConfig,
) -> None:
    """Run on schedule: nightly at config.nightly_time + threshold check every N minutes."""

    def _nightly_job() -> None:
        log.info("Nightly run triggered")
        _run_all(db, llm, embeddings, config)

    def _threshold_job() -> None:
        raw_count = db.get_raw_count()
        if raw_count >= config.raw_threshold:
            log.info("Threshold reached (%d >= %d), running pipeline", raw_count, config.raw_threshold)
            result = run_pipeline(db, llm, embeddings, config, batch_cap=config.batch_cap)
            _print_result(result)
            _print_cost_table(result.step_metrics)

    schedule.every().day.at(config.nightly_time).do(_nightly_job)
    schedule.every(config.check_interval_minutes).minutes.do(_threshold_job)

    log.info(
        "Scheduler started: nightly at %s, threshold check every %d min (>= %d raw)",
        config.nightly_time, config.check_interval_minutes, config.raw_threshold,
    )

    try:
        while True:
            schedule.run_pending()
            time.sleep(30)
    except KeyboardInterrupt:
        log.info("Scheduler stopped")


def _print_stats(db: EditorDB) -> None:
    stats = db.get_stats()
    print("\n--- Graph Stats ---")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print()


def _print_runs(db: EditorDB) -> None:
    print(f"\n{'run id':<38} {'started':<25} {'status':<12} {'batch':>5} {'actions':>7} {'cost':>8}")
    for r in db.fetch_runs():
        summary = json.loads(r["summary"] or "{}")
        cost = summary.get("costUsd")
        print(
            f"{r['id']:<38} {r['startedAt'][:25]:<25} {r['status']:<12} {r['batchSize'] or 0:>5} "
            f"{r['actions']:>7} {'n/a' if cost is None else f'${cost:.4f}':>8}"
        )
    print()


def _print_result(result: PipelineResult) -> None:
    print(f"\n--- Editor Run {result.run_id or ''} ({result.batch_size} memories, {result.duration_seconds:.1f}s) ---")

    if result.dedup:
        d = result.dedup
        print(f"  Dedup: {d.exact_dupes_archived} exact + {d.semantic_dupes_archived} semantic archived, {d.semantic_related} related, {d.llm_calls} LLM calls")

    if result.classify:
        c = result.classify
        print(f"  Classify: {c.memories_classified} memories, {c.entities_classified} entities, {c.llm_calls} LLM calls")

    if result.categories:
        cat = result.categories
        print(f"  Categories: {cat.categories_created} created, {cat.memories_assigned} assigned, {cat.llm_calls} LLM calls")

    if result.contradictions:
        con = result.contradictions
        print(f"  Contradictions: {con.candidates} candidate pairs, {con.superseded} superseded, {con.llm_calls} LLM calls")

    if result.entity_resolution:
        er = result.entity_resolution
        print(
            f"  Entity Resolution: {er.merged} merged ({er.name_matches} name, "
            f"{er.embedding_matches} embedding, {er.llm_matches} LLM), "
            f"{er.llm_distinct} judged distinct, {er.llm_calls} LLM calls"
        )

    if result.relationships:
        rel = result.relationships
        print(f"  Relationships: {rel.created} created, {rel.llm_calls} LLM calls")

    if result.entity_links:
        el = result.entity_links
        print(f"  Entity Links: {el.created} LINKED_TO created, {el.llm_calls} LLM calls")

    if result.confidence:
        conf = result.confidence
        print(f"  Confidence: {conf.boosted} boosted, {conf.decayed} decayed")

    print(f"  Reviewed: {result.reviewed_count}")
    print(f"  Edit actions: {result.edit_actions}")
    print()


def _print_cost_table(metrics: list[StepMetrics]) -> None:
    if metrics:
        print("--- Cost and latency per step ---")
        print(format_step_metrics(metrics))
        print()


if __name__ == "__main__":
    main()
