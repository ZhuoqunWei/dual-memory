"""Run the eval harness.

    python -m editor.evaluation              # replay recorded LLM/embedding calls (no keys)
    python -m editor.evaluation --record     # call the APIs for anything not yet recorded
    python -m editor.evaluation --check      # exit 1 if a metric misses eval/thresholds.json

Needs a scratch Neo4j (eval/docker-compose.yml) at EVAL_NEO4J_URI and
`npm ci` in extensions/dual-memory. See eval/README.md.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

from ..constants import DEFAULT_LLM_EFFORT, DEFAULT_LLM_MODEL
from ..db import EditorDB
from ..pipeline import STEP_ORDER, StepMetrics, run_pipeline, summarize_step_metrics
from . import EVAL_DIR
from .bridge import run_bridge
from .cache import CachedEmbeddingsClient, CachedLLMClient, RecordingCache
from .fixture import Fixture, load_fixture
from .graph import EvalGraph
from .loading import VECTOR_DIMS, embed_all, load_payload
from .metrics import (
    ConflictEdge,
    conflict_metrics,
    follow,
    pairwise_clustering,
    retrieval_metrics,
)

log = logging.getLogger("editor.evaluation")

EMBEDDING_MODEL = "voyage-4-lite"
RETRIEVE_LIMIT = 10
MAX_CYCLES = 20


def main() -> None:
    parser = argparse.ArgumentParser(description="Dual-memory eval harness")
    parser.add_argument("--record", action="store_true", help="Call the APIs for requests missing from eval/cache")
    parser.add_argument("--check", action="store_true", help="Fail if metrics miss eval/thresholds.json")
    parser.add_argument("--prune", action="store_true", help="Drop cache entries this run did not use")
    parser.add_argument("--model", default=DEFAULT_LLM_MODEL, help="Editor LLM model")
    parser.add_argument("--effort", default=DEFAULT_LLM_EFFORT, help="Editor LLM effort")
    parser.add_argument("--skip", nargs="*", default=[], choices=STEP_ORDER, help="Editor steps to skip")
    parser.add_argument("--label", default="latest", help="Name for this run in the report")
    parser.add_argument("--out", type=Path, default=EVAL_DIR / "results" / "latest.json")
    parser.add_argument("--fixture", type=Path, default=EVAL_DIR / "fixture.json")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Neo4j reports harmless schema notices (e.g. unused labels) as warnings.
    logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)

    report = run(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(summary(report))
    print(f"\nFull report: {args.out}")

    if args.check:
        failures = check(report, json.loads((EVAL_DIR / "thresholds.json").read_text()))
        if failures:
            print("\nThreshold check FAILED:\n  " + "\n  ".join(failures))
            sys.exit(1)
        print("\nThreshold check passed.")


def run(args: argparse.Namespace) -> dict:
    started = time.monotonic()
    fx = load_fixture(args.fixture)
    neo4j_env = {
        "NEO4J_URI": os.environ.get("EVAL_NEO4J_URI", "bolt://localhost:7688"),
        "NEO4J_USER": os.environ.get("EVAL_NEO4J_USER", "neo4j"),
        "NEO4J_PASSWORD": os.environ.get("EVAL_NEO4J_PASSWORD", "dualmemory-eval"),
    }
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "") if args.record else ""
    voyage_key = os.environ.get("VOYAGE_API_KEY", "") if args.record else ""
    if args.record and not (anthropic_key and voyage_key):
        sys.exit("--record needs ANTHROPIC_API_KEY and VOYAGE_API_KEY")

    llm_cache = RecordingCache(EVAL_DIR / "cache" / "llm.jsonl", args.record)
    emb_cache = RecordingCache(EVAL_DIR / "cache" / "embeddings.jsonl", args.record)

    graph = EvalGraph(neo4j_env["NEO4J_URI"], neo4j_env["NEO4J_USER"], neo4j_env["NEO4J_PASSWORD"])
    try:
        graph.reset()

        # Embed facts and questions once, up front, like the Writer would.
        harness_emb = CachedEmbeddingsClient(emb_cache, voyage_key, EMBEDDING_MODEL)
        fact_vec = dict(zip([f.id for f in fx.facts], embed_all(harness_emb, [f.content for f in fx.facts])))
        question_texts = [fx.question_text[q.id] for q in fx.questions]
        question_vec = dict(zip([q.id for q in fx.questions], embed_all(harness_emb, question_texts)))

        # Write every session through the real Writer path.
        loaded = run_bridge("load", load_payload(fx, fact_vec), neo4j_env)
        fact_memory: dict[str, str] = {}
        skipped: list[str] = []
        for session, out in zip(fx.sessions, loaded["sessions"]):
            for fact, outcome in zip(session.facts, out["outcomes"]):
                fact_memory[fact.id] = outcome["memoryId"]
                if outcome["status"] == "skipped_similar":
                    skipped.append(fact.id)

        memory_fact = {mid: m["sourceRef"] for mid, m in graph.memories().items()}
        cluster_of = {f.id: f.cluster for f in fx.facts}
        relationship_pairs = graph.relationship_candidate_pairs()

        def ranked_clusters() -> dict[str, list[str]]:
            out = run_bridge("retrieve", {
                "vectorDims": VECTOR_DIMS,
                "limit": RETRIEVE_LIMIT,
                "queries": [{"id": qid, "vector": vec} for qid, vec in question_vec.items()],
            }, neo4j_env)
            return {
                r["id"]: [cluster_of[memory_fact[h["memoryId"]]] for h in r["hits"]]
                for r in out["results"]
            }

        ranked_before = ranked_clusters()

        # Run the Editor until every memory has been reviewed, as `--once --all` does.
        db = EditorDB(neo4j_env["NEO4J_URI"], neo4j_env["NEO4J_USER"], neo4j_env["NEO4J_PASSWORD"])
        llm = CachedLLMClient(llm_cache, anthropic_key, args.model, args.effort)
        editor_emb = CachedEmbeddingsClient(emb_cache, voyage_key, EMBEDDING_MODEL)
        steps = [s for s in STEP_ORDER if s not in args.skip]
        step_metrics: list[StepMetrics] = []
        cycles = 0
        editor_started = time.monotonic()
        try:
            while graph.raw_count() and cycles < MAX_CYCLES:
                result = run_pipeline(
                    db, llm, editor_emb, None,
                    steps=steps, batch_cap=50, mark_reviewed=True,
                )
                step_metrics.extend(result.step_metrics)
                cycles += 1
        finally:
            db.close()
        editor_seconds = time.monotonic() - editor_started

        ranked_after = ranked_clusters()

        report = {
            "label": args.label,
            "model": args.model,
            "effort": args.effort,
            "embedding_model": EMBEDDING_MODEL,
            "steps": steps,
            "fixture": _fixture_stats(fx),
            "writer": {"facts": len(fx.facts), "written": len(fx.facts) - len(skipped), "skipped_similar": len(skipped)},
            "dedup": _score_dedup(fx, graph, fact_memory, memory_fact),
            "conflicts": _score_conflicts(fx, graph, memory_fact),
            "entities": _score_entities(fx, graph),
            "retrieval": {
                "before_editor": retrieval_metrics(fx.questions, ranked_before),
                "after_editor": retrieval_metrics(fx.questions, ranked_after),
            },
            "cost": _cost_report(step_metrics, cycles, editor_seconds, relationship_pairs),
            "harness_seconds": round(time.monotonic() - started, 2),
        }
    finally:
        graph.close()

    if args.prune:
        log.warning("pruned %d LLM and %d embedding cache entries", llm_cache.prune(), emb_cache.prune())
    return report


def _cost_report(metrics: list[StepMetrics], cycles: int, seconds: float, relationship_pairs: int) -> dict:
    rows = [
        {k: (round(v, 6) if isinstance(v, float) else v) for k, v in asdict(m).items()}
        for m in summarize_step_metrics(metrics)
    ]
    return {
        "cycles": cycles,
        "editor_wall_seconds": round(seconds, 2),
        "relationship_candidate_pairs": relationship_pairs,
        "steps": rows,
        "total": rows[-1] if rows else {},
    }


def _fixture_stats(fx: Fixture) -> dict:
    return {
        "sessions": len(fx.sessions),
        "facts": len(fx.facts),
        "gold_unique_facts": len(fx.clusters),
        "chains": len(fx.chains),
        "outdated_facts": sum(len(c.sequence) - 1 for c in fx.chains.values()),
        "entity_names": len(fx.entities),
        "gold_entities": len({e.gold for e in fx.entities.values()}),
        "questions": len(fx.questions),
    }


def _score_dedup(fx: Fixture, graph: EvalGraph, fact_memory: dict[str, str], memory_fact: dict[str, str]) -> dict:
    canonical = graph.canonical_edges()
    rep = {f.id: follow(canonical, fact_memory[f.id]) for f in fx.facts}
    scores = pairwise_clustering({f.id: f.cluster for f in fx.facts}, rep).as_dict()
    content = {f.id: f.content for f in fx.facts}
    cluster_of = {f.id: f.cluster for f in fx.facts}

    # Wrong merges: memories collapsed into a representative from another cluster.
    wrong = sorted({
        (content[fid], content[memory_fact[r]])
        for fid, r in rep.items()
        if cluster_of[fid] != cluster_of[memory_fact[r]]
    })
    missed = []
    by_cluster: dict[str, set[str]] = {}
    for fid, r in rep.items():
        by_cluster.setdefault(cluster_of[fid], set()).add(r)
    for cluster, reps in sorted(by_cluster.items()):
        if len(reps) > 1:
            missed.append([content[memory_fact[r]] for r in sorted(reps, key=lambda r: memory_fact[r])])
    return {
        **scores,
        "archived_by_editor": len(canonical),
        "surviving_memories": len(set(rep.values())),
        "wrong_merges": [{"fact": a, "merged_into": b} for a, b in wrong],
        "missed_duplicates": missed,
    }


def _score_conflicts(fx: Fixture, graph: EvalGraph, memory_fact: dict[str, str]) -> dict:
    cluster_of = {f.id: f.cluster for f in fx.facts}
    content = {f.cluster: f.content for f in fx.facts}
    edges = [
        ConflictEdge(cluster_of[memory_fact[s]], cluster_of[memory_fact[t]], kind)
        for s, t, kind in graph.conflict_edges()
    ]
    superseded: dict[str, tuple[int, int]] = {}
    for m in graph.memories().values():
        if m["status"] == "archived":
            continue
        c = cluster_of[m["sourceRef"]]
        active, sup = superseded.get(c, (0, 0))
        superseded[c] = (active + 1, sup + int(m["validTo"] is not None))
    scores = conflict_metrics(list(fx.chains.values()), edges, fx.ignore_pairs, superseded)

    scores["wrong_edges"] = [
        {"a": content[edges[i].source], "b": content[edges[i].target], "kind": edges[i].kind}
        for i in scores["wrong_edges"]
    ]
    scores["missed"] = [content[c] for c in scores["missed"]]
    return scores


def _score_entities(fx: Fixture, graph: EvalGraph) -> dict:
    names = graph.entities()
    merges = graph.merge_edges()
    gold = {eid: fx.entities[name].gold for eid, name in names.items()}
    root = {eid: follow(merges, eid) for eid in names}
    scores = pairwise_clustering(gold, root).as_dict()
    wrong = [
        {"merged": names[a], "into": names[b]}
        for a, b in sorted(merges.items(), key=lambda kv: names[kv[0]])
        if gold[a] != gold[b]
    ]
    groups: dict[str, set[str]] = {}
    for eid, g in gold.items():
        groups.setdefault(g, set()).add(root[eid])
    missed = sorted(
        sorted(names[eid] for eid in names if gold[eid] == g)
        for g, roots in groups.items() if len(roots) > 1
    )
    return {
        **scores,
        "merges": len(merges),
        "wrong_merges": len(wrong),
        "wrong_merge_rate": round(len(wrong) / len(merges), 4) if merges else 0.0,
        "wrong_merge_examples": wrong,
        "missed_merges": missed,
    }


def _get(report: dict, dotted: str):
    node = report
    for part in dotted.split("."):
        node = node[part]
    return node


def check(report: dict, thresholds: dict) -> list[str]:
    failures = []
    for path, floor in thresholds.get("min", {}).items():
        value = _get(report, path)
        if value is None or value < floor:
            failures.append(f"{path} = {value} (min {floor})")
    for path, ceiling in thresholds.get("max", {}).items():
        value = _get(report, path)
        if value is None or value > ceiling:
            failures.append(f"{path} = {value} (max {ceiling})")
    return failures


def summary(r: dict) -> str:
    d, c, e = r["dedup"], r["conflicts"], r["entities"]
    before, after = r["retrieval"]["before_editor"], r["retrieval"]["after_editor"]
    total = r["cost"]["total"]
    cost = total.get("cost_usd")
    state = c.get("state", {})

    def pct(x: float | None) -> str:
        return "n/a" if x is None else f"{x:.1%}"

    lines = [
        f"## Eval: {r['label']} ({r['model']}, effort {r['effort']})",
        "",
        (
            f"Fixture: {r['fixture']['sessions']} sessions, {r['fixture']['facts']} facts "
            f"({r['fixture']['gold_unique_facts']} unique), {r['fixture']['chains']} update chains, "
            f"{r['fixture']['entity_names']} entity names ({r['fixture']['gold_entities']} real), "
            f"{r['fixture']['questions']} questions"
        ),
        "",
        "| Area | Metric | Value |",
        "|---|---|---|",
        f"| Dedup | precision / recall / F1 | {pct(d['precision'])} / {pct(d['recall'])} / {pct(d['f1'])} |",
        f"| Dedup | Writer skipped / Editor archived | {r['writer']['skipped_similar']} / {d['archived_by_editor']} |",
        f"| Contradictions | precision / recall / F1 | {pct(c['precision'])} / {pct(c['recall'])} / {pct(c['f1'])} |",
        f"| Contradictions | outdated facts superseded | {pct(state.get('outdated_superseded'))} |",
        f"| Contradictions | current facts wrongly superseded | {pct(state.get('current_wrongly_superseded'))} |",
        f"| Entities | pairwise precision / recall | {pct(e['precision'])} / {pct(e['recall'])} |",
        f"| Entities | wrong-merge rate | {pct(e['wrong_merge_rate'])} ({e['wrong_merges']}/{e['merges']}) |",
        f"| Retrieval | recall@5 before → after Editor | {pct(before['recall@5'])} → {pct(after['recall@5'])} |",
        f"| Retrieval | MRR before → after Editor | {before['mrr']:.3f} → {after['mrr']:.3f} |",
        f"| Retrieval | outdated fact ranked first | {pct(before['stale_first_rate'])} → {pct(after['stale_first_rate'])} |",
        (
            f"| Cost | Editor LLM calls / tokens in+out | {total.get('llm_calls', 0)} / "
            f"{total.get('input_tokens', 0)}+{total.get('output_tokens', 0)} |"
        ),
        f"| Cost | Editor cost per fixture run | {'n/a' if cost is None else f'${cost:.4f}'} |",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
