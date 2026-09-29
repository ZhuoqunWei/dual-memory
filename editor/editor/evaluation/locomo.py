"""LoCoMo benchmark: an external number for the whole memory loop.

    python -m editor.evaluation.locomo --limit 1   # one conversation (smoke test)
    python -m editor.evaluation.locomo             # all ten

LoCoMo (Maharana et al., 2024) is ten long conversations between two people,
~27 dated sessions each, with ~200 questions per conversation. Each
conversation is treated as one user's history:

1. every session goes through the Writer's LLM extraction and writeFacts,
   stamped with the session's date;
2. the Editor runs over that conversation's graph;
3. each question goes through Retrieval.retrieveContext(), i.e. the exact
   <graph-memories> block the plugin would inject;
4. a fixed reader answers from that block alone, and an LLM judge grades the
   answer against the gold one (CORRECT/WRONG), plus token F1.

Category 5 (adversarial, premise-wrong questions) is excluded, as in prior
LoCoMo evaluations; categories 1-4 are reported with their usual names.

Needs ANTHROPIC_API_KEY and VOYAGE_API_KEY, the scratch Neo4j, and
eval/locomo/data/locomo10.json (github.com/snap-research/locomo). Every API
call is recorded under eval/locomo/cache, so an interrupted run resumes where
it stopped. The dataset and everything derived from it stay out of git; only
the summary (eval/results/locomo.json) is committed.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import string
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from ..constants import DEFAULT_LLM_EFFORT, DEFAULT_LLM_MODEL
from ..db import EditorDB
from ..llm import LLMUsage, cost_usd
from ..pipeline import StepMetrics, run_pipeline, summarize_step_metrics
from . import EVAL_DIR
from .bridge import run_bridge
from .cache import CachedEmbeddingsClient, CachedLLMClient, RecordingCache, request_key
from .graph import EvalGraph
from .loading import VECTOR_DIMS, embed_all

log = logging.getLogger("editor.evaluation.locomo")

LOCOMO_DIR = EVAL_DIR / "locomo"
DATA = LOCOMO_DIR / "data" / "locomo10.json"
CACHE = LOCOMO_DIR / "cache"

WRITER_MODEL = "claude-sonnet-5"  # the plugin's default extraction model
READER_MODEL, READER_EFFORT = "claude-sonnet-5", "low"
JUDGE_MODEL = "claude-haiku-4-5"
EMBEDDING_MODEL = "voyage-4-lite"
CATEGORIES = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop"}
MAX_CYCLES = 20

READER_SYSTEM = """You answer questions about a long conversation between two people, using only the memories retrieved from it.

Answer with a short phrase: a few words, a name, or a date. Use dates that appear in the memories when a question asks when something happened. If the memories don't contain the answer, answer "unknown".

Return JSON: {"answer": "..."}"""

JUDGE_SYSTEM = """You grade an answer to a question about a conversation against the gold answer.

CORRECT: the generated answer refers to the same thing as the gold answer, even if it is worded differently, shorter, or longer. For dates and times, the same date or period in any format counts.
WRONG: the answer is missing, says it doesn't know, or refers to something else.

Return JSON: {"label": "CORRECT"} or {"label": "WRONG"}"""


def main() -> None:
    parser = argparse.ArgumentParser(description="LoCoMo benchmark for dual-memory")
    parser.add_argument("--limit", type=int, default=None, help="Only the first N conversations")
    parser.add_argument("--workers", type=int, default=6, help="Parallel LLM calls for extraction, answering, judging")
    parser.add_argument("--out", default=str(EVAL_DIR / "results" / "locomo.json"))
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)

    if not DATA.exists():
        sys.exit(f"missing {DATA}; download locomo10.json from github.com/snap-research/locomo")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
    voyage_key = os.environ.get("VOYAGE_API_KEY", "")
    if not (anthropic_key and voyage_key):
        sys.exit("needs ANTHROPIC_API_KEY and VOYAGE_API_KEY")

    conversations = json.loads(DATA.read_text())[: args.limit]
    report = run(conversations, anthropic_key, voyage_key, args.workers)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(summary(report))
    print(f"\nFull report: {args.out}")


def run(conversations: list[dict], anthropic_key: str, voyage_key: str, workers: int) -> dict:
    started = time.monotonic()
    neo4j_env = {
        "NEO4J_URI": os.environ.get("EVAL_NEO4J_URI", "bolt://localhost:7688"),
        "NEO4J_USER": os.environ.get("EVAL_NEO4J_USER", "neo4j"),
        "NEO4J_PASSWORD": os.environ.get("EVAL_NEO4J_PASSWORD", "dualmemory-eval"),
    }
    extractions = RecordingCache(CACHE / "extractions.jsonl", record=True)
    emb_cache = RecordingCache(CACHE / "embeddings.jsonl", record=True)
    llm_cache = RecordingCache(CACHE / "llm.jsonl", record=True)
    embed = CachedEmbeddingsClient(emb_cache, voyage_key, EMBEDDING_MODEL)

    # 1. Writer extraction for every session, in parallel, recorded.
    sessions = [s for c in conversations for s in conversation_sessions(c)]
    writer_usage = extract_all(sessions, extractions, neo4j_env, workers)

    graph = EvalGraph(neo4j_env["NEO4J_URI"], neo4j_env["NEO4J_USER"], neo4j_env["NEO4J_PASSWORD"])
    editor_llm = CachedLLMClient(llm_cache, anthropic_key, DEFAULT_LLM_MODEL, DEFAULT_LLM_EFFORT)
    editor_emb = CachedEmbeddingsClient(emb_cache, voyage_key, EMBEDDING_MODEL)
    step_metrics: list[StepMetrics] = []
    questions: list[dict] = []
    memory_stats = []
    try:
        for conv in conversations:
            memory_stats.append(ingest_and_edit(conv, graph, extractions, embed, editor_llm, editor_emb,
                                                neo4j_env, step_metrics))
            questions.extend(recall_questions(conv, graph, embed, neo4j_env))
            print(f"  {conv['sample_id']}: {memory_stats[-1]}", flush=True)
    finally:
        graph.close()

    # 4. Answer from the recalled context, then judge, in parallel.
    reader = CachedLLMClient(llm_cache, anthropic_key, READER_MODEL, READER_EFFORT)
    judge = CachedLLMClient(llm_cache, anthropic_key, JUDGE_MODEL, "low")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        answers = list(pool.map(lambda q: answer(reader, q), questions))
        for q, a in zip(questions, answers):
            q["generated"] = a
        labels = list(pool.map(lambda q: grade(judge, q), questions))
    for q, label in zip(questions, labels):
        q["correct"] = label
        q["f1"] = token_f1(q["generated"], q["answer"])

    return {
        "benchmark": "LoCoMo (locomo10), categories 1-4",
        "conversations": len(conversations),
        "questions": len(questions),
        "writer_model": WRITER_MODEL,
        "editor_model": f"{DEFAULT_LLM_MODEL} ({DEFAULT_LLM_EFFORT})",
        "reader_model": f"{READER_MODEL} ({READER_EFFORT})",
        "judge_model": JUDGE_MODEL,
        "scores": scores(questions),
        "memories": memory_stats,
        "cost": {
            "writer": usage_row(WRITER_MODEL, writer_usage),
            "editor": {k: (round(v, 6) if isinstance(v, float) else v)
                       for k, v in vars(summarize_step_metrics(step_metrics)[-1]).items()},
            "reader": usage_row(READER_MODEL, reader.usage),
            "judge": usage_row(JUDGE_MODEL, judge.usage),
        },
        "harness_seconds": round(time.monotonic() - started, 1),
        "examples": sample_failures(questions),
    }


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------

def parse_session_date(text: str) -> str:
    """'1:56 pm on 8 May, 2023' -> '2023-05-08T13:56:00Z'."""
    return datetime.strptime(text.strip(), "%I:%M %p on %d %B, %Y").strftime("%Y-%m-%dT%H:%M:00Z")


def conversation_sessions(conv: dict) -> list[dict]:
    """The conversation's sessions in order, as Writer input."""
    c = conv["conversation"]
    numbers = sorted(int(m.group(1)) for k in c if (m := re.fullmatch(r"session_(\d+)", k)))
    out = []
    for n in numbers:
        messages = []
        for turn in c[f"session_{n}"]:
            text = f"{turn['speaker']}: {turn['text']}"
            if turn.get("blip_caption"):
                text += f" [shares a photo: {turn['blip_caption']}]"
            # Both people are "users" of this memory; names carry who said what.
            messages.append({"role": "user", "content": text})
        out.append({
            "id": f"{conv['sample_id']}:session_{n}",
            "number": n,
            "date": parse_session_date(c[f"session_{n}_date_time"]),
            "messages": messages,
        })
    return out


# ----------------------------------------------------------------------
# Pipeline stages
# ----------------------------------------------------------------------

def extraction_key(session: dict) -> str:
    return request_key("extract", WRITER_MODEL, json.dumps(session["messages"], ensure_ascii=False))


def extract_all(sessions: list[dict], cache: RecordingCache, neo4j_env: dict, workers: int) -> LLMUsage:
    missing = [s for s in sessions if cache.get(extraction_key(s)) is None]
    for start in range(0, len(missing), 24):  # record in chunks so a crash loses little
        chunk = missing[start : start + 24]
        out = run_bridge("extract", {
            "model": WRITER_MODEL,
            "concurrency": workers,
            "sessions": [{"id": s["id"], "messages": s["messages"]} for s in chunk],
        }, {**neo4j_env, "ANTHROPIC_API_KEY": os.environ["ANTHROPIC_API_KEY"]})
        for s, r in zip(chunk, out["results"]):
            cache.put({"key": extraction_key(s), "id": s["id"], **r})
        print(f"  extracted {min(start + 24, len(missing))}/{len(missing)} new sessions", flush=True)
    usage = LLMUsage()
    for s in sessions:
        u = cache.get(extraction_key(s))["usage"]
        usage.calls += 1
        usage.input_tokens += u["inputTokens"]
        usage.output_tokens += u["outputTokens"]
    return usage


def ingest_and_edit(conv, graph, extractions, embed, llm, emb, neo4j_env, step_metrics) -> dict:
    """Load one conversation through writeFacts and run the Editor over it."""
    graph.reset()
    payload_sessions = []
    for s in conversation_sessions(conv):
        ex = extractions.get(extraction_key(s))
        facts = [{**f, "sourceRef": f"{s['id']}:{i}"} for i, f in enumerate(ex["facts"])]
        payload_sessions.append({
            "key": s["id"],
            "date": s["date"],
            "summary": ex.get("sessionSummary") or "",
            "channel": "locomo",
            "facts": facts,
            "embeddings": embed_all(embed, [f["content"] for f in facts]) if facts else [],
        })
    written = run_bridge("load", {"vectorDims": VECTOR_DIMS, "sessions": payload_sessions}, neo4j_env)
    extracted = sum(len(s["facts"]) for s in payload_sessions)
    kept = sum(o["status"] == "written" for s in written["sessions"] for o in s["outcomes"])

    db = EditorDB(neo4j_env["NEO4J_URI"], neo4j_env["NEO4J_USER"], neo4j_env["NEO4J_PASSWORD"])
    try:
        cycles = 0
        while graph.raw_count() and cycles < MAX_CYCLES:
            step_metrics.extend(run_pipeline(db, llm, emb, None, batch_cap=50, mark_reviewed=True).step_metrics)
            cycles += 1
    finally:
        db.close()
    active = sum(m["status"] != "archived" for m in graph.memories().values())
    return {"sample_id": conv["sample_id"], "extracted": extracted, "written": kept, "active_after_editor": active}


def recall_questions(conv: dict, graph: EvalGraph, embed, neo4j_env: dict) -> list[dict]:
    """Run every scored question through the plugin's retrieval."""
    qs = [
        {"id": f"{conv['sample_id']}:q{i}", "sample_id": conv["sample_id"], "category": q["category"],
         "question": q["question"], "answer": str(q["answer"]), "evidence": q.get("evidence", [])}
        for i, q in enumerate(conv["qa"]) if q["category"] in CATEGORIES
    ]
    vectors = embed_all(embed, [q["question"] for q in qs])
    out = run_bridge("recall", {
        "vectorDims": VECTOR_DIMS,
        "queries": [{"id": q["id"], "text": q["question"], "vector": v} for q, v in zip(qs, vectors)],
    }, neo4j_env)
    session_of = {mid: m["sourceRef"].rsplit(":", 1)[0] for mid, m in graph.memories().items()}
    for q, r in zip(qs, out["results"]):
        q["context"] = r["context"] or ""
        evidence_sessions = {f"{conv['sample_id']}:session_{e.split(':')[0][1:]}" for e in q["evidence"]
                             if re.fullmatch(r"D\d+:\d+", e)}
        recalled = {session_of.get(mid) for mid in r["memoryIds"]}
        q["evidence_session_recalled"] = bool(evidence_sessions & recalled) if evidence_sessions else None
    return qs


def answer(reader: CachedLLMClient, q: dict) -> str:
    context = q["context"] or "(no memories were retrieved)"
    reply = reader.ask_json(READER_SYSTEM, f"{context}\n\nQuestion: {q['question']}")
    return str(reply.get("answer", "")) if isinstance(reply, dict) else ""


def grade(judge: CachedLLMClient, q: dict) -> bool:
    reply = judge.ask_json(
        JUDGE_SYSTEM,
        f"Question: {q['question']}\nGold answer: {q['answer']}\nGenerated answer: {q['generated']}",
    )
    return isinstance(reply, dict) and str(reply.get("label", "")).upper() == "CORRECT"


# ----------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------

def _tokens(text: str) -> list[str]:
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return [t for t in text.split() if t not in {"a", "an", "the", "and"}]


def token_f1(predicted: str, gold: str) -> float:
    p, g = _tokens(predicted), _tokens(gold)
    common = sum((Counter(p) & Counter(g)).values())
    if not p or not g or not common:
        return 0.0
    precision, recall = common / len(p), common / len(g)
    return 2 * precision * recall / (precision + recall)


def scores(questions: list[dict]) -> dict:
    def block(qs: list[dict]) -> dict:
        evid = [q["evidence_session_recalled"] for q in qs if q["evidence_session_recalled"] is not None]
        return {
            "n": len(qs),
            "judge_accuracy": round(sum(q["correct"] for q in qs) / len(qs), 4) if qs else None,
            "f1": round(sum(q["f1"] for q in qs) / len(qs), 4) if qs else None,
            "evidence_session_recall@8": round(sum(evid) / len(evid), 4) if evid else None,
            "no_context": round(sum(not q["context"] for q in qs) / len(qs), 4) if qs else None,
        }

    return {
        "overall": block(questions),
        "by_category": {name: block([q for q in questions if q["category"] == c]) for c, name in CATEGORIES.items()},
        "by_conversation": {
            sid: block([q for q in questions if q["sample_id"] == sid])["judge_accuracy"]
            for sid in dict.fromkeys(q["sample_id"] for q in questions)
        },
    }


def usage_row(model: str, usage: LLMUsage) -> dict:
    cost = cost_usd(model, usage.input_tokens, usage.output_tokens)
    return {"model": model, "calls": usage.calls, "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens, "cost_usd": None if cost is None else round(cost, 4)}


def sample_failures(questions: list[dict], per_category: int = 3) -> list[dict]:
    out = []
    for c in CATEGORIES:
        wrong = [q for q in questions if q["category"] == c and not q["correct"]][:per_category]
        out.extend({"category": CATEGORIES[c], "question": q["question"], "gold": q["answer"],
                    "generated": q["generated"], "context_chars": len(q["context"])} for q in wrong)
    return out


def summary(r: dict) -> str:
    s = r["scores"]
    lines = [
        f"## LoCoMo: {r['conversations']} conversations, {r['questions']} questions (categories 1-4)",
        "",
        "| Category | n | Judge accuracy | F1 | Evidence session recalled | No context |",
        "|---|---|---|---|---|---|",
    ]
    for name, b in [*s["by_category"].items(), ("overall", s["overall"])]:
        lines.append(
            f"| {name} | {b['n']} | {b['judge_accuracy']:.1%} | {b['f1']:.3f} | "
            f"{b['evidence_session_recall@8'] or 0:.1%} | {b['no_context']:.1%} |"
        )
    c = r["cost"]
    total = sum(x.get("cost_usd") or 0 for x in c.values())
    lines += [
        "",
        "Memories per conversation (extracted → written → after Editor): "
        + ", ".join(f"{m['extracted']}→{m['written']}→{m['active_after_editor']}" for m in r["memories"]),
        f"Cost: writer ${c['writer']['cost_usd']}, editor ${c['editor']['cost_usd']}, "
        f"reader ${c['reader']['cost_usd']}, judge ${c['judge']['cost_usd']} (total ${total:.2f})",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
