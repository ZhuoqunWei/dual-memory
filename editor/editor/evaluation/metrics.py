"""Scoring functions for the eval harness. Pure: no database, no network.

Vocabulary:
- cluster: a gold label for "the same fact". Duplicate facts share a cluster;
  every other fact is its own cluster (keyed by its fact id).
- chain: clusters in time order where each later one makes the earlier ones
  outdated (e.g. lives in Seattle -> lives in Portland).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import asdict, dataclass, field


@dataclass
class PRF:
    precision: float
    recall: float
    f1: float
    true_positives: int
    predicted: int
    gold: int

    def as_dict(self) -> dict:
        return asdict(self)


def prf(true_positives: int, predicted: int, gold: int) -> PRF:
    # With no predictions precision is vacuously 1.0; the counts make that visible.
    precision = true_positives / predicted if predicted else 1.0
    recall = true_positives / gold if gold else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return PRF(round(precision, 4), round(recall, 4), round(f1, 4), true_positives, predicted, gold)


def _pairs(n: int) -> int:
    return n * (n - 1) // 2


def pairwise_clustering(gold: Mapping[Hashable, Hashable], predicted: Mapping[Hashable, Hashable]) -> PRF:
    """Pairwise precision/recall of a predicted clustering against gold.

    A pair of items is predicted-together when they share a predicted label and
    gold-together when they share a gold label. Items missing from `predicted`
    count as singletons.
    """
    items = list(gold)
    gold_pairs = sum(_pairs(n) for n in Counter(gold[i] for i in items).values())
    pred_labels = {i: predicted.get(i, ("singleton", i)) for i in items}
    pred_pairs = sum(_pairs(n) for n in Counter(pred_labels.values()).values())
    both = sum(_pairs(n) for n in Counter((gold[i], pred_labels[i]) for i in items).values())
    return prf(both, pred_pairs, gold_pairs)


def follow(edges: Mapping[str, str], start: str) -> str:
    """Follow single-successor edges (CANONICAL, MERGED_INTO) to the end."""
    seen = {start}
    node = start
    while node in edges:
        node = edges[node]
        if node in seen:  # defensive: a cycle would otherwise loop forever
            break
        seen.add(node)
    return node


@dataclass
class Chain:
    """An attribute's values over time, oldest first.

    `older` facts are only true before the first update (e.g. "Chris Lee is
    User's teammate at Northwind" once User leaves Northwind); `newer` facts imply
    the latest value without stating it ("User is building a pipeline at
    Contoso"). Linking the old side to the new side is correct but only
    `sequence` elements count toward recall.
    """

    sequence: list[str]
    older: list[str] = field(default_factory=list)
    newer: list[str] = field(default_factory=list)


@dataclass
class ConflictEdge:
    source: str  # cluster
    target: str  # cluster
    kind: str  # "supersedes" (source is newer) or "contradicts" (undirected)


def conflict_metrics(
    chains: Sequence[Chain],
    edges: Sequence[ConflictEdge],
    ignore: set[frozenset[str]],
    superseded: Mapping[str, tuple[int, int]] | None = None,
) -> dict:
    """Score contradiction handling against gold update chains.

    Precision: predicted conflict edges linking an earlier and a later position
    of the same chain, over all predicted edges (pairs in `ignore` are neutral).
    Recall: outdated `sequence` elements linked to something later in their chain.
    Direction accuracy: supersedes-edges pointing from newer to older.

    `superseded` maps cluster -> (active memories, of which superseded) and adds
    state metrics: outdated facts fully superseded, current facts wrongly
    superseded, and superseded facts that belong to no chain.
    """
    positions: dict[str, list[tuple[int, int]]] = {}
    for ci, chain in enumerate(chains):
        last = len(chain.sequence) - 1
        for pi, cluster in enumerate(chain.sequence):
            positions.setdefault(cluster, []).append((ci, pi))
        for cluster in chain.older:
            positions.setdefault(cluster, []).append((ci, 0))
        for cluster in chain.newer:
            positions.setdefault(cluster, []).append((ci, last))
    outdated = {c for chain in chains for c in chain.sequence[:-1]}

    def placement(a: str, b: str) -> tuple[int, int] | None:
        """(position of a, position of b) in a chain holding both at different
        positions, or None."""
        for ca, pa in positions.get(a, []):
            for cb, pb in positions.get(b, []):
                if ca == cb and pa != pb:
                    return pa, pb
        return None

    correct = 0
    wrong: list[int] = []
    directed_total = 0
    directed_correct = 0
    detected: set[str] = set()
    for i, e in enumerate(edges):
        if frozenset((e.source, e.target)) in ignore:
            continue
        placed = placement(e.source, e.target)
        if placed is None:
            wrong.append(i)
            continue
        correct += 1
        older = e.source if placed[0] < placed[1] else e.target
        if older in outdated:
            detected.add(older)
        if e.kind == "supersedes":
            directed_total += 1
            directed_correct += int(placed[0] > placed[1])  # source (newer) sits later

    predicted = correct + len(wrong)
    precision = correct / predicted if predicted else 1.0
    recall = len(detected) / len(outdated) if outdated else 1.0
    result = {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(2 * precision * recall / (precision + recall), 4) if precision + recall else 0.0,
        "predicted_edges": predicted,
        "correct_edges": correct,
        "outdated_facts": len(outdated),
        "outdated_detected": len(detected),
        "direction_accuracy": round(directed_correct / directed_total, 4) if directed_total else None,
        "missed": sorted(outdated - detected),
        "wrong_edges": wrong,  # indices into `edges`
    }

    if superseded is not None:
        current = [chain.sequence[-1] for chain in chains]

        def fully(c: str) -> bool:
            active, sup = superseded.get(c, (0, 0))
            return active > 0 and sup == active

        result["state"] = {
            "outdated_superseded": round(sum(fully(c) for c in outdated) / len(outdated), 4) if outdated else None,
            "current_wrongly_superseded": (
                round(sum(superseded.get(c, (0, 0))[1] > 0 for c in current) / len(current), 4)
                if current else None
            ),
            "superseded_outside_chains": sum(1 for c, (_, n) in superseded.items() if n and c not in positions),
        }
    return result


@dataclass
class Question:
    id: str
    type: str
    answer: list[list[str]]  # groups of clusters; any cluster in a group satisfies it
    stale: list[str]


def retrieval_metrics(questions: Sequence[Question], ranked: Mapping[str, Sequence[str]], k: int = 5) -> dict:
    """Recall@k, MRR, and stale-first rate over ranked clusters per question.

    recall@k: share of a question's answer groups satisfied in the top k.
    MRR: reciprocal rank of the first result satisfying any group (0 if none).
    stale_first: for questions with outdated alternatives, how often an outdated
    fact outranks every current answer.
    """
    per_type: dict[str, list[tuple[float, float]]] = {}
    stale_questions = 0
    stale_first = 0
    rows = []
    for q in questions:
        results = list(ranked.get(q.id, []))
        answers = set().union(*q.answer)
        top = set(results[:k])
        recall = sum(1 for group in q.answer if top & set(group)) / len(q.answer)
        rr = next((1 / (i + 1) for i, c in enumerate(results) if c in answers), 0.0)
        per_type.setdefault(q.type, []).append((recall, rr))
        first_relevant = next((c for c in results if c in answers or c in q.stale), None)
        if q.stale:
            stale_questions += 1
            stale_first += int(first_relevant in q.stale)
        rows.append({"id": q.id, f"recall@{k}": round(recall, 4), "rr": round(rr, 4),
                     "stale_first": bool(q.stale) and first_relevant in q.stale})

    def mean(xs: list[float]) -> float:
        return round(sum(xs) / len(xs), 4) if xs else 0.0

    everything = [x for xs in per_type.values() for x in xs]
    return {
        f"recall@{k}": mean([r for r, _ in everything]),
        "mrr": mean([rr for _, rr in everything]),
        "stale_first_rate": round(stale_first / stale_questions, 4) if stale_questions else None,
        "by_type": {
            t: {f"recall@{k}": mean([r for r, _ in xs]), "mrr": mean([rr for _, rr in xs]), "n": len(xs)}
            for t, xs in sorted(per_type.items())
        },
        "questions": rows,
    }
