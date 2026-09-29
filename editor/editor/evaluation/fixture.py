"""Gold-labelled fixture: synthetic sessions of Writer-extracted facts.

The fixture stands in for the Writer's LLM extraction, so the harness measures
what happens after extraction: the Writer's write path, the Editor, retrieval.
Format is documented in eval/README.md.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .metrics import Chain, Question


@dataclass
class Entity:
    name: str
    type: str
    aliases: list[str]
    gold: str  # gold identity; names sharing it are the same real-world entity


@dataclass
class Fact:
    id: str
    session_id: str
    content: str
    kind: str
    entities: list[tuple[str, str]]  # (entity name, role)
    cluster: str
    confidence: float = 0.8
    salience: float = 0.7


@dataclass
class Session:
    id: str
    date: str
    channel: str
    summary: str
    facts: list[Fact] = field(default_factory=list)


@dataclass
class Fixture:
    entities: dict[str, Entity]
    sessions: list[Session]
    chains: dict[str, Chain]
    ignore_pairs: set[frozenset[str]]
    questions: list[Question]
    question_text: dict[str, str]

    @property
    def facts(self) -> list[Fact]:
        return [f for s in self.sessions for f in s.facts]

    @property
    def clusters(self) -> set[str]:
        return {f.cluster for f in self.facts}


class FixtureError(ValueError):
    pass


def load_fixture(path: Path) -> Fixture:
    raw = json.loads(path.read_text())

    entities = {
        name: Entity(name, e["type"], e.get("aliases", []), e.get("gold", name))
        for name, e in raw["entities"].items()
    }

    sessions = []
    for s in raw["sessions"]:
        session = Session(s["id"], s["date"], s.get("channel", "discord"), s.get("summary", ""))
        for f in s["facts"]:
            session.facts.append(Fact(
                id=f["id"],
                session_id=s["id"],
                content=f["content"],
                kind=f["kind"],
                entities=[(name, role) for name, role in f["entities"]],
                cluster=f.get("cluster", f["id"]),
                confidence=f.get("confidence", 0.8),
                salience=f.get("salience", 0.7),
            ))
        sessions.append(session)

    fixture = Fixture(
        entities=entities,
        sessions=sessions,
        chains={
            c["id"]: Chain(c["sequence"], c.get("older", []), c.get("newer", []))
            for c in raw.get("chains", [])
        },
        ignore_pairs={frozenset(p) for p in raw.get("ignore_pairs", [])},
        questions=[
            Question(q["id"], q["type"], q["answer"], q.get("stale", []))
            for q in raw["questions"]
        ],
        question_text={q["id"]: q["text"] for q in raw["questions"]},
    )
    validate(fixture)
    return fixture


def validate(fx: Fixture) -> None:
    """Catch labelling mistakes before they turn into silently wrong metrics."""
    problems: list[str] = []

    ids = [f.id for f in fx.facts]
    if len(ids) != len(set(ids)):
        problems.append("duplicate fact ids")
    dates = [s.date for s in fx.sessions]
    if dates != sorted(dates):
        problems.append("sessions are not in date order")

    for f in fx.facts:
        if f.kind not in {"fact", "decision", "preference", "goal", "emotion", "observation", "event"}:
            problems.append(f"{f.id}: unknown kind {f.kind!r}")
        if not any(role == "subject" for _, role in f.entities):
            problems.append(f"{f.id}: no subject entity")
        for name, _ in f.entities:
            if name not in fx.entities:
                problems.append(f"{f.id}: entity {name!r} not in entity table")

    clusters = fx.clusters
    clustered_ids = {f.id for f in fx.facts if f.cluster != f.id}

    def check_ref(ref: str, where: str) -> None:
        if ref in clustered_ids:
            problems.append(f"{where}: {ref!r} belongs to a cluster; reference the cluster instead")
        elif ref not in clusters:
            problems.append(f"{where}: unknown cluster {ref!r}")

    for chain_id, chain in fx.chains.items():
        if len(chain.sequence) < 2:
            problems.append(f"chain {chain_id}: needs at least two elements")
        for ref in [*chain.sequence, *chain.older, *chain.newer]:
            check_ref(ref, f"chain {chain_id}")
    for pair in fx.ignore_pairs:
        for ref in pair:
            check_ref(ref, f"ignore pair {sorted(pair)}")
    for q in fx.questions:
        for group in q.answer:
            for ref in group:
                check_ref(ref, f"question {q.id}")
        for ref in q.stale:
            check_ref(ref, f"question {q.id} stale")

    if problems:
        raise FixtureError("fixture problems:\n  " + "\n  ".join(problems))
