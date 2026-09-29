"""Step 5: Entity resolution — merge names that refer to the same entity.

Three tiers, cheapest first (the same cascade as dedup):
1. Names. Equal after normalizing case, punctuation, and word order ("Wang Wei"
   / "Wei Wang"), or each lists the other's name as an alias → merge, if the
   types agree.
2. Embeddings of name + type + aliases. Cosine at or above
   ENTITY_AUTO_MERGE_COSINE → merge; below ENTITY_CANDIDATE_COSINE → leave.
   Context stays out of this embedding: with mentioning memories included, two
   names from the same memory ("Framework Laptop" / "Fedora") scored higher than
   any true alias pair on the eval fixture.
3. The LLM, shown each name's mentioning memories, decides the band in between
   plus any pair sharing a name or alias string that tier 1 didn't settle (two
   people called "Chris", "Jordan" the person and the country). "different"
   verdicts are stored as DISTINCT_FROM so they aren't asked again.

Names mentioned together in one memory are never merged by tiers 2 and 3: a
fact doesn't name one entity twice.

Merges go through a union-find so each group collapses into one canonical
entity (the most-mentioned) rather than a chain. "different" verdicts act as
cannot-link constraints: if the LLM's "same" verdicts would put two entities
judged different into one group (a bare "Mei" matched to both "Mei Ortiz" and
"Mei Lin"), none of that group's LLM merges are applied. Names mentioned
together in one memory are cannot-link too, so an LLM slip can't chain them
("VS Code" = "nvim" plus "nvim" = "Neovim" would merge the editors the user
switched between).
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ..constants import (
    ENTITY_AUTO_MERGE_COSINE,
    ENTITY_CANDIDATE_COSINE,
    ENTITY_MAX_LLM_PAIRS,
    LLM_ENTITY_BATCH,
)
from ..llm import verdicts_by_pair

if TYPE_CHECKING:
    from ..audit import AuditLog
    from ..db import EditorDB
    from ..embeddings import EmbeddingsClient
    from ..llm import LLMClient

log = logging.getLogger("editor.steps.entity_resolution")


@dataclass
class EntityResolutionResult:
    merged: int = 0
    name_matches: int = 0
    embedding_matches: int = 0
    llm_matches: int = 0
    llm_distinct: int = 0
    llm_conflicts: int = 0  # "same" verdicts dropped for contradicting a "different"
    llm_calls: int = 0


def run_entity_resolution(
    db: EditorDB,
    audit: AuditLog,
    llm: LLMClient | None = None,
    embeddings: EmbeddingsClient | None = None,
) -> EntityResolutionResult:
    result = EntityResolutionResult()

    entities = db.fetch_entities_for_resolution()
    if len(entities) < 2:
        return result
    groups = UnionFind(len(entities))
    reasons: dict[frozenset[int], str] = {}
    co_mentioned = db.fetch_co_mentioned_entity_pairs()

    def mentioned_together(i: int, j: int) -> bool:
        return frozenset((entities[i]["id"], entities[j]["id"])) in co_mentioned

    # --- Tier 1: names ---
    ambiguous: list[tuple[int, int]] = []
    strings = [name_strings(e) for e in entities]
    for i in range(len(entities)):
        for j in range(i + 1, len(entities)):
            if not strings[i] & strings[j]:
                continue
            a, b = entities[i], entities[j]
            if names_match(a, b) and types_agree(a, b):
                groups.union(i, j)
                reasons[frozenset((i, j))] = f"name match: {a['name']!r} / {b['name']!r}"
                result.name_matches += 1
            elif not mentioned_together(i, j):
                ambiguous.append((i, j))

    # --- Tier 2: descriptor embeddings ---
    uncertain: list[tuple[int, int, float]] = []
    if embeddings is not None:
        sims = _descriptor_similarities(entities, db, embeddings)
        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                cosine = float(sims[i, j])
                if cosine < ENTITY_CANDIDATE_COSINE or groups.same(i, j):
                    continue
                if not types_agree(entities[i], entities[j]) or mentioned_together(i, j):
                    continue
                if cosine >= ENTITY_AUTO_MERGE_COSINE:
                    groups.union(i, j)
                    reasons[frozenset((i, j))] = f"embedding match (cosine={cosine:.3f})"
                    result.embedding_matches += 1
                else:
                    uncertain.append((i, j, cosine))
        ambiguous = [(i, j, float(sims[i, j])) for i, j in ambiguous]
    else:
        ambiguous = [(i, j, 1.0) for i, j in ambiguous]

    # --- Tier 3: LLM for shared-name pairs and the uncertain band ---
    if llm is not None:
        distinct = db.fetch_distinct_pairs()
        llm_same: list[tuple[int, int, str]] = []
        index = {e["id"]: k for k, e in enumerate(entities)}
        judged_different: set[frozenset[int]] = {
            frozenset(index[x] for x in pair)
            for pair in distinct | co_mentioned if pair <= index.keys()
        }
        seen: set[frozenset[int]] = set()
        todo: list[tuple[int, int]] = []
        for i, j, _ in sorted([*ambiguous, *uncertain], key=lambda p: -p[2]):
            key = frozenset((i, j))
            if key in seen or groups.same(i, j):
                continue
            seen.add(key)
            if frozenset((entities[i]["id"], entities[j]["id"])) in distinct:
                continue
            todo.append((i, j))
        todo = todo[:ENTITY_MAX_LLM_PAIRS]

        for start in range(0, len(todo), LLM_ENTITY_BATCH):
            chunk = todo[start : start + LLM_ENTITY_BATCH]
            decisions = _llm_decisions(llm, [(entities[i], entities[j]) for i, j in chunk])
            result.llm_calls += 1
            for (i, j), d in zip(chunk, decisions):
                verdict = d.get("decision")
                reason = d.get("reason", "")
                if verdict == "same":
                    llm_same.append((i, j, reason))
                elif verdict == "different":
                    db.mark_distinct(entities[i]["id"], entities[j]["id"], reason)
                    judged_different.add(frozenset((i, j)))
                    result.llm_distinct += 1

        for i, j, reason in consistent_links(llm_same, judged_different, groups):
            groups.union(i, j)
            reasons[frozenset((i, j))] = f"LLM: {reason}"
            result.llm_matches += 1
        result.llm_conflicts = len(llm_same) - result.llm_matches

    # --- Apply: each group merges into its most-mentioned member ---
    for members in groups.groups():
        if len(members) < 2:
            continue
        target = min(members, key=lambda k: (-entities[k]["mentions"], entities[k]["name"]))
        why = "; ".join(r for pair, r in reasons.items() if pair <= set(members))
        for k in sorted(members):
            if k == target:
                continue
            db.merge_entities(entities[k]["id"], entities[target]["id"])
            result.merged += 1
            audit.log(
                "merge",
                [entities[k]["id"], entities[target]["id"]],
                f"Merged entity '{entities[k]['name']}' into '{entities[target]['name']}' ({why})",
            )

    log.info(
        "Entity resolution: %d merged (%d name, %d embedding, %d LLM), %d judged distinct, %d LLM calls",
        result.merged, result.name_matches, result.embedding_matches, result.llm_matches,
        result.llm_distinct, result.llm_calls,
    )
    return result


def consistent_links(
    links: list[tuple[int, int, str]],
    different: set[frozenset[int]],
    groups: UnionFind,
) -> list[tuple[int, int, str]]:
    """LLM "same" links whose resulting group has no pair judged different.

    Links are grouped into the components they would form on top of the
    existing groups; a component that would newly join a judged-different pair
    keeps none of its links, so the outcome doesn't depend on verdict order.
    """
    trial = groups.copy()
    for i, j, _ in links:
        trial.union(i, j)
    blocked = set()
    for pair in different:
        a, b = tuple(pair)
        if trial.same(a, b) and not groups.same(a, b):
            blocked.add(trial.find(a))
    return [(i, j, r) for i, j, r in links if trial.find(i) not in blocked]


def normalize_name(name: str) -> str:
    """Case-, punctuation-, and word-order-insensitive form of a name."""
    text = unicodedata.normalize("NFKC", name).casefold()
    return " ".join(sorted(re.sub(r"[^\w\s]", " ", text).split()))


def name_strings(entity: dict) -> set[str]:
    return {normalize_name(n) for n in [entity["name"], *(entity.get("aliases") or [])]} - {""}


def names_match(a: dict, b: dict) -> bool:
    """Same normalized name, or each entity lists the other's name as an alias."""
    na, nb = normalize_name(a["name"]), normalize_name(b["name"])
    if na == nb:
        return True
    aliases_a = {normalize_name(x) for x in a.get("aliases") or []}
    aliases_b = {normalize_name(x) for x in b.get("aliases") or []}
    return na in aliases_b and nb in aliases_a


def types_agree(a: dict, b: dict) -> bool:
    return not a.get("type") or not b.get("type") or a["type"] == b["type"]


def descriptor(entity: dict) -> str:
    """What the entity embedding encodes: name, type, and aliases."""
    text = f"{entity['name']} ({entity.get('type') or 'unknown type'})"
    if entity.get("aliases"):
        text += ". Also called: " + ", ".join(entity["aliases"])
    return text


def _descriptor_similarities(entities: list[dict], db: EditorDB, embeddings: EmbeddingsClient) -> np.ndarray:
    """Cosine matrix of descriptor embeddings, re-embedding entities whose
    descriptor changed since their stored embedding."""
    texts = [descriptor(e) for e in entities]
    stale = [k for k, e in enumerate(entities) if e.get("embeddingText") != texts[k] or e.get("embedding") is None]
    for start in range(0, len(stale), 64):
        chunk = stale[start : start + 64]
        for k, vector in zip(chunk, embeddings.embed_batch([texts[k] for k in chunk])):
            entities[k]["embedding"] = vector
            db.set_entity_embedding(entities[k]["id"], vector, texts[k])
    matrix = np.array([e["embedding"] for e in entities], dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / np.where(norms == 0, 1, norms)
    return matrix @ matrix.T


def _describe(label: str, e: dict) -> str:
    line = f"  {label}: {e['name']} ({e.get('type') or 'unknown type'})"
    if e.get("aliases"):
        line += ", also called: " + ", ".join(e["aliases"])
    if e.get("samples"):
        line += "\n     mentioned in: " + "; ".join(s[:120] for s in e["samples"])
    return line


def _llm_decisions(llm: LLMClient, pairs: list[tuple[dict, dict]]) -> list[dict]:
    items = "".join(
        f"\nPair {n}:\n{_describe('A', a)}\n{_describe('B', b)}\n" for n, (a, b) in enumerate(pairs, 1)
    )
    system = """Decide whether each pair of names refers to the same real-world entity in one person's memory graph.

- "same": two names for one entity: a nickname, abbreviation, different word order, or a first name used alone where the context matches.
- "different": distinct entities even though the names overlap, e.g. two people who share a first name, or a person and a country with the same name.
- "unsure": the evidence doesn't settle it. Unsure pairs are left unmerged.

Use the types and the memories that mention each name as evidence.

Return a JSON array with one object per pair:
[{"pair": 1, "decision": "same", "reason": "..."}]"""
    user = f"Review these {len(pairs)} name pairs:{items}"

    try:
        reply = llm.ask_json(system, user)
    except Exception as e:
        log.warning("LLM entity resolution failed: %s", e)
        reply = None
    return verdicts_by_pair(reply, len(pairs))


class UnionFind:
    def __init__(self, n: int) -> None:
        self._parent = list(range(n))

    def copy(self) -> UnionFind:
        other = UnionFind(0)
        other._parent = list(self._parent)
        return other

    def find(self, i: int) -> int:
        while self._parent[i] != i:
            self._parent[i] = self._parent[self._parent[i]]
            i = self._parent[i]
        return i

    def union(self, i: int, j: int) -> None:
        self._parent[self.find(i)] = self.find(j)

    def same(self, i: int, j: int) -> bool:
        return self.find(i) == self.find(j)

    def groups(self) -> list[list[int]]:
        out: dict[int, list[int]] = {}
        for i in range(len(self._parent)):
            out.setdefault(self.find(i), []).append(i)
        return list(out.values())
