"""Neo4j driver wrapper with Editor-specific queries.

Mirrors the TypeScript neo4j-client.ts but adds Editor operations:
dedup (CANONICAL), classification (normalizedKind/Type), categories,
entity resolution (MERGED_INTO), confidence management, and audit logging.
"""

from __future__ import annotations

import json
import logging
import re
from uuid import uuid4

import neo4j

log = logging.getLogger("editor.db")


class EditorDB:
    def __init__(self, uri: str, user: str, password: str) -> None:
        self._driver = neo4j.GraphDatabase.driver(uri, auth=(user, password))
        # Set by start_run: the EditorRun that tags created relationships and
        # collects undo ops for property changes (see rollback.py).
        self.run_id: str | None = None
        self._undo: list[dict] = []

    def close(self) -> None:
        self._driver.close()

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def get_raw_count(self) -> int:
        with self._driver.session() as s:
            r = s.run("MATCH (m:Memory {status: 'raw'}) RETURN count(m) AS c")
            return r.single()["c"]

    def fetch_raw_memories(self, limit: int = 50) -> list[dict]:
        """Fetch oldest raw memories with embeddings, entities, and session info."""
        query = """
        MATCH (m:Memory {status: 'raw'})
        WITH m ORDER BY m.timestamp ASC, m.content ASC LIMIT $limit
        OPTIONAL MATCH (m)-[mention:MENTIONS]->(e:Entity)
        OPTIONAL MATCH (m)-[:PART_OF]->(s:Session)
        WITH m, s,
             collect(DISTINCT {id: e.id, name: e.name, type: e.type, role: mention.role}) AS entities
        RETURN m {.id, .content, .kind, .normalizedKind, .confidence, .salience,
                   .status, .timestamp, .eventTimeStart, .eventTimeEnd,
                   .sourceQuote, .sourceChannel, .sourceAuthor,
                   .embedding} AS memory,
               entities,
               s.id AS sessionId
        """
        with self._driver.session() as s:
            result = s.run(query, limit=limit)
            rows = []
            for record in result:
                mem = dict(record["memory"])
                mem["entities"] = [
                    e for e in record["entities"] if e["id"] is not None
                ]
                mem["sessionId"] = record["sessionId"]
                rows.append(mem)
            return rows

    def fetch_all_memory_embeddings(self) -> list[dict]:
        """Fetch (id, content, embedding) for all non-archived, non-canonical memories."""
        query = """
        MATCH (m:Memory)
        WHERE m.status IN ['raw', 'reviewed']
          AND m.embedding IS NOT NULL
        WITH m
        OPTIONAL MATCH (m)-[c:CANONICAL]->()
        WITH m, count(c) AS canonCount
        WHERE canonCount = 0
        RETURN m.id AS id, m.content AS content, m.embedding AS embedding,
               m.timestamp AS timestamp, m.kind AS kind
        ORDER BY timestamp, content
        """
        # Note: count(c) counts the relationship variable (0 when no match),
        # not count(*) which counts rows (always >= 1 with OPTIONAL MATCH)
        with self._driver.session() as s:
            return [dict(r) for r in s.run(query)]

    def fetch_all_entities(self) -> list[dict]:
        query = """
        MATCH (e:Entity)
        WHERE NOT (e)-[:MERGED_INTO]->()
        RETURN e {.id, .name, .type, .normalizedType, .aliases, .embedding} AS entity
        ORDER BY e.name
        """
        with self._driver.session() as s:
            return [dict(r["entity"]) for r in s.run(query)]

    def fetch_categories(self) -> list[dict]:
        query = """
        MATCH (c:Category)
        OPTIONAL MATCH (m:Memory)-[:IN_CATEGORY]->(c)
        WHERE m.status IN ['raw', 'reviewed']
        WITH c, m ORDER BY m.timestamp, m.content
        WITH c, collect(m.id) AS memberIds, collect(m.embedding) AS memberEmbeddings
        RETURN c.name AS name, c.description AS description,
               memberIds, memberEmbeddings
        ORDER BY name
        """
        with self._driver.session() as s:
            return [dict(r) for r in s.run(query)]

    def fetch_memories_by_shared_entity(self) -> dict[str, list[dict]]:
        """Group memories by entity — for contradiction detection and relationship creation."""
        query = """
        MATCH (m:Memory)-[:MENTIONS]->(e:Entity)
        WHERE m.status IN ['raw', 'reviewed']
        WITH e, m ORDER BY m.timestamp, m.content
        WITH e, collect({id: m.id, content: m.content, kind: m.kind, status: m.status,
                         confidence: m.confidence, sessionId: null}) AS memories
        WHERE size(memories) > 1
        RETURN e.name AS entityName, e.id AS entityId, memories
        ORDER BY entityName
        """
        with self._driver.session() as s:
            groups: dict[str, list[dict]] = {}
            for r in s.run(query):
                groups[r["entityName"]] = [dict(m) for m in r["memories"]]
            return groups

    def get_stats(self) -> dict:
        query = """
        MATCH (m:Memory) WITH count(m) AS total
        OPTIONAL MATCH (r:Memory {status: 'raw'}) WITH total, count(r) AS raw
        OPTIONAL MATCH (e:Entity) WITH total, raw, count(e) AS entities
        OPTIONAL MATCH (s:Session) WITH total, raw, entities, count(s) AS sessions
        OPTIONAL MATCH (c:Category) WITH total, raw, entities, sessions, count(c) AS categories
        OPTIONAL MATCH ()-[can:CANONICAL]->() WITH total, raw, entities, sessions, categories, count(can) AS canonicals
        OPTIONAL MATCH ()-[ea:RELATES_TO]->() WITH total, raw, entities, sessions, categories, canonicals, count(ea) AS relatesToEdges
        RETURN total AS memories, raw, entities, sessions, categories, canonicals, relatesToEdges
        """
        with self._driver.session() as s:
            r = s.run(query).single()
            return dict(r)

    # ------------------------------------------------------------------
    # Dedup
    # ------------------------------------------------------------------

    def create_canonical(self, dupe_id: str, canonical_id: str) -> None:
        """Mark dupe as superseded by canonical. Archives the dupe."""
        query = """
        MATCH (dupe:Memory {id: $dupeId}), (canon:Memory {id: $canonId})
        WITH dupe, canon, dupe.status AS prevStatus
        MERGE (dupe)-[c:CANONICAL]->(canon)
        ON CREATE SET c.editorRun = $run
        MERGE (dupe)-[r:RELATES_TO {type: 'editor:supersedes'}]->(canon)
        ON CREATE SET r.weight = 1.0, r.editorRun = $run
        SET dupe.status = 'archived'
        RETURN prevStatus
        """
        for r in self._write(query, dupeId=dupe_id, canonId=canonical_id):
            self._journal(_restore("Memory", dupe_id, status=r["prevStatus"]))

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def set_normalized_kind(self, memory_id: str, normalized_kind: str) -> None:
        query = """
        MATCH (m:Memory {id: $id}) WITH m, m.normalizedKind AS prev
        SET m.normalizedKind = $nk RETURN prev
        """
        for r in self._write(query, id=memory_id, nk=normalized_kind):
            self._journal(_restore("Memory", memory_id, normalizedKind=r["prev"]))

    def set_entity_normalized_type(self, entity_id: str, normalized_type: str) -> None:
        query = """
        MATCH (e:Entity {id: $id}) WITH e, e.normalizedType AS prev
        SET e.normalizedType = $nt RETURN prev
        """
        for r in self._write(query, id=entity_id, nt=normalized_type):
            self._journal(_restore("Entity", entity_id, normalizedType=r["prev"]))

    # ------------------------------------------------------------------
    # Categories
    # ------------------------------------------------------------------

    def create_category(self, name: str, description: str) -> str:
        query = """
        MERGE (c:Category {name: $name})
        ON CREATE SET c.description = $desc, c.editorRun = $run
        RETURN c.name AS name
        """
        return self._write(query, name=name, desc=description)[0]["name"]

    def assign_category(self, memory_id: str, category_name: str, score: float) -> None:
        query = """
        MATCH (m:Memory {id: $mid}), (c:Category {name: $cname})
        MERGE (m)-[r:IN_CATEGORY]->(c)
        ON CREATE SET r.score = $score, r.assignedBy = 'editor',
                      r.timestamp = datetime(), r.editorRun = $run
        """
        self._write(query, mid=memory_id, cname=category_name, score=score)

    # ------------------------------------------------------------------
    # Entity Resolution
    # ------------------------------------------------------------------

    def fetch_entities_for_resolution(self, samples: int = 3) -> list[dict]:
        """Unmerged entities with type, aliases, mention count, a few mentioning
        memories (oldest first), and their stored descriptor embedding."""
        query = """
        MATCH (e:Entity)
        WHERE NOT (e)-[:MERGED_INTO]->()
        OPTIONAL MATCH (m:Memory)-[:MENTIONS]->(e)
        WHERE m.status IN ['raw', 'reviewed']
        WITH e, m ORDER BY m.timestamp, m.content
        WITH e, collect(m.content) AS contents
        RETURN e.id AS id, e.name AS name, coalesce(e.normalizedType, e.type) AS type,
               coalesce(e.aliases, []) AS aliases, size(contents) AS mentions,
               contents[..$samples] AS samples,
               e.embedding AS embedding, e.embeddingText AS embeddingText
        ORDER BY name
        """
        with self._driver.session() as s:
            return [dict(r) for r in s.run(query, samples=samples)]

    def fetch_co_mentioned_entity_pairs(self) -> set[frozenset[str]]:
        """Entity pairs named together in at least one memory."""
        query = """
        MATCH (a:Entity)<-[:MENTIONS]-(:Memory)-[:MENTIONS]->(b:Entity)
        WHERE a.id < b.id
        RETURN DISTINCT a.id AS a, b.id AS b
        """
        with self._driver.session() as s:
            return {frozenset((r["a"], r["b"])) for r in s.run(query)}

    def set_entity_embedding(self, entity_id: str, embedding: list[float], text: str) -> None:
        """Cache an entity's descriptor embedding (derived data: not journaled)."""
        query = "MATCH (e:Entity {id: $id}) SET e.embedding = $emb, e.embeddingText = $text"
        self._write(query, id=entity_id, emb=embedding, text=text)

    def fetch_distinct_pairs(self) -> set[frozenset[str]]:
        """Entity pairs the Editor already judged to be different."""
        query = "MATCH (a:Entity)-[:DISTINCT_FROM]->(b:Entity) RETURN a.id AS a, b.id AS b"
        with self._driver.session() as s:
            return {frozenset((r["a"], r["b"])) for r in s.run(query)}

    def mark_distinct(self, id_a: str, id_b: str, reason: str) -> None:
        query = """
        MATCH (a:Entity {id: $a}), (b:Entity {id: $b})
        MERGE (a)-[r:DISTINCT_FROM]->(b)
        ON CREATE SET r.reason = $reason, r.createdAt = datetime(), r.editorRun = $run
        """
        self._write(query, a=id_a, b=id_b, reason=reason)

    def merge_entities(self, source_id: str, target_id: str) -> None:
        """Merge source entity into target: MERGED_INTO, rewire MENTIONS and
        LINKED_TO, and add the source's name and aliases to the target's."""
        before = """
        MATCH (src:Entity {id: $srcId}), (tgt:Entity {id: $tgtId})
        OPTIONAL MATCH (m:Memory)-[r:MENTIONS]->(src)
        WITH src, tgt, collect({other: m.id, props: properties(r)}) AS mentions
        OPTIONAL MATCH (src)-[l:LINKED_TO]->(o:Entity)
        WITH src, tgt, mentions, collect({other: o.id, props: properties(l)}) AS outLinks
        OPTIONAL MATCH (i:Entity)-[l2:LINKED_TO]->(src)
        RETURN tgt.aliases AS tgtAliases, src.active AS srcActive, mentions, outLinks,
               collect({other: i.id, props: properties(l2)}) AS inLinks
        """
        merge = """
        MATCH (src:Entity {id: $srcId}), (tgt:Entity {id: $tgtId})
        MERGE (src)-[mi:MERGED_INTO]->(tgt)
        ON CREATE SET mi.editorRun = $run
        SET tgt.aliases = coalesce(tgt.aliases, []) +
                [a IN coalesce(src.aliases, []) + [src.name]
                 WHERE NOT a IN coalesce(tgt.aliases, []) AND a <> tgt.name],
            src.active = false
        WITH src, tgt
        OPTIONAL MATCH (m:Memory)-[r:MENTIONS]->(src)
        FOREACH (_ IN CASE WHEN r IS NULL THEN [] ELSE [1] END |
            MERGE (m)-[nr:MENTIONS]->(tgt)
            ON CREATE SET nr.role = r.role, nr.editorRun = $run
            DELETE r)
        WITH DISTINCT src, tgt
        OPTIONAL MATCH (src)-[l:LINKED_TO]->(other:Entity)
        FOREACH (_ IN CASE WHEN l IS NULL OR other = tgt THEN [] ELSE [1] END |
            MERGE (tgt)-[nl:LINKED_TO]->(other)
            ON CREATE SET nl += properties(l), nl.editorRun = $run)
        FOREACH (_ IN CASE WHEN l IS NULL THEN [] ELSE [1] END | DELETE l)
        WITH DISTINCT src, tgt
        OPTIONAL MATCH (other:Entity)-[l:LINKED_TO]->(src)
        FOREACH (_ IN CASE WHEN l IS NULL OR other = tgt THEN [] ELSE [1] END |
            MERGE (other)-[nl:LINKED_TO]->(tgt)
            ON CREATE SET nl += properties(l), nl.editorRun = $run)
        FOREACH (_ IN CASE WHEN l IS NULL THEN [] ELSE [1] END | DELETE l)
        """
        params = {"srcId": source_id, "tgtId": target_id, "run": self.run_id}

        def work(tx: neo4j.ManagedTransaction) -> neo4j.Record | None:
            prev = tx.run(before, **params).single()
            tx.run(merge, **params).consume()
            return prev

        with self._driver.session() as s:
            prev = s.execute_write(work)
        if prev is None:
            return
        self._journal(
            _restore("Entity", target_id, aliases=prev["tgtAliases"]),
            _restore("Entity", source_id, active=prev["srcActive"]),
            *(_relink("MENTIONS", ("Memory", r["other"]), ("Entity", source_id), r["props"])
              for r in prev["mentions"] if r["other"] is not None),
            *(_relink("LINKED_TO", ("Entity", source_id), ("Entity", r["other"]), r["props"])
              for r in prev["outLinks"] if r["other"] is not None),
            *(_relink("LINKED_TO", ("Entity", r["other"]), ("Entity", source_id), r["props"])
              for r in prev["inLinks"] if r["other"] is not None),
        )

    # ------------------------------------------------------------------
    # Relationships
    # ------------------------------------------------------------------

    def create_relates_to(self, source_id: str, target_id: str, rel_type: str, weight: float) -> None:
        query = """
        MATCH (a:Memory {id: $src}), (b:Memory {id: $tgt})
        MERGE (a)-[r:RELATES_TO {type: $type}]->(b)
        ON CREATE SET r.weight = $weight, r.editorRun = $run
        """
        self._write(query, src=source_id, tgt=target_id, type=rel_type, weight=weight)

    def has_relates_to(self, id_a: str, id_b: str) -> bool:
        query = """
        MATCH (a:Memory {id: $a})-[:RELATES_TO]-(b:Memory {id: $b})
        RETURN count(*) AS c
        """
        with self._driver.session() as s:
            return s.run(query, a=id_a, b=id_b).single()["c"] > 0

    # ------------------------------------------------------------------
    # Entity Links (LINKED_TO)
    # ------------------------------------------------------------------

    def fetch_entity_co_mentions(self, entity_type: str = "Person") -> list[dict]:
        """Find entity pairs co-mentioned in the same memory, with evidence."""
        query = """
        MATCH (m:Memory)-[:MENTIONS]->(e1:Entity)
        MATCH (m)-[:MENTIONS]->(e2:Entity)
        WHERE e1.type = $type AND e2.type = $type
          AND e1.name < e2.name
          AND m.status IN ['raw', 'reviewed']
        WITH e1, e2, m ORDER BY m.timestamp, m.content
        WITH e1, e2, count(m) AS coCount, collect(m.content)[..3] AS samples
        WHERE coCount >= 1
        RETURN e1.id AS id_a, e1.name AS name_a, e1.type AS type_a,
               e2.id AS id_b, e2.name AS name_b, e2.type AS type_b,
               coCount AS co_count, samples AS sample_contents
        ORDER BY coCount DESC, name_a, name_b
        """
        with self._driver.session() as s:
            return [dict(r) for r in s.run(query, type=entity_type)]

    def has_linked_to(self, id_a: str, id_b: str) -> bool:
        query = """
        MATCH (a:Entity {id: $a})-[:LINKED_TO]-(b:Entity {id: $b})
        RETURN count(*) AS c
        """
        with self._driver.session() as s:
            return s.run(query, a=id_a, b=id_b).single()["c"] > 0

    def create_linked_to(
        self, source_id: str, target_id: str,
        relation: str, detail: str, sentiment: str, strength: float,
    ) -> None:
        query = """
        MATCH (a:Entity {id: $src}), (b:Entity {id: $tgt})
        MERGE (a)-[r:LINKED_TO]->(b)
        ON CREATE SET r.relation = $relation, r.detail = $detail,
            r.sentiment = $sentiment, r.strength = $strength,
            r.active = true, r.editorRun = $run
        """
        self._write(query, src=source_id, tgt=target_id, relation=relation,
                    detail=detail, sentiment=sentiment, strength=strength)

    # ------------------------------------------------------------------
    # Temporal validity (supersession)
    # ------------------------------------------------------------------

    def backfill_valid_from(self) -> int:
        """Memories written before validFrom existed are valid from when recorded."""
        query = """
        MATCH (m:Memory) WHERE m.validFrom IS NULL
        SET m.validFrom = m.timestamp
        RETURN m.id AS id
        """
        ids = [r["id"] for r in self._write(query)]
        self._journal(*(_restore("Memory", mid, validFrom=None) for mid in ids))
        return len(ids)

    def fetch_current_memories(self) -> list[dict]:
        """Active, still-valid memories with embeddings and mentioned entity ids."""
        query = """
        MATCH (m:Memory)
        WHERE m.status IN ['raw', 'reviewed'] AND m.validTo IS NULL
          AND m.embedding IS NOT NULL AND NOT (m)-[:CANONICAL]->()
        OPTIONAL MATCH (m)-[:MENTIONS]->(e:Entity)
        WITH m, collect(DISTINCT e.id) AS entityIds
        RETURN m.id AS id, m.content AS content, m.status AS status,
               m.embedding AS embedding, m.validFrom AS validFrom, entityIds
        ORDER BY m.validFrom, m.content
        """
        with self._driver.session() as s:
            return [dict(r) for r in s.run(query)]

    def fetch_conflict_links(self) -> set[frozenset[str]]:
        """Memory pairs already linked as superseding or contradicting."""
        query = """
        MATCH (a:Memory)-[r:SUPERSEDES|RELATES_TO]->(b:Memory)
        WHERE type(r) = 'SUPERSEDES' OR r.type = 'contradicts'
        RETURN a.id AS a, b.id AS b
        """
        with self._driver.session() as s:
            return {frozenset((r["a"], r["b"])) for r in s.run(query)}

    def supersede(self, newer_id: str, older_id: str, reason: str) -> None:
        """Newer memory replaces older: SUPERSEDES edge, and the older memory's
        validTo becomes the newer one's validFrom (keeping an earlier validTo)."""
        query = """
        MATCH (new:Memory {id: $newId}), (old:Memory {id: $oldId})
        WITH new, old, old.validTo AS prevValidTo,
             coalesce(new.validFrom, new.timestamp) AS since
        MERGE (new)-[r:SUPERSEDES]->(old)
        ON CREATE SET r.reason = $reason, r.createdAt = datetime(), r.editorRun = $run
        SET old.validTo = CASE
            WHEN old.validTo IS NULL OR since < old.validTo THEN since
            ELSE old.validTo END
        RETURN prevValidTo
        """
        for r in self._write(query, newId=newer_id, oldId=older_id, reason=reason):
            self._journal(_restore("Memory", older_id, validTo=r["prevValidTo"]))

    # ------------------------------------------------------------------
    # Confidence
    # ------------------------------------------------------------------

    def boost_confidence_once(self, memory_id: str, boost: float) -> bool:
        """Apply the multi-session boost unless this memory already had it."""
        query = """
        MATCH (m:Memory {id: $id}) WHERE NOT coalesce(m.sessionBoosted, false)
        WITH m, m.confidence AS prevConfidence, m.sessionBoosted AS prevBoosted
        SET m.confidence = CASE WHEN m.confidence + $boost > 1.0 THEN 1.0
                                ELSE m.confidence + $boost END,
            m.sessionBoosted = true
        RETURN prevConfidence, prevBoosted
        """
        rows = self._write(query, id=memory_id, boost=boost)
        for r in rows:
            self._journal(_restore("Memory", memory_id, confidence=r["prevConfidence"],
                                   sessionBoosted=r["prevBoosted"]))
        return bool(rows)

    def decay_reviewed_confidence(self, decay: float, floor: float) -> int:
        """Decay confidence for reviewed memories, at most once per calendar day
        so reruns and multi-batch runs don't compound. Returns count affected."""
        query = """
        MATCH (m:Memory {status: 'reviewed'})
        WHERE m.confidence > $floor
          AND (m.confidenceDecayedOn IS NULL OR m.confidenceDecayedOn < date())
        WITH m, m.confidence AS prevConfidence, m.confidenceDecayedOn AS prevDecayedOn
        SET m.confidence = CASE
            WHEN m.confidence - $decay > $floor THEN m.confidence - $decay
            ELSE $floor END,
            m.confidenceDecayedOn = date()
        RETURN m.id AS id, prevConfidence, prevDecayedOn
        """
        rows = self._write(query, decay=decay, floor=floor)
        self._journal(*(
            _restore("Memory", r["id"], confidence=r["prevConfidence"],
                     confidenceDecayedOn=r["prevDecayedOn"])
            for r in rows
        ))
        return len(rows)

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def mark_reviewed_batch(self, memory_ids: list[str]) -> None:
        query = """
        UNWIND $ids AS mid
        MATCH (m:Memory {id: mid})
        WHERE m.status = 'raw'
        SET m.status = 'reviewed'
        RETURN m.id AS id
        """
        rows = self._write(query, ids=memory_ids)
        self._journal(*(_restore("Memory", r["id"], status="raw") for r in rows))

    # ------------------------------------------------------------------
    # Runs, audit, and rollback
    # ------------------------------------------------------------------

    def start_run(self, model: str | None, effort: str | None, batch_size: int) -> str:
        """Open an EditorRun; mutations from now on are tagged and journaled."""
        self.run_id = str(uuid4())
        self._undo = []
        query = """
        CREATE (:EditorRun {id: $run, startedAt: datetime(), status: 'running',
                            model: $model, effort: $effort, batchSize: $batch})
        """
        self._write(query, model=model, effort=effort, batch=batch_size)
        return self.run_id

    def finish_run(self, status: str, summary: dict) -> None:
        query = """
        MATCH (r:EditorRun {id: $run})
        SET r.status = $status, r.finishedAt = datetime(), r.summary = $summary
        """
        self._write(query, status=status, summary=json.dumps(summary))
        self.run_id = None

    def take_undo(self) -> list[dict]:
        ops, self._undo = self._undo, []
        return ops

    @property
    def has_pending_undo(self) -> bool:
        return bool(self._undo)

    def log_edit_action(
        self, action_type: str, targets: list[str], reason: str,
        undo: list[dict] | None = None, seq: int | None = None,
    ) -> str:
        action_id = str(uuid4())
        query = """
        CREATE (a:EditAction {
            id: $id, type: $type, targets: $targets,
            reason: $reason, timestamp: datetime(),
            runId: $run, seq: $seq, undo: $undo
        })
        """
        self._write(query, id=action_id, type=action_type, targets=targets, reason=reason,
                    seq=seq, undo=json.dumps(undo or []))
        return action_id

    def fetch_runs(self, limit: int = 20) -> list[dict]:
        query = """
        MATCH (r:EditorRun)
        OPTIONAL MATCH (a:EditAction {runId: r.id})
        WITH r, count(a) AS actions
        RETURN r.id AS id, toString(r.startedAt) AS startedAt, r.status AS status,
               r.model AS model, r.batchSize AS batchSize, r.summary AS summary, actions
        ORDER BY r.startedAt DESC LIMIT $limit
        """
        with self._driver.session() as s:
            return [dict(r) for r in s.run(query, limit=limit)]

    def later_runs(self, run_id: str) -> list[str]:
        """Runs started after this one that haven't been rolled back."""
        query = """
        MATCH (r:EditorRun {id: $run}), (later:EditorRun)
        WHERE later.startedAt > r.startedAt AND later.status <> 'rolled_back'
        RETURN later.id AS id ORDER BY later.startedAt
        """
        with self._driver.session() as s:
            return [r["id"] for r in s.run(query, run=run_id)]

    def fetch_run_undo(self, run_id: str) -> tuple[dict | None, list[dict]]:
        """(the run, its undo ops in the order to apply them: newest first)."""
        with self._driver.session() as s:
            run = s.run("MATCH (r:EditorRun {id: $run}) RETURN r {.*} AS r", run=run_id).single()
            actions = s.run(
                "MATCH (a:EditAction {runId: $run}) RETURN a.undo AS undo ORDER BY a.seq DESC",
                run=run_id,
            )
            ops = [op for a in actions for op in reversed(json.loads(a["undo"] or "[]"))]
        return (dict(run["r"]) if run else None), ops

    def apply_rollback(self, run_id: str, ops: list[dict]) -> tuple[int, int]:
        """Replay undo ops, then delete what the run created, in one transaction.
        Returns (relationships deleted, nodes deleted)."""

        def work(tx: neo4j.ManagedTransaction) -> tuple[int, int]:
            for op in ops:
                query, params = _undo_query(op)
                tx.run(query, **params).consume()
            rels = tx.run(
                "MATCH ()-[r]->() WHERE r.editorRun = $run DELETE r RETURN count(r) AS c", run=run_id,
            ).single()["c"]
            nodes = tx.run(
                "MATCH (n) WHERE n.editorRun = $run DETACH DELETE n RETURN count(n) AS c", run=run_id,
            ).single()["c"]
            tx.run("MATCH (a:EditAction {runId: $run}) SET a.rolledBack = true", run=run_id).consume()
            tx.run(
                "MATCH (r:EditorRun {id: $run}) SET r.status = 'rolled_back', r.rolledBackAt = datetime()",
                run=run_id,
            ).consume()
            return rels, nodes

        with self._driver.session() as s:
            return s.execute_write(work)

    # ------------------------------------------------------------------
    # Journal plumbing
    # ------------------------------------------------------------------

    def _write(self, query: str, **params) -> list[neo4j.Record]:
        with self._driver.session() as s:
            return list(s.run(query, run=self.run_id, **params))

    def _journal(self, *ops: dict) -> None:
        if self.run_id is not None:
            self._undo.extend(ops)


# Undo ops are JSON: {"op": "set", ...} restores node properties (None removes
# one); {"op": "relink", ...} recreates a relationship the run deleted.
_UNDO_LABELS = {"Memory", "Entity"}
_UNDO_REL_TYPES = {"MENTIONS", "LINKED_TO"}
_PROPERTY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _encode(value: object) -> object:
    """JSON-safe property value; Neo4j temporals become tagged strings."""
    if isinstance(value, neo4j.time.DateTime):
        return {"$datetime": value.iso_format()}
    if isinstance(value, neo4j.time.Date):
        return {"$date": value.iso_format()}
    if isinstance(value, list):
        return [_encode(v) for v in value]
    return value


def _restore(label: str, node_id: str, **props: object) -> dict:
    return {"op": "set", "label": label, "id": node_id,
            "props": {k: _encode(v) for k, v in props.items()}}


def _relink(rel_type: str, start: tuple[str, str], end: tuple[str, str], props: dict | None) -> dict:
    return {"op": "relink", "type": rel_type, "start": list(start), "end": list(end),
            "props": {k: _encode(v) for k, v in (props or {}).items()}}


def _assignments(var: str, props: dict) -> tuple[str, dict]:
    """SET clause restoring `props` on `var`, decoding tagged temporals."""
    parts, params = [], {}
    for i, (key, value) in enumerate(props.items()):
        if not _PROPERTY.match(key):
            raise ValueError(f"bad property name in undo op: {key!r}")
        name = f"v{i}"
        if isinstance(value, dict) and "$datetime" in value:
            parts.append(f"{var}.{key} = datetime(${name})")
            params[name] = value["$datetime"]
        elif isinstance(value, dict) and "$date" in value:
            parts.append(f"{var}.{key} = date(${name})")
            params[name] = value["$date"]
        else:
            parts.append(f"{var}.{key} = ${name}")
            params[name] = value
    return ("SET " + ", ".join(parts)) if parts else "", params


def _undo_query(op: dict) -> tuple[str, dict]:
    if op["op"] == "set" and op["label"] in _UNDO_LABELS:
        clause, params = _assignments("n", op["props"])
        return f"MATCH (n:{op['label']} {{id: $id}}) {clause}", {"id": op["id"], **params}
    if op["op"] == "relink" and op["type"] in _UNDO_REL_TYPES:
        (start_label, start_id), (end_label, end_id) = op["start"], op["end"]
        if {start_label, end_label} - _UNDO_LABELS:
            raise ValueError(f"bad label in undo op: {op}")
        clause, params = _assignments("r", op["props"])
        return (
            (
                f"MATCH (a:{start_label} {{id: $a}}), (b:{end_label} {{id: $b}}) "
                f"CREATE (a)-[r:{op['type']}]->(b) {clause}"
            ),
            {"a": start_id, "b": end_id, **params},
        )
    raise ValueError(f"unknown undo op: {op}")
