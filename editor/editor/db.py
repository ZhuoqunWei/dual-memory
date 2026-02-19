"""Neo4j driver wrapper with Editor-specific queries.

Mirrors the TypeScript neo4j-client.ts but adds Editor operations:
dedup (CANONICAL), classification (normalizedKind/Type), categories,
entity resolution (MERGED_INTO), confidence management, and audit logging.
"""

from __future__ import annotations

import logging
from uuid import uuid4

import neo4j

log = logging.getLogger("editor.db")


class EditorDB:
    def __init__(self, uri: str, user: str, password: str) -> None:
        self._driver = neo4j.GraphDatabase.driver(uri, auth=(user, password))

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
        WITH m ORDER BY m.timestamp ASC LIMIT $limit
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
        """
        with self._driver.session() as s:
            return [dict(r["entity"]) for r in s.run(query)]

    def fetch_categories(self) -> list[dict]:
        query = """
        MATCH (c:Category)
        OPTIONAL MATCH (m:Memory)-[:IN_CATEGORY]->(c)
        WHERE m.status IN ['raw', 'reviewed']
        WITH c, collect(m.id) AS memberIds, collect(m.embedding) AS memberEmbeddings
        RETURN c.name AS name, c.description AS description,
               memberIds, memberEmbeddings
        """
        with self._driver.session() as s:
            return [dict(r) for r in s.run(query)]

    def fetch_memories_by_shared_entity(self) -> dict[str, list[dict]]:
        """Group memories by entity — for contradiction detection and relationship creation."""
        query = """
        MATCH (m:Memory)-[:MENTIONS]->(e:Entity)
        WHERE m.status IN ['raw', 'reviewed']
        WITH e, collect({id: m.id, content: m.content, kind: m.kind,
                         confidence: m.confidence, sessionId: null}) AS memories
        WHERE size(memories) > 1
        RETURN e.name AS entityName, e.id AS entityId, memories
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
        CREATE (dupe)-[:CANONICAL]->(canon)
        CREATE (dupe)-[:RELATES_TO {type: 'editor:supersedes', weight: 1.0}]->(canon)
        SET dupe.status = 'archived'
        """
        with self._driver.session() as s:
            s.run(query, dupeId=dupe_id, canonId=canonical_id)

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def set_normalized_kind(self, memory_id: str, normalized_kind: str) -> None:
        query = "MATCH (m:Memory {id: $id}) SET m.normalizedKind = $nk"
        with self._driver.session() as s:
            s.run(query, id=memory_id, nk=normalized_kind)

    def set_entity_normalized_type(self, entity_id: str, normalized_type: str) -> None:
        query = "MATCH (e:Entity {id: $id}) SET e.normalizedType = $nt"
        with self._driver.session() as s:
            s.run(query, id=entity_id, nt=normalized_type)

    # ------------------------------------------------------------------
    # Categories
    # ------------------------------------------------------------------

    def create_category(self, name: str, description: str) -> str:
        query = """
        CREATE (c:Category {name: $name, description: $desc})
        RETURN c.name AS name
        """
        with self._driver.session() as s:
            r = s.run(query, name=name, desc=description).single()
            return r["name"]

    def assign_category(self, memory_id: str, category_name: str, score: float) -> None:
        query = """
        MATCH (m:Memory {id: $mid}), (c:Category {name: $cname})
        MERGE (m)-[r:IN_CATEGORY]->(c)
        SET r.score = $score, r.assignedBy = 'editor', r.timestamp = datetime()
        """
        with self._driver.session() as s:
            s.run(query, mid=memory_id, cname=category_name, score=score)

    # ------------------------------------------------------------------
    # Entity Resolution
    # ------------------------------------------------------------------

    def merge_entities(self, source_id: str, target_id: str) -> None:
        """Merge source entity into target: MERGED_INTO + rewire MENTIONS + merge aliases."""
        query = """
        MATCH (src:Entity {id: $srcId}), (tgt:Entity {id: $tgtId})
        CREATE (src)-[:MERGED_INTO]->(tgt)
        SET tgt.aliases = [a IN (tgt.aliases + src.aliases) WHERE NOT a IN tgt.aliases | a]
        WITH src, tgt
        MATCH (m:Memory)-[r:MENTIONS]->(src)
        CREATE (m)-[:MENTIONS {role: r.role}]->(tgt)
        DELETE r
        """
        with self._driver.session() as s:
            s.run(query, srcId=source_id, tgtId=target_id)

    # ------------------------------------------------------------------
    # Relationships
    # ------------------------------------------------------------------

    def create_relates_to(self, source_id: str, target_id: str, rel_type: str, weight: float) -> None:
        query = """
        MATCH (a:Memory {id: $src}), (b:Memory {id: $tgt})
        MERGE (a)-[r:RELATES_TO {type: $type}]->(b)
        SET r.weight = $weight
        """
        with self._driver.session() as s:
            s.run(query, src=source_id, tgt=target_id, type=rel_type, weight=weight)

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
          AND e1.id < e2.id
          AND m.status IN ['raw', 'reviewed']
        WITH e1, e2, count(m) AS coCount, collect(m.content)[..3] AS samples
        WHERE coCount >= 1
        RETURN e1.id AS id_a, e1.name AS name_a, e1.type AS type_a,
               e2.id AS id_b, e2.name AS name_b, e2.type AS type_b,
               coCount AS co_count, samples AS sample_contents
        ORDER BY coCount DESC
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
        SET r.relation = $relation, r.detail = $detail,
            r.sentiment = $sentiment, r.strength = $strength,
            r.active = true
        """
        with self._driver.session() as s:
            s.run(query, src=source_id, tgt=target_id,
                  relation=relation, detail=detail,
                  sentiment=sentiment, strength=strength)

    # ------------------------------------------------------------------
    # Confidence
    # ------------------------------------------------------------------

    def set_confidence(self, memory_id: str, confidence: float) -> None:
        query = "MATCH (m:Memory {id: $id}) SET m.confidence = $c"
        with self._driver.session() as s:
            s.run(query, id=memory_id, c=max(0.0, min(1.0, confidence)))

    def decay_reviewed_confidence(self, decay: float, floor: float) -> int:
        """Decay confidence for all reviewed memories. Returns count of affected."""
        query = """
        MATCH (m:Memory {status: 'reviewed'})
        WHERE m.confidence > $floor
        SET m.confidence = CASE
            WHEN m.confidence - $decay > $floor THEN m.confidence - $decay
            ELSE $floor END
        RETURN count(m) AS c
        """
        with self._driver.session() as s:
            return s.run(query, decay=decay, floor=floor).single()["c"]

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def mark_reviewed(self, memory_id: str) -> None:
        query = "MATCH (m:Memory {id: $id}) SET m.status = 'reviewed'"
        with self._driver.session() as s:
            s.run(query, id=memory_id)

    def mark_reviewed_batch(self, memory_ids: list[str]) -> None:
        query = """
        UNWIND $ids AS mid
        MATCH (m:Memory {id: mid})
        WHERE m.status = 'raw'
        SET m.status = 'reviewed'
        """
        with self._driver.session() as s:
            s.run(query, ids=memory_ids)

    # ------------------------------------------------------------------
    # Audit
    # ------------------------------------------------------------------

    def log_edit_action(self, action_type: str, targets: list[str], reason: str) -> str:
        action_id = str(uuid4())
        query = """
        CREATE (a:EditAction {
            id: $id, type: $type, targets: $targets,
            reason: $reason, timestamp: datetime()
        })
        """
        with self._driver.session() as s:
            s.run(query, id=action_id, type=action_type, targets=targets, reason=reason)
        return action_id
