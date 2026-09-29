"""Scratch-graph access for the eval harness: reset it, then read Editor state.

The harness wipes its database on every run, so reset() refuses to touch a
non-empty database it did not create (no :EvalSentinel node). Point it at a
throwaway Neo4j (eval/docker-compose.yml), never the real memory graph.
"""

from __future__ import annotations

import neo4j


class UnsafeDatabase(RuntimeError):
    pass


class EvalGraph:
    def __init__(self, uri: str, user: str, password: str) -> None:
        self._driver = neo4j.GraphDatabase.driver(uri, auth=(user, password))

    def close(self) -> None:
        self._driver.close()

    def _rows(self, query: str, **params) -> list[dict]:
        with self._driver.session() as s:
            return [dict(r) for r in s.run(query, **params)]

    def reset(self) -> None:
        counts = self._rows("""
            MATCH (n) WITH count(n) AS total
            OPTIONAL MATCH (s:EvalSentinel)
            RETURN total, count(s) AS sentinels
        """)[0]
        if counts["total"] and not counts["sentinels"]:
            raise UnsafeDatabase(
                f"refusing to wipe a database with {counts['total']} nodes that the eval "
                "harness did not create. Point EVAL_NEO4J_URI at a scratch instance "
                "(see eval/docker-compose.yml)."
            )
        with self._driver.session() as s:
            s.run("MATCH (n) DETACH DELETE n").consume()
            s.run("CREATE (:EvalSentinel {createdAt: datetime()})").consume()

    # ------------------------------------------------------------------
    # State readers
    # ------------------------------------------------------------------

    def memories(self) -> dict[str, dict]:
        """Memory id -> {sourceRef, status, validTo}."""
        rows = self._rows("""
            MATCH (m:Memory)
            RETURN m.id AS id, m.sourceRef AS sourceRef, m.status AS status,
                   m.validTo AS validTo
        """)
        return {r["id"]: r for r in rows}

    def canonical_edges(self) -> dict[str, str]:
        """Archived duplicate id -> canonical id."""
        rows = self._rows("MATCH (d:Memory)-[:CANONICAL]->(c:Memory) RETURN d.id AS d, c.id AS c")
        return {r["d"]: r["c"] for r in rows}

    def conflict_edges(self) -> list[tuple[str, str, str]]:
        """(source id, target id, kind) for contradiction/supersession edges."""
        return [
            (r["s"], r["t"], r["kind"])
            for r in self._rows("""
                MATCH (a:Memory)-[r:RELATES_TO {type: 'contradicts'}]->(b:Memory)
                RETURN a.id AS s, b.id AS t, 'contradicts' AS kind
                UNION ALL
                MATCH (a:Memory)-[:SUPERSEDES]->(b:Memory)
                RETURN a.id AS s, b.id AS t, 'supersedes' AS kind
            """)
        ]

    def entities(self) -> dict[str, str]:
        """Entity id -> name."""
        return {r["id"]: r["name"] for r in self._rows("MATCH (e:Entity) RETURN e.id AS id, e.name AS name")}

    def merge_edges(self) -> dict[str, str]:
        """Merged entity id -> the entity it was merged into."""
        rows = self._rows("MATCH (a:Entity)-[:MERGED_INTO]->(b:Entity) RETURN a.id AS a, b.id AS b")
        return {r["a"]: r["b"] for r in rows}

    def raw_count(self) -> int:
        return self._rows("MATCH (m:Memory {status: 'raw'}) RETURN count(m) AS c")[0]["c"]

    def relationship_candidate_pairs(self) -> int:
        """How many memory pairs the relationships step would send to the LLM."""
        return self._rows("""
            MATCH (a:Memory)-[:MENTIONS]->(:Entity)<-[:MENTIONS]-(b:Memory)
            WHERE a.id < b.id AND a.status IN ['raw', 'reviewed'] AND b.status IN ['raw', 'reviewed']
            RETURN count(DISTINCT [a.id, b.id]) AS c
        """)[0]["c"]
