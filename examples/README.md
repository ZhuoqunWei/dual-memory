# Examples

This folder contains synthetic data for exploring the Dual-Memory graph model.

## Demo Seed

`demo-seed.cypher` creates a small graph with sessions, memories, entities, categories, memory-to-memory relationships, and entity-to-entity links.

Start Neo4j, then run from the repository root:

```bash
docker exec -i dual-memory-neo4j cypher-shell -u neo4j -p dualmemory2026 < examples/demo-seed.cypher
```

The seed deletes and recreates only demo nodes whose IDs begin with `demo-` plus demo categories named `Demo Learning Progress` and `Demo Career Direction`.

It intentionally does not include vector embeddings, so it is best for schema exploration and screenshots rather than retrieval quality tests.
