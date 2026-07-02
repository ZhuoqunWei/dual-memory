# Dual-Memory

Dual-Memory is a two-agent memory system for AI assistants. It captures useful conversation facts into a Neo4j knowledge graph, then periodically consolidates that graph so retrieval stays cleaner than a write-only memory log.

The project is split into two cooperating parts:

- `extensions/dual-memory`: an OpenClaw memory plugin written in TypeScript. This is the Writer. It extracts facts after conversations, embeds them, writes graph nodes, and retrieves relevant context before future sessions.
- `editor`: a standalone Python agent. This is the Editor. It runs deduplication, classification, categorization, contradiction detection, entity resolution, relationship mining, and confidence scoring over the graph.

## Why This Exists

Most assistant memory systems either append text forever or search a flat vector store. That works early, then gets noisy: duplicates accumulate, stale facts survive, relationships are hard to query, and retrieval quality decays.

Dual-Memory uses a faster Writer and a slower Editor:

1. The Writer prioritizes capture. It runs inline with the assistant lifecycle and records structured memories quickly.
2. The Editor prioritizes coherence. It reviews accumulated memory with broader graph context and keeps the durable semantic layer organized.

## Architecture

```text
Conversation
    |
    v
OpenClaw plugin (Writer) ---> Neo4j graph <--- Python agent (Editor)
        |                         |
        |                         +-- Memory, Entity, Session, Category, EditAction
        +-- recall context
```

Core graph relationships:

- `(Memory)-[:MENTIONS]->(Entity)`
- `(Memory)-[:RELATES_TO]->(Memory)`
- `(Memory)-[:PART_OF]->(Session)`
- `(Memory)-[:IN_CATEGORY]->(Category)`
- `(Memory)-[:CANONICAL]->(Memory)`
- `(Entity)-[:LINKED_TO]->(Entity)`
- `(Entity)-[:MERGED_INTO]->(Entity)`

Retrieval combines vector similarity with graph expansion and confidence/salience weighting:

```text
score = similarity * (0.6 + 0.4 * confidence) * (0.6 + 0.4 * salience)
```

## Repository Layout

```text
.
|-- dual_agent_memory_concept.md      # original design document
|-- TECHNICAL.md                      # implementation-oriented overview
|-- infrastructure/
|   |-- docker-compose.yml            # local Neo4j
|   `-- init-schema.cypher            # schema reference
|-- extensions/dual-memory/           # OpenClaw Writer plugin
|-- examples/demo-seed.cypher         # synthetic graph demo
`-- editor/                           # Python Editor agent
```

## Quick Start

These command blocks assume you start from the repository root.

Install local dependencies:

```bash
cd extensions/dual-memory
npm install
cd ../..

cd editor
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cd ..
```

Start Neo4j:

```bash
cd infrastructure
docker compose up -d
cd ..
```

Configure environment variables. See `.env.example` for the full list.

Run the Editor once:

```bash
cd editor
source .venv/bin/activate
python -m editor.main --once --all
cd ..
```

Show graph stats:

```bash
cd editor
source .venv/bin/activate
python -m editor.main --stats
cd ..
```

The OpenClaw plugin is loaded from `extensions/dual-memory` by an OpenClaw gateway installation.

## Demo Graph

For screenshots or schema exploration, load the synthetic demo graph after Neo4j is running:

```bash
docker exec -i dual-memory-neo4j cypher-shell -u neo4j -p dualmemory2026 < examples/demo-seed.cypher
```

The demo data is intentionally small and contains no embeddings, so use it to inspect the graph model rather than test vector retrieval.

## Local Checks

Run the non-mutating check suite from the repository root:

```bash
./check.sh
```

Run the Neo4j-only TypeScript integration smoke test:

```bash
cd extensions/dual-memory
npm run test:neo4j
```

The full LLM integration test also requires API keys:

```bash
cd extensions/dual-memory
npm run test:integration
```

## Publishing

See [PUBLISHING.md](PUBLISHING.md) for the exact GitHub CLI commands and suggested repository metadata.

## Status

Implemented:

- OpenClaw lifecycle hooks for recall and capture
- Neo4j schema initialization
- Writer extraction through Anthropic or OpenAI
- Voyage AI or OpenAI embeddings
- Hybrid vector plus graph retrieval
- Editor pipeline for dedup, classification, categories, contradictions, entity resolution, memory relationships, entity links, and confidence updates
- EditAction audit logging for editor mutations

Next improvements:

- Broaden unit coverage across the full Editor pipeline
- Add screenshots or graph-browser captures from the synthetic demo data
- Add category-aware retrieval routing
- Add a `before_compaction` capture hook
- Generate entity embeddings for semantic entity search

## License

MIT. See [LICENSE](LICENSE).
