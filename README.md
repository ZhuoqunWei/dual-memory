# Dual-Memory

Dual-Memory is a two-agent memory system for AI assistants. It captures useful conversation facts into a Neo4j knowledge graph, then periodically consolidates that graph so retrieval stays cleaner than a write-only memory log.

The project is split into two cooperating parts:

- `extensions/dual-memory`: an OpenClaw memory plugin written in TypeScript. This is the Writer. It extracts facts after conversations, embeds them, writes graph nodes, and retrieves relevant context before future sessions.
- `editor`: a standalone Python agent. This is the Editor. It runs deduplication, classification, categorization, temporal supersession, entity resolution, relationship mining, and confidence scoring over the graph. Every run can be rolled back.
- `eval`: a gold-labelled harness that scores the whole loop. See [Evaluation](#evaluation).

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
- `(Memory)-[:SUPERSEDES]->(Memory)`: a newer value of the same attribute; the older memory gets `validTo`
- `(Entity)-[:LINKED_TO]->(Entity)`
- `(Entity)-[:MERGED_INTO]->(Entity)`
- `(Entity)-[:DISTINCT_FROM]->(Entity)`: judged different, so never asked again

Retrieval combines vector similarity with graph expansion, confidence/salience weighting, and a penalty for superseded facts (which stay retrievable and are marked outdated in the injected context):

```text
score = cosine * (0.6 + 0.4 * confidence) * (0.6 + 0.4 * salience) * (0.7 if superseded else 1)
```

## Evaluation

`python -m editor.evaluation` loads 40 synthetic sessions (102 facts, 85 unique) through the real Writer write path, runs the Editor, asks 50 questions through the real retrieval, and scores everything against gold labels: which facts are duplicates, which values replaced which, which names are the same entity, and which facts answer each question. LLM replies and embeddings are recorded, so CI replays the run with no API keys. Details in [eval/README.md](eval/README.md).

Each column is one change, measured on the same fixture (Editor on Claude Sonnet 5; 01-06 at low effort, 07-08 at medium):

| | 01 | 02 | 03 | 04 | 05 | 06 | 07 | 08 | Haiku 4.5 |
|---|---|---|---|---|---|---|---|---|---|
| Dedup precision / recall | 55% / 94% | 55% / 94% | 68% / 94% | 81% / 94% | 81% / 94% | 81% / 94% | 89% / 94% | 89% / 94% | 89% / 94% |
| Contradiction precision / recall | 100% / 13% | 100% / 13% | 100% / 7% | 85% / 87% | 77% / 87% | 77% / 87% | 89% / 100% | 89% / 100% | 56% / 80% |
| Outdated facts superseded | 0% | 0% | 0% | 93% | 93% | 93% | 93% | 93% | 100% |
| Current facts wrongly superseded | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 23% |
| Entity wrong-merge rate | 50% (4/8) | 50% (4/8) | 33% (4/12) | 33% (4/12) | 0% (0/9) | 0% (0/9) | 10% (1/10) | 0% (0/9) | 10% (1/10) |
| Entity pairwise recall | 50% | 50% | 67% | 67% | 75% | 75% | 75% | 75% | 75% |
| Retrieval recall@5 | 82% | 82% | 88% | 81% | 80% | 80% | 86% | 86% | 75% |
| Outdated fact ranked first | 67% | 67% | 33% | 0% | 0% | 0% | 0% | 0% | 0% |
| LLM calls / cost per run | 270 / $1.09 | 58 / $0.12 | 57 / $0.13 | 57 / $0.32 | 58 / $0.32 | 58 / $0.32 | 61 / $0.41 | 61 / $0.41 | 60 / $0.14 |

1. **Baseline**: the code as of July 2026.
2. **Relationship pairs from non-hub entities only**: the per-step cost table showed one step spending 95% of the budget on pairs linked only through the user.
3. **True cosine**: Neo4j's vector index reports `(1 + cos) / 2`, so the Writer's "0.92" duplicate gate was really 0.84 and silently dropped updates ("now lives in Portland" never reached the graph). The gate now only catches same-session restatements.
4. **Temporal supersession** replaces symmetric confidence penalties: the newer fact supersedes the older one.
5. **Three-tier entity resolution**: names, then name embeddings, then an LLM with the memories as context, with "different" verdicts as cannot-link constraints.
6. **Idempotent Writer and Editor, rollback**: no metric change by design; covered by integration tests.
7. **Editor effort medium**, chosen from this table.
8. **Names mentioned in one memory can't end up merged** (07 had merged "VS Code" into "Neovim" through "nvim").

The Haiku 4.5 column runs the 07 code: cheapest, but it marks 23% of current facts as outdated.

**External check: LoCoMo.** On the public [LoCoMo](https://github.com/snap-research/locomo) benchmark (10 long conversations, 1,540 questions), run through the same shipped code, the system answers **41.0%** correctly (LLM judge; single-hop 56.8%, multi-hop 33.0%, open-domain 36.5%, temporal 8.1%). Retrieval surfaces a memory from the right session for 79% of questions; what's missing is the detail, because the Writer keeps a few durable facts per conversation and never sees dates. Details in [eval/README.md](eval/README.md#external-benchmark-locomo).

Caveats: one fixture, and the thresholds were tuned while looking at it; three gold labels were corrected after seeing results; every number is a single LLM sample (contradiction precision moved 85% → 77% between 04 and 05 on variance alone). Retrieval recall is lowest on "what was it before?" questions (29%), which the superseded penalty trades away.

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
|-- editor/                           # Python Editor agent (+ editor/evaluation harness)
`-- eval/                             # eval fixture, recordings, results, scratch Neo4j
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

Each Editor run prints a per-step table of LLM calls, tokens, cost, and time. List recent runs, or undo one (runs are undone newest first):

```bash
cd editor
source .venv/bin/activate
python -m editor.main --runs
python -m editor.main --rollback <run_id>
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

Run the eval harness and the idempotency and rollback tests against a scratch Neo4j (the harness wipes it; it refuses any database it didn't create):

```bash
docker compose -f eval/docker-compose.yml up -d
cd editor
source .venv/bin/activate
python -m editor.evaluation --check
EVAL_NEO4J_URI=bolt://localhost:7688 python -m pytest tests/test_idempotency.py tests/test_rollback.py
cd ..
```

## Status

Implemented:

- OpenClaw lifecycle hooks for recall and capture
- Neo4j schema initialization
- Writer extraction through Anthropic or OpenAI
- Voyage AI or OpenAI embeddings
- Hybrid vector plus graph retrieval that prefers currently valid facts
- Editor pipeline for dedup, classification, categories, temporal supersession, three-tier entity resolution, memory relationships, entity links, and confidence updates
- Idempotent writes (session key + fact hash) and Editor reruns
- Per-step LLM cost and latency table on every Editor run
- EditAction audit log with one-command rollback of an Editor run
- Eval harness with recorded calls, replayed in CI with threshold checks

Next improvements:

- Give the Writer the session date and show memory dates in recalled context (LoCoMo temporal: 8.1%)
- Denser extraction as an option, measured on LoCoMo against its cost
- History-aware retrieval for "what was it before?" questions
- Keep implied facts apart in dedup ("accepted an offer" vs "works there")
- Add category-aware retrieval routing
- Add a `before_compaction` capture hook

## License

MIT. See [LICENSE](LICENSE).
