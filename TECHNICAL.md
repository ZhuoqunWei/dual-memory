# Dual-Memory: A Two-Agent Knowledge Graph for Personal AI

## What is this?

A memory system for AI assistants that actually works. Instead of dumping chat logs into a flat file and hoping for the best, Dual-Memory builds a structured knowledge graph from conversations — extracting facts, tracking entities and their relationships, deduplicating noise, and surfacing the right context at the right time.

Built as an [OpenClaw](https://openclaw.dev) plugin + a standalone Python editor agent, backed by Neo4j.

## Architecture

```
Conversation
     │
     ▼
┌──────────┐     extract facts      ┌─────────┐
│  Writer   │ ──────────────────────▶│  Neo4j  │
│ (TypeScript)                       │  Graph  │
│  OpenClaw │ ◀──────────────────────│         │
│  Plugin   │     retrieve context   │ Memory  │
└──────────┘                         │ Entity  │
                                     │ Session │
┌──────────┐     consolidate         │Category │
│  Editor   │ ──────────────────────▶│EditAction│
│ (Python)  │     dedup, merge,      └─────────┘
│ Standalone│     categorize
└──────────┘
```

**Writer** — fast, automatic. Hooks into the AI assistant's lifecycle. When a conversation ends, it extracts facts via LLM, generates embeddings, and writes structured memory nodes to Neo4j. Before the next conversation starts, it retrieves relevant memories and injects them as context.

**Editor** — slow, deliberate. Runs on schedule (nightly or on-demand). Processes raw memories through a 9-step pipeline: deduplication, classification, categorization, temporal supersession, entity resolution, relationship mining, confidence scoring. Turns a messy pile of extractions into a clean knowledge graph. Every run can be rolled back, and every step reports its LLM calls, tokens, cost, and time.

**Eval harness** — a gold-labelled fixture run through the real Writer write path, the Editor, and the real retrieval, scoring dedup, supersession, entity resolution, and retrieval. See `eval/README.md`.

## The Graph Schema

```
(Memory)-[:MENTIONS {role}]->(Entity)
(Memory)-[:RELATES_TO {type, weight}]->(Memory)
(Memory)-[:PART_OF]->(Session)
(Memory)-[:IN_CATEGORY]->(Category)
(Memory)-[:CANONICAL]->(Memory)          -- dedup pointer
(Memory)-[:SUPERSEDES]->(Memory)         -- newer value of the same attribute
(Entity)-[:LINKED_TO {relation}]->(Entity)  -- e.g. friend, teacher
(Entity)-[:MERGED_INTO]->(Entity)
(Entity)-[:DISTINCT_FROM]->(Entity)      -- judged different; not asked again
(EditAction {runId, seq, targets, undo})  -- one per change set, per EditorRun
```

**Memory** nodes carry: content, kind (fact/decision/goal/preference/emotion/observation/event), confidence, salience, embeddings, provenance (sourceQuote, sourceChannel, sourceAuthor), temporal bounds, and validity (`validFrom` when recorded, `validTo` once superseded). `writeKey` = sha256(session, normalized fact) makes rewriting a session a no-op.

**Entity** nodes carry: name, type (Person/Place/Project/Organization/Tool/Concept), aliases (including multilingual or shortened names that resolve to the same person).

**Retrieval scoring**: `cosine × (0.6 + 0.4 × confidence) × (0.6 + 0.4 × salience) × validity` — floor-clamped so new memories can surface while the Editor boosts important ones over time; `validity` is 0.7 for superseded memories, which are also marked "[outdated since …]" in the injected context. Neo4j's vector index reports `(1 + cos) / 2`, so scores are converted back to cosine first.

## Editor Pipeline

| Step | Method | Purpose |
|------|--------|---------|
| Dedup | cosine + LLM | Tier 1: exact (>=0.98), Tier 2: semantic (0.80-0.98 with LLM judge: duplicate / update / related / distinct) |
| Classify | rule-based | Normalize kind/type to canonical enums |
| Categories | LLM | Bottom-up clustering, quorum ≥ 3 to create category |
| Contradictions | cosine + LLM | Nearest still-valid neighbours sharing an entity → update or compatible; the newer memory SUPERSEDES the older and sets its `validTo` |
| Entity Resolution | names → embeddings → LLM | Name match or mutual aliases; name+type+alias embeddings (≥0.95 merge, 0.80-0.95 to the LLM); LLM with mentioning memories. "different" verdicts are cannot-link constraints |
| Relationships | LLM | RELATES_TO edges between memories sharing a non-hub entity, each pair asked once |
| Entity Links | LLM | LINKED_TO edges between entities (friend, teacher, etc.) |
| Confidence | algorithmic | Multi-session boost +0.1 (once), decay -0.05 (once per day) |
| Mark Reviewed | algorithmic | status: raw → reviewed |

Each step is independently runnable via `--step <name>`. Every mutation is journaled on an EditAction; `--rollback <run_id>` undoes a run.

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Writer | TypeScript, OpenClaw plugin SDK |
| Editor | Python 3.11+, standalone process |
| Graph DB | Neo4j 2026 (Docker) |
| Embeddings | Voyage AI voyage-4-lite (1024d) |
| Extraction LLM | Claude Sonnet 5 (Anthropic) |
| Editor LLM | Claude Sonnet 5 at low effort (Anthropic) |

## Implementation Snapshot

- Roughly **4.4k lines** across the TypeScript Writer plugin and Python Editor agent.
- One Neo4j-only TypeScript smoke test for graph operations without API calls.
- One full integration path for OpenAI-backed extraction and embeddings.
- Eval harness with recorded LLM/embedding calls, replayed in CI against a Neo4j service container, plus idempotency and rollback integration tests that use a deterministic fake LLM.
- Editor cost scales with raw memory volume and LLM-dependent steps; the Writer uses one extraction call per captured conversation.

## Key Design Decisions

**Why Neo4j over vector-only (Pinecone, LanceDB)?** Vector similarity finds "related" content. A graph finds "connected" content — causal chains, contradictions, entity networks. "Who introduced Alice to the research group?" is a graph query, not a vector query.

**Why two agents?** The Writer needs to be fast (runs inline with conversation). The Editor needs to be thorough (LLM-assisted dedup, cross-referencing). Different time pressures, different architectures. The Writer errs on over-extraction; the Editor cleans up.

**Why floor-clamped scoring?** Pure `similarity × confidence × salience` with default values (0.5, 0.5) gives `similarity × 0.25` — buries new memories before the Editor has a chance to boost them. Floor clamping gives `similarity × 0.64` minimum.

**Why not Kafka/message queue?** Single user, single instance. The `agent_end` hook runs post-conversation with no backpressure concern. Direct Neo4j writes are simpler and debuggable. Can add a queue later (it won't be needed).

## Running It

```bash
# Neo4j
cd infrastructure
docker compose up -d
cd ..

# Editor (one-time cleanup)
cd editor && source .venv/bin/activate
python -m editor.main --once --all

# Editor (stats)
python -m editor.main --stats

# Editor runs, and undoing one
python -m editor.main --runs
python -m editor.main --rollback <run_id>

# Eval harness (scratch Neo4j on :7688)
docker compose -f ../eval/docker-compose.yml up -d
python -m editor.evaluation --check

# Writer is auto-loaded by OpenClaw gateway on restart
```

## What's Next

- A held-out fixture and a public benchmark run (LongMemEval, LoCoMo); today's thresholds were chosen looking at the one fixture
- History-aware retrieval: "what was it before?" questions lose to the superseded weight
- Dedup that keeps implied facts apart ("accepted an offer" vs "works there")
- Category-scoped retrieval (categories exist but not yet used in search routing)
- `before_compaction` emergency hook (save memories before OpenClaw truncates context)
