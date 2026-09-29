# Development Log — Dual-Agent Memory System

Captures decisions, discoveries, and gotchas as they happen.

---

## 2026-02-18 — Phase 0 Complete

### Key Discovery: OpenClaw Plugin API is richer than documented

Public docs suggest limited hooks (only `command:new`, `agent:bootstrap`, `gateway:startup`). Actual source code at `/opt/homebrew/lib/node_modules/openclaw/dist/plugin-sdk/index.d.ts` reveals **14 lifecycle hooks**:

```
before_agent_start | agent_end | before_compaction | after_compaction
message_received | message_sending | message_sent
before_tool_call | after_tool_call | tool_result_persist
session_start | session_end | gateway_start | gateway_stop
```

This changes the architecture — no file watcher needed. `agent_end` gives us full conversation messages.

### Key Discovery: memory-lancedb is our blueprint

The existing LanceDB memory plugin at `extensions/memory-lancedb/index.ts` already uses:
- `api.on("before_agent_start", ...)` for auto-recall (returns `{ prependContext: "..." }`)
- `api.on("agent_end", ...)` for auto-capture (processes `event.messages`)
- `api.registerTool(...)` for explicit tools (memory_recall, memory_store, memory_forget)
- `api.registerService(...)` for lifecycle management
- `api.registerCli(...)` for CLI commands

We follow this exact pattern but swap LanceDB for Neo4j.

### Key Discovery: Plugin package structure

```
extensions/<plugin-name>/
├── index.ts                # Default export with { id, name, kind, configSchema, register(api) }
├── openclaw.plugin.json    # { id, kind, configSchema }
├── package.json            # { openclaw: { extensions: ["./index.ts"] } }
└── ...other source files
```

Plugin is loaded via `jiti` (TypeScript JIT). Uses `@sinclair/typebox` for parameter schemas. Config supports `${ENV_VAR}` resolution.

### Decision: No Kafka for Phase 1

Original design included Kafka. Unnecessary for MVP:
- `agent_end` hook runs post-session, not during — no backpressure concern
- Direct Neo4j writes are simpler
- Single-user scale doesn't need message bus
- Can add Kafka in Phase 2 if needed (it won't be)

### Decision: Pre-filter retrieval (Approach A)

For two-phase retrieval, use graph traversal first (Category → Memory), then compute cosine similarity within that subgraph. Not ANN index search with post-filter.

Reason: each category will have hundreds to low thousands of memories. Exact computation is fast at this scale, deterministic, and leverages the directory layer (which is the whole point).

### Workspace observations

Observed an existing OpenClaw workspace with:
- Manually curated memory file — bilingual, tracks people/servers/chat rules
- Daily session logs (only a few days old)
- Agent personality file — casual, bilingual (Chinese/English)
- Multi-platform integration active (Discord, CLI)
- The "lobotomy problem" hasn't hit yet (small log volume), but the flat Markdown structure will degrade fast with daily use

---

## 2026-02-18 — Schema Deep Design

### The core tension

Schema must be rich enough for deep queries (causal chains, emotion tracking, cross-session reasoning) but simple enough that a Writer LLM can reliably produce correct output. Tested against three user profiles (student, founder, researcher) to ensure generality.

### Decision: Memory.kind as open string, not separate node types

Debated: should Decision, Event, Emotion be separate node types?

**No.** All are Memory nodes with a `kind` field. Reasons:
- Writer LLM only needs to classify one field, not decide which node type to create
- Adding new kinds (e.g., "hypothesis" for a researcher) = zero schema change
- Queries like "all decisions" = `WHERE kind = 'decision'` — simple
- Editor can reclassify kind without deleting/recreating nodes

### Decision: One RELATES_TO with type field, not separate relationship types

Debated: `CAUSED_BY`, `FOLLOWS`, `CONTRADICTS` as separate Neo4j relationship types vs one `RELATES_TO` with a `type` property.

Chose **single RELATES_TO**. At personal graph scale (<100K nodes), zero performance difference. But gains:
- Adding new relation types = new string value, no migration
- Weight property on every relation enables cross-type ranking
- Writer can produce types the Editor hasn't seen yet
- Simpler Cypher for multi-type traversal

### Decision: Entity.aliases as String[] for multilingual

"老王" and "Wang Wei" must resolve to the same Entity. `aliases` array is the simplest correct solution. Writer writes both names, Editor merges on alias overlap.

Not a separate AliasNode — that adds joins for zero benefit at this scale.

### Decision: MENTIONS.role limited to 4 values

subject | object | context | source. Tested with real conversations from the user's daily logs. These 4 cover every case encountered. More granular roles (instrument, beneficiary, location) would tank LLM extraction accuracy.

### Validated against real query scenarios

All five target queries work cleanly against the schema:
1. Learning progress → kind filter + Entity.type + temporal
2. Decisions + reasons → kind='decision' + RELATES_TO {type: 'caused_by'}
3. Person × topic cross → Entity alias match + Entity.type filter
4. Emotion tracking → kind='emotion' + temporal + causal chain
5. Cross-session reasoning → RELATES_TO chain traversal with temporal bounds

---

## 2026-02-18 — Schema v2: Peer Review Patches

External review identified 6 gaps in v1. All addressed:

### Patch 1: sourceRef (provenance pointer)
Added `sourceRef` (e.g. "discord:channelId:messageId") to Memory. Without it, `sourceQuote` alone can't pinpoint the exact message in a 50-message session. Essential for "replay context" and debugging extraction errors.

### Patch 2: eventTimeStart/End (time ranges)
Replaced single `eventTime` with `eventTimeStart` + `eventTimeEnd`. Many memories are naturally ranges: "spent February reading DDIA", "in NYC this week", "currently job hunting". `eventTimeEnd = null` means point-in-time.

### Patch 3: LINKED_TO.active — Editor-only with consistency rules
Added `active` (Boolean) and `until` (DateTime?) to LINKED_TO. Rules:
- Writer **never** writes `active`. It always defaults to `true`.
- Editor maintains `active`. Can set `false` without `until` (ended but time unknown).
- `active=false` + `until=null` is valid. `active=true` + `until != null` is invalid (Editor must enforce).

### Patch 4: Retrieval scoring formula — floor clamp
Changed from `sim × confidence × salience` to `sim × (0.6 + 0.4*conf) × (0.6 + 0.4*sal)`.
Pure multiplication with defaults (0.5, 0.5) gives `sim × 0.25` — buries new memories. Floor clamp gives `sim × 0.64` — new memories can still surface while Editor boosts important ones.

### Patch 5: Canonical pointer on Memory
Originally added as `canonicalId` string field. Later upgraded to `CANONICAL` relationship in v2.1 (see Patch 7) to avoid string↔edge drift. When Editor supersedes a memory, old one gets `CANONICAL` edge pointing to new one. Retrieval filters `WHERE NOT (m)-[:CANONICAL]->()`. Avoids walking supersedes chains at query time.

### Patch 6: status += suppressed
Added `suppressed` to status enum. Covers: privacy requests, extraction errors, user deletion. Suppressed memories are excluded from retrieval but retained for audit. Different from `archived` (old but possibly true).

### Bonus: RELATES_TO type naming convention
Split into fact-layer (bare names: caused_by, follows, etc.) and editor-layer (`editor:` prefix: editor:summarizes, editor:supersedes). Default retrieval queries filter `WHERE NOT type STARTS WITH 'editor:'`. Prevents version/aggregation edges from polluting reasoning chains.

### Decision: normalizedKind canonical set
Writer's `kind` is open. Editor's `normalizedKind` converges to: fact | decision | preference | goal | emotion | observation | event. Max ~10-15 values. If this set grows past 20, something is wrong and the Editor prompt needs tightening.

---

## 2026-02-18 — Schema v2.1: Pre-Commit Hardening

Second review round. Focused on eliminating redundancy and strengthening enforcement.

### Patch 7: canonicalId → CANONICAL relationship
Removed `canonicalId` string field from Memory. Replaced with `(Memory)-[:CANONICAL]->(Memory)` relationship. Reasoning: same redundancy problem we already solved for sourceSession — if you have both a string pointer and a relationship, they will eventually drift. `editor:supersedes` remains as the version chain / audit trail. `CANONICAL` is the O(1) index for retrieval filtering.

### Patch 8: eventTimeStart null = unknown
Changed semantics from "null = same as timestamp" to "null = unknown/not specified". The old definition created false certainty — if Writer can't extract event time, that's unknown, not "it happened right now". If known to equal extraction time, set it explicitly.

### Patch 9: sourceAuthor provenance field
Added `sourceAuthor: String?` to Memory. Hashed user ID or speaker identifier. Disambiguates multi-speaker channels (Discord, group chats). Essential for "who said this?" queries and trust scoring.

### Patch 10: Validation rules (Writer vs Editor boundary)
Added mechanical enforcement section to design doc. Key rules:
- Writer MUST NOT create `editor:*` RELATES_TO types, `CANONICAL` edges, or write `LINKED_TO.active`
- Writer SHOULD create at least one `MENTIONS {role: 'subject'}` per Memory
- Editor MUST NOT modify provenance fields (immutable)
- Editor MUST log every mutation as EditAction
- All `RELATES_TO` edges require `weight` — no weightless edges

### Patch 11: Constraints & indexes
Added full constraint/index specification to design doc and updated `init-schema.cypher`:
- Uniqueness: Memory.id, Entity.id, Session.id, EditAction.id, Category.name
- Performance: Memory(status, timestamp, eventTimeStart, expiresAt, normalizedKind), Entity(name, normalizedType)
- Fulltext: Entity.name for alias resolution
- Vector: both Memory.embedding and Entity.embedding

### Housekeeping: PII sanitization
Replaced all personal names, workspace paths, and identifying details with generic examples before committing to version control.

---

## 2026-02-19 — Phase 1: Writer MVP

### Architecture
OpenClaw plugin with 3 lifecycle integration points:
- `before_agent_start` → retrieval (inject relevant memories into agent context)
- `agent_end` → writer (LLM fact extraction → Neo4j writes)
- 3 agent tools: `memory_graph_search`, `memory_graph_store`, `memory_graph_forget`

### Files built
```
extensions/dual-memory/
├── index.ts              # Plugin entry: hooks, tools, CLI, service lifecycle
├── neo4j-client.ts       # Neo4j driver wrapper: schema init, CRUD, retrieval
├── writer.ts             # LLM fact extraction (Anthropic + OpenAI support)
├── retrieval.ts          # Two-phase retrieval + context formatting
├── embeddings.ts         # Voyage AI / OpenAI embeddings wrapper
├── config.ts             # Config parsing with env var resolution
├── openclaw.plugin.json  # Plugin manifest
├── package.json          # Dependencies: neo4j-driver, openai, voyageai, @anthropic-ai/sdk
├── test-neo4j.ts         # Integration test (Neo4j only, mock embeddings)
└── test-integration.ts   # Full integration test (requires API keys)
```

### Key decisions

**Dual LLM provider support.** User uses Claude (Anthropic) for conversations. Writer extraction now supports both Anthropic and OpenAI as extraction providers.

**Application-layer filtering for ANN queries.** Neo4j 2026.01.4 has syntax limitations on `WHERE` clauses after `CALL ... YIELD`. Moved status/canonical/expiry filtering to application layer with 3x over-fetch to compensate. Entity fetching is a separate query.

**No CANONICAL check in ANN.** The ANN vector search path doesn't filter CANONICAL relationships in Cypher (would require pattern existence check that Neo4j rejects after YIELD). Filtering happens application-side. For category-scoped queries (Phase 2), the `NOT EXISTS` syntax works.

### Integration test results
All passing against live Neo4j 2026.01.4:
- ✅ Schema initialization (17 indexes + constraints)
- ✅ Session creation
- ✅ Fact writing with entities and MENTIONS relationships
- ✅ ANN vector retrieval with floor-clamped scoring
- ✅ Entity attachment to retrieval results
- ✅ Memory suppression (soft delete) correctly excludes from retrieval
- ✅ Single memory store (explicit tool path)
- ✅ TypeScript type-checking clean (zero errors)

### What's NOT done yet
- No `before_compaction` emergency hook (Phase 3)
- No category-scoped retrieval (no categories exist until Editor creates them)
- No entity embedding generation (entities stored without embeddings for now)
- No RELATES_TO creation between facts (Writer extracts them but doesn't write them yet — needs content-matching logic)
- Not yet installed as an actual OpenClaw plugin (need API keys configured)

---

## 2026-02-19 — Voyage AI Embeddings

### Decision: Voyage AI as default embedding provider

Anthropic has no embeddings API. The choices were:
- **Voyage AI** — Anthropic's recommended embedding partner. `voyage-4-lite` (1024d) — optimized for retrieval tasks, cheaper than OpenAI, lower dimensionality means faster vector search and smaller indexes
- **OpenAI** — `text-embedding-3-small` (1536d) — solid but requires an OpenAI API key just for embeddings
- **Local** — Ollama + nomic-embed-text — zero cost but adds deployment complexity

Chose Voyage AI. Reasons:
1. Anthropic recommends them — semantic alignment with the extraction LLM (Claude)
2. 1024d vs 1536d means ~33% smaller vector indexes, faster ANN search
3. Batch embedding natively supported (no chunking needed)
4. One less OpenAI dependency for users who are all-in on Anthropic

### Changes
- Rewrote `embeddings.ts` — dual provider support (Voyage AI + OpenAI)
- Updated `config.ts` — `embedding.provider: "voyage" | "openai"` (default: "voyage"), env var resolution for `VOYAGE_API_KEY`
- Updated `neo4j-client.ts` — dynamic `vectorDims` constructor param (default 1024)
- Updated `index.ts` — passes `vectorDimsForModel()` result to Neo4j client
- Updated `openclaw.plugin.json` — added embedding provider option, Voyage models
- Dropped old 1536d vector indexes, recreated at 1024d
- Updated `init-schema.cypher` to default to 1024d
- Updated `test-neo4j.ts` to use 1024d mock embeddings
- Installed `voyageai` npm package
- All tests passing, TypeScript compiles clean

### Supported embedding models
| Provider | Model | Dimensions |
|----------|-------|-----------|
| Voyage AI | voyage-4-lite (default) | 1024 |
| Voyage AI | voyage-4 | 1024 |
| Voyage AI | voyage-4-large | 1024 |
| Voyage AI | voyage-code-3 | 1024 |
| Voyage AI | voyage-3-lite | 512 |
| Voyage AI | voyage-3 | 1024 |
| OpenAI | text-embedding-3-small | 1536 |
| OpenAI | text-embedding-3-large | 3072 |

---

## 2026-02-19 — Phase 2: Editor Agent

### Architecture
Standalone Python process (`editor/editor/`) connecting to the same Neo4j instance. Runs via `python -m editor.main --once --all`. Uses venv at `editor/.venv/` with Python 3.13 (homebrew).

### 9-Step Pipeline
| Step | Type | What it does |
|------|------|-------------|
| dedup | algorithmic + LLM | Tier 1: cosine >= 0.98 exact archive. Tier 2: 0.80-0.98 LLM-assisted |
| classify | rule-based | Normalize kind/type to canonical sets |
| categories | LLM | Cluster similar memories, create Category nodes (quorum ≥ 3) |
| contradictions | LLM | Detect conflicting facts, lower confidence |
| entity_resolution | algorithmic | Alias overlap merge → MERGED_INTO edge |
| relationships | LLM | Create RELATES_TO edges between co-entity memories |
| entity_links | LLM | Create LINKED_TO edges between entities (friend, teacher, etc.) |
| confidence | algorithmic | Multi-session boost +0.1, decay -0.05/cycle |
| mark_reviewed | algorithmic | Set status = 'reviewed' |

### First Run Results
- 190 raw memories → 52 surviving (73% dedup rate)
- 25 entity merges, 558 RELATES_TO edges, 179 EditActions
- Second run (after more sessions): 300 total → 54 active (82% cleanup rate)

### Neo4j 2026 Gotchas
- **`count(*)` with OPTIONAL MATCH**: Always returns ≥ 1 (counts the row). Must count the relationship variable: `count(c)` not `count(*)`.
- **`NOT IN` invalid syntax**: Use `WHERE m.status IN ['raw', 'reviewed']` instead of `NOT IN ['archived', 'suppressed']`.
- **Cartesian product warnings** on `MATCH (a {id: $x}), (b {id: $y})`: Harmless for ID lookups, safe to ignore.

---

## 2026-02-19 — LINKED_TO Entity Relationships

### Problem
Schema designed LINKED_TO edges (Entity ↔ Entity) in v2.1 but never implemented. Retrieval showed entities but not how they relate (for example, friend or teacher relationships between people).

### Implementation (3 layers)
- **Editor** (`steps/entity_links.py`): Mines co-mentioned entity pairs from memories, LLM classifies relationship type. Batches 8 pairs/call.
- **Writer** (`writer.ts`): Added `entityLinks` to extraction prompt + `EntityLink` type. LLM extracts relationships alongside facts.
- **Writer persistence** (`neo4j-client.ts`): `writeEntityLinks()` using `MERGE (a)-[r:LINKED_TO]->(b)`.
- **Retrieval** (`neo4j-client.ts` + `retrieval.ts`): Fetches LINKED_TO edges via entity queries, formats relationships such as "Alice is colleague of Bob".

### Results
17 LINKED_TO edges created across creator, teacher, friend, and acquaintance-style relationships.

---

## 2026-02-19 — Salience Filtering (Cost Reduction)

### Problem
Writer prompt said "prefer over-extraction to under-extraction" → ~8 facts/session, 73% archived as duplicates. Every fact costs: embedding + Neo4j write + downstream Editor LLM calls.

### Fix (Writer-side only, zero new LLM calls)
1. **Prompt tightened**: "Extract ONLY facts worth remembering long-term." Added 6 skip criteria (greetings, meta-conversation, mechanical steps, obvious facts, repeated facts, salience < 0.50). Added target "3-6 facts per conversation" + counter-example showing empty extraction.
2. **Post-extraction filter**: `SALIENCE_FLOOR = 0.50` — drops facts the LLM itself labeled as trivial before embedding/persisting.
3. **Logging**: Shows `extracted N facts, persisted M memories` for monitoring.

### Entity Cleanup
- Merged duplicate person entities through aliases (MERGED_INTO edge already existed)
- Deleted synthetic test entities from local experiments
- Archived 9 meta-observation memories (system documenting its own schema)

---

## 2026-09-28 — Eval Harness, Supersession, Entity Resolution, Idempotency, Rollback

### Why
Every number the project had was a count ("73% archived as duplicates"), not a correctness claim. Built a gold-labelled harness first, then measured each change against it. Full results: `eval/results/`, summary in the README.

### Harness (`editor/editor/evaluation/`, `eval/`)
- 40 synthetic sessions, 102 facts (85 unique), 13 update chains, 64 entity names for 53 real entities, 50 retrieval questions. Facts go in through the real TypeScript `writeFacts` and questions through the real hybrid `retrieve()` via `eval-bridge.ts`, so it measures the shipped code, not a Python copy.
- LLM replies and embeddings are recorded to `eval/cache/` keyed by a hash of the full request. CI replays against a Neo4j service container with no keys; a changed prompt fails as a cache miss. The miss is a `BaseException` on purpose: steps catch `Exception`, which would have turned a stale cache into a silently skipped step.
- Prompts had to be byte-stable across runs, which meant `ORDER BY` on every Editor read that feeds a prompt (several ordered by random UUIDs or not at all).
- The harness wipes its database, so it refuses any non-empty database without its `:EvalSentinel` node, and reads its connection only from `EVAL_NEO4J_*`, never `editor/.env` (which points at the real graph).

### What the harness found
- **Neo4j vector scores are `(1 + cos) / 2`, not cosine.** Both `db.index.vector.queryNodes` and `vector.similarity.cosine()`. The Writer's pre-write gate at "0.92" was really cosine 0.84, and it silently dropped updates: "User now lives in Portland, Oregon" never reached the graph because it was close to "User lives in Seattle". 22 of 102 facts skipped at 54.8% dedup precision. Hybrid retrieval also ranked vector seeds (normalized score) against graph expansions (raw cosine). No cosine threshold separates restatements from updates (updates up to 0.945, real paraphrases down to 0.88), so the gate is now same-session only, where the re-extraction duplicates come from, and cross-session dedup is the Editor's LLM tier.
- **Relationships step was 95% of Editor spend** (227 of 270 LLM calls, $1.03 of $1.09 per run) because the user-as-hub entity links nearly every pair. Pairs now come from non-hub entities only (same 30-mention cut-off retrieval uses), each asked once. $1.09 → $0.12 with every quality metric unchanged.
- **Dedup's LLM calls updates "duplicates"** ("B reflects updated fact about Tidepool's language, superseding A") and archives the history away before supersession sees it. Added an explicit `update` verdict and a stricter duplicate definition.
- **Models sometimes wrap the verdict list** as `{"results": [...]}`; dedup then treated the whole batch as distinct and zipped verdicts to pairs by position. Now matched by pair number (`verdicts_by_pair`).
- **`merge_entities` dropped the target's aliases** (`[a IN tgt.aliases + src.aliases WHERE NOT a IN tgt.aliases]` keeps only the new ones) and CREATEd duplicate MENTIONS.
- **Confidence decay ran on every run**, and `--once --all` is one run per 50-memory batch, so a nightly run decayed several times.
- **`--all --step X` looped forever** (a single step never marks the batch reviewed). Now rejected.
- **`claude-sonnet-4-20250514` is retired** (404). Default is now `claude-sonnet-5`, which rejects `temperature` and thinks by default; clients strip sampling params and run adaptive thinking at low effort.

### Temporal supersession
`validFrom` on write; the contradictions step asks about each batch memory's nearest still-valid neighbours sharing an entity and labels pairs update/compatible (naming the attribute first; asking for it cut wrong edges from 7 to 3). Newer supersedes older: `SUPERSEDES` edge plus `validTo`, no confidence penalty. Retrieval weights superseded memories ×0.7 and marks them "[outdated since …]" in context. The weight is a real trade: 0.5 buried history questions, 0.8 let stale facts rank first again.

### Three-tier entity resolution
Name tier (normalized equality or mutual aliases) → embedding tier → LLM. The plan said "name plus context"; measured on the fixture, adding context made names from the same memory look identical (Framework Laptop / Fedora 0.967, above every true alias pair), so the embedding is name + type + aliases and context goes to the LLM. Two more guards from the data: names mentioned in one memory are never merged, and LLM "different" verdicts are cannot-link constraints (a bare "Mei" matched to both "Mei Ortiz" and "Mei Lin" had merged the sister with the coworker through the union-find).

### Idempotency and rollback
Writer: Session MERGEd on OpenClaw's session id, Memory MERGEd on `writeKey = sha256(session, normalized fact)`, MENTIONS/PART_OF MERGEd, new mentions follow `MERGED_INTO`. Editor: decay once per day, boost once, MERGE everywhere. Each run is an `EditorRun`; created relationships/nodes carry `editorRun`, property changes and deleted relationships are journaled as undo ops on EditActions; `--rollback <run>` replays them newest-first and deletes the tagged items in one transaction. Integration tests (`tests/test_idempotency.py`, `tests/test_rollback.py`) use a deterministic fake LLM; the Writer and decay tests fail on the pre-change code.

### Model and effort, chosen by the harness
| Editor model | Dedup P | Contradiction P / R | Current facts wrongly superseded | Recall@5 | Cost / run |
|---|---|---|---|---|---|
| Sonnet 5, low effort | 81% | 77% / 87% | 0% | 80% | $0.32 |
| Sonnet 5, medium effort | 89% | 89% / 100% | 0% | 86% | $0.41 |
| Haiku 4.5 | 89% | 56% / 80% | 23% | 75% | $0.14 |

Default effort is now medium. Haiku 4.5 is cheapest but marks almost a quarter of current facts outdated, which is the failure that hurts retrieval most. (`editor/.env` in my setup still pins Haiku; worth switching.) The review's "< $0.08 per run" target isn't met on this fixture: it is 102 facts in one run, about $0.004 per memory; a nightly run over a day's 10-20 new memories lands around $0.04-0.08.

The medium run exposed one more chain: the LLM said "VS Code" = "nvim", the embedding tier merged "nvim" into "Neovim", and the editors the user switched between became one entity. Names mentioned in one memory are now cannot-link constraints like "different" verdicts (run 08: wrong merges back to 0/9).

### Honest caveats
- Thresholds and prompts were tuned while looking at this fixture; there is no held-out split. Three gold labels were corrected after seeing results (listed in `eval/README.md`).
- Single-sample numbers: contradiction precision moved 85% → 77% between two runs whose only change was entity resolution, from different candidate pairs and LLM variance.
- Retrieval recall@5 on "what was it before?" questions is 29%: they lose to the superseded weight. Dedup still merges "accepted an offer" into "works there" at every setting tried.

---
