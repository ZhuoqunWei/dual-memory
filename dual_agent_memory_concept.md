# Dual-Agent Self-Organizing Memory for AI Agents
## Project Concept & Design Document

*February 2026*

---

## The Problem

AI agents forget things.

Not because they're stupid — because their memory systems aren't designed to last. Most agent frameworks, including OpenClaw, store memory as plain text files or flat vector databases. Every session adds more content. Over time the memory gets noisy: duplicate facts, contradictions, stale information, no sense of what's important vs what's throwaway. When the context window fills up, everything gets compacted and the agent effectively gets a lobotomy. The community that uses OpenClaw has a name for this: the lobotomy problem.

The deeper issue is that current memory systems are **write-only**. Things go in, nothing gets cleaned up. There's no process that looks at the accumulated memory and asks: is this still true? have we seen this before? where does this belong? The memory grows but it doesn't mature.

---

## The Insight

Human memory doesn't work like a write-only log. It has two distinct processes running at different speeds:

**Episodic memory** — fast, automatic, captures what just happened. You don't have to try to remember your day. It just gets recorded.

**Semantic memory** — slow, deliberate, consolidates episodic experience into durable knowledge. This is the process that takes "I burned my hand on the stove three times" and turns it into "stoves are hot, be careful." It runs mostly during sleep.

The interesting thing is that episodic memory is weak — high recall, low precision, full of noise. Semantic memory is strong — lower recall, much higher precision, organized for retrieval. The strong emerges from the weak over time, through a consolidation process.

This is the architecture worth building.

---

## The Idea: Two Agents, One Graph

Instead of one agent doing everything, split the memory responsibility across two agents with fundamentally different roles:

### The Writer
Runs after every conversation session. Its job is capture — extract facts from what just happened and write them to the knowledge graph. It operates fast, it's allowed to be messy, it never deletes anything. It's optimized for coverage. Think of it as the journalist: get it down, get it all down, don't overthink it.

### The Editor
Runs on a schedule — nightly, or when the graph hits a size threshold. Its job is curation — read what the Writer produced, find problems, fix them, and reorganize the graph so future retrieval is faster and more accurate. It has something the Writer never has: perspective across time. It can see that two nodes are describing the same person, that a fact recorded last Tuesday was contradicted by something recorded yesterday, that a cluster of memories all belong under the same category. Think of it as the editor: slow down, ask if this is actually right, make it coherent.

The Writer is weak in the sense that it only ever sees one session at a time. The Editor is strong not because it's a bigger model, but because it sees the whole graph — all sessions, all time. That broader context is what makes it capable of producing organization that neither agent could achieve alone. This is the same intuition behind weak-to-strong generalization in AI alignment research: a weaker supervisor can guide a stronger model to generalize beyond what the supervisor itself knows.

---

## The Knowledge Graph

The memory lives in Neo4j as a property graph. This matters — a graph is the right data structure here because memory is fundamentally relational. Facts connect to entities, entities connect to each other, categories contain memories, and those connections are as important as the content itself.

### Design Principles

1. **Few node types, rich relationships.** Nodes are nouns. Relationships carry the intelligence.
2. **Open enums with normalization.** `Memory.kind` and `Entity.type` are open strings from the Writer. `normalizedKind` / `normalizedType` are Editor-controlled stable values for reliable queries. Writer generates freely; Editor converges to a canonical set.
3. **Every relationship has weight.** Importance is always queryable and rankable.
4. **Aliases on Entity.** Multilingual by default. "老王" and "Wang Wei" are the same person.
5. **No domain-specific node types.** A "Decision" is `Memory {kind: "decision"}`, not a Decision node. This keeps the schema universal.
6. **Provenance on everything.** Every Memory traces back to a specific message. Every category assignment traces to who assigned it.

### Node Types

**Memory** — a single extracted fact, decision, observation, or emotional state.
```
(Memory {
  id:                   String      // UUID
  content:              String      // "Alice decided to switch majors, motivated by career and personal growth"

  // --- Classification (dual-track) ---
  kind:                 String      // Writer's raw classification: fact | decision | preference | goal | emotion | observation | event
                                    // Open enum — Writer can produce new kinds
  normalizedKind:       String?     // Editor's stable classification (null = not yet reviewed)
                                    // Converges to canonical set: fact | decision | preference | goal | emotion | observation | event

  // --- Temporal (dual timestamps) ---
  timestamp:            DateTime    // When extracted (system time)
  eventTimeStart:       DateTime?   // When the event happened (null = unknown/not specified)
                                    // If known to equal extraction time, set explicitly to timestamp value
  eventTimeEnd:         DateTime?   // End of event time range (null = point-in-time or unknown)
  expiresAt:            DateTime?   // Expiry for time-sensitive memories (null = never expires)

  // --- Scoring ---
  confidence:           Float       // 0.0-1.0, certainty that this fact is true. Starts at 0.5.
  salience:             Float       // 0.0-1.0, importance/relevance. Starts at 0.5.
                                    // confidence × salience are orthogonal: "Alice ate a sandwich" = high confidence, low salience
                                    // "Alice might be considering switching careers" = low confidence, high salience

  // --- Lifecycle ---
  status:               String      // raw | reviewed | archived | suppressed
                                    // suppressed = excluded from retrieval but retained for audit (privacy, errors, user request)
  lastAccessed:         DateTime?   // Last retrieval time (null = never retrieved)

  // --- Provenance ---
  sourceRef:            String?     // Stable pointer: "discord:channelId:messageId" or "cli:sessionId:turnN"
  sourceQuote:          String?     // Original text snippet (capped length) for auditability
  sourceChannel:        String?     // Channel where this was captured: discord | telegram | cli
  sourceAuthor:         String?     // Who said it: hashed user ID or speaker identifier
                                    // Disambiguates multi-speaker channels. Hash for privacy.

  // --- Search ---
  embedding:            Vector      // 1536d, text-embedding-3-small
})
```

**Entity** — a named thing that memories refer to.
```
(Entity {
  id:                   String      // UUID
  name:                 String      // Primary display name: "老王"
  type:                 String      // Writer's raw type
  normalizedType:       String?     // Editor's stable type (null = not yet reviewed)
                                    // Canonical set: Person | Place | Project | Organization | Tool | Concept
  aliases:              String[]    // ["Wang Wei", "老王", "wangwei"] — multilingual
  key:                  String?     // Stable external identifier for disambiguation
                                    // e.g. "person:phone_hash", "org:domain", "project:repo_url"
                                    // null = no external key available
  firstSeen:            DateTime    // First appearance in the graph
  embedding:            Vector      // For entity disambiguation
})
```

**Session** — a record of one conversation.
```
(Session {
  id:                   String
  date:                 DateTime
  summary:              String
  messageCount:         Int
  channel:              String      // discord | telegram | cli | whatsapp
})
```

**Category** — the directory layer. Maintained exclusively by the Editor.
```
(Category {
  name:                 String      // "Learning", "People", "Career Decisions"
  description:          String
})
```

**EditAction** — audit trail for Editor operations. Lives in the graph so the Editor can query its own history.
```
(EditAction {
  id:                   String
  type:                 String      // merge | reclassify | archive | split | reweight | suppress
  targets:              String[]    // Node IDs affected
  reason:               String      // LLM-generated explanation
  timestamp:            DateTime
})
```

### Relationships

```
// === Memory ↔ Entity ===
(Memory)-[:MENTIONS {role}]->(Entity)
  // role: "subject" | "object" | "context" | "source"
  // 4 roles max. Writer can reliably classify these.
  // "Alice switched to CS":        Alice=subject, CS=object
  // "老王 said the approach is bad": 老王=source, approach=object

// === Memory ↔ Memory ===
(Memory)-[:RELATES_TO {type, weight}]->(Memory)
  // Fact-layer types (participate in reasoning/retrieval):
  //   "caused_by" | "follows" | "contradicts" | "supports" | "elaborates"
  // Editor-layer types (version/aggregation, excluded from reasoning by default):
  //   "editor:summarizes" | "editor:supersedes"
  //
  // weight: 0.0-1.0, strength of connection
  // Open enum — extensible without schema migration.
  // Naming convention: fact-layer = bare name, editor-layer = "editor:" prefix.

// === Memory ↔ Session ===
(Memory)-[:PART_OF]->(Session)

// === Memory ↔ Category ===
(Memory)-[:IN_CATEGORY {score, assignedBy, timestamp}]->(Category)
  // score:      0.0-1.0, how strongly this memory belongs to this category
  // assignedBy: "writer" | "editor"
  // timestamp:  when assigned
  // A Memory can belong to multiple Categories.

// === Entity ↔ Entity ===
(Entity)-[:LINKED_TO {relation, detail, sentiment, strength, since, until, active}]->(Entity)
  // relation:   "friend" | "colleague" | "uses" | "part_of" — open enum
  // detail:     "calls him 老王" — human-readable context
  // sentiment:  "positive" | "negative" | "neutral" | "mixed"
  // strength:   0.0-1.0
  // since:      DateTime?
  // until:      DateTime?   // When relationship ended (null = ongoing OR unknown)
  // active:     Boolean     // Editor-only field. true = current, false = ended.
  //                         // Rule: Writer never writes active. Editor maintains it.
  //                         // active=false + until=null is valid (ended but time unknown).

// === Memory canonical pointer ===
(Memory)-[:CANONICAL]->(Memory)
  // Superseded memory points to its canonical replacement.
  // At most 1 outgoing CANONICAL per memory. Editor-maintained.
  // editor:supersedes (RELATES_TO) is the audit trail / version chain.
  // CANONICAL is the index for O(1) lookup.

// === Entity ↔ Entity (merge semantics) ===
(Entity)-[:MERGED_INTO]->(Entity)
  // Non-canonical entity points to canonical. Queries follow MERGED_INTO to resolve.

// === Category ↔ Category ===
(Category)-[:SUBCATEGORY_OF]->(Category)
```

### Retrieval Scoring Formula

```
score = similarity * (0.6 + 0.4 * confidence) * (0.6 + 0.4 * salience)
```

Uses floor-clamped weighting instead of pure multiplication. A brand-new memory (confidence=0.5, salience=0.5) gets scored at `sim * 0.8 * 0.8 = sim * 0.64` — not `sim * 0.25` which pure multiplication would give. This prevents new-but-important memories from being buried.

### RELATES_TO Type Convention

| Layer | Types | Used in retrieval? | Who writes? |
|-------|-------|--------------------|-------------|
| Fact | caused_by, follows, contradicts, supports, elaborates | Yes | Writer + Editor |
| Editor | editor:summarizes, editor:supersedes | No (unless explicitly requested) | Editor only |

Queries that traverse reasoning chains filter to fact-layer types by default:
```cypher
MATCH (a)-[:RELATES_TO]->(b) WHERE NOT r.type STARTS WITH 'editor:'
```

### Canonical Resolution Patterns

**Memory:** `CANONICAL` relationship. If a memory has an outgoing CANONICAL edge, it's been superseded. Retrieval skips non-canonical memories:
```cypher
WHERE NOT (m)-[:CANONICAL]->() AND m.status NOT IN ['archived', 'suppressed']
```

Resolve canonical from a superseded memory:
```cypher
MATCH (m:Memory)-[:CANONICAL]->(canonical:Memory)
```

**Entity:** `MERGED_INTO` relationship. Resolve by following the chain:
```cypher
MATCH (e:Entity)-[:MERGED_INTO*0..]->(canonical:Entity)
WHERE NOT (canonical)-[:MERGED_INTO]->()
```

### Validation Rules (Writer vs Editor boundary)

These rules are mechanically enforced at the application layer:

**Writer constraints:**
- MUST set `status: 'raw'` on all Memory nodes
- MUST set `confidence: 0.5`, `salience: 0.5` as defaults
- MUST NOT create RELATES_TO with `type` starting with `editor:`
- MUST NOT write `CANONICAL` relationships
- MUST NOT write `LINKED_TO.active` — field defaults to `true` implicitly
- MUST NOT write `normalizedKind` or `normalizedType`
- MUST NOT modify or delete existing nodes
- SHOULD create at least one `MENTIONS {role: 'subject'}` per Memory (unless genuinely no subject exists, e.g. weather observations)

**Editor constraints:**
- MUST log every mutation as an `EditAction` node
- MUST NOT modify `sourceRef`, `sourceQuote`, `sourceAuthor`, `sourceChannel` (provenance is immutable)
- MUST maintain `LINKED_TO.active` consistency: `active=true` + `until != null` is invalid
- MUST set `normalizedKind` / `normalizedType` only from canonical sets
- OWNS: `CANONICAL`, `MERGED_INTO`, `editor:*` RELATES_TO types, `IN_CATEGORY`, `SUBCATEGORY_OF`

**Shared rules:**
- All node IDs are UUIDs, generated at creation time
- `RELATES_TO.weight` is required (no weightless edges)
- `MENTIONS.role` is one of: `subject | object | context | source`
- No hard deletes — only `status: 'archived'` or `status: 'suppressed'`

### What's Deliberately NOT in the Schema

- **No Emotion/Decision/Event node types.** All are `Memory` with different `kind` values. One node type = simpler Writer, simpler queries, no ambiguity about which type to create.
- **No Timeline node.** Timelines emerge from `RELATES_TO {type: "follows"}` chains + eventTimeStart ordering.
- **No contradictionCount field.** Derived from `RELATES_TO {type: "contradicts"}` edges. Ground truth lives in edges, not counters.
- **No sourceSession field.** Redundant with `PART_OF` relationship. One source of truth, not two.
- **No canonicalId field.** Canonical pointer is the `CANONICAL` relationship, not a string field. Avoids drift between field and edge. `editor:supersedes` is the version chain; `CANONICAL` is the index.
- **No multi-tenant fields.** Scope by `Session.channel` or add `namespace` later. Don't over-engineer for single-user MVP.

### The Category Layer

The `Category` layer is the key architectural decision. Most memory systems do blind vector search over everything — expensive, and the results get noisier as the graph grows. With a directory layer, retrieval becomes two-phase: find the right category first, then do vector search only within that subgraph. The Editor maintains this directory. It's the map of the memory. Category assignments are relationships with metadata (`IN_CATEGORY` with score and provenance), not simple containment edges — making them traceable, rankable, and allowing multi-category membership.

---

## How the Two Agents Work Together

### The Writer's Process (post-session)

1. Takes the session transcript
2. Makes an LLM call to extract structured facts: entity-relation-entity triples
3. Writes Memory nodes and Entity nodes to Neo4j
4. Sets `status: raw` and `confidence: 0.5` on everything it creates
5. Flags new nodes for the Editor to review
6. Never deletes, never modifies existing nodes

### The Editor's Process (scheduled)

1. Queries all nodes with `status: raw`
2. For each flagged node and its neighborhood, makes an LLM call to reason:
   - Does this duplicate an existing Memory node? → merge them
   - Does this contradict an existing fact? → lower confidence on both, flag for review
   - Does this appear in multiple sessions? → boost confidence
   - Which Category does this belong to? → create or update `CONTAINS` relationship
   - Should a new Category be created? → only if confidence threshold is met
3. Updates confidence scores across the graph
4. Maintains the Category taxonomy
5. Archives (soft delete) nodes that are superseded or contradicted and low confidence
6. Marks processed nodes `status: reviewed`

### The Handshake Protocol

**Writer always appends, never deletes.**
**Editor never touches live session data, only flagged nodes.**
**Nothing is ever hard deleted — only archived.**

This separation of concerns is what makes the system safe. If the Editor makes a bad decision and merges two nodes that shouldn't be merged, nothing is lost — it's recoverable. The Writer's raw output is always preserved in the session record.

---

## The Feedback Loop

This is the part that makes the system interesting over time.

Session 1: Writer captures messy facts. Editor cleans them up, organizes them into categories.

Session 2: Writer retrieves context before the session starts. Instead of scanning everything, it hits the Editor-maintained directory — fast, precise retrieval. The writer's session prompt already has better context. So the facts it extracts this session are better structured, because it's reading cleaner memory.

Session 3: Editor has less work to do. The Writer's output is already closer to the right categories. Confidence scores accumulate. The graph becomes more reliable.

Over time, the system self-organizes. The weak signal from the Writer gets elevated into strong, durable knowledge by the Editor. And the Writer improves not because it changed, but because the Editor shaped the environment it reads from.

---

## The Directory Layer in Detail

This is the most novel piece and worth explaining carefully.

Most RAG systems work like this: you have a query, you embed it, you find the nearest vectors in your database. Simple. But as the database grows, "nearest vectors" gets noisy — you find things that are semantically similar but contextually irrelevant.

The directory layer adds a structured index on top of the vector index. Categories are explicitly maintained nodes in the graph. When the agent needs to retrieve memory, it first routes the query to the right category — "this is a question about work projects, look in the Work Projects subgraph" — then does vector search only within that subgraph.

The Editor's most important job is keeping this directory coherent. It asks:
- Is this category too broad? Split it.
- Are these two categories overlapping too much? Merge them.
- Is this Memory node in the wrong category? Reclassify it.

The result is that retrieval gets faster and more precise as the graph grows, instead of slower and noisier. The directory is the map. The Editor draws and maintains the map. The Writer and the retrieval system use the map to navigate.

---

## What's Novel About This

**Most agent memory systems are write-only.** This one has a dedicated maintenance process.

**Most graph memory systems do flat search.** This one has a two-phase retrieval with a maintained directory layer.

**The dual-agent architecture mirrors how human memory actually works** — fast episodic capture, slow semantic consolidation, running at different speeds, producing something neither could produce alone.

**The weak-to-strong dynamic produces emergent organization.** The Writer's local, session-level observations get elevated into global, graph-level knowledge structure through the Editor's broader perspective. This is the same principle that makes weak-to-strong generalization interesting in AI alignment — a weaker process can bootstrap a stronger outcome if the feedback loop is designed well.

**The system improves without retraining.** No model weights change. The improvement comes entirely from the graph getting better organized over time.

---

## Open Questions (Worth Thinking Through)

**How expensive is the Editor?** It makes LLM calls to reason about graph structure. Batching is the answer — Editor only processes flagged nodes and their neighborhoods, not the whole graph. But the right batch size needs to be figured out empirically.

**When does a new Category get created?** Too liberal and the taxonomy fragments. Too conservative and everything ends up in one catch-all category. Probably needs a confidence threshold — only create a new Category when enough Memory nodes are clearly similar and don't fit existing categories.

**What happens when the Editor is wrong?** It merges two nodes that shouldn't be merged, or puts something in the wrong category. Soft deletes and versioning protect against data loss, but how do you detect and recover from Editor errors? Possibly: the Writer can flag a contradiction if new session data conflicts with an Editor-consolidated node, triggering a re-review.

**Multi-agent extension.** If multiple OpenClaw agents share the same Neo4j instance — say, a personal assistant and a work assistant — the Editor could maintain a shared graph while each Writer scopes its writes to its own subgraph. The Editor sees across both. That's a more complex version of the same idea.

---

## OpenClaw Integration — Confirmed Architecture (Phase 0 Findings)

Phase 0 source code analysis of OpenClaw v2026.2.3 revealed a significantly richer plugin API than publicly documented. The original design assumed file watchers and limited hooks. The actual API provides everything natively.

### Plugin API Surface

The Writer and retrieval system are implemented as a **single OpenClaw plugin** using the `OpenClawPluginApi`:

```typescript
type OpenClawPluginApi = {
  registerTool: (tool, opts?) => void;       // Agent-callable tools
  registerCli: (registrar, opts?) => void;    // CLI commands
  registerService: (service) => void;         // Background services
  resolvePath: (input: string) => string;     // Path resolution
  on: <K extends PluginHookName>(hookName: K, handler, opts?) => void;  // Lifecycle hooks
  logger: PluginLogger;
  pluginConfig?: Record<string, unknown>;
};
```

### Lifecycle Hooks — The Integration Points

OpenClaw exposes 14 lifecycle hooks. Three are critical for this project:

| Hook | Role | Event Data |
|------|------|-----------|
| `before_agent_start` | **Retrieval** — inject memories into context | `{ prompt: string, messages?: unknown[] }` → return `{ prependContext: string }` |
| `agent_end` | **Writer** — extract facts from completed session | `{ messages: unknown[], success: boolean, durationMs?: number }` |
| `session_end` | **Writer (backup)** — alternative trigger point | `{ sessionId: string, messageCount: number, durationMs?: number }` |

Additional useful hooks:
- `before_compaction` — emergency fact extraction before context window compaction
- `after_tool_call` — track tool usage patterns
- `gateway_start` / `gateway_stop` — Editor scheduler lifecycle

### Writer Integration (replaces file watcher design)

The original design proposed watching Markdown log files. This is unnecessary — the `agent_end` hook provides the full conversation messages directly:

```typescript
api.on("agent_end", async (event) => {
  // event.messages contains the full session conversation
  // event.success indicates whether the session completed normally
  // Extract facts via LLM, write to Neo4j
});
```

This is the same pattern used by the existing `memory-lancedb` plugin's auto-capture feature.

### Retrieval Integration (replaces agent:bootstrap design)

The `before_agent_start` hook allows injecting context before the agent processes a user message:

```typescript
api.on("before_agent_start", async (event) => {
  // event.prompt = the user's current message
  // Query Neo4j: classify prompt → category → vector search within subgraph
  const memories = await queryNeo4j(event.prompt);
  return {
    prependContext: `<graph-memories>\n${memories}\n</graph-memories>`
  };
});
```

### Plugin Package Structure

```
extensions/dual-memory/
├── index.ts              # Plugin definition with register()
├── writer.ts             # Fact extraction logic
├── retrieval.ts          # Two-phase Neo4j query
├── neo4j-client.ts       # Neo4j driver wrapper
├── embeddings.ts         # OpenAI embedding calls
├── config.ts             # Plugin config schema
├── openclaw.plugin.json  # Plugin manifest
└── package.json          # With openclaw.extensions field
```

### Key Design Change: No Kafka in Phase 1

The original design included Kafka between Writer and Neo4j. Phase 0 analysis shows this is unnecessary for the MVP:

- The `agent_end` hook is synchronous enough — the Writer runs after each session, not during
- Direct Neo4j writes from the plugin are simpler and sufficient at this scale
- Kafka can be introduced in Phase 2 if throughput becomes an issue (unlikely for single-user)

---

## Resolved Design Decisions (from Phase 0 analysis)

### Editor Cost & Batching

The Editor processes only `WHERE status = 'raw'` nodes. Batch by shared entity — group raw memories that mention the same entities and process them together. Each Editor run: 15-20 LLM calls max, ~$0.02-0.05 per run.

### Category Creation

Bottom-up emergence with quorum rule:
- Create a new Category when 3+ Memory nodes share high semantic similarity (cosine > 0.85) AND don't fit existing categories
- Merge trigger: when two Categories have >60% membership overlap
- Flat taxonomy initially; allow `SUBCATEGORY_OF` only after parent has 10+ members

### Editor Error Recovery

The Writer is source of truth, the Editor is an interpretation layer:
- Every Editor action logged as `EditAction` node: `{type, targets, reason, timestamp}`
- Writer flags `CONTRADICTS` relationship when new facts conflict with Editor-consolidated nodes
- Confidence decay: -0.05 per Editor cycle for unreinforced consolidated facts
- Rollback: replay original Memory nodes from Session records

### Editor Scheduling

Dual trigger:
- **Nightly** at 2:00 AM local time (regular consolidation)
- **Threshold** when raw node count ≥ 40 (responsive to heavy use)
- Check interval: every 5 minutes
- Batch cap: 50 nodes per run (oldest first)
- Concurrency: single lock prevents overlapping runs

### Neo4j Vector Index

Using Neo4j 2026.01.4 Community Edition with native vector indexes:

```cypher
CREATE VECTOR INDEX memoryEmbeddings IF NOT EXISTS
FOR (m:Memory) ON (m.embedding)
OPTIONS { indexConfig: {
  `vector.dimensions`: 1536,
  `vector.similarity_function`: 'cosine'
}}
```

Two-phase retrieval uses **pre-filtering** (Approach A) with the floor-clamped scoring formula:

```cypher
MATCH (mem:Memory)-[:IN_CATEGORY {score: catScore}]->(cat:Category)
WHERE cat.name IN $categoryNames
  AND mem.status NOT IN ['archived', 'suppressed']
  AND NOT (mem)-[:CANONICAL]->()
  AND (mem.expiresAt IS NULL OR mem.expiresAt > datetime())
WITH mem, vector.similarity.cosine(mem.embedding, $queryVector) AS sim,
     catScore
WITH mem, sim * (0.6 + 0.4 * mem.confidence) * (0.6 + 0.4 * mem.salience) AS score
ORDER BY score DESC LIMIT 10
RETURN mem.content, mem.id, mem.kind, mem.normalizedKind, score
```

Fallback (low category-classification confidence): full ANN search via `db.index.vector.queryNodes`.

### Constraints & Indexes

Required before inserting any data:

```cypher
// === Uniqueness constraints (also create indexes) ===
CREATE CONSTRAINT memory_id IF NOT EXISTS FOR (m:Memory) REQUIRE m.id IS UNIQUE;
CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE;
CREATE CONSTRAINT session_id IF NOT EXISTS FOR (s:Session) REQUIRE s.id IS UNIQUE;
CREATE CONSTRAINT editaction_id IF NOT EXISTS FOR (a:EditAction) REQUIRE a.id IS UNIQUE;
CREATE CONSTRAINT category_name IF NOT EXISTS FOR (c:Category) REQUIRE c.name IS UNIQUE;

// === Performance indexes ===
CREATE INDEX memory_status IF NOT EXISTS FOR (m:Memory) ON (m.status);
CREATE INDEX memory_timestamp IF NOT EXISTS FOR (m:Memory) ON (m.timestamp);
CREATE INDEX memory_eventTimeStart IF NOT EXISTS FOR (m:Memory) ON (m.eventTimeStart);
CREATE INDEX memory_expiresAt IF NOT EXISTS FOR (m:Memory) ON (m.expiresAt);
CREATE INDEX memory_normalizedKind IF NOT EXISTS FOR (m:Memory) ON (m.normalizedKind);
CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name);
CREATE INDEX entity_normalizedType IF NOT EXISTS FOR (e:Entity) ON (e.normalizedType);

// === Fulltext index for entity resolution ===
CREATE FULLTEXT INDEX entity_names IF NOT EXISTS
FOR (e:Entity) ON EACH [e.name]
// Note: aliases is a String[] — fulltext over arrays requires
// application-layer search or a separate AliasNode in the future.

// === Vector indexes ===
CREATE VECTOR INDEX memoryEmbeddings IF NOT EXISTS
FOR (m:Memory) ON (m.embedding)
OPTIONS { indexConfig: {
  `vector.dimensions`: 1536,
  `vector.similarity_function`: 'cosine'
}};

CREATE VECTOR INDEX entityEmbeddings IF NOT EXISTS
FOR (e:Entity) ON (e.embedding)
OPTIONS { indexConfig: {
  `vector.dimensions`: 1536,
  `vector.similarity_function`: 'cosine'
}};
```

### Schema Design Notes

The schema went through three iterations (v0 → v1 → v2) driven by use-case validation and peer review. Key changes from v1 to v2:

| Change | v1 | v2 | Why |
|--------|----|----|-----|
| Temporal | Single `timestamp` | `timestamp` + `eventTimeStart/End` + `expiresAt` | "switched majors last year" ≠ today. "in NYC this week" must expire. |
| Classification | Open enum only | `kind` (Writer) + `normalizedKind` (Editor) | Writer generates freely, Editor converges to stable set for reliable queries |
| Scoring | `similarity × confidence` | `sim × (0.6 + 0.4*conf) × (0.6 + 0.4*sal)` | Pure multiplication buries new memories. Floor clamp prevents this. |
| Category | `Category-[:CONTAINS]->Memory` | `Memory-[:IN_CATEGORY {score, assignedBy}]->Category` | Assignment is traceable, rankable, multi-category |
| Provenance | `sourceSession` field | `PART_OF` relationship + `sourceRef` + `sourceQuote` | No redundancy, stable pointer to exact message |
| Versioning | None | `CANONICAL` rel + `editor:supersedes` + `MERGED_INTO` | Explicit merge/version semantics in the graph, no string↔edge drift |
| Status | raw/reviewed/archived | + `suppressed` | Privacy, errors, user deletion requests |
| Entity relationships | `since` only | + `until` + `active` (Editor-only) | Relationships can end |
| RELATES_TO types | All flat | `editor:` prefix convention for edit-layer types | Separates reasoning from versioning in queries |

Validated against query scenarios:

- **"我做过哪些重要决定"** → `normalizedKind = 'decision'` + `RELATES_TO {type: 'caused_by'}` for reasons
- **"我和老王聊过什么技术话题"** → `Entity.aliases` for multilingual + `MENTIONS.role` for precise joins
- **"我最近情绪怎么样"** → `normalizedKind = 'emotion'` + `eventTimeStart` temporal aggregation
- **"这个决定和两个月前那个想法有什么联系"** → fact-layer `RELATES_TO` chain traversal across sessions
- **"我上个月的学习进度"** → `eventTimeStart` range + `Entity.type = 'Project'/'Concept'`

---

## Tech Stack (Updated)

| Component | Choice | Reason |
|-----------|--------|--------|
| Knowledge Graph | Neo4j 2026.01.4 Community | Native vector indexes, graph queries, free |
| Writer Agent | TypeScript (OpenClaw plugin) | Runs inside OpenClaw via `agent_end` hook |
| Retrieval | TypeScript (same plugin) | Runs via `before_agent_start` hook |
| Editor Agent | Python | Better LLM tooling, graph reasoning, scheduled |
| Editor Scheduler | Python (schedule lib) | Nightly + threshold triggers |
| Embeddings | text-embedding-3-small (1536d) | Cheap, within Neo4j's 4096 max |
| Infrastructure | Docker Compose | Neo4j only (no Kafka for MVP) |

---

## Build Phases (Updated)

**Phase 0 — Study** ✅ COMPLETE
- Installed OpenClaw, explored source code
- Read memory-core, memory-lancedb plugins
- Mapped plugin API, all 14 lifecycle hooks
- Confirmed `agent_end` + `before_agent_start` as integration points
- Identified memory-lancedb as architectural blueprint

**Phase 1 — Writer MVP** (~8hrs)
- OpenClaw plugin scaffold with `openclaw.plugin.json`
- Neo4j Docker Compose setup with vector index
- Writer: `agent_end` hook → LLM fact extraction → Neo4j writes
- Retrieval: `before_agent_start` hook → Neo4j vector search → context injection
- Tools: `memory_graph_search`, `memory_graph_store` for explicit agent use
- Direct Neo4j writes (no Kafka)

**Phase 2 — Editor Agent** (~10hrs)
- Python Editor agent with batched LLM reasoning
- Category creation, deduplication, contradiction detection
- Editor scheduler (nightly + threshold)
- EditAction audit trail
- Confidence score management

**Phase 3 — Polish** (~6hrs)
- Confidence decay for stale facts
- `before_compaction` hook for emergency saves
- Benchmarks vs baseline (memory-core, memory-lancedb)
- Migration tool from existing memory files / daily logs
- Documentation

---

*The goal is a memory system that doesn't just accumulate — it matures.*
