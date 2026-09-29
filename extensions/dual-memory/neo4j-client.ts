/**
 * Neo4j driver wrapper with schema initialization and typed queries.
 * All graph operations go through this client.
 */

import neo4j, { type Driver, type Session, type ManagedTransaction } from "neo4j-driver";
import { createHash, randomUUID } from "node:crypto";

// ============================================================================
// Types — matches v2.1 schema
// ============================================================================

export type MemoryNode = {
  id: string;
  content: string;
  kind: string;
  normalizedKind?: string;
  timestamp: string; // ISO 8601
  eventTimeStart?: string;
  eventTimeEnd?: string;
  expiresAt?: string;
  confidence: number;
  salience: number;
  status: "raw" | "reviewed" | "archived" | "suppressed";
  lastAccessed?: string;
  sourceRef?: string;
  sourceQuote?: string;
  sourceChannel?: string;
  sourceAuthor?: string;
  embedding?: number[];
};

export type EntityNode = {
  id: string;
  name: string;
  type: string;
  normalizedType?: string;
  aliases: string[];
  key?: string;
  firstSeen: string;
  embedding?: number[];
};

export type SessionNode = {
  id: string;
  key?: string; // OpenClaw session id; makes createSession idempotent
  date: string;
  summary: string;
  messageCount: number;
  channel: string;
};

export type MentionsRel = {
  role: "subject" | "object" | "context" | "source";
};

export type RelatesToRel = {
  type: string;
  weight: number;
};

export type ExtractedFact = {
  content: string;
  kind: string;
  confidence: number;
  salience: number;
  eventTimeStart?: string;
  eventTimeEnd?: string;
  sourceRef?: string;
  sourceQuote?: string;
  sourceChannel?: string;
  sourceAuthor?: string;
  entities: {
    name: string;
    type: string;
    aliases?: string[];
    role: MentionsRel["role"];
  }[];
  relatesTo?: {
    targetContent: string; // matched by content similarity
    type: string;
    weight: number;
  }[];
};

export type SimilarMemory = {
  id: string;
  content: string;
  similarity: number;
};

/** Per-fact result of writeFacts, in input order. */
export type FactWriteOutcome =
  | { status: "written"; memoryId: string }
  | { status: "already_written"; memoryId: string }
  | { status: "skipped_similar"; memoryId: string; similarity: number };

export type WriteFactsResult = {
  memoryIds: string[]; // newly created Memory ids
  outcomes: FactWriteOutcome[];
};

export type EntityLink = {
  source: string;  // entity name
  target: string;  // entity name
  relation: string;
  detail: string;
  sentiment: "positive" | "negative" | "neutral" | "mixed";
  strength: number;
};

export type RetrievalResult = {
  id: string;
  content: string;
  kind: string;
  normalizedKind?: string;
  score: number;
  entities?: string[];
  entityRelations?: string[];  // e.g., ["Alice is colleague of Bob"]
  supersededAt?: string;  // set when a newer memory replaced this one
};

// ============================================================================
// Schema initialization statements (from init-schema.cypher)
// ============================================================================

const SCHEMA_STATEMENTS = [
  // Uniqueness constraints
  "CREATE CONSTRAINT memory_id IF NOT EXISTS FOR (m:Memory) REQUIRE m.id IS UNIQUE",
  "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
  "CREATE CONSTRAINT session_id IF NOT EXISTS FOR (s:Session) REQUIRE s.id IS UNIQUE",
  "CREATE CONSTRAINT session_key IF NOT EXISTS FOR (s:Session) REQUIRE s.key IS UNIQUE",
  "CREATE CONSTRAINT memory_write_key IF NOT EXISTS FOR (m:Memory) REQUIRE m.writeKey IS UNIQUE",
  "CREATE CONSTRAINT editaction_id IF NOT EXISTS FOR (a:EditAction) REQUIRE a.id IS UNIQUE",
  "CREATE CONSTRAINT editorrun_id IF NOT EXISTS FOR (r:EditorRun) REQUIRE r.id IS UNIQUE",
  "CREATE CONSTRAINT category_name IF NOT EXISTS FOR (c:Category) REQUIRE c.name IS UNIQUE",
  // Performance indexes
  "CREATE INDEX memory_status IF NOT EXISTS FOR (m:Memory) ON (m.status)",
  "CREATE INDEX memory_timestamp IF NOT EXISTS FOR (m:Memory) ON (m.timestamp)",
  "CREATE INDEX memory_eventTimeStart IF NOT EXISTS FOR (m:Memory) ON (m.eventTimeStart)",
  "CREATE INDEX memory_expiresAt IF NOT EXISTS FOR (m:Memory) ON (m.expiresAt)",
  "CREATE INDEX memory_normalizedKind IF NOT EXISTS FOR (m:Memory) ON (m.normalizedKind)",
  "CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)",
  "CREATE INDEX entity_normalizedType IF NOT EXISTS FOR (e:Entity) ON (e.normalizedType)",
  "CREATE INDEX session_date IF NOT EXISTS FOR (s:Session) ON (s.date)",
  "CREATE INDEX editaction_run IF NOT EXISTS FOR (a:EditAction) ON (a.runId)",
];

// Vector indexes are created dynamically based on embedding model dimensions
function vectorIndexStatements(dims: number): string[] {
  return [
    `CREATE VECTOR INDEX memoryEmbeddings IF NOT EXISTS
     FOR (m:Memory) ON (m.embedding)
     OPTIONS { indexConfig: { \`vector.dimensions\`: ${dims}, \`vector.similarity_function\`: 'cosine' }}`,
    `CREATE VECTOR INDEX entityEmbeddings IF NOT EXISTS
     FOR (e:Entity) ON (e.embedding)
     OPTIONS { indexConfig: { \`vector.dimensions\`: ${dims}, \`vector.similarity_function\`: 'cosine' }}`,
  ];
}

/**
 * Score multiplier for memories a newer memory has superseded. They stay
 * retrievable for questions about the past, but rank below current facts.
 */
export const SUPERSEDED_WEIGHT = 0.7;

/** Cypher expression: the date a memory stopped being valid, or null. */
const SUPERSEDED_AT = (m: string) =>
  `CASE WHEN ${m}.validTo IS NOT NULL AND ${m}.validTo <= datetime()
        THEN toString(date(${m}.validTo)) END AS supersededAt`;

function validityWeight(supersededAt: string | undefined): number {
  return supersededAt ? SUPERSEDED_WEIGHT : 1;
}

/**
 * Upsert an entity by name and mention it from a memory. A name the Editor
 * merged away resolves to the entity it was merged into.
 */
const MENTION_ENTITY = `
  MERGE (e:Entity {name: $name})
  ON CREATE SET e.id = $entityId, e.type = $type,
                e.aliases = $aliases, e.firstSeen = datetime()
  ON MATCH SET e.aliases = CASE
    WHEN size([a IN $aliases WHERE NOT a IN e.aliases]) > 0
    THEN e.aliases + [a IN $aliases WHERE NOT a IN e.aliases]
    ELSE e.aliases END
  WITH e
  OPTIONAL MATCH (e)-[:MERGED_INTO*1..5]->(root:Entity)
  WHERE NOT (root)-[:MERGED_INTO]->()
  WITH coalesce(root, e) AS target
  MATCH (m:Memory {id: $memoryId})
  MERGE (m)-[r:MENTIONS]->(target)
  ON CREATE SET r.role = $role`;

/**
 * Idempotency key for a fact written in a session: rerunning the Writer on the
 * same session (agent_end fires after every turn) finds it instead of writing
 * a copy. Case, spacing, and trailing punctuation don't count as a new fact.
 */
function writeKey(sessionId: string, content: string): string {
  const normalized = content.toLowerCase().replace(/\s+/g, " ").trim().replace(/[.!]+$/, "");
  return createHash("sha256").update(`${sessionId}\n${normalized}`).digest("hex");
}

/** Cosine above which a same-session fact counts as a restatement. */
export const WRITER_DEDUP_COSINE = 0.95;

/**
 * Neo4j's vector index reports cosine similarity rescaled to [0, 1] as
 * (1 + cos) / 2. Thresholds and scores here are plain cosine, so convert.
 */
function toCosine(score: number): number {
  return 2 * score - 1;
}

// ============================================================================
// Client
// ============================================================================

export class Neo4jClient {
  private driver: Driver;
  private initialized = false;
  private initPromise: Promise<void> | null = null;
  private vectorDims: number;

  constructor(
    uri: string,
    user: string,
    password: string,
    vectorDims: number = 1024,
  ) {
    this.driver = neo4j.driver(uri, neo4j.auth.basic(user, password));
    this.vectorDims = vectorDims;
  }

  /**
   * Ensure schema is initialized (constraints, indexes, vector indexes).
   * Safe to call multiple times — idempotent.
   */
  async ensureSchema(): Promise<void> {
    if (this.initialized) return;
    if (this.initPromise) return this.initPromise;
    this.initPromise = this._initSchema();
    return this.initPromise;
  }

  private async _initSchema(): Promise<void> {
    const session = this.driver.session();
    try {
      // Run constraints and regular indexes
      for (const stmt of SCHEMA_STATEMENTS) {
        await session.run(stmt);
      }
      // Run vector index creation (dimensions depend on embedding model)
      for (const stmt of vectorIndexStatements(this.vectorDims)) {
        await session.run(stmt);
      }
      this.initialized = true;
    } finally {
      await session.close();
    }
  }

  // ========================================================================
  // Session (conversation) operations
  // ========================================================================

  /**
   * Create the Session node for a conversation. With a `key` (OpenClaw's
   * session id) this is idempotent: agent_end fires after every turn, and each
   * call for the same conversation returns the same Session.
   */
  async createSession(data: Omit<SessionNode, "id">): Promise<SessionNode> {
    await this.ensureSchema();
    const session = this.driver.session();
    try {
      const result = await session.executeWrite((tx: ManagedTransaction) =>
        tx.run(
          `MERGE (s:Session {key: $key})
           ON CREATE SET s.id = $id, s.date = datetime($date), s.channel = $channel
           SET s.summary = $summary, s.messageCount = $messageCount
           RETURN s.id AS id`,
          { ...data, id: randomUUID(), key: data.key ?? randomUUID() },
        ),
      );
      return { ...data, id: result.records[0].get("id") as string };
    } finally {
      await session.close();
    }
  }

  // ========================================================================
  // Memory operations (Writer)
  // ========================================================================

  /**
   * Find an active memory from the same session that is a near-restatement
   * (cosine >= threshold). Returns the closest match, or null.
   *
   * Scoped to the session because that is where repeats come from: every
   * agent_end re-extracts the whole transcript. Across sessions a similar
   * fact may be an update ("lives in Seattle" / "now lives in Portland" score
   * 0.86), so those go to the Editor, whose LLM tier can tell the difference.
   */
  async findSimilar(
    embedding: number[],
    sessionId: string,
    threshold: number = WRITER_DEDUP_COSINE,
  ): Promise<SimilarMemory | null> {
    await this.ensureSchema();
    const session = this.driver.session();
    try {
      const result = await session.executeRead((tx: ManagedTransaction) =>
        tx.run(
          `CALL db.index.vector.queryNodes('memoryEmbeddings', 20, $embedding)
           YIELD node AS mem, score
           WITH mem, 2 * score - 1 AS cosine
           WHERE cosine >= $threshold AND mem.status IN ['raw', 'reviewed']
             AND EXISTS { (mem)-[:PART_OF]->(:Session {id: $sessionId}) }
           RETURN mem.id AS id, mem.content AS content, cosine
           ORDER BY cosine DESC LIMIT 1`,
          { embedding, sessionId, threshold },
        ),
      );
      const record = result.records[0];
      if (!record) return null;
      return {
        id: record.get("id") as string,
        content: record.get("content") as string,
        similarity: record.get("cosine") as number,
      };
    } finally {
      await session.close();
    }
  }

  /**
   * Write extracted facts to Neo4j as Memory nodes + Entity nodes + relationships.
   * This is the Writer's main operation.
   *
   * `observedAt` stamps the memories (defaults to now); the eval harness uses it
   * to replay sessions at their original dates.
   */
  async writeFacts(
    facts: ExtractedFact[],
    sessionId: string,
    embeddings: number[][],
    options: { observedAt?: string } = {},
  ): Promise<WriteFactsResult> {
    await this.ensureSchema();
    const observedAt = options.observedAt ?? new Date().toISOString();
    const memoryIds: string[] = [];
    const outcomes: FactWriteOutcome[] = [];
    const session = this.driver.session();

    try {
      for (let i = 0; i < facts.length; i++) {
        const fact = facts[i];
        const embedding = embeddings[i];
        const key = writeKey(sessionId, fact.content);

        // Idempotency: this session already wrote this fact (whatever the
        // Editor has since done to it).
        const already = await session.executeRead((tx: ManagedTransaction) =>
          tx.run(`MATCH (m:Memory {writeKey: $key}) RETURN m.id AS id`, { key }),
        );
        if (already.records.length > 0) {
          outcomes.push({ status: "already_written", memoryId: already.records[0].get("id") as string });
          continue;
        }

        // Pre-write dedup: skip if similar memory already exists
        const existing = await this.findSimilar(embedding, sessionId);
        if (existing) {
          console.log(
            `[dual-memory] Skipped duplicate: "${fact.content.slice(0, 60)}..." ≈ "${existing.content.slice(0, 60)}..."`,
          );
          outcomes.push({ status: "skipped_similar", memoryId: existing.id, similarity: existing.similarity });
          continue;
        }

        let memoryId: string = randomUUID();

        const created = await session.executeWrite(async (tx: ManagedTransaction) => {
          // 1. Create Memory node (MERGE on writeKey: a concurrent run for the
          //    same session can't create a second copy)
          const res = await tx.run(
            `MERGE (m:Memory {writeKey: $writeKey})
             ON CREATE SET
              m.id = $id,
              m.content = $content,
              m.kind = $kind,
              m.timestamp = datetime($observedAt),
              m.validFrom = datetime($observedAt),
              m.eventTimeStart = CASE WHEN $eventTimeStart IS NOT NULL THEN datetime($eventTimeStart) ELSE null END,
              m.eventTimeEnd = CASE WHEN $eventTimeEnd IS NOT NULL THEN datetime($eventTimeEnd) ELSE null END,
              m.confidence = $confidence,
              m.salience = $salience,
              m.status = 'raw',
              m.sourceRef = $sourceRef,
              m.sourceQuote = $sourceQuote,
              m.sourceChannel = $sourceChannel,
              m.sourceAuthor = $sourceAuthor,
              m.embedding = $embedding
             RETURN m.id AS id`,
            {
              writeKey: key,
              id: memoryId,
              observedAt,
              content: fact.content,
              kind: fact.kind,
              confidence: fact.confidence,
              salience: fact.salience,
              eventTimeStart: fact.eventTimeStart ?? null,
              eventTimeEnd: fact.eventTimeEnd ?? null,
              sourceRef: fact.sourceRef ?? null,
              sourceQuote: fact.sourceQuote ?? null,
              sourceChannel: fact.sourceChannel ?? null,
              sourceAuthor: fact.sourceAuthor ?? null,
              embedding,
            },
          );
          if (res.records[0].get("id") !== memoryId) {
            memoryId = res.records[0].get("id") as string;
            return false;
          }

          // 2. Link to Session
          await tx.run(
            `MATCH (m:Memory {id: $memoryId}), (s:Session {id: $sessionId})
             MERGE (m)-[:PART_OF]->(s)`,
            { memoryId, sessionId },
          );

          // 3. Create/merge Entity nodes and MENTIONS relationships
          for (const entity of fact.entities) {
            await tx.run(MENTION_ENTITY, {
              name: entity.name,
              entityId: randomUUID(),
              type: entity.type,
              aliases: entity.aliases ?? [entity.name],
              memoryId,
              role: entity.role,
            });
          }
          return true;
        });

        if (created) memoryIds.push(memoryId);
        outcomes.push({ status: created ? "written" : "already_written", memoryId });
      }
    } finally {
      await session.close();
    }

    return { memoryIds, outcomes };
  }

  /**
   * Write entity-to-entity LINKED_TO relationships extracted by the Writer.
   */
  async writeEntityLinks(links: EntityLink[]): Promise<void> {
    if (links.length === 0) return;
    await this.ensureSchema();
    const session = this.driver.session();

    try {
      for (const link of links) {
        await session.executeWrite((tx: ManagedTransaction) =>
          tx.run(
            `MATCH (a:Entity {name: $source}), (b:Entity {name: $target})
             MERGE (a)-[r:LINKED_TO]->(b)
             SET r.relation = $relation, r.detail = $detail,
                 r.sentiment = $sentiment, r.strength = $strength`,
            {
              source: link.source,
              target: link.target,
              relation: link.relation,
              detail: link.detail,
              sentiment: link.sentiment,
              strength: link.strength,
            },
          ),
        );
      }
    } finally {
      await session.close();
    }
  }

  // ========================================================================
  // Retrieval operations
  // ========================================================================

  /**
   * Hub entity threshold — entities with more than this many active memories
   * are too generic for graph expansion (e.g., "User" connects to 137 memories).
   * They're still included in results display, just not used for expansion.
   */
  private static readonly HUB_ENTITY_THRESHOLD = 30;

  /**
   * Hybrid retrieval: vector seeds → graph expansion via MENTIONS → rerank.
   *
   * Step 1: ANN vector search for seed memories (top 5).
   * Step 2: Extract non-hub entities from seeds, expand via MENTIONS edges
   *         to find related memories not in the seed set.
   * Step 3: Score expansions using vector similarity + entity overlap bonus.
   * Step 4: Merge seeds + expansions, dedupe, rerank, return top-k.
   *
   * Falls back to pure vector search if graph expansion yields nothing.
   */
  async retrieve(
    queryVector: number[],
    limit: number = 10,
    _categoryNames?: string[],
  ): Promise<RetrievalResult[]> {
    await this.ensureSchema();
    const session = this.driver.session();

    try {
      // ── Step 1: Vector seed recall ──────────────────────────────────
      const seedCount = Math.max(5, Math.ceil(limit * 0.6));
      const annLimit = neo4j.int(seedCount * 3); // over-fetch for filtering

      const annResult = await session.executeRead((tx: ManagedTransaction) =>
        tx.run(
          `CALL db.index.vector.queryNodes('memoryEmbeddings', $annLimit, $queryVector)
           YIELD node AS mem, score AS sim
           RETURN mem.id AS id, mem.content AS content, mem.kind AS kind,
                  mem.normalizedKind AS normalizedKind,
                  mem.status AS status, mem.confidence AS confidence,
                  mem.salience AS salience, mem.expiresAt AS expiresAt,
                  ${SUPERSEDED_AT("mem")},
                  sim`,
          { queryVector, annLimit },
        ),
      );

      const now = new Date().toISOString();
      const seeds = annResult.records
        .filter((r) => {
          const status = r.get("status");
          if (status === "archived" || status === "suppressed") return false;
          const expiresAt = r.get("expiresAt");
          if (expiresAt && expiresAt < now) return false;
          return true;
        })
        .map((r) => {
          const sim = toCosine(r.get("sim") as number);
          const confidence = (r.get("confidence") as number) ?? 0.5;
          const salience = (r.get("salience") as number) ?? 0.5;
          const supersededAt = (r.get("supersededAt") as string | null) ?? undefined;
          const score =
            sim * (0.6 + 0.4 * confidence) * (0.6 + 0.4 * salience) * validityWeight(supersededAt);
          return {
            id: r.get("id") as string,
            content: r.get("content") as string,
            kind: r.get("kind") as string,
            normalizedKind: r.get("normalizedKind") as string | undefined,
            score,
            supersededAt,
            source: "vector" as const,
          };
        })
        .sort((a, b) => b.score - a.score)
        .slice(0, seedCount);

      if (seeds.length === 0) return [];

      // ── Step 2: Extract entities from seeds, filter hubs ────────────
      const seedIds = seeds.map((s) => s.id);

      const entityResult = await session.executeRead((tx: ManagedTransaction) =>
        tx.run(
          `UNWIND $seedIds AS memId
           MATCH (m:Memory {id: memId})-[:MENTIONS]->(e:Entity)
           WHERE e.active IS NULL OR e.active = true
           WITH e.name AS name, count(DISTINCT m) AS seedHits
           // Count total active memories for hub detection
           OPTIONAL MATCH (e2:Entity {name: name})<-[:MENTIONS]-(allMem:Memory)
           WHERE allMem.status IN ['active', 'raw', 'reviewed']
           WITH name, seedHits, count(DISTINCT allMem) AS totalMems
           RETURN name, seedHits, totalMems`,
          { seedIds },
        ),
      );

      // Separate hub entities from expansion-eligible entities
      const expansionEntities: string[] = [];
      for (const r of entityResult.records) {
        const name = r.get("name") as string;
        const totalMems = (r.get("totalMems") as number) ?? 0;
        if (totalMems <= Neo4jClient.HUB_ENTITY_THRESHOLD) {
          expansionEntities.push(name);
        }
      }

      // ── Step 3: Graph expansion via MENTIONS ────────────────────────
      type ScoredResult = {
        id: string;
        content: string;
        kind: string;
        normalizedKind?: string;
        score: number;
        supersededAt?: string;
        source: "vector" | "graph";
      };

      let expansions: ScoredResult[] = [];

      if (expansionEntities.length > 0) {
        const expandResult = await session.executeRead((tx: ManagedTransaction) =>
          tx.run(
            `UNWIND $entityNames AS eName
             MATCH (e:Entity {name: eName})<-[:MENTIONS]-(m:Memory)
             WHERE m.status IN ['active', 'raw', 'reviewed']
               AND NOT m.id IN $seedIds
               AND (m.expiresAt IS NULL OR m.expiresAt > datetime())
             WITH m, collect(DISTINCT eName) AS sharedEntities
             RETURN m.id AS id, m.content AS content, m.kind AS kind,
                    m.normalizedKind AS normalizedKind,
                    m.confidence AS confidence, m.salience AS salience,
                    m.embedding AS embedding,
                    ${SUPERSEDED_AT("m")},
                    sharedEntities`,
            { entityNames: expansionEntities, seedIds },
          ),
        );

        // Score expansions: vector similarity + entity overlap bonus
        const seen = new Set<string>();
        for (const r of expandResult.records) {
          const id = r.get("id") as string;
          if (seen.has(id)) continue;
          seen.add(id);

          const embedding = r.get("embedding") as number[] | null;
          const confidence = (r.get("confidence") as number) ?? 0.5;
          const salience = (r.get("salience") as number) ?? 0.5;
          const sharedEntities = r.get("sharedEntities") as string[];

          // Vector similarity to query (compute in JS to avoid extra ANN call)
          let sim = 0;
          if (embedding && embedding.length === queryVector.length) {
            sim = cosineSimilarity(queryVector, embedding);
          }

          // Entity overlap bonus: 0.1 per shared non-hub entity, capped at 0.3
          const entityBonus = Math.min(0.3, sharedEntities.length * 0.1);

          // Combined score: base vector score + entity bonus
          const supersededAt = (r.get("supersededAt") as string | null) ?? undefined;
          const baseScore = sim * (0.6 + 0.4 * confidence) * (0.6 + 0.4 * salience);
          const score = (baseScore + entityBonus) * validityWeight(supersededAt);

          expansions.push({
            id,
            content: r.get("content") as string,
            kind: r.get("kind") as string,
            normalizedKind: r.get("normalizedKind") as string | undefined,
            score,
            supersededAt,
            source: "graph",
          });
        }
      }

      // ── Step 4: Merge, dedupe, rerank ───────────────────────────────
      const merged = new Map<string, ScoredResult>();
      for (const s of seeds) {
        merged.set(s.id, s);
      }
      for (const e of expansions) {
        const existing = merged.get(e.id);
        if (!existing || e.score > existing.score) {
          merged.set(e.id, e);
        }
      }

      const ranked = [...merged.values()]
        .sort((a, b) => b.score - a.score)
        .slice(0, limit);

      const graphCount = ranked.filter((r) => r.source === "graph").length;
      if (graphCount > 0) {
        console.log(
          `[dual-memory] Hybrid retrieval: ${ranked.length - graphCount} vector + ${graphCount} graph-expanded results`,
        );
      }

      // ── Fetch entity decorations for final results ──────────────────
      // Only include LINKED_TO relations where the target entity is also
      // mentioned by at least one memory in the result set (relevance filter).
      // Cap at 3 relations per entity to limit context size.
      if (ranked.length > 0) {
        const finalIds = ranked.map((r) => r.id);
        const entResult = await session.executeRead((tx: ManagedTransaction) =>
          tx.run(
            `// Collect all entities mentioned by result memories
             MATCH (resultMem:Memory)-[:MENTIONS]->(resultEnt:Entity)
             WHERE resultMem.id IN $ids AND (resultEnt.active IS NULL OR resultEnt.active = true)
             WITH collect(DISTINCT resultEnt.name) AS resultEntityNames
             // Now decorate each memory
             UNWIND $ids AS memId
             MATCH (m:Memory {id: memId})
             OPTIONAL MATCH (m)-[:MENTIONS]->(e:Entity)
             WHERE e.active IS NULL OR e.active = true
             WITH m, resultEntityNames, collect(DISTINCT e) AS ents
             UNWIND ents AS e
             OPTIONAL MATCH (e)-[r:LINKED_TO]->(other:Entity)
             WHERE (other.active IS NULL OR other.active = true)
               AND other.name IN resultEntityNames
             WITH m, e, collect(DISTINCT {name: other.name, relation: r.relation})[..3] AS links
             RETURN m.id AS id,
                    collect(DISTINCT e.name) AS entities,
                    collect(DISTINCT {from: e.name, links: links}) AS entityLinks`,
            { ids: finalIds },
          ),
        );

        const entityMap = new Map<string, string[]>();
        const relMap = new Map<string, string[]>();
        for (const r of entResult.records) {
          entityMap.set(r.get("id"), r.get("entities"));
          const entityLinks = r.get("entityLinks") as Array<{
            from: string;
            links: Array<{ name: string; relation: string }>;
          }>;
          const relStrings: string[] = [];
          for (const el of entityLinks) {
            for (const link of el.links) {
              if (link.name && link.relation) {
                relStrings.push(`${el.from} is ${link.relation} of ${link.name}`);
              }
            }
          }
          if (relStrings.length > 0) {
            relMap.set(r.get("id"), [...new Set(relStrings)]);
          }
        }

        return ranked.map((r) => ({
          id: r.id,
          content: r.content,
          kind: r.kind,
          normalizedKind: r.normalizedKind,
          score: r.score,
          supersededAt: r.supersededAt,
          entities: entityMap.get(r.id) ?? [],
          entityRelations: relMap.get(r.id),
        }));
      }

      return ranked.map((r) => ({
        id: r.id,
        content: r.content,
        kind: r.kind,
        normalizedKind: r.normalizedKind,
        score: r.score,
        supersededAt: r.supersededAt,
      }));
    } finally {
      await session.close();
    }
  }

  /**
   * Store a single memory from explicit tool use (memory_graph_store).
   */
  async storeMemory(
    content: string,
    kind: string,
    embedding: number[],
    entities: ExtractedFact["entities"],
    sessionId?: string,
  ): Promise<string> {
    await this.ensureSchema();
    const memoryId = randomUUID();
    const session = this.driver.session();

    try {
      await session.executeWrite(async (tx: ManagedTransaction) => {
        await tx.run(
          `CREATE (m:Memory {
            id: $id,
            content: $content,
            kind: $kind,
            timestamp: datetime(),
            validFrom: datetime(),
            confidence: 0.5,
            salience: 0.5,
            status: 'raw',
            embedding: $embedding
          })`,
          { id: memoryId, content, kind, embedding },
        );

        // Link to session if provided
        if (sessionId) {
          await tx.run(
            `MATCH (m:Memory {id: $memoryId}), (s:Session {id: $sessionId})
             CREATE (m)-[:PART_OF]->(s)`,
            { memoryId, sessionId },
          );
        }

        // Create entity mentions
        for (const entity of entities) {
          await tx.run(MENTION_ENTITY, {
            name: entity.name,
            entityId: randomUUID(),
            type: entity.type,
            aliases: entity.aliases ?? [entity.name],
            memoryId,
            role: entity.role,
          });
        }
      });
    } finally {
      await session.close();
    }

    return memoryId;
  }

  /**
   * Get memory count for stats.
   */
  async getStats(): Promise<{
    memories: number;
    entities: number;
    sessions: number;
    categories: number;
    raw: number;
  }> {
    await this.ensureSchema();
    const session = this.driver.session();
    try {
      const result = await session.executeRead((tx: ManagedTransaction) =>
        tx.run(`
          OPTIONAL MATCH (m:Memory) WITH count(m) AS memories
          OPTIONAL MATCH (e:Entity) WITH memories, count(e) AS entities
          OPTIONAL MATCH (s:Session) WITH memories, entities, count(s) AS sessions
          OPTIONAL MATCH (c:Category) WITH memories, entities, sessions, count(c) AS categories
          OPTIONAL MATCH (r:Memory {status: 'raw'}) WITH memories, entities, sessions, categories, count(r) AS raw
          RETURN memories, entities, sessions, categories, raw
        `),
      );
      const r = result.records[0];
      return {
        memories: (r.get("memories") as any)?.toNumber?.() ?? 0,
        entities: (r.get("entities") as any)?.toNumber?.() ?? 0,
        sessions: (r.get("sessions") as any)?.toNumber?.() ?? 0,
        categories: (r.get("categories") as any)?.toNumber?.() ?? 0,
        raw: (r.get("raw") as any)?.toNumber?.() ?? 0,
      };
    } finally {
      await session.close();
    }
  }

  /**
   * Suppress a memory (soft delete for privacy/GDPR).
   */
  async suppressMemory(memoryId: string): Promise<boolean> {
    await this.ensureSchema();
    const session = this.driver.session();
    try {
      const result = await session.executeWrite((tx: ManagedTransaction) =>
        tx.run(
          `MATCH (m:Memory {id: $memoryId})
           SET m.status = 'suppressed'
           RETURN m.id`,
          { memoryId },
        ),
      );
      return result.records.length > 0;
    } finally {
      await session.close();
    }
  }

  /**
   * Close the driver connection.
   */
  async close(): Promise<void> {
    await this.driver.close();
  }
}

// ============================================================================
// Utility functions
// ============================================================================

/**
 * Cosine similarity between two vectors (used for graph expansion scoring).
 * Both vectors must have the same length.
 */
function cosineSimilarity(a: number[], b: number[]): number {
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  const denom = Math.sqrt(normA) * Math.sqrt(normB);
  return denom === 0 ? 0 : dot / denom;
}
