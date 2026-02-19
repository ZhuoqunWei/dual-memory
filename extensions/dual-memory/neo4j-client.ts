/**
 * Neo4j driver wrapper with schema initialization and typed queries.
 * All graph operations go through this client.
 */

import neo4j, { type Driver, type Session, type ManagedTransaction } from "neo4j-driver";
import { randomUUID } from "node:crypto";

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

export type RetrievalResult = {
  id: string;
  content: string;
  kind: string;
  normalizedKind?: string;
  score: number;
  entities?: string[];
};

// ============================================================================
// Schema initialization statements (from init-schema.cypher)
// ============================================================================

const SCHEMA_STATEMENTS = [
  // Uniqueness constraints
  "CREATE CONSTRAINT memory_id IF NOT EXISTS FOR (m:Memory) REQUIRE m.id IS UNIQUE",
  "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
  "CREATE CONSTRAINT session_id IF NOT EXISTS FOR (s:Session) REQUIRE s.id IS UNIQUE",
  "CREATE CONSTRAINT editaction_id IF NOT EXISTS FOR (a:EditAction) REQUIRE a.id IS UNIQUE",
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
];

// Vector indexes need separate handling (OPTIONS clause)
const VECTOR_INDEX_STATEMENTS = [
  `CREATE VECTOR INDEX memoryEmbeddings IF NOT EXISTS
   FOR (m:Memory) ON (m.embedding)
   OPTIONS { indexConfig: { \`vector.dimensions\`: 1536, \`vector.similarity_function\`: 'cosine' }}`,
  `CREATE VECTOR INDEX entityEmbeddings IF NOT EXISTS
   FOR (e:Entity) ON (e.embedding)
   OPTIONS { indexConfig: { \`vector.dimensions\`: 1536, \`vector.similarity_function\`: 'cosine' }}`,
];

// ============================================================================
// Client
// ============================================================================

export class Neo4jClient {
  private driver: Driver;
  private initialized = false;
  private initPromise: Promise<void> | null = null;

  constructor(
    uri: string,
    user: string,
    password: string,
  ) {
    this.driver = neo4j.driver(uri, neo4j.auth.basic(user, password));
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
      // Run vector index creation
      for (const stmt of VECTOR_INDEX_STATEMENTS) {
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

  async createSession(data: Omit<SessionNode, "id">): Promise<SessionNode> {
    await this.ensureSchema();
    const id = randomUUID();
    const session = this.driver.session();
    try {
      await session.executeWrite((tx: ManagedTransaction) =>
        tx.run(
          `CREATE (s:Session {
            id: $id, date: datetime($date), summary: $summary,
            messageCount: $messageCount, channel: $channel
          })`,
          { id, ...data },
        ),
      );
      return { id, ...data };
    } finally {
      await session.close();
    }
  }

  // ========================================================================
  // Memory operations (Writer)
  // ========================================================================

  /**
   * Write extracted facts to Neo4j as Memory nodes + Entity nodes + relationships.
   * This is the Writer's main operation.
   */
  async writeFacts(
    facts: ExtractedFact[],
    sessionId: string,
    embeddings: number[][],
  ): Promise<string[]> {
    await this.ensureSchema();
    const memoryIds: string[] = [];
    const session = this.driver.session();

    try {
      for (let i = 0; i < facts.length; i++) {
        const fact = facts[i];
        const embedding = embeddings[i];
        const memoryId = randomUUID();
        memoryIds.push(memoryId);

        await session.executeWrite(async (tx: ManagedTransaction) => {
          // 1. Create Memory node
          await tx.run(
            `CREATE (m:Memory {
              id: $id,
              content: $content,
              kind: $kind,
              timestamp: datetime(),
              eventTimeStart: CASE WHEN $eventTimeStart IS NOT NULL THEN datetime($eventTimeStart) ELSE null END,
              eventTimeEnd: CASE WHEN $eventTimeEnd IS NOT NULL THEN datetime($eventTimeEnd) ELSE null END,
              expiresAt: null,
              confidence: $confidence,
              salience: $salience,
              status: 'raw',
              lastAccessed: null,
              sourceRef: $sourceRef,
              sourceQuote: $sourceQuote,
              sourceChannel: $sourceChannel,
              sourceAuthor: $sourceAuthor,
              embedding: $embedding
            })`,
            {
              id: memoryId,
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

          // 2. Link to Session
          await tx.run(
            `MATCH (m:Memory {id: $memoryId}), (s:Session {id: $sessionId})
             CREATE (m)-[:PART_OF]->(s)`,
            { memoryId, sessionId },
          );

          // 3. Create/merge Entity nodes and MENTIONS relationships
          for (const entity of fact.entities) {
            // Try to find existing entity by name or alias
            await tx.run(
              `MERGE (e:Entity {name: $name})
               ON CREATE SET
                 e.id = $entityId,
                 e.type = $type,
                 e.aliases = $aliases,
                 e.firstSeen = datetime()
               ON MATCH SET
                 e.aliases = CASE
                   WHEN size([a IN $aliases WHERE NOT a IN e.aliases]) > 0
                   THEN e.aliases + [a IN $aliases WHERE NOT a IN e.aliases]
                   ELSE e.aliases
                 END
               WITH e
               MATCH (m:Memory {id: $memoryId})
               CREATE (m)-[:MENTIONS {role: $role}]->(e)`,
              {
                name: entity.name,
                entityId: randomUUID(),
                type: entity.type,
                aliases: entity.aliases ?? [entity.name],
                memoryId,
                role: entity.role,
              },
            );
          }
        });
      }
    } finally {
      await session.close();
    }

    return memoryIds;
  }

  // ========================================================================
  // Retrieval operations
  // ========================================================================

  /**
   * Two-phase retrieval: category routing → vector search within subgraph.
   * Falls back to full ANN search if no categories match.
   */
  async retrieve(
    queryVector: number[],
    limit: number = 10,
    categoryNames?: string[],
  ): Promise<RetrievalResult[]> {
    await this.ensureSchema();
    const session = this.driver.session();

    try {
      // Phase 1: If categories provided, do category-scoped search
      if (categoryNames && categoryNames.length > 0) {
        const result = await session.executeRead((tx: ManagedTransaction) =>
          tx.run(
            `MATCH (mem:Memory)-[:IN_CATEGORY]->(cat:Category)
             WHERE cat.name IN $categoryNames
               AND mem.status NOT IN ['archived', 'suppressed']
               AND NOT EXISTS { (mem)-[:CANONICAL]->() }
               AND (mem.expiresAt IS NULL OR mem.expiresAt > datetime())
             WITH mem, vector.similarity.cosine(mem.embedding, $queryVector) AS sim
             WITH mem, sim * (0.6 + 0.4 * mem.confidence) * (0.6 + 0.4 * mem.salience) AS score
             WHERE score > 0.1
             ORDER BY score DESC LIMIT $limit
             OPTIONAL MATCH (mem)-[:MENTIONS]->(e:Entity)
             RETURN mem.id AS id, mem.content AS content, mem.kind AS kind,
                    mem.normalizedKind AS normalizedKind, score,
                    collect(DISTINCT e.name) AS entities`,
            { categoryNames, queryVector, limit: neo4j.int(limit) },
          ),
        );

        if (result.records.length > 0) {
          return result.records.map((r) => ({
            id: r.get("id"),
            content: r.get("content"),
            kind: r.get("kind"),
            normalizedKind: r.get("normalizedKind"),
            score: r.get("score"),
            entities: r.get("entities"),
          }));
        }
      }

      // Phase 2 / fallback: full ANN search across all memories
      // Note: We fetch more than needed and filter in application to avoid Cypher syntax
      // limitations around WHERE after CALL/YIELD in some Neo4j versions.
      const annLimit = neo4j.int(limit * 3); // over-fetch to account for filtering
      const result = await session.executeRead((tx: ManagedTransaction) =>
        tx.run(
          `CALL db.index.vector.queryNodes('memoryEmbeddings', $annLimit, $queryVector)
           YIELD node AS mem, score AS sim
           RETURN mem.id AS id, mem.content AS content, mem.kind AS kind,
                  mem.normalizedKind AS normalizedKind,
                  mem.status AS status, mem.confidence AS confidence,
                  mem.salience AS salience, mem.expiresAt AS expiresAt,
                  sim`,
          { queryVector, annLimit },
        ),
      );

      // Apply filters and scoring in application layer
      const now = new Date().toISOString();
      const scored = result.records
        .filter((r) => {
          const status = r.get("status");
          if (status === "archived" || status === "suppressed") return false;
          const expiresAt = r.get("expiresAt");
          if (expiresAt && expiresAt < now) return false;
          return true;
        })
        .map((r) => {
          const sim = r.get("sim") as number;
          const confidence = (r.get("confidence") as number) ?? 0.5;
          const salience = (r.get("salience") as number) ?? 0.5;
          const score = sim * (0.6 + 0.4 * confidence) * (0.6 + 0.4 * salience);
          return {
            id: r.get("id") as string,
            content: r.get("content") as string,
            kind: r.get("kind") as string,
            normalizedKind: r.get("normalizedKind") as string | undefined,
            score,
          };
        })
        .sort((a, b) => b.score - a.score)
        .slice(0, limit);

      // Now fetch entities for the top results (separate query to keep it clean)
      if (scored.length > 0) {
        const ids = scored.map((s) => s.id);
        const entResult = await session.executeRead((tx: ManagedTransaction) =>
          tx.run(
            `UNWIND $ids AS memId
             MATCH (m:Memory {id: memId})
             OPTIONAL MATCH (m)-[:MENTIONS]->(e:Entity)
             RETURN m.id AS id, collect(DISTINCT e.name) AS entities`,
            { ids },
          ),
        );
        const entityMap = new Map<string, string[]>();
        for (const r of entResult.records) {
          entityMap.set(r.get("id"), r.get("entities"));
        }
        return scored.map((s) => ({
          ...s,
          entities: entityMap.get(s.id) ?? [],
        }));
      }

      return scored;

      return result.records.map((r) => ({
        id: r.get("id"),
        content: r.get("content"),
        kind: r.get("kind"),
        normalizedKind: r.get("normalizedKind"),
        score: r.get("score"),
        entities: r.get("entities"),
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
          await tx.run(
            `MERGE (e:Entity {name: $name})
             ON CREATE SET e.id = $entityId, e.type = $type,
                           e.aliases = $aliases, e.firstSeen = datetime()
             ON MATCH SET e.aliases = CASE
               WHEN size([a IN $aliases WHERE NOT a IN e.aliases]) > 0
               THEN e.aliases + [a IN $aliases WHERE NOT a IN e.aliases]
               ELSE e.aliases END
             WITH e
             MATCH (m:Memory {id: $memoryId})
             CREATE (m)-[:MENTIONS {role: $role}]->(e)`,
            {
              name: entity.name,
              entityId: randomUUID(),
              type: entity.type,
              aliases: entity.aliases ?? [entity.name],
              memoryId,
              role: entity.role,
            },
          );
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
