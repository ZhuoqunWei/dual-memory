// ============================================================================
// Dual-Agent Memory — Neo4j Schema Initialization (v2.1)
// Run once after first Neo4j startup
// ============================================================================

// === Uniqueness constraints (also create indexes) ===

CREATE CONSTRAINT memory_id IF NOT EXISTS
FOR (m:Memory) REQUIRE m.id IS UNIQUE;

CREATE CONSTRAINT entity_id IF NOT EXISTS
FOR (e:Entity) REQUIRE e.id IS UNIQUE;

CREATE CONSTRAINT session_id IF NOT EXISTS
FOR (s:Session) REQUIRE s.id IS UNIQUE;

CREATE CONSTRAINT editaction_id IF NOT EXISTS
FOR (a:EditAction) REQUIRE a.id IS UNIQUE;

CREATE CONSTRAINT category_name IF NOT EXISTS
FOR (c:Category) REQUIRE c.name IS UNIQUE;

// === Performance indexes ===

CREATE INDEX memory_status IF NOT EXISTS
FOR (m:Memory) ON (m.status);

CREATE INDEX memory_timestamp IF NOT EXISTS
FOR (m:Memory) ON (m.timestamp);

CREATE INDEX memory_eventTimeStart IF NOT EXISTS
FOR (m:Memory) ON (m.eventTimeStart);

CREATE INDEX memory_expiresAt IF NOT EXISTS
FOR (m:Memory) ON (m.expiresAt);

CREATE INDEX memory_normalizedKind IF NOT EXISTS
FOR (m:Memory) ON (m.normalizedKind);

CREATE INDEX entity_name IF NOT EXISTS
FOR (e:Entity) ON (e.name);

CREATE INDEX entity_normalizedType IF NOT EXISTS
FOR (e:Entity) ON (e.normalizedType);

CREATE INDEX session_date IF NOT EXISTS
FOR (s:Session) ON (s.date);

// === Fulltext index for entity resolution ===

CREATE FULLTEXT INDEX entity_names IF NOT EXISTS
FOR (e:Entity) ON EACH [e.name];
// Note: aliases is a String[] — fulltext over arrays requires
// application-layer search or a separate AliasNode in the future.

// === Vector indexes ===

CREATE VECTOR INDEX memoryEmbeddings IF NOT EXISTS
FOR (m:Memory) ON (m.embedding)
OPTIONS {
  indexConfig: {
    `vector.dimensions`: 1536,
    `vector.similarity_function`: 'cosine'
  }
};

CREATE VECTOR INDEX entityEmbeddings IF NOT EXISTS
FOR (e:Entity) ON (e.embedding)
OPTIONS {
  indexConfig: {
    `vector.dimensions`: 1536,
    `vector.similarity_function`: 'cosine'
  }
};
