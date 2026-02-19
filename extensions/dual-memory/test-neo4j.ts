/**
 * Neo4j-only integration test — verifies graph operations work without OpenAI.
 * Uses random vectors as mock embeddings.
 *
 * Requires: Neo4j running at bolt://localhost:7687
 * Usage: npx tsx test-neo4j.ts
 */

import { Neo4jClient, type ExtractedFact } from "./neo4j-client.js";

const NEO4J_URI = "bolt://localhost:7687";
const NEO4J_USER = "neo4j";
const NEO4J_PASSWORD = "dualmemory2026";

const VECTOR_DIMS = 1024; // Voyage AI default (voyage-4-lite)

/** Generate a random 1024-dim vector (mock embedding). */
function mockEmbedding(): number[] {
  const v = Array.from({ length: VECTOR_DIMS }, () => Math.random() * 2 - 1);
  // Normalize to unit vector for cosine similarity
  const norm = Math.sqrt(v.reduce((s, x) => s + x * x, 0));
  return v.map((x) => x / norm);
}

async function test() {
  console.log("=== Neo4j Integration Test (no OpenAI) ===\n");

  // 1. Connect
  console.log("1. Connecting to Neo4j...");
  const db = new Neo4jClient(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, VECTOR_DIMS);
  await db.ensureSchema();
  console.log("   ✓ Connected and schema initialized\n");

  // 2. Create session
  console.log("2. Creating session...");
  const session = await db.createSession({
    date: new Date().toISOString(),
    summary: "Test session: learning Rust",
    messageCount: 4,
    channel: "cli",
  });
  console.log(`   ✓ Session: ${session.id.slice(0, 8)}...\n`);

  // 3. Write facts
  console.log("3. Writing facts...");
  const facts: ExtractedFact[] = [
    {
      content: "User has been learning Rust for the past two weeks",
      kind: "fact",
      confidence: 0.8,
      salience: 0.6,
      sourceQuote: "I've been learning Rust for the past two weeks",
      sourceChannel: "cli",
      entities: [
        { name: "User", type: "Person", aliases: ["User"], role: "subject" },
        { name: "Rust", type: "Tool", aliases: ["Rust", "Rust language"], role: "object" },
      ],
    },
    {
      content: "Alice recommended Rust to the user because she uses it at TechCorp",
      kind: "fact",
      confidence: 0.8,
      salience: 0.5,
      sourceQuote: "My friend Alice recommended it",
      sourceChannel: "cli",
      entities: [
        { name: "Alice", type: "Person", aliases: ["Alice"], role: "source" },
        { name: "Rust", type: "Tool", aliases: ["Rust"], role: "object" },
        { name: "TechCorp", type: "Organization", aliases: ["TechCorp"], role: "context" },
      ],
    },
    {
      content: "User decided to build a CLI tool as their first Rust project",
      kind: "decision",
      confidence: 0.9,
      salience: 0.8,
      sourceQuote: "I decided to build a CLI tool as my first project",
      sourceChannel: "cli",
      entities: [
        { name: "User", type: "Person", aliases: ["User"], role: "subject" },
        { name: "Rust", type: "Tool", aliases: ["Rust"], role: "context" },
      ],
    },
    {
      content: "User is feeling really motivated about their career direction",
      kind: "emotion",
      confidence: 0.85,
      salience: 0.7,
      sourceQuote: "I'm feeling really motivated about this career direction",
      sourceChannel: "cli",
      entities: [
        { name: "User", type: "Person", aliases: ["User"], role: "subject" },
      ],
    },
  ];

  const embeddings = facts.map(() => mockEmbedding());
  const memoryIds = await db.writeFacts(facts, session.id, embeddings);
  console.log(`   ✓ Wrote ${memoryIds.length} memories`);
  for (let i = 0; i < memoryIds.length; i++) {
    console.log(`     ${memoryIds[i].slice(0, 8)}... [${facts[i].kind}] ${facts[i].content.slice(0, 50)}...`);
  }
  console.log();

  // 4. Retrieval (ANN fallback — no categories exist yet)
  console.log("4. Testing retrieval (ANN fallback)...");
  // Use a vector similar to the first fact for testing
  const queryVector = embeddings[0].map((v) => v + (Math.random() - 0.5) * 0.1);
  const queryNorm = Math.sqrt(queryVector.reduce((s, x) => s + x * x, 0));
  const normalizedQuery = queryVector.map((x) => x / queryNorm);

  const results = await db.retrieve(normalizedQuery, 5);
  console.log(`   ✓ Retrieved ${results.length} results`);
  for (const r of results) {
    console.log(
      `     [${r.kind}] ${r.content.slice(0, 50)}... (score=${r.score.toFixed(3)}) entities=[${r.entities?.join(", ")}]`,
    );
  }
  console.log();

  // 5. Stats
  console.log("5. Graph statistics...");
  const stats = await db.getStats();
  console.log(`   Memories: ${stats.memories} (${stats.raw} raw)`);
  console.log(`   Entities: ${stats.entities}`);
  console.log(`   Sessions: ${stats.sessions}`);
  console.log(`   Categories: ${stats.categories}`);
  console.log();

  // 6. Suppress test
  console.log("6. Testing suppress...");
  const suppressed = await db.suppressMemory(memoryIds[0]);
  console.log(`   ✓ Suppressed ${memoryIds[0].slice(0, 8)}...: ${suppressed}`);

  // Verify suppressed memory is excluded from retrieval
  const afterSuppress = await db.retrieve(normalizedQuery, 5);
  const suppressedInResults = afterSuppress.some((r) => r.id === memoryIds[0]);
  console.log(`   ✓ Suppressed memory in results: ${suppressedInResults} (should be false)`);
  console.log();

  // 7. Store single memory test
  console.log("7. Testing storeMemory...");
  const storeId = await db.storeMemory(
    "User prefers dark mode in all IDEs",
    "preference",
    mockEmbedding(),
    [
      { name: "User", type: "Person", aliases: ["User"], role: "subject" },
    ],
    session.id,
  );
  console.log(`   ✓ Stored: ${storeId.slice(0, 8)}...\n`);

  // Final stats
  console.log("8. Final statistics...");
  const finalStats = await db.getStats();
  console.log(`   Memories: ${finalStats.memories} (${finalStats.raw} raw)`);
  console.log(`   Entities: ${finalStats.entities}`);
  console.log(`   Sessions: ${finalStats.sessions}`);
  console.log();

  // Cleanup
  await db.close();
  console.log("=== All tests passed! ===");
}

test().catch((err) => {
  console.error("Test failed:", err);
  process.exit(1);
});
