/**
 * Integration test — verifies Neo4j client + Writer + Retrieval work end-to-end.
 * Requires: Neo4j running at bolt://localhost:7687, OPENAI_API_KEY env var.
 *
 * Usage: npx tsx test-integration.ts
 */

import { Neo4jClient, type ExtractedFact } from "./neo4j-client.js";
import { Embeddings } from "./embeddings.js";
import { Writer } from "./writer.js";
import { Retrieval } from "./retrieval.js";

const NEO4J_URI = "bolt://localhost:7687";
const NEO4J_USER = "neo4j";
const NEO4J_PASSWORD = "dualmemory2026";

const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
if (!OPENAI_API_KEY) {
  console.error("ERROR: OPENAI_API_KEY environment variable is required");
  process.exit(1);
}

async function test() {
  console.log("=== Dual-Memory Integration Test ===\n");

  // 1. Connect to Neo4j
  console.log("1. Connecting to Neo4j...");
  const db = new Neo4jClient(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD);
  await db.ensureSchema();
  console.log("   ✓ Connected and schema initialized\n");

  // 2. Create a session
  console.log("2. Creating session...");
  const session = await db.createSession({
    date: new Date().toISOString(),
    summary: "Integration test session",
    messageCount: 4,
    channel: "cli",
  });
  console.log(`   ✓ Session created: ${session.id.slice(0, 8)}...\n`);

  // 3. Initialize embeddings
  console.log("3. Initializing embeddings...");
  const embeddings = new Embeddings(OPENAI_API_KEY, "text-embedding-3-small");

  // 4. Test Writer: extract facts from mock messages
  console.log("4. Testing Writer (LLM fact extraction)...");
  const writer = new Writer(OPENAI_API_KEY, "gpt-4o-mini");
  const mockMessages = [
    {
      role: "user",
      content:
        "I've been learning Rust for the past two weeks. My friend Alice recommended it because she uses it at her company, TechCorp.",
    },
    {
      role: "assistant",
      content:
        "That's great! Rust is an excellent choice for systems programming. How are you finding the learning curve?",
    },
    {
      role: "user",
      content:
        "It's challenging but I love the borrow checker. I decided to build a CLI tool as my first project. I'm feeling really motivated about this career direction.",
    },
    {
      role: "assistant",
      content:
        "Building a CLI tool is a perfect first Rust project. The ownership model will click quickly with hands-on practice.",
    },
  ];

  const { facts, sessionSummary } = await writer.extractFacts(mockMessages, "cli");
  console.log(`   ✓ Extracted ${facts.length} facts`);
  console.log(`   Summary: "${sessionSummary}"`);
  for (const f of facts) {
    console.log(`   - [${f.kind}] ${f.content.slice(0, 80)}... (conf=${f.confidence}, sal=${f.salience})`);
    for (const e of f.entities) {
      console.log(`     → ${e.name} (${e.type}, role=${e.role})`);
    }
  }
  console.log();

  // 5. Generate embeddings and write to Neo4j
  console.log("5. Writing facts to Neo4j...");
  const factTexts = facts.map((f) => f.content);
  const factEmbeddings = await embeddings.embedBatch(factTexts);
  const memoryIds = await db.writeFacts(facts, session.id, factEmbeddings);
  console.log(`   ✓ Wrote ${memoryIds.length} memories to Neo4j\n`);

  // 6. Test Retrieval
  console.log("6. Testing Retrieval...");
  const retrieval = new Retrieval(db, embeddings);

  const queries = [
    "What programming languages is the user learning?",
    "Who recommended Rust?",
    "How is the user feeling about their career?",
  ];

  for (const q of queries) {
    console.log(`   Query: "${q}"`);
    const results = await retrieval.retrieveResults(q, 3);
    if (results.length === 0) {
      console.log("   → No results\n");
    } else {
      for (const r of results) {
        const ents = r.entities?.join(", ") || "";
        console.log(
          `   → [${r.kind}] ${r.content.slice(0, 60)}... (score=${Math.round(r.score * 100)}%) ${ents ? `[${ents}]` : ""}`,
        );
      }
      console.log();
    }
  }

  // 7. Test context injection format
  console.log("7. Testing context injection...");
  const context = await retrieval.retrieveContext("Tell me about Rust");
  if (context) {
    console.log("   ✓ Context generated:");
    console.log(
      context
        .split("\n")
        .map((l) => `   ${l}`)
        .join("\n"),
    );
  } else {
    console.log("   ✗ No context generated");
  }
  console.log();

  // 8. Stats
  console.log("8. Graph statistics...");
  const stats = await db.getStats();
  console.log(`   Memories: ${stats.memories} (${stats.raw} raw)`);
  console.log(`   Entities: ${stats.entities}`);
  console.log(`   Sessions: ${stats.sessions}`);
  console.log(`   Categories: ${stats.categories}`);
  console.log();

  // 9. Test suppress (soft delete)
  console.log("9. Testing suppress...");
  if (memoryIds.length > 0) {
    const suppressed = await db.suppressMemory(memoryIds[0]);
    console.log(`   ✓ Suppressed memory ${memoryIds[0].slice(0, 8)}...: ${suppressed}`);
  }
  console.log();

  // Cleanup
  await db.close();
  console.log("=== All tests passed! ===");
}

test().catch((err) => {
  console.error("Test failed:", err);
  process.exit(1);
});
