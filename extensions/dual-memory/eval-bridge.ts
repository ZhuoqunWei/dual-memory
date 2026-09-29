/**
 * Eval bridge — lets the Python eval harness drive the real Writer write path
 * and the real retrieval against a scratch Neo4j. Not part of the plugin
 * (OpenClaw only loads index.ts).
 *
 * Usage:
 *   tsx eval-bridge.ts load <in.json> <out.json>
 *   tsx eval-bridge.ts retrieve <in.json> <out.json>
 *
 * Env: NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD. Results go to <out.json> because
 * the client logs to stdout.
 */

import { readFileSync, writeFileSync } from "node:fs";
import { Neo4jClient, type ExtractedFact, type FactWriteOutcome } from "./neo4j-client.js";

type LoadInput = {
  vectorDims: number;
  sessions: Array<{
    key: string;
    date: string;
    summary: string;
    channel: string;
    facts: ExtractedFact[];
    embeddings: number[][];
  }>;
};

type RetrieveInput = {
  vectorDims: number;
  limit: number;
  queries: Array<{ id: string; vector: number[] }>;
};

async function load(db: Neo4jClient, input: LoadInput) {
  const sessions: Array<{ sessionId: string; outcomes: FactWriteOutcome[] }> = [];
  for (const s of input.sessions) {
    const node = await db.createSession({
      key: s.key,
      date: s.date,
      summary: s.summary,
      messageCount: s.facts.length,
      channel: s.channel,
    });
    const { outcomes } = await db.writeFacts(s.facts, node.id, s.embeddings, {
      observedAt: s.date,
    });
    sessions.push({ sessionId: node.id, outcomes });
  }
  return { sessions };
}

async function retrieve(db: Neo4jClient, input: RetrieveInput) {
  const results = [];
  for (const q of input.queries) {
    const hits = await db.retrieve(q.vector, input.limit);
    results.push({ id: q.id, hits: hits.map((h) => ({ memoryId: h.id, score: h.score })) });
  }
  return { results };
}

const [command, inPath, outPath] = process.argv.slice(2);
if (!command || !inPath || !outPath) {
  console.error("usage: tsx eval-bridge.ts <load|retrieve> <in.json> <out.json>");
  process.exit(2);
}

const input = JSON.parse(readFileSync(inPath, "utf8"));
const db = new Neo4jClient(
  process.env.NEO4J_URI ?? "",
  process.env.NEO4J_USER ?? "neo4j",
  process.env.NEO4J_PASSWORD ?? "",
  input.vectorDims,
);

try {
  let output: unknown;
  if (command === "load") output = await load(db, input as LoadInput);
  else if (command === "retrieve") output = await retrieve(db, input as RetrieveInput);
  else throw new Error(`unknown command: ${command}`);
  writeFileSync(outPath, JSON.stringify(output));
} finally {
  await db.close();
}
