/**
 * Eval bridge — lets the Python eval harness drive the real Writer write path
 * and the real retrieval against a scratch Neo4j. Not part of the plugin
 * (OpenClaw only loads index.ts).
 *
 * Usage:
 *   tsx eval-bridge.ts load <in.json> <out.json>
 *   tsx eval-bridge.ts retrieve <in.json> <out.json>
 *   tsx eval-bridge.ts extract <in.json> <out.json>   (Writer LLM extraction)
 *   tsx eval-bridge.ts recall <in.json> <out.json>    (the context block the plugin injects)
 *
 * Env: NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD. Results go to <out.json> because
 * the client logs to stdout.
 */

import { readFileSync, writeFileSync } from "node:fs";
import type { Embeddings } from "./embeddings.js";
import { Neo4jClient, type ExtractedFact, type FactWriteOutcome } from "./neo4j-client.js";
import { Retrieval } from "./retrieval.js";
import { Writer } from "./writer.js";

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

type ExtractInput = {
  model?: string;
  concurrency: number;
  sessions: Array<{ id: string; messages: Array<{ role: string; content: string }> }>;
};

/** Runs the Writer's LLM extraction; needs ANTHROPIC_API_KEY. No database. */
async function extract(input: ExtractInput) {
  const writer = new Writer("anthropic", process.env.ANTHROPIC_API_KEY ?? "", input.model);
  const results: unknown[] = new Array(input.sessions.length);
  let next = 0;
  const worker = async () => {
    while (next < input.sessions.length) {
      const i = next++;
      const s = input.sessions[i];
      const { facts, entityLinks, sessionSummary, usage } = await writer.extractFacts(s.messages);
      results[i] = { id: s.id, facts, entityLinks, sessionSummary, usage };
    }
  };
  await Promise.all(Array.from({ length: input.concurrency }, worker));
  return { results };
}

type RecallInput = {
  vectorDims: number;
  queries: Array<{ id: string; text: string; vector: number[] }>;
};

/** Retrieval.retrieveContext with the query embedding supplied by the caller. */
async function recall(db: Neo4jClient, input: RecallInput) {
  const results = [];
  for (const q of input.queries) {
    const embeddings = { embed: async () => q.vector } as unknown as Embeddings;
    const context = await new Retrieval(db, embeddings).retrieveContext(q.text);
    const hits = await db.retrieve(q.vector, 8);
    results.push({ id: q.id, context, memoryIds: hits.map((h) => h.id) });
  }
  return { results };
}

const [command, inPath, outPath] = process.argv.slice(2);
if (!command || !inPath || !outPath) {
  console.error("usage: tsx eval-bridge.ts <load|retrieve|extract|recall> <in.json> <out.json>");
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
  else if (command === "extract") output = await extract(input as ExtractInput);
  else if (command === "recall") output = await recall(db, input as RecallInput);
  else throw new Error(`unknown command: ${command}`);
  writeFileSync(outPath, JSON.stringify(output));
} finally {
  await db.close();
}
