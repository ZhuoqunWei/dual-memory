/**
 * Dual-Agent Memory — OpenClaw Plugin
 *
 * Self-organizing memory system with Neo4j knowledge graph.
 * Writer (this plugin): fast episodic capture after every session.
 * Editor (separate Python agent): slow semantic consolidation on schedule.
 *
 * Integration points:
 *   - before_agent_start → retrieval (inject relevant memories)
 *   - agent_end → writer (extract facts from conversation)
 *   - Tools: memory_graph_search, memory_graph_store, memory_graph_forget
 */

import type { OpenClawPluginApi } from "openclaw/plugin-sdk";
import { Type } from "@sinclair/typebox";
import { parseConfig, vectorDimsForModel } from "./config.js";
import { Neo4jClient } from "./neo4j-client.js";
import { Embeddings } from "./embeddings.js";
import { Writer } from "./writer.js";
import { Retrieval } from "./retrieval.js";

// ============================================================================
// Plugin Definition
// ============================================================================

const dualMemoryPlugin = {
  id: "dual-memory",
  name: "Dual-Agent Memory (Neo4j)",
  description:
    "Self-organizing knowledge graph memory with Writer/Editor dual-agent architecture",
  kind: "memory" as const,

  register(api: OpenClawPluginApi) {
    // ========================================================================
    // Initialization
    // ========================================================================

    const cfg = parseConfig(api.pluginConfig);
    const vectorDims = vectorDimsForModel(cfg.embedding.model);
    const db = new Neo4jClient(cfg.neo4j.uri, cfg.neo4j.user, cfg.neo4j.password, vectorDims);
    const embeddings = new Embeddings(
      cfg.embedding.provider,
      cfg.embedding.apiKey,
      cfg.embedding.model,
    );
    const writer = new Writer(
      cfg.extraction.provider,
      cfg.extraction.apiKey,
      cfg.extraction.model,
    );
    const retrieval = new Retrieval(db, embeddings);

    // Track current session ID for linking memories
    let currentSessionId: string | null = null;

    api.logger.info("dual-memory: plugin registered (lazy Neo4j init)");

    // ========================================================================
    // Lifecycle Hooks
    // ========================================================================

    // --- Auto-Recall: inject relevant memories before agent starts ---
    if (cfg.autoRecall) {
      api.on("before_agent_start", async (event) => {
        if (!event.prompt || event.prompt.length < 5) return;

        try {
          const context = await retrieval.retrieveContext(event.prompt);
          if (!context) return;

          api.logger.info?.("dual-memory: injecting memories into context");

          return { prependContext: context };
        } catch (err) {
          api.logger.warn(`dual-memory: recall failed: ${String(err)}`);
        }
      });
    }

    // --- Auto-Capture: extract facts after agent ends ---
    if (cfg.autoCapture) {
      api.on("agent_end", async (event, ctx) => {
        if (!event.success || !event.messages || event.messages.length === 0) {
          return;
        }

        try {
          // Determine channel from context
          const channel = ctx.messageProvider ?? "cli";

          // Extract facts via LLM
          const { facts, sessionSummary } = await writer.extractFacts(
            event.messages,
            channel,
          );

          if (facts.length === 0) {
            api.logger.info?.("dual-memory: no facts extracted from session");
            return;
          }

          // Create session node
          const sessionNode = await db.createSession({
            date: new Date().toISOString(),
            summary: sessionSummary || "Conversation session",
            messageCount: event.messages.length,
            channel,
          });
          currentSessionId = sessionNode.id;

          // Generate embeddings for all facts
          const factTexts = facts.map((f) => f.content);
          const factEmbeddings = await embeddings.embedBatch(factTexts);

          // Write to Neo4j
          const memoryIds = await db.writeFacts(
            facts,
            sessionNode.id,
            factEmbeddings,
          );

          api.logger.info(
            `dual-memory: captured ${memoryIds.length} memories from ${event.messages.length} messages`,
          );
        } catch (err) {
          api.logger.warn(`dual-memory: capture failed: ${String(err)}`);
        }
      });
    }

    // ========================================================================
    // Tools
    // ========================================================================

    // --- memory_graph_search: semantic search over the knowledge graph ---
    api.registerTool(
      {
        name: "memory_graph_search",
        label: "Graph Memory Search",
        description:
          "Search the knowledge graph for relevant memories. Use when you need context about past conversations, decisions, preferences, people, or events. Returns scored results with entity connections.",
        parameters: Type.Object({
          query: Type.String({
            description: "Natural language search query",
          }),
          limit: Type.Optional(
            Type.Number({
              description: "Max results (default: 8)",
              minimum: 1,
              maximum: 20,
            }),
          ),
        }),
        async execute(_toolCallId, params) {
          const { query, limit = 8 } = params as {
            query: string;
            limit?: number;
          };

          const results = await retrieval.retrieveResults(query, limit);

          if (results.length === 0) {
            return {
              content: [
                { type: "text", text: "No relevant memories found in the knowledge graph." },
              ],
              details: { count: 0 },
            };
          }

          const text = results
            .map((r, i) => {
              const kind = r.normalizedKind ?? r.kind;
              const entities =
                r.entities && r.entities.length > 0
                  ? ` (${r.entities.join(", ")})`
                  : "";
              return `${i + 1}. [${kind}]${entities} ${r.content} (${Math.round(r.score * 100)}%)`;
            })
            .join("\n");

          return {
            content: [
              {
                type: "text",
                text: `Found ${results.length} memories:\n\n${text}`,
              },
            ],
            details: {
              count: results.length,
              results: results.map((r) => ({
                id: r.id,
                content: r.content,
                kind: r.kind,
                score: r.score,
                entities: r.entities,
              })),
            },
          };
        },
      },
      { name: "memory_graph_search" },
    );

    // --- memory_graph_store: explicitly store a fact ---
    api.registerTool(
      {
        name: "memory_graph_store",
        label: "Graph Memory Store",
        description:
          "Store an important fact, decision, preference, or observation in the knowledge graph. Use when the user explicitly asks to remember something or when critical information should be preserved.",
        parameters: Type.Object({
          content: Type.String({
            description: "The fact or information to store",
          }),
          kind: Type.Optional(
            Type.String({
              description:
                "Type: fact | decision | preference | goal | emotion | observation | event (default: fact)",
            }),
          ),
          entities: Type.Optional(
            Type.Array(
              Type.Object({
                name: Type.String({ description: "Entity name" }),
                type: Type.Optional(
                  Type.String({
                    description: "Entity type: Person | Place | Project | Organization | Tool | Concept",
                  }),
                ),
                role: Type.Optional(
                  Type.String({
                    description: "Relation to fact: subject | object | context | source",
                  }),
                ),
              }),
              { description: "Entities mentioned in this fact" },
            ),
          ),
        }),
        async execute(_toolCallId, params) {
          const {
            content,
            kind = "fact",
            entities = [],
          } = params as {
            content: string;
            kind?: string;
            entities?: Array<{
              name: string;
              type?: string;
              role?: string;
            }>;
          };

          const embedding = await embeddings.embed(content);

          const typedEntities = entities.map((e) => ({
            name: e.name,
            type: e.type || "Concept",
            aliases: [e.name],
            role: (["subject", "object", "context", "source"].includes(e.role ?? "")
              ? e.role
              : "subject") as "subject" | "object" | "context" | "source",
          }));

          const memoryId = await db.storeMemory(
            content,
            kind,
            embedding,
            typedEntities,
            currentSessionId ?? undefined,
          );

          return {
            content: [
              {
                type: "text",
                text: `Stored in knowledge graph: "${content.slice(0, 80)}${content.length > 80 ? "..." : ""}"`,
              },
            ],
            details: { action: "created", id: memoryId, kind },
          };
        },
      },
      { name: "memory_graph_store" },
    );

    // --- memory_graph_forget: suppress a memory (soft delete) ---
    api.registerTool(
      {
        name: "memory_graph_forget",
        label: "Graph Memory Forget",
        description:
          "Suppress a memory from the knowledge graph. The memory is marked as suppressed (not deleted) for audit purposes. Use when the user asks to forget something.",
        parameters: Type.Object({
          query: Type.Optional(
            Type.String({ description: "Search to find memory to suppress" }),
          ),
          memoryId: Type.Optional(
            Type.String({ description: "Specific memory ID to suppress" }),
          ),
        }),
        async execute(_toolCallId, params) {
          const { query, memoryId } = params as {
            query?: string;
            memoryId?: string;
          };

          if (memoryId) {
            const success = await db.suppressMemory(memoryId);
            if (success) {
              return {
                content: [
                  { type: "text", text: `Memory ${memoryId.slice(0, 8)}... suppressed.` },
                ],
                details: { action: "suppressed", id: memoryId },
              };
            }
            return {
              content: [{ type: "text", text: `Memory ${memoryId} not found.` }],
              details: { error: "not_found" },
            };
          }

          if (query) {
            const results = await retrieval.retrieveResults(query, 5);

            if (results.length === 0) {
              return {
                content: [{ type: "text", text: "No matching memories found." }],
                details: { found: 0 },
              };
            }

            // If high-confidence single match, suppress it
            if (results.length === 1 || results[0].score > 0.9) {
              const target = results[0];
              const success = await db.suppressMemory(target.id);
              if (success) {
                return {
                  content: [
                    {
                      type: "text",
                      text: `Suppressed: "${target.content.slice(0, 60)}..."`,
                    },
                  ],
                  details: { action: "suppressed", id: target.id },
                };
              }
            }

            // Otherwise list candidates
            const list = results
              .map(
                (r) =>
                  `- [${r.id.slice(0, 8)}...] ${r.content.slice(0, 60)}... (${Math.round(r.score * 100)}%)`,
              )
              .join("\n");

            return {
              content: [
                {
                  type: "text",
                  text: `Found ${results.length} candidates. Specify memoryId to suppress:\n${list}`,
                },
              ],
              details: {
                action: "candidates",
                candidates: results.map((r) => ({
                  id: r.id,
                  content: r.content,
                  score: r.score,
                })),
              },
            };
          }

          return {
            content: [
              { type: "text", text: "Provide a query or memoryId to suppress." },
            ],
            details: { error: "missing_param" },
          };
        },
      },
      { name: "memory_graph_forget" },
    );

    // ========================================================================
    // CLI Commands
    // ========================================================================

    api.registerCli(
      ({ program }) => {
        const mem = program
          .command("graph-memory")
          .description("Dual-agent graph memory commands");

        mem
          .command("stats")
          .description("Show knowledge graph statistics")
          .action(async () => {
            const stats = await db.getStats();
            console.log("Knowledge Graph Statistics:");
            console.log(`  Memories: ${stats.memories} (${stats.raw} raw/unreviewed)`);
            console.log(`  Entities: ${stats.entities}`);
            console.log(`  Sessions: ${stats.sessions}`);
            console.log(`  Categories: ${stats.categories}`);
          });

        mem
          .command("search")
          .description("Search the knowledge graph")
          .argument("<query>", "Search query")
          .option("--limit <n>", "Max results", "8")
          .action(async (query: string, opts: { limit: string }) => {
            const results = await retrieval.retrieveResults(
              query,
              parseInt(opts.limit),
            );
            if (results.length === 0) {
              console.log("No results found.");
              return;
            }
            for (const r of results) {
              const kind = r.normalizedKind ?? r.kind;
              const entities =
                r.entities && r.entities.length > 0
                  ? ` → ${r.entities.join(", ")}`
                  : "";
              console.log(
                `[${kind}] ${r.content} (${Math.round(r.score * 100)}%)${entities}`,
              );
            }
          });
      },
      { commands: ["graph-memory"] },
    );

    // ========================================================================
    // Service (lifecycle)
    // ========================================================================

    api.registerService({
      id: "dual-memory",
      start: async () => {
        // Lazily initialize schema on first use
        await db.ensureSchema();
        api.logger.info(
          `dual-memory: Neo4j connected (${cfg.neo4j.uri}), schema initialized`,
        );
      },
      stop: async () => {
        await db.close();
        api.logger.info("dual-memory: Neo4j connection closed");
      },
    });
  },
};

export default dualMemoryPlugin;
