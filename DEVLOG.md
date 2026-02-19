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

User's OpenClaw workspace at `~/.openclaw/workspace/`:
- `MEMORY.md` — manually curated, in Chinese, tracks people/servers/chat rules
- `memory/2026-02-05.md`, `memory/2026-02-06.md` — daily logs (only 2 days)
- Agent personality in `SOUL.md` — casual, bilingual (Chinese/English)
- Discord integration active (multiple servers)
- The "lobotomy problem" hasn't hit yet (only 2 days of logs), but the flat Markdown structure will degrade fast with daily use

---
