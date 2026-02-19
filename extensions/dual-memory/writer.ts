/**
 * Writer agent — extracts structured facts from session messages using LLM.
 *
 * Runs after every conversation session (agent_end hook).
 * Follows Writer validation rules:
 *   - Sets status: 'raw', confidence: 0.5, salience: 0.5
 *   - Never creates editor:* RELATES_TO types or CANONICAL edges
 *   - Should create at least one MENTIONS {role: 'subject'} per Memory
 */

import OpenAI from "openai";
import Anthropic from "@anthropic-ai/sdk";
import type { ExtractedFact, EntityLink } from "./neo4j-client.js";

// ============================================================================
// Types
// ============================================================================

type Message = {
  role: string;
  content: string | Array<{ type: string; text?: string }>;
};

type LLMExtractionResult = {
  facts: Array<{
    content: string;
    kind: string;
    confidence: number;
    salience: number;
    eventTimeStart?: string;
    eventTimeEnd?: string;
    sourceQuote?: string;
    entities: Array<{
      name: string;
      type: string;
      aliases?: string[];
      role: "subject" | "object" | "context" | "source";
    }>;
    relations?: Array<{
      targetContent: string;
      type: string;
      weight: number;
    }>;
  }>;
  entityLinks?: Array<{
    source: string;
    target: string;
    relation: string;
    detail?: string;
    sentiment?: "positive" | "negative" | "neutral" | "mixed";
    strength?: number;
  }>;
  sessionSummary: string;
};

// ============================================================================
// Extraction prompt
// ============================================================================

const EXTRACTION_SYSTEM_PROMPT = `You are a fact extraction agent for a knowledge graph memory system. Your job is to extract structured facts from conversation transcripts.

## Output Format
Return a JSON object with:
- "facts": array of extracted facts
- "entityLinks": array of relationships between entities (optional, only when relationships are mentioned)
- "sessionSummary": one sentence summary of the conversation

## Fact Schema
Each fact has:
- "content": concise statement of the fact (1-2 sentences)
- "kind": one of: fact | decision | preference | goal | emotion | observation | event
- "confidence": 0.0-1.0 how certain this fact is (default 0.5)
- "salience": 0.0-1.0 how important/relevant this is (default 0.5)
- "eventTimeStart": ISO 8601 if the fact refers to a specific time (null if unknown)
- "eventTimeEnd": ISO 8601 if the fact spans a time range (null if point-in-time)
- "sourceQuote": short quote from the original text (max 100 chars)
- "entities": array of entities mentioned (MUST have at least one with role "subject" unless truly impossible)
- "relations": optional array of relations to other facts in this batch

## Entity Schema
Each entity has:
- "name": primary display name
- "type": Person | Place | Project | Organization | Tool | Concept
- "aliases": array of alternative names (include multilingual variants)
- "role": "subject" | "object" | "context" | "source"

## Relation Schema (between facts in same batch)
- "targetContent": content string of the target fact (must match another fact in the array)
- "type": "caused_by" | "follows" | "contradicts" | "supports" | "elaborates"
  NEVER use types starting with "editor:" — those are reserved for the Editor agent.
- "weight": 0.0-1.0 strength of connection

## Entity Link Schema (relationships between entities)
When the conversation reveals relationships between entities (e.g., "Bob is my friend", "三哥在深圳"), include them:
- "source": name of the first entity (must match an entity name in facts)
- "target": name of the second entity (must match an entity name in facts)
- "relation": relationship type (e.g., "friend", "sibling", "colleague", "teacher", "student", "creator", "uses", "member_of")
- "detail": short description (e.g., "calls him 三哥", "recommended Rust")
- "sentiment": "positive" | "negative" | "neutral" | "mixed"
- "strength": 0.0-1.0 how strong/close the relationship is
Do NOT write "active" field — it defaults to true.

## Rules
1. Extract ONLY facts worth remembering long-term. Skip trivial, ephemeral, or obvious information.
2. Every fact SHOULD have at least one entity with role "subject".
3. Confidence: high (0.8+) for stated facts, medium (0.5) for inferred, low (0.3) for uncertain.
4. Salience: high (0.8+) for decisions/goals/key facts, medium (0.5) for context, low (0.3) for trivial.
5. SKIP these — do NOT extract:
   - Greetings, small talk, thanks, or pleasantries
   - Meta-conversation about the AI ("can you help me with...", "let me think...")
   - Mechanical steps ("ran npm install", "opened the file") unless they reveal a decision or preference
   - Facts obvious from entities alone ("User uses Python" when entire conversation is about Python)
   - Repeated/restated versions of the same fact — pick the most complete phrasing once
   - Observations with salience below 0.35
6. Aim for 3-6 facts per conversation. Short conversations might yield 0-2. More than 8 is almost always too many.
7. Do NOT create "editor:" prefixed relation types — only fact-layer types.
8. Bilingual support: keep names in their original language + add transliterations as aliases.
9. If no meaningful facts, return empty facts array. This is perfectly fine.

## Example
Input: "I decided to learn Rust because it's safer than C++. My friend Bob recommended it."
Output:
{
  "facts": [
    {
      "content": "User decided to learn Rust, motivated by safety advantages over C++",
      "kind": "decision",
      "confidence": 0.9,
      "salience": 0.8,
      "sourceQuote": "I decided to learn Rust because it's safer than C++",
      "entities": [
        {"name": "User", "type": "Person", "aliases": [], "role": "subject"},
        {"name": "Rust", "type": "Tool", "aliases": ["Rust language"], "role": "object"},
        {"name": "C++", "type": "Tool", "aliases": ["cpp"], "role": "context"}
      ]
    },
    {
      "content": "Bob recommended Rust to user",
      "kind": "fact",
      "confidence": 0.8,
      "salience": 0.5,
      "sourceQuote": "My friend Bob recommended it",
      "entities": [
        {"name": "Bob", "type": "Person", "aliases": [], "role": "source"},
        {"name": "Rust", "type": "Tool", "aliases": [], "role": "object"}
      ],
      "relations": [
        {"targetContent": "User decided to learn Rust, motivated by safety advantages over C++", "type": "supports", "weight": 0.7}
      ]
    }
  ],
  "entityLinks": [
    {"source": "Bob", "target": "User", "relation": "friend", "detail": "recommended Rust to user", "sentiment": "positive", "strength": 0.6}
  ],
  "sessionSummary": "User discussed decision to learn Rust based on safety and friend's recommendation."
}

## Counter-Example (do NOT extract these)
Input: "Hey Claude, can you help me? I need to fix a bug. Let me paste the error... ok I ran npm install and it worked. Thanks!"
Output:
{
  "facts": [],
  "entityLinks": [],
  "sessionSummary": "User fixed a bug by running npm install."
}`;

// ============================================================================
// Writer
// ============================================================================

export class Writer {
  private provider: "anthropic" | "openai";
  private openai?: OpenAI;
  private anthropic?: Anthropic;
  private model: string;

  constructor(
    provider: "anthropic" | "openai",
    apiKey: string,
    model?: string,
  ) {
    this.provider = provider;
    this.model = model ?? (provider === "anthropic" ? "claude-sonnet-4-20250514" : "gpt-4o-mini");

    if (provider === "anthropic") {
      this.anthropic = new Anthropic({ apiKey });
    } else {
      this.openai = new OpenAI({ apiKey });
    }
  }

  /**
   * Extract facts from session messages.
   */
  async extractFacts(
    messages: unknown[],
    channel: string = "cli",
  ): Promise<{ facts: ExtractedFact[]; entityLinks: EntityLink[]; sessionSummary: string }> {
    // 1. Extract text from messages
    const transcript = this.buildTranscript(messages);
    if (!transcript || transcript.length < 20) {
      return { facts: [], entityLinks: [], sessionSummary: "" };
    }

    // 2. Call LLM for extraction
    const extraction = await this.callLLM(transcript);
    if (!extraction || extraction.facts.length === 0) {
      return { facts: [], entityLinks: [], sessionSummary: extraction?.sessionSummary ?? "" };
    }

    // 3. Convert to ExtractedFact format with Writer defaults
    const facts: ExtractedFact[] = extraction.facts.map((f) => ({
      content: f.content,
      kind: f.kind,
      confidence: Math.min(1, Math.max(0, f.confidence ?? 0.5)),
      salience: Math.min(1, Math.max(0, f.salience ?? 0.5)),
      eventTimeStart: f.eventTimeStart ?? undefined,
      eventTimeEnd: f.eventTimeEnd ?? undefined,
      sourceQuote: f.sourceQuote?.slice(0, 100),
      sourceChannel: channel,
      entities: (f.entities ?? []).map((e) => ({
        name: e.name,
        type: e.type || "Concept",
        aliases: e.aliases,
        role: this.validateRole(e.role),
      })),
      relatesTo: (f.relations ?? [])
        .filter((r) => !r.type.startsWith("editor:")) // Enforce: no editor: types
        .map((r) => ({
          targetContent: r.targetContent,
          type: r.type,
          weight: Math.min(1, Math.max(0, r.weight ?? 0.5)),
        })),
    }));

    // 4. Post-extraction filter: drop low-salience facts
    const SALIENCE_FLOOR = 0.35;
    const filteredFacts = facts.filter((f) => f.salience >= SALIENCE_FLOOR);
    if (filteredFacts.length < facts.length) {
      const dropped = facts.length - filteredFacts.length;
      console.log(`[dual-memory] Dropped ${dropped} low-salience facts (< ${SALIENCE_FLOOR})`);
    }

    // Convert entityLinks
    const entityLinks: EntityLink[] = (extraction.entityLinks ?? []).map((l) => ({
      source: l.source,
      target: l.target,
      relation: l.relation,
      detail: l.detail ?? "",
      sentiment: l.sentiment ?? "neutral",
      strength: Math.min(1, Math.max(0, l.strength ?? 0.5)),
    }));

    return { facts: filteredFacts, entityLinks, sessionSummary: extraction.sessionSummary };
  }

  /**
   * Build a readable transcript from raw messages.
   */
  private buildTranscript(messages: unknown[]): string {
    const lines: string[] = [];

    for (const msg of messages) {
      if (!msg || typeof msg !== "object") continue;
      const m = msg as Record<string, unknown>;

      const role = m.role as string;
      if (role !== "user" && role !== "assistant") continue;

      const content = m.content;
      let text = "";

      if (typeof content === "string") {
        text = content;
      } else if (Array.isArray(content)) {
        for (const block of content) {
          if (
            block &&
            typeof block === "object" &&
            (block as Record<string, unknown>).type === "text" &&
            typeof (block as Record<string, unknown>).text === "string"
          ) {
            text += (block as Record<string, unknown>).text as string;
          }
        }
      }

      if (!text) continue;

      // Skip injected memory context
      if (text.includes("<graph-memories>")) continue;
      if (text.includes("<relevant-memories>")) continue;

      // Truncate very long messages (keep first 2000 chars)
      if (text.length > 2000) {
        text = text.slice(0, 2000) + "... [truncated]";
      }

      lines.push(`[${role}]: ${text}`);
    }

    return lines.join("\n\n");
  }

  /**
   * Call the LLM for fact extraction.
   */
  private async callLLM(transcript: string): Promise<LLMExtractionResult | null> {
    try {
      let content: string | null = null;

      if (this.provider === "anthropic" && this.anthropic) {
        const response = await this.anthropic.messages.create({
          model: this.model,
          max_tokens: 4096,
          temperature: 0.1,
          system: EXTRACTION_SYSTEM_PROMPT + "\n\nIMPORTANT: Respond with ONLY a valid JSON object, no markdown fences.",
          messages: [
            {
              role: "user",
              content: `Extract facts from this conversation:\n\n${transcript}`,
            },
          ],
        });

        const textBlock = response.content.find((b) => b.type === "text");
        content = textBlock && "text" in textBlock ? textBlock.text : null;
      } else if (this.openai) {
        const response = await this.openai.chat.completions.create({
          model: this.model,
          temperature: 0.1,
          response_format: { type: "json_object" },
          messages: [
            { role: "system", content: EXTRACTION_SYSTEM_PROMPT },
            {
              role: "user",
              content: `Extract facts from this conversation:\n\n${transcript}`,
            },
          ],
        });

        content = response.choices[0]?.message?.content ?? null;
      }

      if (!content) return null;

      // Strip markdown fences if present (Anthropic sometimes wraps in ```json)
      const cleaned = content.replace(/^```(?:json)?\n?/m, "").replace(/\n?```$/m, "").trim();
      const parsed = JSON.parse(cleaned) as LLMExtractionResult;

      // Basic validation
      if (!Array.isArray(parsed.facts)) return null;

      return parsed;
    } catch (err) {
      // Log but don't throw — writer failures should be silent
      console.error("[dual-memory] Writer LLM extraction failed:", err);
      return null;
    }
  }

  private validateRole(role: string): "subject" | "object" | "context" | "source" {
    const valid = ["subject", "object", "context", "source"];
    return valid.includes(role) ? (role as any) : "context";
  }
}
