/**
 * Retrieval module — two-phase memory retrieval for context injection.
 *
 * Runs before every agent start (before_agent_start hook).
 * Phase 1: Category routing (when categories exist)
 * Phase 2: Full ANN vector search (fallback)
 *
 * Scoring: sim × (0.6 + 0.4 × confidence) × (0.6 + 0.4 × salience)
 */

import type { Neo4jClient, RetrievalResult } from "./neo4j-client.js";
import type { Embeddings } from "./embeddings.js";

// ============================================================================
// Types
// ============================================================================

export type RetrievalConfig = {
  maxResults: number;
  minScore: number;
  includeEntities: boolean;
};

const DEFAULT_CONFIG: RetrievalConfig = {
  maxResults: 8,
  minScore: 0.1,
  includeEntities: true,
};

// ============================================================================
// Retrieval
// ============================================================================

export class Retrieval {
  constructor(
    private db: Neo4jClient,
    private embeddings: Embeddings,
    private config: RetrievalConfig = DEFAULT_CONFIG,
  ) {}

  /**
   * Retrieve relevant memories for a user prompt.
   * Returns formatted context string for injection.
   */
  async retrieveContext(prompt: string): Promise<string | null> {
    if (!prompt || prompt.length < 3) return null;

    try {
      // Embed the query
      const queryVector = await this.embeddings.embed(prompt);

      // Query Neo4j (two-phase handled internally by neo4j-client)
      const results = await this.db.retrieve(
        queryVector,
        this.config.maxResults,
      );

      // Filter by minimum score
      const filtered = results.filter((r) => r.score >= this.config.minScore);

      if (filtered.length === 0) return null;

      // Format for context injection
      return this.formatContext(filtered);
    } catch (err) {
      console.error("[dual-memory] Retrieval failed:", err);
      return null;
    }
  }

  /**
   * Retrieve raw results (for tool use, not context injection).
   */
  async retrieveResults(query: string, limit?: number): Promise<RetrievalResult[]> {
    if (!query || query.length < 3) return [];

    try {
      const queryVector = await this.embeddings.embed(query);
      const results = await this.db.retrieve(
        queryVector,
        limit ?? this.config.maxResults,
      );
      return results.filter((r) => r.score >= this.config.minScore);
    } catch (err) {
      console.error("[dual-memory] Retrieval failed:", err);
      return [];
    }
  }

  /**
   * Format retrieval results as context for prepending to agent prompt.
   */
  private formatContext(results: RetrievalResult[]): string {
    const lines = results.map((r, i) => {
      const kind = r.normalizedKind ?? r.kind;
      const entities =
        this.config.includeEntities && r.entities && r.entities.length > 0
          ? ` (${r.entities.join(", ")})`
          : "";
      const score = Math.round(r.score * 100);
      return `${i + 1}. [${kind}]${entities} ${r.content} (${score}%)`;
    });

    // Collect unique entity relations across all results
    const allRelations = new Set<string>();
    for (const r of results) {
      if (r.entityRelations) {
        for (const rel of r.entityRelations) {
          allRelations.add(rel);
        }
      }
    }

    const sections = [
      "<graph-memories>",
      "Relevant memories from your knowledge graph:",
      ...lines,
    ];

    if (allRelations.size > 0) {
      sections.push("");
      sections.push("Entity relationships:");
      for (const rel of allRelations) {
        sections.push(`- ${rel}`);
      }
    }

    sections.push("</graph-memories>");
    return sections.join("\n");
  }
}
