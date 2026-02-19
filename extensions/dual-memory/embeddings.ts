/**
 * Embeddings wrapper — supports Voyage AI and OpenAI.
 *
 * Default: Voyage AI (voyage-4-lite, 1024d) — Anthropic's recommended partner.
 * Fallback: OpenAI (text-embedding-3-small, 1536d).
 */

import OpenAI from "openai";
import { VoyageAIClient } from "voyageai";

export type EmbeddingProvider = "voyage" | "openai";

export class Embeddings {
  private provider: EmbeddingProvider;
  private openai?: OpenAI;
  private voyage?: VoyageAIClient;
  private model: string;

  constructor(
    provider: EmbeddingProvider,
    apiKey: string,
    model: string,
  ) {
    this.provider = provider;
    this.model = model;

    if (provider === "voyage") {
      this.voyage = new VoyageAIClient({ apiKey });
    } else {
      this.openai = new OpenAI({ apiKey });
    }
  }

  async embed(text: string): Promise<number[]> {
    if (this.provider === "voyage" && this.voyage) {
      const response = await this.voyage.embed({
        input: text,
        model: this.model,
        inputType: "document",
      });
      return response.data![0].embedding!;
    }

    // OpenAI path
    const response = await this.openai!.embeddings.create({
      model: this.model,
      input: text,
    });
    return response.data[0].embedding;
  }

  async embedBatch(texts: string[]): Promise<number[][]> {
    if (texts.length === 0) return [];

    if (this.provider === "voyage" && this.voyage) {
      // Voyage supports batch embedding natively
      const response = await this.voyage.embed({
        input: texts,
        model: this.model,
        inputType: "document",
      });
      return response.data!.map((d) => d.embedding!);
    }

    // OpenAI path
    const response = await this.openai!.embeddings.create({
      model: this.model,
      input: texts,
    });
    return response.data
      .sort((a, b) => a.index - b.index)
      .map((d) => d.embedding);
  }
}
