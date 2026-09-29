/**
 * Plugin configuration types and parsing.
 *
 * Embeddings:
 *   - "voyage" (default): Voyage AI (voyage-4-lite, 1024d)
 *   - "openai": OpenAI (text-embedding-3-small, 1536d)
 *
 * Extraction (fact extraction LLM):
 *   - "anthropic" (default): Claude via Anthropic API
 *   - "openai": GPT models via OpenAI API
 */

import type { EmbeddingProvider } from "./embeddings.js";
import { DEFAULT_ANTHROPIC_MODEL } from "./writer.js";

export type DualMemoryConfig = {
  neo4j: {
    uri: string;
    user: string;
    password: string;
  };
  embedding: {
    provider: EmbeddingProvider;
    apiKey: string;
    model: string;
  };
  extraction: {
    provider: "anthropic" | "openai";
    apiKey: string;
    model: string;
  };
  autoCapture: boolean;
  autoRecall: boolean;
};

const EMBEDDING_DIMENSIONS: Record<string, number> = {
  // Voyage AI models (voyage-4 series default to 1024)
  "voyage-4-lite": 1024,
  "voyage-4": 1024,
  "voyage-4-large": 1024,
  "voyage-code-3": 1024,
  "voyage-3-lite": 512,
  "voyage-3": 1024,
  // OpenAI models
  "text-embedding-3-small": 1536,
  "text-embedding-3-large": 3072,
};

export function vectorDimsForModel(model: string): number {
  const dims = EMBEDDING_DIMENSIONS[model];
  if (!dims) {
    throw new Error(
      `Unsupported embedding model: ${model}. Supported: ${Object.keys(EMBEDDING_DIMENSIONS).join(", ")}`,
    );
  }
  return dims;
}

function resolveEnvVars(value: string): string {
  return value.replace(/\$\{([^}]+)\}/g, (_, envVar) => {
    const envValue = process.env[envVar];
    if (!envValue) {
      throw new Error(`Environment variable ${envVar} is not set`);
    }
    return envValue;
  });
}

function resolveApiKey(
  explicit: string | undefined,
  envVarName: string,
  errorMsg: string,
): string {
  if (explicit) return resolveEnvVars(explicit);
  if (process.env[envVarName]) return process.env[envVarName]!;
  throw new Error(errorMsg);
}

export function parseConfig(value: unknown): DualMemoryConfig {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error("dual-memory config required");
  }

  const cfg = value as Record<string, unknown>;

  // Neo4j config
  const neo4j = cfg.neo4j as Record<string, unknown> | undefined;
  if (!neo4j || typeof neo4j.password !== "string") {
    throw new Error("neo4j.password is required");
  }

  // Embedding config (defaults to Voyage AI)
  const embedding = cfg.embedding as Record<string, unknown> | undefined;
  const embeddingProvider: EmbeddingProvider =
    (embedding?.provider as string) === "openai" ? "openai" : "voyage";

  const defaultEmbeddingModel =
    embeddingProvider === "voyage" ? "voyage-4-lite" : "text-embedding-3-small";
  const embeddingModel =
    typeof embedding?.model === "string" ? embedding.model : defaultEmbeddingModel;
  vectorDimsForModel(embeddingModel); // validate

  const embeddingApiKey = resolveApiKey(
    embedding?.apiKey as string | undefined,
    embeddingProvider === "voyage" ? "VOYAGE_API_KEY" : "OPENAI_API_KEY",
    `embedding.apiKey is required (or set ${embeddingProvider === "voyage" ? "VOYAGE_API_KEY" : "OPENAI_API_KEY"})`,
  );

  // Extraction config (defaults to Anthropic)
  const extraction = cfg.extraction as Record<string, unknown> | undefined;
  const extractionProvider =
    (extraction?.provider as string) === "openai"
      ? ("openai" as const)
      : ("anthropic" as const);

  const extractionApiKey = resolveApiKey(
    extraction?.apiKey as string | undefined,
    extractionProvider === "anthropic" ? "ANTHROPIC_API_KEY" : "OPENAI_API_KEY",
    `extraction.apiKey is required (or set ${extractionProvider === "anthropic" ? "ANTHROPIC_API_KEY" : "OPENAI_API_KEY"})`,
  );

  const extractionModel =
    typeof extraction?.model === "string"
      ? extraction.model
      : extractionProvider === "anthropic"
        ? DEFAULT_ANTHROPIC_MODEL
        : "gpt-4o-mini";

  return {
    neo4j: {
      uri: typeof neo4j.uri === "string" ? neo4j.uri : "bolt://localhost:7687",
      user: typeof neo4j.user === "string" ? neo4j.user : "neo4j",
      password: resolveEnvVars(neo4j.password),
    },
    embedding: {
      provider: embeddingProvider,
      apiKey: embeddingApiKey,
      model: embeddingModel,
    },
    extraction: {
      provider: extractionProvider,
      apiKey: extractionApiKey,
      model: extractionModel,
    },
    autoCapture: cfg.autoCapture !== false,
    autoRecall: cfg.autoRecall !== false,
  };
}
