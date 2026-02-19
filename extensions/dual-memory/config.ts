/**
 * Plugin configuration types and parsing.
 *
 * Supports two LLM providers for fact extraction:
 *   - "anthropic" (default): uses Claude via Anthropic API
 *   - "openai": uses GPT models via OpenAI API
 *
 * Embeddings always use OpenAI (text-embedding-3-small) since
 * Anthropic doesn't have an embeddings API. Voyage AI support
 * can be added later.
 */

export type DualMemoryConfig = {
  neo4j: {
    uri: string;
    user: string;
    password: string;
  };
  embedding: {
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
  "text-embedding-3-small": 1536,
  "text-embedding-3-large": 3072,
};

export function vectorDimsForModel(model: string): number {
  const dims = EMBEDDING_DIMENSIONS[model];
  if (!dims) {
    throw new Error(`Unsupported embedding model: ${model}`);
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

  // Embedding config (OpenAI for now — no Claude embeddings API)
  const embedding = cfg.embedding as Record<string, unknown> | undefined;
  if (!embedding || typeof embedding.apiKey !== "string") {
    throw new Error("embedding.apiKey is required");
  }

  const embeddingModel =
    typeof embedding.model === "string" ? embedding.model : "text-embedding-3-small";
  vectorDimsForModel(embeddingModel); // validate

  // Extraction config (defaults to Anthropic)
  const extraction = cfg.extraction as Record<string, unknown> | undefined;
  const extractionProvider =
    (extraction?.provider as string) === "openai" ? "openai" as const : "anthropic" as const;

  // For extraction API key: check extraction.apiKey, fall back to ANTHROPIC_API_KEY env var
  let extractionApiKey: string;
  if (extraction?.apiKey && typeof extraction.apiKey === "string") {
    extractionApiKey = resolveEnvVars(extraction.apiKey);
  } else if (process.env.ANTHROPIC_API_KEY) {
    extractionApiKey = process.env.ANTHROPIC_API_KEY;
  } else {
    throw new Error(
      "extraction.apiKey is required (or set ANTHROPIC_API_KEY env var)",
    );
  }

  const extractionModel =
    typeof extraction?.model === "string"
      ? extraction.model
      : extractionProvider === "anthropic"
        ? "claude-sonnet-4-20250514"
        : "gpt-4o-mini";

  return {
    neo4j: {
      uri: typeof neo4j.uri === "string" ? neo4j.uri : "bolt://localhost:7687",
      user: typeof neo4j.user === "string" ? neo4j.user : "neo4j",
      password: resolveEnvVars(neo4j.password),
    },
    embedding: {
      apiKey: resolveEnvVars(embedding.apiKey),
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
