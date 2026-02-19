/**
 * Plugin configuration types and parsing
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
  autoCapture: boolean;
  autoRecall: boolean;
  extractionModel: string;
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

  // Embedding config
  const embedding = cfg.embedding as Record<string, unknown> | undefined;
  if (!embedding || typeof embedding.apiKey !== "string") {
    throw new Error("embedding.apiKey is required");
  }

  const embeddingModel =
    typeof embedding.model === "string" ? embedding.model : "text-embedding-3-small";
  vectorDimsForModel(embeddingModel); // validate

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
    autoCapture: cfg.autoCapture !== false,
    autoRecall: cfg.autoRecall !== false,
    extractionModel:
      typeof cfg.extractionModel === "string" ? cfg.extractionModel : "gpt-4o-mini",
  };
}
