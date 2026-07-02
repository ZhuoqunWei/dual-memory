// Synthetic Dual-Memory demo graph.
// This is safe to publish and contains no real user data.
//
// Usage:
//   cypher-shell -u neo4j -p dualmemory2026 -f examples/demo-seed.cypher
//
// The demo omits vector embeddings so it is best for schema exploration,
// graph screenshots, and Cypher query examples rather than ANN retrieval.

MATCH (n)
WHERE n.id STARTS WITH "demo-"
DETACH DELETE n;

MATCH (c:Category)
WHERE c.name IN ["Demo Learning Progress", "Demo Career Direction"]
DETACH DELETE c;

CREATE (s1:Session {
  id: "demo-session-1",
  date: datetime("2026-02-01T10:00:00Z"),
  summary: "User discussed learning Rust and planning a first project.",
  messageCount: 8,
  channel: "cli"
});

CREATE (s2:Session {
  id: "demo-session-2",
  date: datetime("2026-02-03T15:30:00Z"),
  summary: "User revisited Rust plans and connected them to career goals.",
  messageCount: 10,
  channel: "cli"
});

CREATE (user:Entity {
  id: "demo-entity-user",
  name: "User",
  type: "Person",
  normalizedType: "Person",
  aliases: ["User"],
  firstSeen: datetime("2026-02-01T10:00:00Z")
});

CREATE (alice:Entity {
  id: "demo-entity-alice",
  name: "Alice",
  type: "Person",
  normalizedType: "Person",
  aliases: ["Alice"],
  firstSeen: datetime("2026-02-01T10:00:00Z")
});

CREATE (rust:Entity {
  id: "demo-entity-rust",
  name: "Rust",
  type: "Tool",
  normalizedType: "Tool",
  aliases: ["Rust", "Rust language"],
  firstSeen: datetime("2026-02-01T10:00:00Z")
});

CREATE (cliProject:Entity {
  id: "demo-entity-cli-project",
  name: "CLI Tool Project",
  type: "Project",
  normalizedType: "Project",
  aliases: ["CLI project"],
  firstSeen: datetime("2026-02-01T10:00:00Z")
});

CREATE (career:Entity {
  id: "demo-entity-career",
  name: "Career Direction",
  type: "Concept",
  normalizedType: "Concept",
  aliases: ["career planning"],
  firstSeen: datetime("2026-02-03T15:30:00Z")
});

CREATE (m1:Memory {
  id: "demo-memory-rust-learning",
  content: "User has been learning Rust and is interested in its safety guarantees.",
  kind: "fact",
  normalizedKind: "fact",
  timestamp: datetime("2026-02-01T10:05:00Z"),
  confidence: 0.85,
  salience: 0.70,
  status: "reviewed",
  sourceChannel: "cli",
  sourceQuote: "I've been learning Rust because I like its safety model."
});

CREATE (m2:Memory {
  id: "demo-memory-alice-recommendation",
  content: "Alice recommended Rust to the user as a practical systems programming language.",
  kind: "fact",
  normalizedKind: "fact",
  timestamp: datetime("2026-02-01T10:08:00Z"),
  confidence: 0.80,
  salience: 0.55,
  status: "reviewed",
  sourceChannel: "cli",
  sourceQuote: "Alice recommended Rust for systems work."
});

CREATE (m3:Memory {
  id: "demo-memory-cli-decision",
  content: "User decided to build a small CLI tool as a first Rust project.",
  kind: "decision",
  normalizedKind: "decision",
  timestamp: datetime("2026-02-01T10:12:00Z"),
  confidence: 0.90,
  salience: 0.85,
  status: "reviewed",
  sourceChannel: "cli",
  sourceQuote: "I decided my first Rust project will be a small CLI tool."
});

CREATE (m4:Memory {
  id: "demo-memory-career-goal",
  content: "User wants the Rust project to support a broader career direction in systems engineering.",
  kind: "goal",
  normalizedKind: "goal",
  timestamp: datetime("2026-02-03T15:36:00Z"),
  confidence: 0.75,
  salience: 0.80,
  status: "reviewed",
  sourceChannel: "cli",
  sourceQuote: "This feels aligned with systems engineering."
});

CREATE (learning:Category {
  name: "Demo Learning Progress",
  description: "Skills, study plans, and progress on technical learning."
});

CREATE (careerCat:Category {
  name: "Demo Career Direction",
  description: "Long-term goals and decisions related to career development."
});

CREATE (m1)-[:PART_OF]->(s1);
CREATE (m2)-[:PART_OF]->(s1);
CREATE (m3)-[:PART_OF]->(s1);
CREATE (m4)-[:PART_OF]->(s2);

CREATE (m1)-[:MENTIONS {role: "subject"}]->(user);
CREATE (m1)-[:MENTIONS {role: "object"}]->(rust);
CREATE (m2)-[:MENTIONS {role: "source"}]->(alice);
CREATE (m2)-[:MENTIONS {role: "object"}]->(rust);
CREATE (m3)-[:MENTIONS {role: "subject"}]->(user);
CREATE (m3)-[:MENTIONS {role: "object"}]->(cliProject);
CREATE (m3)-[:MENTIONS {role: "context"}]->(rust);
CREATE (m4)-[:MENTIONS {role: "subject"}]->(user);
CREATE (m4)-[:MENTIONS {role: "context"}]->(career);
CREATE (m4)-[:MENTIONS {role: "context"}]->(rust);

CREATE (m2)-[:RELATES_TO {type: "supports", weight: 0.70}]->(m1);
CREATE (m3)-[:RELATES_TO {type: "follows", weight: 0.80}]->(m1);
CREATE (m4)-[:RELATES_TO {type: "elaborates", weight: 0.75}]->(m3);

CREATE (m1)-[:IN_CATEGORY {score: 0.90, assignedBy: "editor", timestamp: datetime("2026-02-04T02:00:00Z")}]->(learning);
CREATE (m3)-[:IN_CATEGORY {score: 0.85, assignedBy: "editor", timestamp: datetime("2026-02-04T02:00:00Z")}]->(learning);
CREATE (m4)-[:IN_CATEGORY {score: 0.92, assignedBy: "editor", timestamp: datetime("2026-02-04T02:00:00Z")}]->(careerCat);

CREATE (alice)-[:LINKED_TO {
  relation: "mentor",
  detail: "recommended Rust as a practical systems language",
  sentiment: "positive",
  strength: 0.60,
  active: true
}]->(user);
