"""Canonical sets, thresholds, and magic numbers for the Editor."""

# LLM defaults. claude-sonnet-5 replaces the retired claude-sonnet-4-20250514.
# Effort chosen with the eval harness: medium over low raised dedup precision
# 81% -> 90%, supersession precision 77% -> 89%, and retrieval recall@5
# 80% -> 87%, for $0.32 -> $0.41 per 102-fact run (eval/results/).
DEFAULT_LLM_MODEL = "claude-sonnet-5"
DEFAULT_LLM_EFFORT = "medium"

# Anthropic first-party list prices, USD per 1M tokens (input, output), 2026-06.
LLM_PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-sonnet-5": (2.00, 10.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-opus-5": (5.00, 25.00),
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-4-6": (3.00, 15.00),
}

# Canonical classification sets (from design doc v2.1)
CANONICAL_KINDS: set[str] = {
    "fact", "decision", "preference", "goal", "emotion", "observation", "event",
}

CANONICAL_ENTITY_TYPES: set[str] = {
    "Person", "Place", "Project", "Organization", "Tool", "Concept",
}

# Known synonyms for rule-based classification (no LLM needed)
KIND_SYNONYMS: dict[str, str] = {
    "feelings": "emotion",
    "feeling": "emotion",
    "mood": "emotion",
    "choice": "decision",
    "plan": "goal",
    "plans": "goal",
    "target": "goal",
    "objective": "goal",
    "opinion": "preference",
    "like": "preference",
    "dislike": "preference",
    "belief": "fact",
    "info": "fact",
    "information": "fact",
    "notice": "observation",
    "note": "observation",
    "happening": "event",
    "incident": "event",
    "occurrence": "event",
}

ENTITY_TYPE_SYNONYMS: dict[str, str] = {
    "person": "Person",
    "people": "Person",
    "user": "Person",
    "place": "Place",
    "location": "Place",
    "city": "Place",
    "country": "Place",
    "project": "Project",
    "repo": "Project",
    "repository": "Project",
    "org": "Organization",
    "organization": "Organization",
    "company": "Organization",
    "tool": "Tool",
    "library": "Tool",
    "framework": "Tool",
    "language": "Tool",
    "software": "Tool",
    "concept": "Concept",
    "topic": "Concept",
    "idea": "Concept",
    "skill": "Concept",
}

# Dedup thresholds
EXACT_DEDUP_COSINE = 0.98
SEMANTIC_DEDUP_COSINE = 0.80

# Category thresholds
CATEGORY_COSINE = 0.85
CATEGORY_QUORUM = 3
CATEGORY_ASSIGN_MIN_COSINE = 0.70
CATEGORY_MERGE_OVERLAP = 0.60
CATEGORY_SUBCATEGORY_MIN_MEMBERS = 10

# Confidence
CONFIDENCE_DECAY = 0.05
CONFIDENCE_FLOOR = 0.1
CONFIDENCE_MULTI_SESSION_BOOST = 0.1

# Supersession: conflict candidates are each batch memory's nearest still-valid
# memories sharing an entity. On the eval fixture every true update pair is
# within 4 neighbours at cosine >= 0.62; these leave headroom for bigger graphs.
CONFLICT_CANDIDATE_COSINE = 0.55
CONFLICT_NEIGHBORS = 8
CONFLICT_MAX_PAIRS = 200

# Entity resolution, tier 2 (name + type + aliases embeddings). On the eval
# fixture true alias pairs scored 0.84-1.00 and the closest distinct pairs 0.91
# (Tokyo/Kyoto, "Chris Lee"/"Chris Park"), so only near-identical names merge
# without the LLM. Chosen by looking at that fixture; there is no held-out set.
ENTITY_AUTO_MERGE_COSINE = 0.95
ENTITY_CANDIDATE_COSINE = 0.80
ENTITY_MAX_LLM_PAIRS = 60
LLM_ENTITY_BATCH = 10

# Relationships: skip hub entities (same cut-off as retrieval's graph
# expansion) and cap LLM pairs per run.
RELATIONSHIP_HUB_MENTIONS = 30
RELATIONSHIP_MAX_PAIRS = 120

# Batch sizes
BATCH_CAP = 50
LLM_DEDUP_BATCH = 12
LLM_CLASSIFY_BATCH = 20
LLM_CONFLICT_BATCH = 10
