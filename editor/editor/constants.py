"""Canonical sets, thresholds, and magic numbers for the Editor."""

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
CONFIDENCE_CONTRADICTION_PENALTY = 0.15

# Batch sizes
BATCH_CAP = 50
LLM_DEDUP_BATCH = 12
LLM_CLASSIFY_BATCH = 20
LLM_CONTRADICTION_BATCH = 15
