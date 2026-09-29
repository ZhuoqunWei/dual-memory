"""Step 3: Category creation and IN_CATEGORY assignment.

Bottom-up: cluster similar memories → LLM names the cluster → create Category node.
Top-down: assign uncategorized memories to closest existing category.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ..constants import CATEGORY_ASSIGN_MIN_COSINE, CATEGORY_COSINE, CATEGORY_QUORUM

if TYPE_CHECKING:
    from ..audit import AuditLog
    from ..db import EditorDB
    from ..embeddings import EmbeddingsClient
    from ..llm import LLMClient

log = logging.getLogger("editor.steps.categories")


@dataclass
class CategoryResult:
    categories_created: int = 0
    memories_assigned: int = 0
    llm_calls: int = 0


def run_categories(
    batch: list[dict],
    db: EditorDB,
    audit: AuditLog,
    llm: LLMClient,
    embeddings: EmbeddingsClient,
) -> CategoryResult:
    result = CategoryResult()

    # --- Phase A: Assign to existing categories ---
    existing_cats = db.fetch_categories()
    assigned_ids: set[str] = set()

    if existing_cats:
        for cat in existing_cats:
            # Compute category centroid from member embeddings
            member_embs = [e for e in cat["memberEmbeddings"] if e is not None]
            if not member_embs:
                continue
            centroid = np.mean(member_embs, axis=0)
            centroid = centroid / (np.linalg.norm(centroid) + 1e-8)

            for mem in batch:
                if mem["id"] in assigned_ids:
                    continue
                emb = mem.get("embedding")
                if emb is None:
                    continue

                emb_normed = np.array(emb) / (np.linalg.norm(emb) + 1e-8)
                cosine = float(np.dot(centroid, emb_normed))

                if cosine >= CATEGORY_ASSIGN_MIN_COSINE:
                    db.assign_category(mem["id"], cat["name"], cosine)
                    assigned_ids.add(mem["id"])
                    result.memories_assigned += 1

    # --- Phase B: Discover new categories from unassigned memories ---
    unassigned = [m for m in batch if m["id"] not in assigned_ids and m.get("embedding")]
    if len(unassigned) < CATEGORY_QUORUM:
        if result.memories_assigned:
            audit.log("reclassify", list(assigned_ids),
                       f"Assigned {result.memories_assigned} memories to existing categories")
        log.info("Categories: %d assigned, %d created", result.memories_assigned, result.categories_created)
        return result

    # Build pairwise cosine matrix for unassigned memories
    emb_matrix = np.array([m["embedding"] for m in unassigned], dtype=np.float32)
    norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    normed = emb_matrix / norms
    sim = normed @ normed.T

    # Find clusters: greedy complete-linkage-ish approach
    clusters = _find_clusters(sim, len(unassigned), CATEGORY_COSINE, CATEGORY_QUORUM)

    for cluster_indices in clusters:
        cluster_memories = [unassigned[i] for i in cluster_indices]

        # Ask LLM to name this category
        cat_info = _llm_name_category(llm, cluster_memories)
        result.llm_calls += 1

        if not cat_info:
            continue

        cat_name = cat_info["name"]
        cat_desc = cat_info.get("description", "")

        # Check if category already exists (by name)
        existing_names = {c["name"] for c in existing_cats}
        if cat_name in existing_names:
            # Assign to existing
            for mem in cluster_memories:
                db.assign_category(mem["id"], cat_name, 0.85)
                assigned_ids.add(mem["id"])
                result.memories_assigned += 1
            continue

        # Create new category
        db.create_category(cat_name, cat_desc)
        result.categories_created += 1

        # Assign cluster members
        for mem in cluster_memories:
            db.assign_category(mem["id"], cat_name, 0.85)
            assigned_ids.add(mem["id"])
            result.memories_assigned += 1

        audit.log(
            "reclassify",
            [m["id"] for m in cluster_memories],
            f"Created category '{cat_name}' with {len(cluster_memories)} members",
        )

    log.info("Categories: %d assigned, %d created, %d LLM calls",
             result.memories_assigned, result.categories_created, result.llm_calls)
    return result


def _find_clusters(
    sim: np.ndarray, n: int, threshold: float, min_size: int,
) -> list[list[int]]:
    """Greedy clustering: find groups where all pairwise cosine >= threshold."""
    used: set[int] = set()
    clusters: list[list[int]] = []

    # Sort potential seeds by average similarity (most connected first)
    avg_sim = np.mean(sim, axis=1)
    seed_order = np.argsort(-avg_sim, kind="stable")

    for seed in seed_order:
        if int(seed) in used:
            continue

        # Start cluster with seed
        cluster = [int(seed)]

        # Try adding each unused node
        for candidate in seed_order:
            candidate = int(candidate)
            if candidate in used or candidate == int(seed):
                continue

            # Check if candidate is similar enough to ALL current cluster members
            if all(sim[candidate, member] >= threshold for member in cluster):
                cluster.append(candidate)

        if len(cluster) >= min_size:
            clusters.append(cluster)
            used.update(cluster)

    return clusters


def _llm_name_category(llm: LLMClient, memories: list[dict]) -> dict | None:
    """Ask LLM to name a cluster of memories."""
    items = "\n".join(f"- {m['content'][:120]}" for m in memories)

    system = """Given a cluster of related memories, suggest a category name and description.

Requirements:
- name: 2-4 words, title case (e.g., "Learning Progress", "Career Decisions", "Social Relationships")
- description: 1 sentence explaining what memories belong here

Return JSON: {"name": "...", "description": "..."}"""

    user = f"Name this cluster of {len(memories)} related memories:\n{items}"

    try:
        result = llm.ask_json(system, user)
        if isinstance(result, dict) and "name" in result:
            return result
    except Exception as e:
        log.warning("LLM category naming failed: %s", e)

    return None
