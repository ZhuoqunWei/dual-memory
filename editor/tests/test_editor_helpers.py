"""Fast tests for pure Editor helper logic."""

from __future__ import annotations

import unittest

import numpy as np

from editor.steps.classify import _resolve_entity_type, _resolve_kind
from editor.steps.dedup import _find_pairs_above_threshold


class ClassifyHelperTests(unittest.TestCase):
    def test_resolve_kind_accepts_canonical_and_synonyms(self) -> None:
        self.assertEqual(_resolve_kind("decision"), "decision")
        self.assertEqual(_resolve_kind("plan"), "goal")
        self.assertEqual(_resolve_kind("mood"), "emotion")
        self.assertIsNone(_resolve_kind("unknown-kind"))

    def test_resolve_entity_type_accepts_canonical_and_synonyms(self) -> None:
        self.assertEqual(_resolve_entity_type("Person"), "Person")
        self.assertEqual(_resolve_entity_type("repo"), "Project")
        self.assertEqual(_resolve_entity_type("framework"), "Tool")
        self.assertIsNone(_resolve_entity_type("mystery"))


class DedupHelperTests(unittest.TestCase):
    def test_find_pairs_above_threshold_excludes_self_and_respects_upper_bound(self) -> None:
        sim_matrix = np.array([[1.0, 0.99, 0.90, 0.75]], dtype=np.float32)
        pairs = _find_pairs_above_threshold(
            sim_matrix=sim_matrix,
            batch_indices=[0],
            all_ids=["batch", "exact", "semantic", "low"],
            threshold=0.80,
            exclude=set(),
            upper_bound=0.98,
        )

        self.assertEqual(pairs, [(0, 2, float(sim_matrix[0, 2]))])

    def test_find_pairs_above_threshold_skips_excluded_ids(self) -> None:
        sim_matrix = np.array([[1.0, 0.95]], dtype=np.float32)
        pairs = _find_pairs_above_threshold(
            sim_matrix=sim_matrix,
            batch_indices=[0],
            all_ids=["batch", "excluded"],
            threshold=0.80,
            exclude={"excluded"},
        )

        self.assertEqual(pairs, [])


if __name__ == "__main__":
    unittest.main()
