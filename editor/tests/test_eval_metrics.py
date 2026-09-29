"""Tests for the eval harness scoring functions."""

from __future__ import annotations

import unittest

from editor.evaluation.metrics import (
    Chain,
    ConflictEdge,
    Question,
    conflict_metrics,
    follow,
    pairwise_clustering,
    retrieval_metrics,
)


class PairwiseClusteringTests(unittest.TestCase):
    def test_over_merging_costs_precision_not_recall(self) -> None:
        scores = pairwise_clustering({"a": 1, "b": 1, "c": 2}, {"a": "X", "b": "X", "c": "X"})
        self.assertEqual((scores.true_positives, scores.predicted, scores.gold), (1, 3, 1))
        self.assertAlmostEqual(scores.precision, 1 / 3, places=3)
        self.assertEqual(scores.recall, 1.0)

    def test_items_missing_from_prediction_are_singletons(self) -> None:
        scores = pairwise_clustering({"a": 1, "b": 1}, {"a": "X"})
        self.assertEqual(scores.predicted, 0)
        self.assertEqual(scores.recall, 0.0)


class FollowTests(unittest.TestCase):
    def test_follows_chain_and_survives_cycles(self) -> None:
        self.assertEqual(follow({"a": "b", "b": "c"}, "a"), "c")
        self.assertEqual(follow({}, "a"), "a")
        self.assertIn(follow({"a": "b", "b": "a"}, "a"), {"a", "b"})


class ConflictMetricsTests(unittest.TestCase):
    chains = (Chain(["v1", "v2", "v3"], older=["old-only"], newer=["implies-v3"]),)

    def test_edges_scored_against_chain_positions(self) -> None:
        edges = [
            ConflictEdge("v2", "v1", "supersedes"),        # correct, right direction
            ConflictEdge("v2", "v3", "supersedes"),        # correct, wrong direction
            ConflictEdge("implies-v3", "old-only", "contradicts"),  # correct, detects nothing
            ConflictEdge("v1", "unrelated", "contradicts"),  # wrong
            ConflictEdge("v3", "implies-v3", "contradicts"),  # same position: wrong
            ConflictEdge("x", "y", "contradicts"),         # ignored
        ]
        scores = conflict_metrics(self.chains, edges, {frozenset(("x", "y"))})
        self.assertEqual(scores["predicted_edges"], 5)
        self.assertEqual(scores["correct_edges"], 3)
        self.assertEqual(scores["wrong_edges"], [3, 4])
        self.assertEqual(scores["outdated_detected"], 2)  # v1 and v2
        self.assertEqual(scores["missed"], [])
        self.assertEqual(scores["direction_accuracy"], 0.5)

    def test_a_middle_link_does_not_hide_a_missed_later_step(self) -> None:
        scores = conflict_metrics(self.chains, [ConflictEdge("v2", "v1", "supersedes")], set())
        self.assertEqual(scores["missed"], ["v2"])
        self.assertEqual(scores["recall"], 0.5)

    def test_supersession_state(self) -> None:
        superseded = {"v1": (1, 1), "v2": (2, 1), "v3": (1, 1), "stray": (1, 1)}
        state = conflict_metrics(self.chains, [], set(), superseded)["state"]
        self.assertEqual(state["outdated_superseded"], 0.5)  # v2 only half superseded
        self.assertEqual(state["current_wrongly_superseded"], 1.0)
        self.assertEqual(state["superseded_outside_chains"], 1)


class RetrievalMetricsTests(unittest.TestCase):
    def test_recall_mrr_and_stale_first(self) -> None:
        questions = [
            Question("q1", "current", [["new"]], ["old"]),
            Question("q2", "multi", [["a"], ["b"]], []),
        ]
        ranked = {"q1": ["old", "x", "new"], "q2": ["b", "z"]}
        scores = retrieval_metrics(questions, ranked, k=5)
        self.assertEqual(scores["recall@5"], 0.75)  # q1: 1.0, q2: 0.5
        self.assertAlmostEqual(scores["mrr"], (1 / 3 + 1) / 2, places=3)
        self.assertEqual(scores["stale_first_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
