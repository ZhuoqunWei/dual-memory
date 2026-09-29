"""Tests for supersession candidate selection and ordering."""

from __future__ import annotations

import unittest

from editor.steps.contradictions import candidate_pairs, order_by_time


def memory(mid: str, embedding: list[float], status: str = "reviewed", entities=("user",), day: str = "2026-01-01") -> dict:
    return {"id": mid, "content": mid, "status": status, "embedding": embedding,
            "entityIds": list(entities), "validFrom": day}


class CandidatePairsTests(unittest.TestCase):
    def test_filters_neighbours(self) -> None:
        new = memory("new", [1.0, 0.0], status="raw")
        memories = [
            new,
            memory("similar", [0.9, 0.1]),
            memory("raw-elsewhere", [0.9, 0.1], status="raw"),  # not in this batch
            memory("other-entity", [0.9, 0.1], entities=("someone-else",)),
            memory("dissimilar", [0.0, 1.0]),
            memory("linked", [0.9, 0.1]),
        ]
        pairs = candidate_pairs([new], memories, {frozenset(("new", "linked"))})
        self.assertEqual([(a["id"], b["id"]) for a, b, _ in pairs], [("new", "similar")])

    def test_pair_within_batch_is_asked_once(self) -> None:
        a = memory("a", [1.0, 0.0], status="raw")
        b = memory("b", [0.9, 0.1], status="raw")
        self.assertEqual(len(candidate_pairs([a, b], [a, b], set())), 1)


class OrderByTimeTests(unittest.TestCase):
    def test_newer_wins_regardless_of_llm_pick(self) -> None:
        old = memory("old", [1.0], day="2026-01-01")
        new = memory("new", [1.0], day="2026-06-01")
        self.assertEqual(order_by_time(old, new, "A"), (new, old))

    def test_same_day_uses_llm_pick_or_gives_up(self) -> None:
        a = memory("a", [1.0])
        b = memory("b", [1.0])
        self.assertEqual(order_by_time(a, b, "B"), (b, a))
        self.assertIsNone(order_by_time(a, b, None))


if __name__ == "__main__":
    unittest.main()
