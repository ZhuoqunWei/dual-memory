"""Tests for entity resolution's name tier and union-find."""

from __future__ import annotations

import unittest

from editor.steps.entity_resolution import (
    UnionFind,
    consistent_links,
    name_strings,
    names_match,
    normalize_name,
    types_agree,
)


def entity(name: str, aliases: list[str] | None = None, type_: str = "Person") -> dict:
    return {"name": name, "aliases": aliases or [], "type": type_}


class NameTierTests(unittest.TestCase):
    def test_normalization_ignores_case_punctuation_and_order(self) -> None:
        self.assertEqual(normalize_name("Wang Wei"), normalize_name("wei,  WANG"))
        self.assertNotEqual(normalize_name("Contoso"), normalize_name("Contoso Ltd."))

    def test_mutual_aliases_match_but_one_way_aliases_do_not(self) -> None:
        self.assertTrue(names_match(entity("Northwind", ["Northwind Logistics"]),
                                    entity("Northwind Logistics", ["Northwind"])))
        # "Portland" is an alias of "Portland, Maine", but not the other way round.
        self.assertFalse(names_match(entity("Portland", ["PDX"]), entity("Portland, Maine", ["Portland"])))

    def test_shared_first_name_is_only_a_candidate(self) -> None:
        lee, park = entity("Chris Lee", ["Chris"]), entity("Chris Park", ["Chris"])
        self.assertTrue(name_strings(lee) & name_strings(park))
        self.assertFalse(names_match(lee, park))

    def test_types_must_agree_when_both_known(self) -> None:
        self.assertFalse(types_agree(entity("Jordan", type_="Place"), entity("Jordan Blake")))
        self.assertTrue(types_agree(entity("x", type_=""), entity("y")))


class UnionFindTests(unittest.TestCase):
    def test_groups_are_transitive(self) -> None:
        uf = UnionFind(4)
        uf.union(0, 1)
        uf.union(1, 2)
        self.assertTrue(uf.same(0, 2))
        self.assertEqual(sorted(sorted(g) for g in uf.groups()), [[0, 1, 2], [3]])


class ConsistentLinksTests(unittest.TestCase):
    def test_links_joining_entities_judged_different_are_all_dropped(self) -> None:
        # 0 = "Mei", 1 = "Mei Ortiz", 2 = "Mei Lin", 3 / 4 = an unrelated alias pair
        links = [(0, 1, "sister"), (0, 2, "coworker"), (3, 4, "alias")]
        kept = consistent_links(links, {frozenset((1, 2))}, UnionFind(5))
        self.assertEqual(kept, [(3, 4, "alias")])

    def test_existing_groups_count_toward_conflicts(self) -> None:
        groups = UnionFind(3)
        groups.union(0, 1)  # e.g. a name match made earlier
        self.assertEqual(consistent_links([(1, 2, "")], {frozenset((0, 2))}, groups), [])

    def test_constraints_inside_an_existing_group_do_not_block(self) -> None:
        groups = UnionFind(3)
        groups.union(0, 1)  # already one entity, whatever the constraint says
        self.assertEqual(consistent_links([(1, 2, "x")], {frozenset((0, 1))}, groups), [(1, 2, "x")])


if __name__ == "__main__":
    unittest.main()
