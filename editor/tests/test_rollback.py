"""Rolling back an Editor run restores the graph it started from.

Neo4j integration tests; see tests/support.py for how to run them.
"""

from __future__ import annotations

import unittest

from editor.pipeline import run_pipeline
from editor.rollback import RollbackRefused, rollback_run

from .support import (
    FakeEmbeddings,
    FakeLLM,
    editor_db,
    fingerprint,
    requires_neo4j,
    reset_and_load,
)


@requires_neo4j
class RollbackTest(unittest.TestCase):
    def test_rollback_restores_the_graph_exactly(self) -> None:
        graph, _ = reset_and_load(sessions=30)
        db = editor_db()
        try:
            llm, emb = FakeLLM(), FakeEmbeddings()
            start = fingerprint(graph)
            first = run_pipeline(db, llm, emb, None)
            after_first = fingerprint(graph)
            second = run_pipeline(db, llm, emb, None)
            self.assertGreater(first.edit_actions, 0)
            self.assertGreater(second.edit_actions, 0)
            self.assertNotEqual(fingerprint(graph), after_first)

            # Last-first: the first run can't be undone while the second stands.
            with self.assertRaises(RollbackRefused):
                rollback_run(db, first.run_id)

            rollback_run(db, second.run_id)
            self.assertEqual(fingerprint(graph), after_first)
            rollback_run(db, first.run_id)
            self.assertEqual(fingerprint(graph), start)

            with self.assertRaises(RollbackRefused):
                rollback_run(db, first.run_id)
        finally:
            db.close()
            graph.close()


if __name__ == "__main__":
    unittest.main()
