"""Rerunning the Writer or the Editor must not duplicate or compound anything.

Neo4j integration tests; see tests/support.py for how to run them.
"""

from __future__ import annotations

import unittest

from editor.evaluation.bridge import run_bridge
from editor.pipeline import run_pipeline
from editor.steps.confidence import run_confidence
from editor.audit import AuditLog

from .support import (
    NEO4J_ENV,
    FakeEmbeddings,
    FakeLLM,
    duplicate_relationships,
    editor_db,
    fingerprint,
    requires_neo4j,
    reset_and_load,
)


@requires_neo4j
class WriterIdempotencyTest(unittest.TestCase):
    def test_rewriting_a_session_writes_nothing_new(self) -> None:
        graph, payload = reset_and_load(sessions=12)
        try:
            before = fingerprint(graph)

            # The Editor archives one memory; a rerun must not bring it back.
            victim = next(iter(graph.memories()))
            graph._rows("MATCH (m:Memory {id: $id}) SET m.status = 'archived'", id=victim)
            before_archived = fingerprint(graph)
            self.assertNotEqual(before, before_archived)

            out = run_bridge("load", payload, NEO4J_ENV)
            statuses = {o["status"] for s in out["sessions"] for o in s["outcomes"]}
            self.assertEqual(statuses, {"already_written"})
            self.assertEqual(fingerprint(graph), before_archived)
        finally:
            graph.close()


@requires_neo4j
class EditorIdempotencyTest(unittest.TestCase):
    def test_rerunning_an_unfinished_batch_converges_and_then_changes_nothing(self) -> None:
        graph, _ = reset_and_load(sessions=20)
        db = editor_db()
        try:
            llm, emb = FakeLLM(), FakeEmbeddings()
            # As if each run crashed before marking the batch reviewed.
            actions = []
            for _ in range(4):
                result = run_pipeline(db, llm, emb, None, mark_reviewed=False)
                actions.append(result.edit_actions)
                if result.edit_actions == 0:
                    break
            self.assertEqual(actions[-1], 0, f"no fixpoint after {len(actions)} passes: {actions}")

            settled = fingerprint(graph)
            self.assertEqual(run_pipeline(db, llm, emb, None, mark_reviewed=False).edit_actions, 0)
            self.assertEqual(fingerprint(graph), settled)
            self.assertEqual(duplicate_relationships(graph), [])
        finally:
            db.close()
            graph.close()

    def test_confidence_decays_once_per_day(self) -> None:
        graph, _ = reset_and_load(sessions=5)
        db = editor_db()
        try:
            graph._rows("MATCH (m:Memory) SET m.status = 'reviewed'")
            audit = AuditLog(db)
            first = run_confidence([], db, audit)
            second = run_confidence([], db, audit)
            self.assertGreater(first.decayed, 0)
            self.assertEqual(second.decayed, 0)
        finally:
            db.close()
            graph.close()


if __name__ == "__main__":
    unittest.main()
