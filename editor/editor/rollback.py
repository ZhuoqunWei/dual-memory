"""Undo an Editor run.

    python -m editor.main --runs               # recent runs
    python -m editor.main --rollback <run_id>  # undo one

Every pipeline run is an EditorRun node. Relationships and nodes it creates are
tagged editorRun=<id>; property changes and relationships it deletes are
recorded as undo ops on its EditActions. Rollback replays those ops newest-first
and then deletes everything carrying the tag, in one transaction.

Runs are undone last-first, like migrations: a later run may have built on
this one's changes, so rolling back past it needs --force. Writer activity is
never undone, and a property the Writer changed after the run (an entity's
aliases, say) is restored to its pre-run value.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .db import EditorDB


class RollbackRefused(RuntimeError):
    pass


@dataclass
class RollbackResult:
    run_id: str
    undo_ops: int
    relationships_deleted: int
    nodes_deleted: int


def rollback_run(db: EditorDB, run_id: str, force: bool = False) -> RollbackResult:
    run, ops = db.fetch_run_undo(run_id)
    if run is None:
        raise RollbackRefused(f"no Editor run {run_id!r}")
    if run["status"] == "rolled_back":
        raise RollbackRefused(f"run {run_id} is already rolled back")
    later = db.later_runs(run_id)
    if later and not force:
        raise RollbackRefused(
            f"{len(later)} later run(s) may depend on this one; roll back "
            f"{', '.join(later)} first, or pass --force"
        )
    rels, nodes = db.apply_rollback(run_id, ops)
    return RollbackResult(run_id, len(ops), rels, nodes)
