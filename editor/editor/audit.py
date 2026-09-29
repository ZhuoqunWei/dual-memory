"""EditAction audit logging — every Editor mutation is recorded.

Each EditAction carries the undo ops for the changes made since the previous
one (see EditorDB's journal), numbered within its run so rollback can replay
them newest-first.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .db import EditorDB

log = logging.getLogger("editor.audit")


class AuditLog:
    """Accumulates EditAction entries during a pipeline run."""

    def __init__(self, db: EditorDB) -> None:
        self._db = db
        self.actions: list[str] = []

    def log(self, action_type: str, targets: list[str], reason: str) -> str:
        action_id = self._db.log_edit_action(
            action_type, targets, reason,
            undo=self._db.take_undo(), seq=len(self.actions) + 1,
        )
        self.actions.append(action_id)
        log.info("EditAction[%s] %s targets=%s reason=%s", action_id[:8], action_type, len(targets), reason[:80])
        return action_id

    def flush(self, step: str) -> None:
        """Give changes a step made without logging them an EditAction of their own."""
        if self._db.has_pending_undo:
            self.log("unlogged", [], f"{step}: changes not logged individually")

    @property
    def count(self) -> int:
        return len(self.actions)
