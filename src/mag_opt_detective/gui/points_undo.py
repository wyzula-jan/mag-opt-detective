"""Undo history of the picked points.

Every change of ``AppController.points`` (record, remove, new, rename or delete a curve,
import, a new table on Process) runs inside ``AppController.point_edit(text)``. The controller
takes a :class:`PointsState` before and after the block, and :class:`PointsUndoStack` pushes
one command that restores either of them, so a batch of edits (e.g. the points an auto-pick
accepts) is one undo step. Snapshots share the column arrays that did not change, so a step
costs about one column of the table.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Hashable
from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import QObject
from PySide6.QtGui import QUndoCommand, QUndoStack

from mag_opt_detective.core.points import PointTable

logger = logging.getLogger("mag_opt_detective")

UNDO_LIMIT = 500
_MERGE_ID = 1  # QUndoCommand.id() of commands that may merge with the one before


def _frozen(values: np.ndarray) -> np.ndarray:
    values = np.array(values, dtype=float)
    values.flags.writeable = False
    return values


def _same(a: np.ndarray, b: np.ndarray) -> bool:
    return a is b or (a.shape == b.shape and np.array_equal(a, b, equal_nan=True))


@dataclass(frozen=True, eq=False)
class PointsState:
    """The point table (``field`` None: no table yet) and the current curve at one moment.

    The arrays are read-only; :meth:`table` copies them into a new table.
    """

    field: np.ndarray | None
    columns: tuple[tuple[str, np.ndarray], ...]
    curve: str

    @classmethod
    def capture(
        cls, table: PointTable | None, curve: str, like: PointsState | None = None
    ) -> PointsState:
        """A snapshot of *table*; arrays equal to those of *like* are shared, not copied."""
        if table is None:
            return cls(None, (), curve)
        old = dict(like.columns) if like is not None else {}
        same_field = like is not None and like.field is not None and _same(like.field, table.field)
        field = like.field if same_field else _frozen(table.field)
        columns = []
        for name in table.names:
            values, kept = table.column(name), old.get(name)
            if kept is None or not _same(kept, values):
                kept = _frozen(values)
            columns.append((name, kept))
        return cls(field, tuple(columns), curve)

    def table(self) -> PointTable | None:
        if self.field is None:
            return None
        return PointTable(self.field.copy(), dict(self.columns))

    def same_as(self, other: PointsState) -> bool:
        if self.curve != other.curve or (self.field is None) != (other.field is None):
            return False
        if self.field is not None and not _same(self.field, other.field):
            return False
        if [name for name, _ in self.columns] != [name for name, _ in other.columns]:
            return False
        return all(_same(a, b) for (_, a), (_, b) in zip(self.columns, other.columns, strict=True))


class PointsCommand(QUndoCommand):
    """One undo step: restores the state before (undo) or after (redo) an edit.

    The edit is already applied when the command is pushed, so the first redo does nothing.
    Commands with the same *merge* key (not None) in a row become one step.
    """

    def __init__(
        self,
        restore: Callable[[PointsState], None],
        text: str,
        before: PointsState,
        after: PointsState,
        merge: Hashable | None = None,
    ):
        super().__init__(text)
        self._restore = restore
        self.before = before
        self.after = after
        self.merge = merge
        self._applied = True

    def id(self) -> int:
        return _MERGE_ID if self.merge is not None else -1

    def mergeWith(self, other: QUndoCommand) -> bool:
        if not isinstance(other, PointsCommand) or other.merge != self.merge:
            return False
        self.after = other.after
        self.setText(other.text())
        self.setObsolete(self.after.same_as(self.before))  # e.g. a name typed back: no step
        return True

    def redo(self) -> None:
        if self._applied:
            self._applied = False
            return
        logger.info("Redo: %s", self.text())
        self._restore(self.after)

    def undo(self) -> None:
        logger.info("Undo: %s", self.text())
        self._restore(self.before)


class PointsUndoStack(QUndoStack):
    """The undo stack of the point edits; *restore* puts a :class:`PointsState` back.

    Use ``AppController.point_edit`` to edit points; :meth:`record` is its back end.
    """

    def __init__(self, restore: Callable[[PointsState], None], parent: QObject | None = None):
        super().__init__(parent)
        self.setUndoLimit(UNDO_LIMIT)
        self._restore = restore
        self._last: PointsState | None = None  # newest snapshot; its arrays are reused

    def capture(self, table: PointTable | None, curve: str) -> PointsState:
        """Snapshot of *table* and *curve* (unchanged columns share the last snapshot's)."""
        self._last = PointsState.capture(table, curve, self._last)
        return self._last

    def record(
        self, text: str, before: PointsState, after: PointsState, merge: Hashable | None = None
    ) -> bool:
        """Push the edit *before* -> *after* as one step; False (no step) if nothing changed."""
        if after.same_as(before):
            return False
        self.push(PointsCommand(self._put_back, text, before, after, merge))
        return True

    def _put_back(self, state: PointsState) -> None:
        self._last = state
        self._restore(state)
