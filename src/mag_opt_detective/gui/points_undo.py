"""Undo history of the picked points.

Every change of ``AppController.points`` (record, remove, new, rename or delete a curve,
import, a new table on Process) runs inside ``AppController.point_edit(text)``. The controller
takes a :class:`PointsState` before and after the block, and :class:`PointsUndoStack` keeps
one step that restores either of them, so a batch of edits (e.g. the points an auto-pick
accepts) is one undo step. Snapshots share the column arrays that did not change, so a step
costs about one column of the table.

The history is a Python list with the QUndoStack methods it needs, not a QUndoStack: one owns
the Python commands pushed to it, and the garbage collector can free them under it, which
crashes when a window or controller is collected.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Hashable
from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QAction

from mag_opt_detective.core.points import PointTable

logger = logging.getLogger("mag_opt_detective")

UNDO_LIMIT = 500  # steps kept; the oldest go first


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


@dataclass(eq=False)
class UndoStep:
    """One step: the states before and after an edit called *text*.

    Steps with the same *merge* key (not None) in a row become one (the letters of a name).
    """

    text: str
    before: PointsState
    after: PointsState
    merge: Hashable | None = None


class PointsUndoStack(QObject):
    """The undo history of the point edits; *restore* puts a :class:`PointsState` back.

    It has the QUndoStack methods used here (``count``, ``text``, ``undo``, ``undoText``, ...)
    and emits :attr:`changed` when the steps or the position in them change. Use
    ``AppController.point_edit`` to edit points; :meth:`record` is its back end.
    """

    changed = Signal()

    def __init__(self, restore: Callable[[PointsState], None], parent: QObject | None = None):
        super().__init__(parent)
        self._restore = restore
        self._steps: list[UndoStep] = []
        self._index = 0  # the steps before it are done, the others undone
        self._last: PointsState | None = None  # newest snapshot; its arrays are reused

    def capture(self, table: PointTable | None, curve: str) -> PointsState:
        """Snapshot of *table* and *curve* (unchanged columns share the last snapshot's)."""
        self._last = PointsState.capture(table, curve, self._last)
        return self._last

    def record(
        self, text: str, before: PointsState, after: PointsState, merge: Hashable | None = None
    ) -> bool:
        """Add the edit *before* -> *after* as a step; False (no step) if nothing changed.

        The steps undone go. With the *merge* key of the last step the edit joins it, which
        keeps its text; a step that ends where it began goes away (a name typed back).
        """
        if after.same_as(before):
            return False
        del self._steps[self._index :]
        last = self._steps[-1] if self._steps else None
        if merge is not None and last is not None and last.merge == merge:
            last.after = after
            if last.after.same_as(last.before):
                self._steps.pop()
        else:
            self._steps.append(UndoStep(text, before, after, merge))
            del self._steps[: max(0, len(self._steps) - UNDO_LIMIT)]
        self._index = len(self._steps)
        self.changed.emit()
        return True

    # --- like QUndoStack ---------------------------------------------------------------
    def count(self) -> int:
        return len(self._steps)

    def index(self) -> int:
        return self._index

    def text(self, index: int) -> str:
        return self._steps[index].text

    def canUndo(self) -> bool:
        return self._index > 0

    def canRedo(self) -> bool:
        return self._index < len(self._steps)

    def undoText(self) -> str:
        return self._steps[self._index - 1].text if self.canUndo() else ""

    def redoText(self) -> str:
        return self._steps[self._index].text if self.canRedo() else ""

    def undo(self) -> None:
        if self.canUndo():
            self._index -= 1
            step = self._steps[self._index]
            logger.info("Undo: %s", step.text)
            self._put_back(step.before)
            self.changed.emit()

    def redo(self) -> None:
        if self.canRedo():
            step = self._steps[self._index]
            self._index += 1
            logger.info("Redo: %s", step.text)
            self._put_back(step.after)
            self.changed.emit()

    def clear(self) -> None:
        self._steps.clear()
        self._index = 0
        self.changed.emit()

    def createUndoAction(self, parent: QObject, prefix: str = "Undo") -> QAction:
        return StepAction(self, prefix, redo=False, parent=parent)

    def createRedoAction(self, parent: QObject, prefix: str = "Redo") -> QAction:
        return StepAction(self, prefix, redo=True, parent=parent)

    def _put_back(self, state: PointsState) -> None:
        self._last = state
        self._restore(state)


class StepAction(QAction):
    """Undo (or Redo) of a :class:`PointsUndoStack`, named after its step, off without one."""

    def __init__(self, stack: PointsUndoStack, prefix: str, redo: bool, parent: QObject):
        super().__init__(prefix, parent)
        self._stack, self._prefix, self._redo = stack, prefix, redo
        self.triggered.connect(stack.redo if redo else stack.undo)
        stack.changed.connect(self.refresh)
        self.refresh()

    def refresh(self) -> None:
        stack = self._stack
        text = stack.redoText() if self._redo else stack.undoText()
        self.setText(f"{self._prefix} {text}" if text else self._prefix)
        self.setEnabled(stack.canRedo() if self._redo else stack.canUndo())
