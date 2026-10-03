"""The measurement files of the Sample and Reference panels.

:class:`FileTableModel` lists the in-field files with the field read from each name and the
prefix all names share (shown once above the table); :func:`check_fields` reports gaps in the
field steps. :class:`SweepFilesBox` holds the zero-field list and the in-field table of one
measurement and takes dropped files or folders: names whose first field tag is 0 T (e.g.
``..._a00p000T_a16p000T.txt``) go to the zero-field list, the others to the in-field table.
"""

from __future__ import annotations

import html
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PySide6.QtCore import (
    QAbstractTableModel,
    QModelIndex,
    QPersistentModelIndex,
    QRect,
    QSize,
    Qt,
    Signal,
)
from PySide6.QtGui import QKeySequence, QPainter
from PySide6.QtWidgets import (
    QAbstractItemView,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QSizePolicy,
    QTableView,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from mag_opt_detective.core.readers import FIELD_PATTERN, parse_field, sort_paths
from mag_opt_detective.gui import icons
from mag_opt_detective.gui.controller import FieldRange, common_prefix
from mag_opt_detective.gui.panels.common import (
    Card,
    Divider,
    DropZone,
    ElidedLabel,
    FileDrops,
    LinkButton,
    Note,
    Tag,
    UnitField,
    clear_layout,
    labelled,
    mono_font,
    scaled_font,
    section_label,
)
from mag_opt_detective.gui.theme import current_tokens
from mag_opt_detective.gui.widgets import FloatEdit, open_files, parse_float

_ROOT = QModelIndex()
MAX_ROWS = 9  # rows the in-field table shows before it scrolls
MAX_LISTED = 4  # missing fields named in the gap check


# ---------------------------------------------------------------------- pure helpers
def is_zero_field(path: str | Path) -> bool:
    """A zero-field spectrum: the first field tag of the name is 0 T (``..._a00p000T...``)."""
    matches = FIELD_PATTERN.findall(Path(path).name)
    return bool(matches) and int(matches[0][0]) == 0 and int(matches[0][1]) == 0


def split_zero(paths: Iterable[str | Path]) -> tuple[list[str], list[str]]:
    """``(zero-field, in-field)`` paths, each sorted by field."""
    paths = [str(p) for p in paths]
    zero = [p for p in paths if is_zero_field(p)]
    field = [p for p in paths if not is_zero_field(p)]
    return sort_paths(zero), sort_paths(field)


def merged(old: Sequence[str], new: Iterable[str]) -> list[str]:
    """*old* plus the paths of *new* not in it yet, sorted by field."""
    return sort_paths(dict.fromkeys([*old, *(str(p) for p in new)]))


def fmt_field(b: float) -> str:
    return f"{round(b, 6):g}"


def breakable(text: str, run: int = 12) -> str:
    """*text* with zero-width spaces where a label may wrap it: after ``_``, ``-`` and ``.``,
    and inside longer runs, so a long name wraps instead of widening its panel."""
    out, length = [], 0
    for char in text:
        out.append(char)
        length += 1
        if char in "_-." or length >= run:
            out.append("\u200b")
            length = 0
    return "".join(out)


def _listing(values: Sequence[float]) -> str:
    shown = [fmt_field(b) for b in values[:MAX_LISTED]]
    more = len(values) - len(shown)
    if more:
        return f"{', '.join(shown)} T and {more} more"
    if len(shown) == 1:
        return f"{shown[0]} T"
    return f"{', '.join(shown[:-1])} and {shown[-1]} T"


@dataclass(frozen=True)
class FieldCheck:
    """Result of :func:`check_fields`: ``ok`` and a sentence for the panel."""

    ok: bool
    text: str


def check_fields(fields: Sequence[float | None]) -> FieldCheck | None:
    """Check the fields read from the file names for gaps; None without files.

    The step is the most common spacing (the smallest one on a tie); spacings that are whole
    multiples of it are missing fields, others make the steps uneven.
    """
    if not fields:
        return None
    unnamed = sum(b is None for b in fields)
    if unnamed:
        files = "file name has" if unnamed == 1 else "file names have"
        return FieldCheck(
            False, f"{unnamed} {files} no field (like …_a01p250T): use a custom range."
        )
    values = np.sort(np.asarray(fields, dtype=float))
    if values.size == 1:
        return FieldCheck(True, f"One field, {fmt_field(values[0])} T.")
    steps = np.diff(values)
    tol = 1e-6 * max(1.0, float(np.abs(values).max()))
    repeated = values[1:][steps <= tol]
    if repeated.size:
        return FieldCheck(False, f"Repeated field: {_listing(list(np.unique(repeated)))}.")
    rounded, counts = np.unique(np.round(steps, 6), return_counts=True)
    step = float(rounded[np.argmax(counts)])  # unique() sorts: the smallest wins a tie
    multiples = np.rint(steps / step)
    if not np.allclose(steps, multiples * step, rtol=0, atol=max(tol, 1e-3 * step)):
        return FieldCheck(
            False,
            f"Uneven field steps, {fmt_field(steps.min())} to {fmt_field(steps.max())} T.",
        )
    missing = [
        values[i] + k * step for i, n in enumerate(multiples.astype(int)) for k in range(1, n)
    ]
    if not missing:
        return FieldCheck(True, f"Even {fmt_field(step)} T steps, no missing fields.")
    return FieldCheck(False, f"Missing {_listing(missing)} ({fmt_field(step)} T steps).")


# ---------------------------------------------------------------------- in-field table
class FileTableModel(QAbstractTableModel):
    """In-field files, sorted by field: B (T) read from the name, and the name without the
    common prefix (:meth:`prefix`)."""

    pathsChanged = Signal()
    HEADERS = ("B (T)", "File")
    B, FILE = 0, 1

    def __init__(self, parent=None):
        super().__init__(parent)
        self._paths: list[str] = []
        self._names: list[str] = []
        self._fields: list[float | None] = []
        self._prefix = ""

    def paths(self) -> list[str]:
        return list(self._paths)

    def fields(self) -> list[float | None]:
        return list(self._fields)

    def prefix(self) -> str:
        return self._prefix

    def set_paths(self, paths: Iterable[str | Path]) -> None:
        self.beginResetModel()
        self._paths = sort_paths(paths)
        self._names = [Path(p).name for p in self._paths]
        self._fields = [parse_field(p) for p in self._paths]
        self._prefix = common_prefix(self._names)
        self.endResetModel()
        self.pathsChanged.emit()

    def remove_rows(self, rows: Iterable[int]) -> None:
        drop = set(rows)
        if drop:
            self.set_paths([p for i, p in enumerate(self._paths) if i not in drop])

    def rowCount(self, parent: QModelIndex | QPersistentModelIndex = _ROOT) -> int:
        return 0 if parent.isValid() else len(self._paths)

    def columnCount(self, parent: QModelIndex | QPersistentModelIndex = _ROOT) -> int:
        return 0 if parent.isValid() else len(self.HEADERS)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        row, col = index.row(), index.column()
        field = self._fields[row]
        if role == Qt.ItemDataRole.DisplayRole:
            if col == self.B:
                return "–" if field is None else fmt_field(field)
            return self._names[row][len(self._prefix) :]
        if role == Qt.ItemDataRole.ToolTipRole:
            if col == self.B and field is None:
                return "No field in this name"
            return self._paths[row]
        if role == Qt.ItemDataRole.TextAlignmentRole:
            side = Qt.AlignmentFlag.AlignRight if col == self.B else Qt.AlignmentFlag.AlignLeft
            return int(side | Qt.AlignmentFlag.AlignVCenter)
        if role == Qt.ItemDataRole.ForegroundRole:
            tokens = current_tokens()
            if col == self.B:
                return tokens["warn"] if field is None else tokens["fg"]
            return tokens["muted"]
        return None

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.DisplayRole:
            return self.HEADERS[section]
        return None


class _FlatHeader(QHeaderView):
    """A flat header in small muted capitals with a line below (as in the mockup)."""

    def __init__(self, parent=None):
        super().__init__(Qt.Orientation.Horizontal, parent)
        self.setFont(scaled_font(self, 0.82, bold=True))
        self.setHighlightSections(False)
        self.setSectionsClickable(False)

    def sizeHint(self) -> QSize:
        hint = super().sizeHint()
        return QSize(hint.width(), self.fontMetrics().height() + 8)

    def sectionSizeFromContents(self, index: int) -> QSize:
        text = str(self.model().headerData(index, self.orientation())).upper()
        return QSize(self.fontMetrics().horizontalAdvance(text) + 16, self.sizeHint().height())

    def paintSection(self, painter: QPainter, rect: QRect, index: int) -> None:
        tokens = current_tokens()
        painter.save()
        painter.fillRect(rect, tokens["surface"])
        painter.fillRect(rect.left(), rect.bottom(), rect.width(), 1, tokens["line"])
        painter.setPen(tokens["muted"])
        painter.setFont(self.font())
        text = str(self.model().headerData(index, self.orientation())).upper()
        align = Qt.AlignmentFlag.AlignRight if index == 0 else Qt.AlignmentFlag.AlignLeft
        painter.drawText(rect.adjusted(8, 0, -8, 0), align | Qt.AlignmentFlag.AlignVCenter, text)
        painter.restore()

    def paintEvent(self, event) -> None:
        painter = QPainter(self.viewport())
        painter.fillRect(self.viewport().rect(), current_tokens()["surface"])
        painter.end()
        super().paintEvent(event)


class FileTable(QTableView):
    """Table of in-field files; Delete or Backspace removes the selected rows."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.file_model = FileTableModel(self)
        self.setModel(self.file_model)
        self.pathsChanged = self.file_model.pathsChanged
        self.setHorizontalHeader(_FlatHeader(self))
        header = self.horizontalHeader()
        header.setSectionResizeMode(FileTableModel.B, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(FileTableModel.FILE, QHeaderView.ResizeMode.Stretch)
        header.setMinimumSectionSize(40)
        self.verticalHeader().hide()
        self.setFont(mono_font())
        self.verticalHeader().setDefaultSectionSize(self.fontMetrics().height() + 5)
        self.setShowGrid(False)
        self.setAlternatingRowColors(True)
        self.setWordWrap(False)
        self.setTextElideMode(Qt.TextElideMode.ElideMiddle)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setFrameShape(QTableView.Shape.NoFrame)
        self.setTabKeyNavigation(False)  # Tab leaves the table; the arrow keys move in it
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setAccessibleName("In-field files")
        self.setToolTip("Delete or Backspace removes the selected files")
        self.file_model.modelReset.connect(self._fit_height)
        self._fit_height()

    # the list API of the old file lists
    def paths(self) -> list[str]:
        return self.file_model.paths()

    def set_paths(self, paths) -> None:
        self.file_model.set_paths(paths)

    def clear_paths(self) -> None:
        self.file_model.set_paths([])

    def count(self) -> int:
        return self.file_model.rowCount()

    def remove_selected(self) -> None:
        rows = {index.row() for index in self.selectionModel().selectedRows()}
        self.file_model.remove_rows(rows)

    def _fit_height(self) -> None:
        rows = min(max(self.file_model.rowCount(), 1), MAX_ROWS)
        height = self.horizontalHeader().sizeHint().height() + rows * (
            self.verticalHeader().defaultSectionSize()
        )
        self.setFixedHeight(height + 2)

    def keyPressEvent(self, event) -> None:
        delete = event.matches(QKeySequence.StandardKey.Delete)
        if (delete or event.key() == Qt.Key.Key_Backspace) and self.selectionModel().selectedRows():
            self.remove_selected()
            return
        super().keyPressEvent(event)


# ---------------------------------------------------------------------- zero-field list
class ZeroList(QWidget):
    """One or two zero-field files (Before / After the sweep), each with a remove button."""

    pathsChanged = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._paths: list[str] = []
        self._prefix = ""
        self.card = Card()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.card)
        self.rows: list[QWidget] = []
        self.setAccessibleName("Zero-field files")

    def paths(self) -> list[str]:
        return list(self._paths)

    def count(self) -> int:
        return len(self._paths)

    def set_paths(self, paths) -> None:
        self._paths = sort_paths(paths)
        self._rebuild()
        self.pathsChanged.emit()

    def clear_paths(self) -> None:
        self.set_paths([])

    def remove(self, path: str) -> None:
        self.set_paths([p for p in self._paths if p != path])

    def set_prefix(self, prefix: str) -> None:
        """Hide *prefix* (the in-field files' common prefix) in names that start with it."""
        if prefix != self._prefix:
            self._prefix = prefix
            self._rebuild()

    def tags(self) -> list[str]:
        if len(self._paths) == 2:
            return ["Before", "After"]
        return [""] * len(self._paths)

    def _rebuild(self) -> None:
        box = self.card.box
        clear_layout(box)
        self.rows = []
        for i, (path, tag) in enumerate(zip(self._paths, self.tags(), strict=True)):
            if i:
                box.addWidget(Divider())
            row = QWidget()
            line = QHBoxLayout(row)
            line.setContentsMargins(8, 4, 4, 4)
            line.setSpacing(8)
            if tag:
                label = Tag(tag)
                label.setFixedWidth(max(Tag(t).sizeHint().width() for t in self.tags()))
                line.addWidget(label, 0, Qt.AlignmentFlag.AlignVCenter)
            name = Path(path).name
            if self._prefix and name.startswith(self._prefix) and name != self._prefix:
                name = name[len(self._prefix) :]
            label = ElidedLabel(name, mode=Qt.TextElideMode.ElideLeft)  # names differ at the end
            label.setProperty("kit", "muted")
            label.setFont(mono_font())
            label.setToolTip(path)
            line.addWidget(label, 1)
            remove = QToolButton()
            remove.setProperty("kit", "tool")
            remove.setFixedSize(22, 22)
            remove.setIconSize(QSize(12, 12))
            icons.set_icon(remove, "x", "faint")
            remove.setToolTip(f"Remove {Path(path).name}")
            remove.setAccessibleName(f"Remove {Path(path).name}")
            remove.clicked.connect(lambda _checked=False, p=path: self.remove(p))
            line.addWidget(remove)
            box.addWidget(row)
            self.rows.append(row)


# ---------------------------------------------------------------------- the box
def count_text(n: int) -> str:
    return "no files" if n == 0 else f"{n} file" + ("s" if n != 1 else "")


def zero_note(n: int) -> tuple[str, str]:
    """``(text, level)`` about the zero-field files: drift correction or a problem."""
    if n == 0:
        return "One file, or two measured before and after the sweep.", "muted"
    if n == 1:
        return "One spectrum for the whole sweep, so drift is not corrected.", "muted"
    if n == 2:
        return "Measured before and after the sweep, so drift is corrected linearly.", "ok"
    return f"{n} files: keep one, or two (before and after the sweep).", "warn"


class _FilesHead(QWidget):
    """Section label, file count and link buttons in one row."""

    def __init__(self, title: str, *buttons: LinkButton):
        super().__init__()
        self.count_label = QLabel(count_text(0))
        self.count_label.setProperty("kit", "muted")
        self.count_label.setFont(scaled_font(self.count_label, 0.94))
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        row.addWidget(section_label(title))
        row.addWidget(self.count_label, 1)
        for button in buttons:
            row.addWidget(button)

    def set_count(self, n: int) -> None:
        self.count_label.setText(count_text(n))


class SweepFilesBox(QWidget):
    """Zero-field list and in-field table of one measurement, with their checks.

    ``changed`` fires when either list changes. Dropping files or a folder anywhere on the box
    sorts zero-field names into the zero list and the others into the table, replacing each
    list that receives files; files dropped on the zero list all go there.
    """

    changed = Signal()

    def __init__(self, what: str = "sample", drop_zone: bool = True, parent=None):
        super().__init__(parent)
        self.what = what
        self.add_zero_button = LinkButton("Add…", "Add zero-field spectra")
        self.add_field_button = LinkButton("Add…", "Add in-field spectra")
        self.clear_button = LinkButton("Clear", "Remove all in-field files")
        self.zero_head = _FilesHead("Zero field", self.add_zero_button)
        self.field_head = _FilesHead("In field", self.add_field_button, self.clear_button)

        self.zero_list = ZeroList()
        self.zero_drop = DropZone(f"Drop the {what}'s zero-field file or files here", icon=None)
        self.zero_note = Note()
        self.prefix_label = QLabel()
        self.prefix_label.setProperty("kit", "muted")
        self.prefix_label.setFont(mono_font(0.86))
        self.prefix_label.setWordWrap(True)
        self.prefix_label.setTextFormat(Qt.TextFormat.RichText)
        self.field_list = FileTable()
        self.field_card = Card()
        self.field_card.box.addWidget(self.field_list)
        self.field_drop = DropZone(
            "Drop the reference sweep. It is interpolated onto the sample fields."
            if what == "reference"
            else f"Drop the {what}'s in-field files here",
            icon=None,
        )
        self.gap_note = Note()
        self.drop_zone = DropZone("Drop files or a sweep folder here to replace both lists.")
        self.drop_zone.setVisible(drop_zone)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)
        zero = QVBoxLayout()
        zero.setSpacing(7)
        for widget in (self.zero_head, self.zero_list, self.zero_drop, self.zero_note):
            zero.addWidget(widget)
        field = QVBoxLayout()
        field.setSpacing(7)
        for widget in (
            self.field_head,
            self.prefix_label,
            self.field_card,
            self.field_drop,
            self.gap_note,
        ):
            field.addWidget(widget)
        layout.addLayout(zero)
        layout.addLayout(field)
        layout.addWidget(self.drop_zone)

        self.add_zero_button.clicked.connect(self.load_zero_dialog)
        self.add_field_button.clicked.connect(self.add_field_dialog)
        self.clear_button.clicked.connect(self.field_list.clear_paths)
        self.drop_zone.clicked.connect(self.open_sweep_dialog)
        self.zero_drop.clicked.connect(self.load_zero_dialog)
        self.field_drop.clicked.connect(self.add_field_dialog)
        self.drop_zone.filesDropped.connect(self.drop_files)
        self.field_drop.filesDropped.connect(self.drop_files)
        self.zero_drop.filesDropped.connect(self.zero_list.set_paths)
        FileDrops(self, self.drop_files)
        FileDrops(self.field_list.viewport(), self.drop_files)
        FileDrops(self.zero_list, self.zero_list.set_paths)
        self.zero_list.pathsChanged.connect(self._on_changed)
        self.field_list.pathsChanged.connect(self._on_changed)
        self._gap_check = True
        self._sync()

    # --- files -------------------------------------------------------------------------
    def zero_paths(self) -> list[str]:
        return self.zero_list.paths()

    def field_paths(self) -> list[str]:
        return self.field_list.paths()

    def set_files(self, zero: Sequence[str], field: Sequence[str]) -> None:
        """Show these files (sorted); emits ``changed`` only if something differs."""
        if self.zero_paths() != sort_paths(zero):
            self.zero_list.set_paths(zero)
        if self.field_paths() != sort_paths(field):
            self.field_list.set_paths(field)

    def drop_files(self, paths: Sequence[str]) -> None:
        """Replace the lists by the dropped files (zero-field names to the zero list)."""
        zero, field = split_zero(paths)
        if zero:
            self.zero_list.set_paths(zero)
        if field:
            self.field_list.set_paths(field)

    def add_files(self, paths: Sequence[str]) -> None:
        """Add files to the lists (zero-field names to the zero list)."""
        zero, field = split_zero(paths)
        if zero:
            self.zero_list.set_paths(merged(self.zero_paths(), zero))
        if field:
            self.field_list.set_paths(merged(self.field_paths(), field))

    def load_zero_dialog(self) -> None:
        """Replace the zero-field list by files chosen in a dialog."""
        paths = open_files(self, f"Zero-field spectra of the {self.what}")
        if paths:
            self.zero_list.set_paths(paths)

    def add_field_dialog(self) -> None:
        paths = open_files(self, f"Add spectra to the {self.what}")
        if paths:
            self.add_files(paths)

    def open_sweep_dialog(self) -> None:
        """Replace the lists by the files of a sweep chosen in a dialog."""
        paths = open_files(self, f"Open the {self.what} sweep (select all its files)")
        if paths:
            self.drop_files(paths)

    # --- checks ------------------------------------------------------------------------
    def set_gap_check(self, on: bool) -> None:
        """Show the gap check under the table (off while the fields come from a custom range)."""
        self._gap_check = on
        self._sync()

    def _on_changed(self) -> None:
        self._sync()
        self.changed.emit()

    def _sync(self) -> None:
        n_zero, n_field = self.zero_list.count(), self.field_list.count()
        self.zero_head.set_count(n_zero)
        self.field_head.set_count(n_field)
        self.zero_list.setVisible(bool(n_zero))
        self.zero_drop.setVisible(not n_zero)
        self.zero_note.set_text(*zero_note(n_zero))
        self.zero_note.setVisible(bool(n_zero))
        self.field_card.setVisible(bool(n_field))
        self.field_drop.setVisible(not n_field)
        self.clear_button.setEnabled(bool(n_field))
        prefix = self.field_list.file_model.prefix()
        title = "Common prefix" if n_field > 1 else "Prefix"
        shown = html.escape(breakable(prefix))
        self.prefix_label.setText(f"{title} <b>{shown}</b>" if prefix else "")
        self.prefix_label.setToolTip(prefix)
        self.prefix_label.setAccessibleName(f"{title} {prefix}" if prefix else "")
        self.prefix_label.setVisible(bool(prefix))
        self.zero_list.set_prefix(prefix)
        check = check_fields(self.field_list.file_model.fields()) if self._gap_check else None
        self.gap_note.setVisible(check is not None)
        if check is not None:
            self.gap_note.set_text(check.text, "ok" if check.ok else "warn")


# ---------------------------------------------------------------------- custom field range
class FieldRangeInputs(QWidget):
    """Start / step / end of a custom field range in T, with a note that checks the number
    of values against the number of files (:meth:`set_file_count`)."""

    changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.start = FloatEdit(0.25, "Start B")
        self.step = FloatEdit(0.25, "Step B")
        self.end = FloatEdit(16.0, "End B")
        self.fields = {edit: UnitField(edit, "T") for edit in (self.start, self.step, self.end)}
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(6)
        for col, (text, edit) in enumerate(
            (("Start", self.start), ("Step", self.step), ("End", self.end))
        ):
            grid.addWidget(labelled(text, self.fields[edit]), 0, col)
            grid.setColumnStretch(col, 1)
        self.note = Note()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(7)
        layout.addLayout(grid)
        layout.addWidget(self.note)
        self._files = 0
        for edit in self.fields:
            edit.textChanged.connect(self._on_text)
        self._check()

    def edits(self) -> tuple[FloatEdit, FloatEdit, FloatEdit]:
        return self.start, self.step, self.end

    def field_range(self) -> FieldRange:
        return FieldRange(*(parse_float(edit.text()) for edit in self.edits()))

    def show_range(self, rng: FieldRange) -> None:
        """Show *rng*; a field that already holds its value keeps what was typed."""
        for edit, value in zip(self.edits(), (rng.start, rng.step, rng.end), strict=True):
            if parse_float(edit.text()) != value:
                edit.setText("" if value is None else fmt_field(value))

    def set_file_count(self, n: int) -> None:
        if n != self._files:
            self._files = n
            self._check()

    def _on_text(self) -> None:
        self._check()
        self.changed.emit()

    def _check(self) -> None:
        start, step, end = (parse_float(edit.text()) for edit in self.edits())
        bad: set[FloatEdit] = set()
        if None in (start, step, end):
            bad = {e for e, v in zip(self.edits(), (start, step, end), strict=True) if v is None}
            text, level = "Enter the start, step and end.", "warn"
        elif step <= 0:
            bad = {self.step}
            text, level = "Step must be a positive number.", "warn"
        elif end < start:
            bad = {self.end}
            text, level = "End must not be below start.", "warn"
        else:
            n = round((end - start) / step) + 1
            if not self._files:
                text, level = f"{n} values, {fmt_field(start)} – {fmt_field(end)} T.", "muted"
            elif n == self._files:
                text, level = f"{n} values, matches the {self._files} files.", "ok"
            else:
                text, level = f"{n} values, but there are {self._files} files.", "warn"
        for edit, field in self.fields.items():
            field.set_invalid(edit in bad)
        self.note.set_text(text, level)
