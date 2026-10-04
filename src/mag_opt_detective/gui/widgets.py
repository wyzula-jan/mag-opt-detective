"""Small reusable widgets."""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path

import shiboken6
from PySide6.QtCore import (
    QAbstractTableModel,
    QLocale,
    QModelIndex,
    QObject,
    QPersistentModelIndex,
    QPoint,
    QRect,
    QSize,
    Qt,
    Signal,
)
from PySide6.QtGui import QAction, QDoubleValidator, QKeySequence, QPainter
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLayout,
    QLayoutItem,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QSizePolicy,
    QSlider,
    QWidget,
)

from mag_opt_detective.core.points import PointTable
from mag_opt_detective.core.readers import sort_paths
from mag_opt_detective.core.units import Unit, from_cm1, to_cm1
from mag_opt_detective.gui.theme import current_tokens

SPECTRA_FILTER = "Spectra (*.txt *.dat *.[0-9] *.[0-9][0-9]);;All files (*)"
TABLE_FILTER = "Tab-separated table (*.csv *.tsv *.txt);;All files (*)"
IMAGE_FILTER = "PNG image (*.png);;SVG image (*.svg)"

_last_dir = ""
_ROOT = QModelIndex()


def last_dir() -> str:
    """Folder of the last file opened or saved (start folder of the next dialog)."""
    return _last_dir


def set_last_dir(path: str) -> None:
    global _last_dir
    _last_dir = str(path or "")


def _remember(path: str) -> None:
    global _last_dir
    if path:
        _last_dir = str(Path(path).parent)


def open_files(parent, title: str, filter: str = SPECTRA_FILTER) -> list[str]:
    paths, _ = QFileDialog.getOpenFileNames(parent, title, _last_dir, filter)
    if paths:
        _remember(paths[0])
    return paths


def open_file(parent, title: str, filter: str = TABLE_FILTER) -> str:
    path, _ = QFileDialog.getOpenFileName(parent, title, _last_dir, filter)
    _remember(path)
    return path


def save_file(parent, title: str, filter: str = TABLE_FILTER) -> str:
    path, _ = QFileDialog.getSaveFileName(parent, title, _last_dir, filter)
    _remember(path)
    return path


class FileListWidget(QListWidget):
    """List of measurement files; accepts files (or folders) dropped from the file manager.

    Dropping or loading replaces the content; files are sorted by the field in their name.
    """

    pathsChanged = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._paths: list[str] = []
        self.setAcceptDrops(True)
        self.setDragDropMode(QAbstractItemView.DragDropMode.DropOnly)
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setAlternatingRowColors(True)
        self.setToolTip("Drop files here (Del removes the selected ones)")

    def paths(self) -> list[str]:
        return list(self._paths)

    def set_paths(self, paths) -> None:
        self._paths = sort_paths(paths)
        self.clear()
        for p in self._paths:
            item = QListWidgetItem(Path(p).name)
            item.setToolTip(p)
            self.addItem(item)
        self.pathsChanged.emit()

    def clear_paths(self) -> None:
        self.set_paths([])

    @staticmethod
    def _dropped_files(mime) -> list[str]:
        files: list[str] = []
        for url in mime.urls():
            if not url.isLocalFile():
                continue
            path = Path(url.toLocalFile())
            if path.is_dir():
                files.extend(
                    str(p) for p in path.iterdir() if p.is_file() and not p.name.startswith(".")
                )
            elif path.is_file():
                files.append(str(path))
        return files

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.setDropAction(Qt.DropAction.CopyAction)
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        files = self._dropped_files(event.mimeData()) if event.mimeData().hasUrls() else []
        if not files:
            event.ignore()
            return
        event.setDropAction(Qt.DropAction.CopyAction)
        event.accept()
        self.set_paths(files)

    def keyPressEvent(self, event):
        if event.matches(QKeySequence.StandardKey.Delete) or event.key() == Qt.Key.Key_Backspace:
            rows = {self.row(item) for item in self.selectedItems()}
            if rows:
                self.set_paths([p for i, p in enumerate(self._paths) if i not in rows])
                return
        super().keyPressEvent(event)


class FloatEdit(QLineEdit):
    """Line edit for a float (C locale, scientific notation allowed)."""

    def __init__(self, value: float | None = None, name: str = "value", parent=None):
        super().__init__(parent)
        self._name = name
        validator = QDoubleValidator(self)
        validator.setLocale(QLocale.c())
        validator.setNotation(QDoubleValidator.Notation.ScientificNotation)
        self.setValidator(validator)
        if value is not None:
            self.set_value(value)

    def set_value(self, value: float) -> None:
        self.setText(f"{value:g}")

    def value_or_none(self) -> float | None:
        text = self.text().strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            raise ValueError(f"{self._name}: {text!r} is not a number") from None

    def value(self) -> float:
        value = self.value_or_none()
        if value is None:
            raise ValueError(f"{self._name} is empty")
        return value


def parse_float(text: str) -> float | None:
    """A typed number (C locale), or None when the text is empty or not a number."""
    try:
        value = float(text.strip())
    except ValueError:
        return None
    return value if math.isfinite(value) else None


class EnergyEdit(FloatEdit):
    """Number field for an energy: shown in the display unit, kept in cm^-1.

    :meth:`set_unit` shows the kept value in another unit without changing it, so switching
    units back and forth never accumulates rounding. The settings protocol stores cm^-1
    (an empty field stores ""), so restoring it does not depend on the restored unit.
    ``valueChanged`` fires when the text changes the kept value.
    """

    valueChanged = Signal()

    def __init__(self, value_cm1: float | None = None, name: str = "value", parent=None):
        super().__init__(None, name, parent)
        self._unit = Unit.CM1
        self._cm1 = value_cm1
        self._rendering = False
        self.textChanged.connect(self._on_text)
        self._render()

    def unit(self) -> Unit:
        return self._unit

    def set_unit(self, unit: Unit | str) -> None:
        self._unit = Unit(unit)
        self._render()

    def cm1(self) -> float | None:
        """The kept value in cm^-1 (None when empty or not a number)."""
        return self._cm1

    def set_cm1(self, value: float | None) -> None:
        self._cm1 = None if value is None else float(value)
        self._render()
        self.valueChanged.emit()

    def set_value(self, value: float) -> None:
        """Set the value in the display unit."""
        self.set_cm1(float(to_cm1(value, self._unit)))

    def _render(self) -> None:
        self._rendering = True
        try:
            text = "" if self._cm1 is None else f"{float(from_cm1(self._cm1, self._unit)):.6g}"
            self.setText(text)
        finally:
            self._rendering = False

    def _on_text(self, text: str) -> None:
        if self._rendering:
            return
        value = parse_float(text)
        self._cm1 = None if value is None else float(to_cm1(value, self._unit))
        self.valueChanged.emit()

    def settings_value(self) -> float | str:
        return "" if self._cm1 is None else self._cm1

    def set_settings_value(self, value) -> bool:
        if value == "" or value is None:
            self.set_cm1(None)
            return True
        number = value if isinstance(value, float | int) else parse_float(str(value))
        if number is None or isinstance(number, bool) or not math.isfinite(number):
            return False
        self.set_cm1(float(number))
        return True


class CheckableSetting:
    """Settings protocol for a checkable QAction (``Persistence.bind`` accepts it)."""

    def __init__(self, action: QAction):
        self.action = action

    def settings_value(self) -> bool:
        return self.action.isChecked()

    def set_settings_value(self, value) -> bool:
        if isinstance(value, str) and value.lower() in ("true", "false"):
            value = value.lower() == "true"
        if not isinstance(value, bool):
            return False
        self.action.setChecked(value)
        return True


class Separator(QWidget):
    """A one-pixel line in the theme's line colour."""

    def __init__(self, orientation=Qt.Orientation.Horizontal, parent=None):
        super().__init__(parent)
        self._horizontal = orientation == Qt.Orientation.Horizontal
        if self._horizontal:
            self.setFixedHeight(1)
            self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        else:
            self.setFixedWidth(1)
            self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), current_tokens()["line"])
        painter.end()


def read_layout_items[T](layout: QLayout, read: Callable[[QLayoutItem], T]) -> list[T]:
    """``read(item)`` for each item of *layout*, in order.

    In a Qt layout class only Qt holds the items, and it deletes the ones that are no QObject
    (widget items, spacers) without telling PySide6, which keeps their Python wrappers and may
    later hand one back for a new object at the same address (on Linux a status bar's own
    layout came back as a spacer). So their wrappers are dropped once read; *read* must not
    keep the item. Items of a Python layout (e.g. :class:`FlowLayout`) are its own to keep.
    """
    qt_layout = type(layout).__module__.startswith("PySide6.")
    values = []
    for i in range(layout.count()):
        item = layout.itemAt(i)
        values.append(read(item))
        if qt_layout and not isinstance(item, QObject) and not shiboken6.createdByPython(item):
            shiboken6.invalidate(item)
    return values


class FlowLayout(QLayout):
    """Lays its items out left to right and wraps them onto new rows when space runs out."""

    def __init__(self, parent: QWidget | None = None, spacing: int = 6, row_spacing: int = 6):
        super().__init__(parent)
        self._items: list[QLayoutItem] = []
        self._spacing = spacing
        self._row_spacing = row_spacing
        self.setContentsMargins(0, 0, 0, 0)

    def addItem(self, item: QLayoutItem) -> None:
        self._items.append(item)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index: int) -> QLayoutItem | None:
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index: int) -> QLayoutItem | None:
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def expandingDirections(self) -> Qt.Orientation:
        return Qt.Orientation(0)

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        return self._arrange(QRect(0, 0, width, 0), apply=False)

    def setGeometry(self, rect: QRect) -> None:
        super().setGeometry(rect)
        self._arrange(rect, apply=True)

    def sizeHint(self) -> QSize:
        return self.minimumSize()

    def minimumSize(self) -> QSize:
        """The widest item (items are never squeezed below their size hint)."""
        size = QSize()
        for item in self._items:
            size = size.expandedTo(item.sizeHint())
        m = self.contentsMargins()
        return size + QSize(m.left() + m.right(), m.top() + m.bottom())

    def _arrange(self, rect: QRect, apply: bool) -> int:
        """Place the items (centred vertically in their row); returns the height used."""
        m = self.contentsMargins()
        area = rect.adjusted(m.left(), m.top(), -m.right(), -m.bottom())
        rows: list[list[tuple[QLayoutItem, QSize]]] = [[]]
        x = area.x()
        for item in self._items:
            if item.isEmpty():
                continue
            hint = item.sizeHint()
            if rows[-1] and x + hint.width() > area.right() + 1:
                rows.append([])
                x = area.x()
            rows[-1].append((item, hint))
            x += hint.width() + self._spacing
        y = area.y()
        for row in rows:
            height = max((hint.height() for _item, hint in row), default=0)
            x = area.x()
            for item, hint in row:
                if apply:
                    item.setGeometry(QRect(QPoint(x, y + (height - hint.height()) // 2), hint))
                x += hint.width() + self._spacing
            y += height + self._row_spacing
        used = y - self._row_spacing if rows[0] else y
        return used - rect.y() + m.bottom()


class SliderSpin(QWidget):
    """A spin box with a slider next to it for quick, live changes."""

    valueChanged = Signal(float)

    def __init__(self, minimum: float, maximum: float, step: float, value: float, suffix=""):
        super().__init__()
        self._step = step
        self.spin = QDoubleSpinBox()
        self.spin.setLocale(QLocale.c())  # same decimal point as the other number fields
        self.spin.setRange(minimum, maximum)
        self.spin.setSingleStep(step)
        self.spin.setDecimals(max(0, -math.floor(math.log10(step))))
        self.spin.setSuffix(suffix)
        self.spin.setValue(value)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(round(minimum / step), round(maximum / step))
        self.slider.setValue(round(value / step))
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.slider, stretch=1)
        layout.addWidget(self.spin)
        self.slider.valueChanged.connect(lambda i: self.spin.setValue(i * self._step))
        self.spin.valueChanged.connect(self._on_spin)

    def _on_spin(self, value: float) -> None:
        self.slider.blockSignals(True)
        self.slider.setValue(round(value / self._step))
        self.slider.blockSignals(False)
        self.valueChanged.emit(value)

    def value(self) -> float:
        return self.spin.value()

    def setValue(self, value: float) -> None:
        self.spin.setValue(value)


class PointTableModel(QAbstractTableModel):
    """Read-only view of a :class:`PointTable` (rows = field, columns = curves).

    The table keeps cm^-1; the energies are shown in :meth:`unit`.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._table: PointTable | None = None
        self._unit = Unit.CM1

    def table(self) -> PointTable | None:
        return self._table

    def unit(self) -> Unit:
        return self._unit

    def set_unit(self, unit: Unit | str) -> None:
        unit = Unit(unit)
        if unit is not self._unit:
            self.beginResetModel()
            self._unit = unit
            self.endResetModel()

    def set_table(self, table: PointTable | None) -> None:
        self.beginResetModel()
        self._table = table
        self.endResetModel()

    def refresh(self) -> None:
        self.beginResetModel()
        self.endResetModel()

    def rowCount(self, parent: QModelIndex | QPersistentModelIndex = _ROOT) -> int:
        if parent.isValid() or self._table is None:
            return 0
        return self._table.field.size

    def columnCount(self, parent: QModelIndex | QPersistentModelIndex = _ROOT) -> int:
        if parent.isValid() or self._table is None:
            return 0
        return len(self._table.names)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if self._table is None or not index.isValid():
            return None
        if role in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.ToolTipRole):
            value = self._table.column(self._table.names[index.column()])[index.row()]
            return "" if math.isnan(value) else f"{from_cm1(value, self._unit):.5g}"
        if role == Qt.ItemDataRole.TextAlignmentRole:
            return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        return None

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if self._table is None or role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            return self._table.names[section]
        return f"{self._table.field[section]:g} T"
