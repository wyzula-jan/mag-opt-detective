"""Small reusable widgets."""

from __future__ import annotations

import math
from pathlib import Path

from PySide6.QtCore import (
    QAbstractTableModel,
    QLocale,
    QModelIndex,
    QPersistentModelIndex,
    Qt,
    Signal,
)
from PySide6.QtGui import QDoubleValidator, QKeySequence
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QSlider,
    QWidget,
)

from mag_opt_detective.core.points import PointTable
from mag_opt_detective.core.readers import sort_paths
from mag_opt_detective.core.units import Unit, from_cm1

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
