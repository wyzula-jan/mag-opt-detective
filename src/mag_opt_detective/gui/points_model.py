"""Table model of the Points panel: the current curve's points, sorted by field."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QAbstractTableModel, QModelIndex, QPersistentModelIndex, Qt

from mag_opt_detective.core.units import Unit, from_cm1
from mag_opt_detective.gui.display import unit_text

_ROOT = QModelIndex()


class CurvePointsModel(QAbstractTableModel):
    """Rows of (B in T, E in :meth:`unit`, remove) for one curve, sorted by B.

    The energies are kept in cm^-1; a unit switch only changes what is shown. The third
    column holds no text: the view draws a remove button there (:meth:`field_at` gives the
    row's field).
    """

    FIELD, ENERGY, REMOVE = range(3)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._field = np.array([])
        self._energy = np.array([])  # cm^-1
        self._unit = Unit.CM1

    def unit(self) -> Unit:
        return self._unit

    def set_unit(self, unit: Unit | str) -> None:
        unit = Unit(unit)
        if unit is not self._unit:
            self._unit = unit
            if self._field.size:
                last = self.index(len(self) - 1, self.ENERGY)
                self.dataChanged.emit(self.index(0, self.ENERGY), last)
            self.headerDataChanged.emit(Qt.Orientation.Horizontal, self.ENERGY, self.ENERGY)

    def set_points(self, field: np.ndarray, energy_cm1: np.ndarray) -> bool:
        """Show the points (*field* in T, *energy_cm1* in cm^-1), in any order.

        Returns False (and leaves the model alone) if exactly these points are shown already.
        """
        order = np.argsort(field, kind="stable")
        field = np.asarray(field, dtype=float)[order]
        energy = np.asarray(energy_cm1, dtype=float)[order]
        if np.array_equal(field, self._field) and np.array_equal(energy, self._energy):
            return False
        self.beginResetModel()
        self._field, self._energy = field, energy
        self.endResetModel()
        return True

    def field_at(self, row: int) -> float:
        return float(self._field[row])

    def energy_at(self, row: int) -> float:
        """Energy of *row* in the display unit."""
        return float(from_cm1(self._energy[row], self._unit))

    def __len__(self) -> int:
        return int(self._field.size)

    def rowCount(self, parent: QModelIndex | QPersistentModelIndex = _ROOT) -> int:
        return 0 if parent.isValid() else len(self)

    def columnCount(self, parent: QModelIndex | QPersistentModelIndex = _ROOT) -> int:
        return 0 if parent.isValid() else 3

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        row, column = index.row(), index.column()
        if role == Qt.ItemDataRole.DisplayRole:
            if column == self.FIELD:
                return f"{self.field_at(row):.6g}"
            if column == self.ENERGY:
                return f"{self.energy_at(row):.5g}"
            return None
        if role in (Qt.ItemDataRole.ToolTipRole, Qt.ItemDataRole.AccessibleTextRole):
            if column == self.REMOVE:
                return f"Remove the point at B = {self.field_at(row):g} T"
            return self.data(index, Qt.ItemDataRole.DisplayRole)
        if role == Qt.ItemDataRole.TextAlignmentRole:
            return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        if role == Qt.ItemDataRole.UserRole:
            return self.field_at(row)
        return None

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if orientation != Qt.Orientation.Horizontal:
            return None
        if role == Qt.ItemDataRole.DisplayRole:
            return ("B (T)", f"E ({unit_text(self._unit)})", "")[section]
        if role == Qt.ItemDataRole.TextAlignmentRole and section != self.REMOVE:
            return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        return None
