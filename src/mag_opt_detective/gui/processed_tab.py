"""Processed tab: slots of processed maps that can be re-plotted and merged."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QGridLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

N_SLOTS = 16


# table columns
COL_USE, COL_NAME, COL_EMIN, COL_EMAX, COL_BMIN, COL_BMAX = range(6)
HEADERS = ["Use", "Dataset", "E min", "E max", "B min", "B max"]


class ProcessedTab(QWidget):
    """Table of slots (use, name, energy and field cut) plus the actions working on them."""

    loadRequested = Signal(int)
    saveRequested = Signal(int)
    plotRequested = Signal(int)
    mergeEnergyRequested = Signal()
    mergeFieldRequested = Signal()
    averageRequested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        grid = QGridLayout()

        self.slot_spin = QSpinBox()
        self.slot_spin.setRange(0, N_SLOTS - 1)
        self.slot_spin.setPrefix("Slot ")
        self.load_button = QPushButton("Load to Slot…")
        self.save_button = QPushButton("Save Ratio to Slot")
        self.plot_button = QPushButton("Plot Slot")
        self.merge_energy_button = QPushButton("Merge by Energy")
        self.merge_energy_button.setToolTip(
            "Join slots measured in different spectral ranges, each cut to E min – max"
        )
        self.merge_field_button = QPushButton("Merge by Field")
        self.merge_field_button.setToolTip(
            "Join slots measured over different field ranges, each cut to B min – max"
        )
        self.average_button = QPushButton("Average")
        self.average_button.setToolTip(
            "Average repeated measurements in the ticked slots (same field values)"
        )
        self.full_energy = QCheckBox("Full energy")
        self.full_energy.setChecked(True)
        self.full_energy.setToolTip("Unchecked: Plot Slot uses the E min / E max of the slot")
        self.auto_field = QCheckBox("Field from header")
        self.auto_field.setChecked(True)
        self.auto_field.setToolTip("Unchecked: use the custom field range of the Sample tab")

        grid.addWidget(self.slot_spin, 0, 0)
        grid.addWidget(self.load_button, 0, 1)
        grid.addWidget(self.save_button, 0, 2)
        grid.addWidget(self.plot_button, 0, 3)
        grid.addWidget(self.full_energy, 1, 0)
        grid.addWidget(self.auto_field, 1, 1)
        grid.addWidget(self.merge_energy_button, 1, 2)
        grid.addWidget(self.merge_field_button, 1, 3)
        grid.addWidget(self.average_button, 2, 3)
        layout.addLayout(grid)

        hint = QLabel(
            "Load exported R(B)/R(0) tables. Merging and averaging use the slots ticked in Use, "
            "each cut to its E and B range (empty = no cut)."
        )
        hint.setWordWrap(True)
        hint.setEnabled(False)
        layout.addWidget(hint)

        self.table = QTableWidget(N_SLOTS, len(HEADERS))
        self.table.setHorizontalHeaderLabels(HEADERS)
        self.table.setVerticalHeaderLabels([str(i) for i in range(N_SLOTS)])
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(COL_NAME, QHeaderView.ResizeMode.Stretch)
        for row in range(N_SLOTS):
            use = QTableWidgetItem()
            use.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
            use.setCheckState(Qt.CheckState.Unchecked)
            self.table.setItem(row, COL_USE, use)
            name = QTableWidgetItem("")
            name.setFlags(name.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(row, COL_NAME, name)
            for col in (COL_EMIN, COL_EMAX, COL_BMIN, COL_BMAX):
                self.table.setItem(row, col, QTableWidgetItem(""))
        self.table.currentCellChanged.connect(lambda row, *_: self.slot_spin.setValue(max(row, 0)))
        layout.addWidget(self.table, stretch=1)

        self.load_button.clicked.connect(lambda: self.loadRequested.emit(self.slot()))
        self.save_button.clicked.connect(lambda: self.saveRequested.emit(self.slot()))
        self.plot_button.clicked.connect(lambda: self.plotRequested.emit(self.slot()))
        self.merge_energy_button.clicked.connect(self.mergeEnergyRequested)
        self.merge_field_button.clicked.connect(self.mergeFieldRequested)
        self.average_button.clicked.connect(self.averageRequested)

    def slot(self) -> int:
        return self.slot_spin.value()

    def set_slot_name(self, slot: int, name: str) -> None:
        """Name a filled slot; filling a slot also ticks its Use box."""
        self.table.item(slot, COL_NAME).setText(name)
        self.table.item(slot, COL_NAME).setToolTip(name)
        self.set_used(slot, True)

    def slot_name(self, slot: int) -> str:
        return self.table.item(slot, COL_NAME).text()

    def is_used(self, slot: int) -> bool:
        return self.table.item(slot, COL_USE).checkState() == Qt.CheckState.Checked

    def set_used(self, slot: int, used: bool) -> None:
        state = Qt.CheckState.Checked if used else Qt.CheckState.Unchecked
        self.table.item(slot, COL_USE).setCheckState(state)

    def used_slots(self, filled) -> list[int]:
        """The filled slots that are ticked in the Use column, in slot order."""
        return [slot for slot in sorted(filled) if self.is_used(slot)]

    def _range(self, slot: int, cols, labels) -> tuple[float | None, float | None]:
        values = []
        for col, label in zip(cols, labels, strict=True):
            text = self.table.item(slot, col).text().strip()
            try:
                values.append(float(text) if text else None)
            except ValueError:
                raise ValueError(f"slot {slot}: {label} {text!r} is not a number") from None
        return values[0], values[1]

    def energy_range(self, slot: int) -> tuple[float | None, float | None]:
        """E min / E max as typed (in the panel's energy unit; None = no cut)."""
        return self._range(slot, (COL_EMIN, COL_EMAX), ("E min", "E max"))

    def field_range(self, slot: int) -> tuple[float | None, float | None]:
        return self._range(slot, (COL_BMIN, COL_BMAX), ("B min", "B max"))

    def _set_range(self, slot: int, cols, lo: float | None, hi: float | None) -> None:
        for col, value in zip(cols, (lo, hi), strict=True):
            self.table.item(slot, col).setText("" if value is None else f"{value:g}")

    def set_energy_range(self, slot: int, lo: float | None, hi: float | None) -> None:
        self._set_range(slot, (COL_EMIN, COL_EMAX), lo, hi)

    def set_field_range(self, slot: int, lo: float | None, hi: float | None) -> None:
        self._set_range(slot, (COL_BMIN, COL_BMAX), lo, hi)
