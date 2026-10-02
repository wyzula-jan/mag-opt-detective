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


class ProcessedTab(QWidget):
    """Table of slots (name, E min, E max) plus the actions working on them."""

    loadRequested = Signal(int)
    saveRequested = Signal(int)
    plotRequested = Signal(int)
    mergeRequested = Signal()

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
        self.merge_button = QPushButton("Merge by Energy")
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
        grid.addWidget(self.merge_button, 1, 3)
        layout.addLayout(grid)

        hint = QLabel(
            "Load exported R(B)/R(0) tables. Merge joins all filled slots, each cut to E min – max."
        )
        hint.setWordWrap(True)
        hint.setEnabled(False)
        layout.addWidget(hint)

        self.table = QTableWidget(N_SLOTS, 3)
        self.table.setHorizontalHeaderLabels(["Dataset", "E min", "E max"])
        self.table.setVerticalHeaderLabels([str(i) for i in range(N_SLOTS)])
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for row in range(N_SLOTS):
            item = QTableWidgetItem("")
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(row, 0, item)
            for col in (1, 2):
                self.table.setItem(row, col, QTableWidgetItem(""))
        self.table.currentCellChanged.connect(lambda row, *_: self.slot_spin.setValue(max(row, 0)))
        layout.addWidget(self.table, stretch=1)

        self.load_button.clicked.connect(lambda: self.loadRequested.emit(self.slot()))
        self.save_button.clicked.connect(lambda: self.saveRequested.emit(self.slot()))
        self.plot_button.clicked.connect(lambda: self.plotRequested.emit(self.slot()))
        self.merge_button.clicked.connect(self.mergeRequested)

    def slot(self) -> int:
        return self.slot_spin.value()

    def set_slot_name(self, slot: int, name: str) -> None:
        self.table.item(slot, 0).setText(name)
        self.table.item(slot, 0).setToolTip(name)

    def slot_name(self, slot: int) -> str:
        return self.table.item(slot, 0).text()

    def energy_range(self, slot: int) -> tuple[float | None, float | None]:
        values = []
        for col, label in ((1, "E min"), (2, "E max")):
            text = self.table.item(slot, col).text().strip()
            try:
                values.append(float(text) if text else None)
            except ValueError:
                raise ValueError(f"slot {slot}: {label} {text!r} is not a number") from None
        return values[0], values[1]

    def set_energy_range(self, slot: int, lo: float | None, hi: float | None) -> None:
        for col, value in ((1, lo), (2, hi)):
            self.table.item(slot, col).setText("" if value is None else f"{value:g}")
