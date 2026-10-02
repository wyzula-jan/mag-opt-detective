"""Right panel: plot selection and the plot tabs."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from mag_opt_detective.core.pipeline import PlotKind
from mag_opt_detective.core.processing import Axis
from mag_opt_detective.gui.plots import ColorMapPlot, StackedPlot
from mag_opt_detective.gui.widgets import FloatEdit


class PlotPanel(QWidget):
    selectionChanged = Signal()  # plot type, derivative or stacked options changed
    referenceSelectionChanged = Signal()

    KINDS = (
        (PlotKind.RATIO, "R(B)/R(0)", "Ctrl+1"),
        (PlotKind.DATA, "Data", "Ctrl+2"),
        (PlotKind.AVERAGE, "R(B)/R(B-AVR)", "Ctrl+3"),
    )
    ORDERS = ((0, "No", "Alt+1"), (1, "1st", "Alt+2"), (2, "2nd", "Alt+3"))
    AUTO_COLOURS = "Auto"
    COLOURS = (AUTO_COLOURS, "magma", "viridis", "inferno", "plasma", "turbo", "grey", "bipolar")

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        grid = QGridLayout()
        self.kind_group = QButtonGroup(self)
        self.kind_buttons: dict[PlotKind, QRadioButton] = {}
        grid.addWidget(QLabel("Plot type:"), 0, 0)
        for col, (kind, text, shortcut) in enumerate(self.KINDS, start=1):
            button = QRadioButton(text)
            button.setShortcut(QKeySequence(shortcut))
            button.setToolTip(shortcut)
            self.kind_group.addButton(button)
            self.kind_buttons[kind] = button
            grid.addWidget(button, 0, col)
        self.kind_buttons[PlotKind.RATIO].setChecked(True)
        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems(self.COLOURS)
        self.cmap_combo.setToolTip("Colour map. Auto: magma for maps, grey for derivatives")
        grid.addWidget(QLabel("Colours:"), 0, len(self.KINDS) + 2, Qt.AlignmentFlag.AlignRight)
        grid.addWidget(self.cmap_combo, 0, len(self.KINDS) + 3)

        self.order_group = QButtonGroup(self)
        self.order_buttons: dict[int, QRadioButton] = {}
        grid.addWidget(QLabel("Derivative:"), 1, 0)
        for col, (order, text, shortcut) in enumerate(self.ORDERS, start=1):
            button = QRadioButton(text)
            button.setShortcut(QKeySequence(shortcut))
            button.setToolTip(shortcut)
            self.order_group.addButton(button)
            self.order_buttons[order] = button
            grid.addWidget(button, 1, col)
        self.order_buttons[0].setChecked(True)
        self.axis_combo = QComboBox()
        self.axis_combo.addItems(["along Energy", "along Field"])
        grid.addWidget(self.axis_combo, 1, len(self.ORDERS) + 1)
        self.per_unit = QCheckBox("per unit")
        self.per_unit.setToolTip(
            "Divide by the real step: d/dE per energy unit or d/dB per tesla.\n"
            "Off: per data point, as in the old versions (matches the Tools limits).\n"
            "Derivatives per unit are colour-scaled automatically."
        )
        grid.addWidget(self.per_unit, 1, len(self.ORDERS) + 2)
        grid.setColumnStretch(len(self.ORDERS) + 3, 1)
        layout.addLayout(grid)

        self.tabs = QTabWidget()
        self.color_map = ColorMapPlot()
        self.tabs.addTab(self.color_map, "Color Map")

        stacked_tab = QWidget()
        stacked_layout = QVBoxLayout(stacked_tab)
        row = QHBoxLayout()
        self.stacked_enabled = QCheckBox("Enable")
        self.stacked_enabled.setChecked(True)
        self.offset = FloatEdit(0.01, "Offset")
        self.offset.setMaximumWidth(100)
        self.replot_button = QPushButton("Replot")
        row.addWidget(self.stacked_enabled)
        row.addWidget(QLabel("Offset:"))
        row.addWidget(self.offset)
        row.addWidget(self.replot_button)
        row.addStretch(1)
        stacked_layout.addLayout(row)
        self.stacked = StackedPlot()
        stacked_layout.addWidget(self.stacked, stretch=1)
        self.tabs.addTab(stacked_tab, "Stacked Plot")

        reference_tab = QWidget()
        reference_layout = QVBoxLayout(reference_tab)
        row = QHBoxLayout()
        self.ref_ratio = QRadioButton("Reference R(B)/R(0)")
        self.ref_data = QRadioButton("Reference data")
        self.ref_ratio.setChecked(True)
        self.ref_group = QButtonGroup(self)
        self.ref_group.addButton(self.ref_ratio)
        self.ref_group.addButton(self.ref_data)
        row.addWidget(self.ref_ratio)
        row.addWidget(self.ref_data)
        row.addStretch(1)
        reference_layout.addLayout(row)
        self.reference_map = ColorMapPlot()
        reference_layout.addWidget(self.reference_map, stretch=1)
        self.tabs.addTab(reference_tab, "Reference")
        layout.addWidget(self.tabs, stretch=1)

        for group in (self.kind_group, self.order_group):
            group.buttonToggled.connect(
                lambda _b, checked: checked and self.selectionChanged.emit()
            )
        self.axis_combo.currentIndexChanged.connect(self.selectionChanged)
        self.per_unit.toggled.connect(self.selectionChanged)
        self.cmap_combo.currentIndexChanged.connect(self.selectionChanged)
        self.cmap_combo.currentIndexChanged.connect(self.referenceSelectionChanged)
        self.stacked_enabled.toggled.connect(self.selectionChanged)
        self.replot_button.clicked.connect(self.selectionChanged)
        self.offset.editingFinished.connect(self.selectionChanged)
        self.ref_group.buttonToggled.connect(
            lambda _b, checked: checked and self.referenceSelectionChanged.emit()
        )

    def kind(self) -> PlotKind:
        return next(k for k, b in self.kind_buttons.items() if b.isChecked())

    def order(self) -> int:
        return next(o for o, b in self.order_buttons.items() if b.isChecked())

    def axis(self) -> Axis:
        return Axis.FIELD if self.axis_combo.currentIndex() == 1 else Axis.ENERGY

    def colormap(self, order: int = 0) -> str:
        choice = self.cmap_combo.currentText()
        if choice == self.AUTO_COLOURS:
            return "magma" if order == 0 else "grey"
        return choice

    def current_view(self):
        """The plot widget of the visible tab."""
        return (self.color_map, self.stacked, self.reference_map)[self.tabs.currentIndex()]

    def physical(self) -> bool:
        return self.per_unit.isChecked()

    def reference_kind(self) -> PlotKind:
        return PlotKind.DATA if self.ref_data.isChecked() else PlotKind.RATIO
