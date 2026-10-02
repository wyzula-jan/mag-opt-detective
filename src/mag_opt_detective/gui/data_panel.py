"""Left panel: global options, the data tabs and the console."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGridLayout,
    QLabel,
    QPushButton,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from mag_opt_detective.core.units import Unit
from mag_opt_detective.gui.console import ConsoleWidget
from mag_opt_detective.gui.measurement_tab import MeasurementTab
from mag_opt_detective.gui.processed_tab import ProcessedTab
from mag_opt_detective.gui.tools_tab import ToolsTab

FIELD_FROM_NAMES = "Field from file names"
FIELD_CUSTOM = "Custom field range"


class DataPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        top = QGridLayout()
        self.field_source = QComboBox()
        self.field_source.addItems([FIELD_FROM_NAMES, FIELD_CUSTOM])
        self.field_source.setToolTip(
            "File type (OPUS or text) is detected automatically.\n"
            "Field from names expects e.g. ..._a01p250T.txt (= 1.25 T)."
        )
        self.unit = QComboBox()
        self.unit.addItems([u.value for u in Unit])
        self.process_button = QPushButton("Process")
        self.export_button = QPushButton("Export…")
        self.export_type_suffix = QCheckBox("Add data type to export name")
        self.export_type_suffix.setChecked(True)
        top.addWidget(QLabel("Field:"), 0, 0)
        top.addWidget(self.field_source, 0, 1, 1, 2)
        top.addWidget(QLabel("Energy unit:"), 1, 0)
        top.addWidget(self.unit, 1, 1)
        top.addWidget(self.process_button, 1, 2)
        top.addWidget(self.export_type_suffix, 2, 0, 1, 2)
        top.addWidget(self.export_button, 2, 2)
        top.setColumnStretch(1, 1)
        layout.addLayout(top)

        self.tabs = QTabWidget()
        self.sample = MeasurementTab()
        self.reference = MeasurementTab(reference=True)
        self.tools = ToolsTab()
        self.processed = ProcessedTab()
        self.tabs.addTab(self.sample, "Sample")
        self.tabs.addTab(self.reference, "Reference")
        self.tabs.addTab(self.tools, "Tools")
        self.tabs.addTab(self.processed, "Processed")

        console_box = QWidget()
        console_layout = QVBoxLayout(console_box)
        console_layout.setContentsMargins(0, 0, 0, 0)
        console_layout.addWidget(QLabel("Console"))
        self.console = ConsoleWidget()
        console_layout.addWidget(self.console)

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(self.tabs)
        splitter.addWidget(console_box)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter, stretch=1)

        self.field_source.currentIndexChanged.connect(self._on_field_source_changed)
        self._on_field_source_changed()

    def _on_field_source_changed(self) -> None:
        custom = self.custom_field()
        self.sample.set_custom_field_enabled(custom)
        self.reference.set_custom_field_enabled(custom)

    def custom_field(self) -> bool:
        return self.field_source.currentText() == FIELD_CUSTOM

    def energy_unit(self) -> Unit:
        return Unit(self.unit.currentText())
