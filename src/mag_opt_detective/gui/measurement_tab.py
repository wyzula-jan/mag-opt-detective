"""Sample / Reference tabs: file lists and field range."""

from __future__ import annotations

import numpy as np
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from mag_opt_detective.core.pipeline import ReferenceMode
from mag_opt_detective.gui.widgets import FileListWidget, FloatEdit, open_files


class FieldRangeBox(QGroupBox):
    """Start / step / end of a custom field range (used when names carry no field)."""

    def __init__(self, parent=None):
        super().__init__("Custom field range (T)", parent)
        self.start = FloatEdit(0.25, "Start B")
        self.step = FloatEdit(0.25, "Step B")
        self.end = FloatEdit(16.0, "End B")
        layout = QGridLayout(self)
        for col, (text, edit) in enumerate(
            [("Start", self.start), ("Step", self.step), ("End", self.end)]
        ):
            layout.addWidget(QLabel(text), 0, col)
            layout.addWidget(edit, 1, col)

    def field(self) -> np.ndarray:
        start, step, end = self.start.value(), self.step.value(), self.end.value()
        if step <= 0:
            raise ValueError("field step must be positive")
        if end < start:
            raise ValueError("end field must not be smaller than start field")
        n = round((end - start) / step) + 1
        return start + step * np.arange(n)


class MeasurementTab(QWidget):
    """Zero-field and in-field file lists of one measurement (sample or reference)."""

    def __init__(self, reference: bool = False, parent=None):
        super().__init__(parent)
        self.is_reference = reference
        layout = QVBoxLayout(self)

        self.field_range = FieldRangeBox()
        layout.addWidget(self.field_range)

        if reference:
            layout.addWidget(self._build_reference_box())

        self.zero_list = FileListWidget()
        self.field_list = FileListWidget()
        self.load_zero_button = QPushButton("Load Zero Field…")
        self.load_field_button = QPushButton("Load Field…")
        self.load_zero_button.clicked.connect(self.load_zero_dialog)
        self.load_field_button.clicked.connect(self.load_field_dialog)

        for button, file_list, hint in (
            (self.load_zero_button, self.zero_list, "1 file, or 2 (before and after the sweep)"),
            (self.load_field_button, self.field_list, ""),
        ):
            row = QHBoxLayout()
            row.addWidget(button)
            clear = QPushButton("Clear")
            clear.clicked.connect(file_list.clear_paths)
            row.addWidget(clear)
            count = QLabel(hint)
            count.setEnabled(False)
            file_list.pathsChanged.connect(
                lambda c=count, fl=file_list, h=hint: c.setText(
                    f"{fl.count()} files" if fl.count() else h
                )
            )
            row.addWidget(count)
            row.addStretch(1)
            layout.addLayout(row)
            if hint:
                file_list.setToolTip(f"{hint}. Drop files here (Del removes the selected ones)")
            layout.addWidget(file_list, stretch=1 if file_list is self.field_list else 0)
        self.zero_list.setMaximumHeight(70)

    def _build_reference_box(self) -> QGroupBox:
        box = QGroupBox("Reference")
        grid = QGridLayout(box)
        self.ref_none = QRadioButton("No reference")
        self.ref_separate = QRadioButton("Reference correction")
        self.ref_self = QRadioButton("Use data as reference")
        self.ref_none.setChecked(True)
        self.ref_group = QButtonGroup(self)
        for i, button in enumerate((self.ref_none, self.ref_separate, self.ref_self)):
            self.ref_group.addButton(button)
            grid.addWidget(button, 0, i)
        self.smooth = QCheckBox("SG smooth")
        self.sg_window = QSpinBox()
        self.sg_window.setRange(3, 9999)
        self.sg_window.setSingleStep(2)
        self.sg_window.setValue(11)
        self.sg_poly = QSpinBox()
        self.sg_poly.setRange(0, 10)
        self.sg_poly.setValue(2)
        grid.addWidget(self.smooth, 1, 0)
        window_row = QHBoxLayout()
        window_row.addWidget(QLabel("Window"))
        window_row.addWidget(self.sg_window)
        grid.addLayout(window_row, 1, 1)
        poly_row = QHBoxLayout()
        poly_row.addWidget(QLabel("Poly order"))
        poly_row.addWidget(self.sg_poly)
        grid.addLayout(poly_row, 1, 2)
        return box

    def set_custom_field_enabled(self, enabled: bool) -> None:
        self.field_range.setEnabled(enabled)

    def zero_paths(self) -> list[str]:
        return self.zero_list.paths()

    def field_paths(self) -> list[str]:
        return self.field_list.paths()

    def load_zero_dialog(self) -> None:
        paths = open_files(self, "Load zero-field spectra")
        if paths:
            self.zero_list.set_paths(paths)

    def load_field_dialog(self) -> None:
        paths = open_files(self, "Load in-field spectra")
        if paths:
            self.field_list.set_paths(paths)

    # reference options -------------------------------------------------
    def reference_mode(self) -> ReferenceMode:
        if not self.is_reference or self.ref_none.isChecked():
            return ReferenceMode.NONE
        if self.ref_separate.isChecked():
            return ReferenceMode.SEPARATE
        return ReferenceMode.SELF

    def set_reference_mode(self, mode: ReferenceMode) -> None:
        {
            ReferenceMode.NONE: self.ref_none,
            ReferenceMode.SEPARATE: self.ref_separate,
            ReferenceMode.SELF: self.ref_self,
        }[mode].setChecked(True)
