"""Tools tab: plot limits, baseline correction and the point extractor."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from PySide6.QtCore import QRegularExpression, Signal
from PySide6.QtGui import QFont, QRegularExpressionValidator
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QStackedWidget,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from mag_opt_detective.core.pipeline import PlotKind
from mag_opt_detective.gui.widgets import FloatEdit, PointTableModel, SliderSpin

Range = tuple[float, float]


class EnergyMode(Enum):
    AUTO = "auto"
    CUSTOM = "custom"  # only the view range
    CUT = "cut"  # data outside the range is dropped when processing


class PointMode(Enum):
    OFF = "off"
    RECORD = "record"
    REMOVE = "remove"


def level_key(kind: PlotKind | str, order: int = 0) -> str:
    """Key of the intensity-limit row for a plot: derivatives share one row per order."""
    return {1: "der1", 2: "der2"}.get(order, str(kind))


@dataclass(frozen=True)
class PlotLimits:
    field_range: Range | None = None
    energy_mode: EnergyMode = EnergyMode.AUTO
    energy_range: Range = (0.0, 200.0)
    custom_levels: bool = True
    levels: dict[str, Range] = field(default_factory=dict)
    stacked_range: Range | None = None

    def levels_for(self, kind: PlotKind | str, order: int = 0) -> Range | None:
        if not self.custom_levels:
            return None
        return self.levels.get(level_key(kind, order))

    @property
    def energy_view(self) -> Range | None:
        return None if self.energy_mode is EnergyMode.AUTO else self.energy_range

    @property
    def energy_cut(self) -> tuple[float | None, float | None]:
        return self.energy_range if self.energy_mode is EnergyMode.CUT else (None, None)


def _title(text: str) -> QLabel:
    label = QLabel(text)
    font = QFont(label.font())
    font.setBold(True)
    label.setFont(font)
    return label


def _radio_group(parent, *labels: str, checked: int = 0) -> tuple[QButtonGroup, list[QRadioButton]]:
    group = QButtonGroup(parent)
    buttons = [QRadioButton(text) for text in labels]
    for i, button in enumerate(buttons):
        group.addButton(button, i)
    buttons[checked].setChecked(True)
    return group, buttons


class PlotLimitsPage(QWidget):
    """Field, energy and intensity ranges of the plots."""

    changed = Signal()

    _LEVEL_ROWS = (
        (str(PlotKind.RATIO), "R(B)/R(0)", 0.9, 1.1),
        (str(PlotKind.DATA), "Data", 0.0, 2.0),
        (str(PlotKind.AVERAGE), "R(B)/R(B-AVR)", 0.9, 1.1),
        (str(PlotKind.STEP), "R(B)/R(B-ΔB)", 0.98, 1.02),
        ("der1", "1st derivative", -0.01, 0.01),
        ("der2", "2nd derivative", -0.001, 0.001),
    )

    def __init__(self, parent=None):
        super().__init__(parent)
        grid = QGridLayout(self)
        row = 0

        grid.addWidget(_title("Magnetic field range"), row, 0, 1, 3)
        row += 1
        self.field_group, (self.field_auto, self.field_custom) = _radio_group(
            self, "Autoscale", "Custom"
        )
        grid.addWidget(self.field_auto, row, 1)
        grid.addWidget(self.field_custom, row, 2)
        row += 1
        self.field_min, self.field_max = FloatEdit(0, "B min"), FloatEdit(16, "B max")
        row = self._range_row(grid, row, "B min – max", self.field_min, self.field_max)

        grid.addWidget(_title("Energy range"), row, 0, 1, 3)
        row += 1
        self.energy_group, (self.energy_auto, self.energy_custom, self.energy_cut) = _radio_group(
            self, "Autoscale", "Custom", "Cut"
        )
        self.energy_cut.setToolTip("Drop data outside the range when processing")
        for col, button in enumerate((self.energy_auto, self.energy_custom, self.energy_cut)):
            grid.addWidget(button, row, col)
        row += 1
        self.energy_min, self.energy_max = FloatEdit(0, "E min"), FloatEdit(200, "E max")
        row = self._range_row(grid, row, "E min – max", self.energy_min, self.energy_max)

        grid.addWidget(_title("Colour map intensity"), row, 0, 1, 3)
        row += 1
        self.level_group, (self.level_auto, self.level_custom) = _radio_group(
            self, "Autoscale", "Custom", checked=1
        )
        grid.addWidget(self.level_auto, row, 1)
        grid.addWidget(self.level_custom, row, 2)
        row += 1
        self.level_edits: dict[str, tuple[FloatEdit, FloatEdit]] = {}
        for key, text, lo, hi in self._LEVEL_ROWS:
            edits = FloatEdit(lo, f"{text} min"), FloatEdit(hi, f"{text} max")
            self.level_edits[key] = edits
            row = self._range_row(grid, row, text, *edits)

        grid.addWidget(_title("Stacked plot intensity"), row, 0, 1, 3)
        row += 1
        self.stacked_group, (self.stacked_auto, self.stacked_custom) = _radio_group(
            self, "Autoscale", "Custom", checked=1
        )
        grid.addWidget(self.stacked_auto, row, 1)
        grid.addWidget(self.stacked_custom, row, 2)
        row += 1
        self.stacked_min, self.stacked_max = FloatEdit(0.9, "Y min"), FloatEdit(3, "Y max")
        row = self._range_row(grid, row, "Y min – max", self.stacked_min, self.stacked_max)
        grid.setRowStretch(row, 1)

        for group in (self.field_group, self.energy_group, self.level_group, self.stacked_group):
            group.buttonToggled.connect(lambda _b, checked: checked and self.changed.emit())

    def set_levels(self, key: str, lo: float, hi: float) -> None:
        """Store a colour range (e.g. dragged on the histogram) and switch to Custom."""
        lo_edit, hi_edit = self.level_edits[key]
        lo_edit.set_value(lo)
        hi_edit.set_value(hi)
        if self.level_custom.isChecked():
            self.changed.emit()
        else:
            self.level_custom.setChecked(True)  # emits changed

    def level_label(self, key: str) -> str:
        return next(text for k, text, _lo, _hi in self._LEVEL_ROWS if k == key)

    def _range_row(self, grid: QGridLayout, row: int, text: str, lo: FloatEdit, hi: FloatEdit):
        grid.addWidget(QLabel(text), row, 0)
        grid.addWidget(lo, row, 1)
        grid.addWidget(hi, row, 2)
        lo.editingFinished.connect(self.changed)
        hi.editingFinished.connect(self.changed)
        return row + 1

    @staticmethod
    def _range(lo: FloatEdit, hi: FloatEdit) -> Range:
        a, b = lo.value(), hi.value()
        if a >= b:
            raise ValueError(f"invalid range {a:g} – {b:g}: minimum must be below maximum")
        return a, b

    def limits(self) -> PlotLimits:
        if self.energy_custom.isChecked():
            energy_mode = EnergyMode.CUSTOM
        elif self.energy_cut.isChecked():
            energy_mode = EnergyMode.CUT
        else:
            energy_mode = EnergyMode.AUTO
        return PlotLimits(
            field_range=(
                self._range(self.field_min, self.field_max)
                if self.field_custom.isChecked()
                else None
            ),
            energy_mode=energy_mode,
            energy_range=(
                self._range(self.energy_min, self.energy_max)
                if energy_mode is not EnergyMode.AUTO
                else (0.0, 0.0)
            ),
            custom_levels=self.level_custom.isChecked(),
            levels={key: self._range(*edits) for key, edits in self.level_edits.items()},
            stacked_range=(
                self._range(self.stacked_min, self.stacked_max)
                if self.stacked_custom.isChecked()
                else None
            ),
        )


class CorrectionsPage(QWidget):
    """Baseline normalization and the point extractor."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)

        baseline = QGroupBox("Baseline correction (normalize region to 1)")
        grid = QGridLayout(baseline)
        self.baseline_group, (self.baseline_off, self.baseline_on) = _radio_group(
            self, "Disabled", "Enabled"
        )
        grid.addWidget(self.baseline_off, 0, 1)
        grid.addWidget(self.baseline_on, 0, 2)
        self.baseline_min = FloatEdit(name="Baseline min")
        self.baseline_max = FloatEdit(name="Baseline max")
        self.baseline_min.setPlaceholderText("min")
        self.baseline_max.setPlaceholderText("max")
        grid.addWidget(QLabel("Region min – max"), 1, 0)
        grid.addWidget(self.baseline_min, 1, 1)
        grid.addWidget(self.baseline_max, 1, 2)
        layout.addWidget(baseline)

        points = QGroupBox("Point extractor")
        grid = QGridLayout(points)
        self.column_name = QLineEdit("LL 1")
        self.column_name.setValidator(
            QRegularExpressionValidator(QRegularExpression(r"[A-Za-z0-9_ .+-]{1,32}"), self)
        )
        self.column_name.setToolTip("Name of the curve the clicked points are stored in")
        self.init_table = QCheckBox("New table on next Process")
        self.init_table.setChecked(True)
        grid.addWidget(QLabel("Curve"), 0, 0)
        grid.addWidget(self.column_name, 0, 1, 1, 2)
        grid.addWidget(self.init_table, 0, 3)

        self.point_group, (self.point_off, self.point_record, self.point_remove) = _radio_group(
            self, "Off", "Record", "Remove"
        )
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Click mode:"))
        for button in (self.point_off, self.point_record, self.point_remove):
            mode_row.addWidget(button)
        mode_row.addStretch(1)
        grid.addLayout(mode_row, 1, 0, 1, 4)

        self.show_all_button = QPushButton("Show All Points")
        self.show_all_button.setCheckable(True)
        self.drop_button = QPushButton("Drop Curve")
        self.load_button = QPushButton("Load Points…")
        self.export_button = QPushButton("Export Points…")
        for col, button in enumerate(
            (self.show_all_button, self.drop_button, self.load_button, self.export_button)
        ):
            grid.addWidget(button, 2, col)

        self.model = PointTableModel(self)
        self.table = QTableView()
        self.table.setModel(self.model)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        grid.addWidget(self.table, 3, 0, 1, 4)
        layout.addWidget(points, stretch=1)

    def baseline_region(self) -> Range | None:
        if not self.baseline_on.isChecked():
            return None
        lo, hi = self.baseline_min.value(), self.baseline_max.value()
        if lo >= hi:
            raise ValueError(f"baseline region {lo:g} – {hi:g}: minimum must be below maximum")
        return lo, hi

    def point_mode(self) -> PointMode:
        if self.point_record.isChecked():
            return PointMode.RECORD
        if self.point_remove.isChecked():
            return PointMode.REMOVE
        return PointMode.OFF

    def curve_name(self) -> str:
        name = self.column_name.text().strip()
        if not name:
            raise ValueError("enter a curve name for the picked points")
        return name


@dataclass(frozen=True)
class DiracModel:
    velocity: float  # 10^5 m/s
    delta: float  # half-gap, meV
    n_lines: int


class ModelsPage(QWidget):
    """Massive Dirac Landau-level transitions drawn over the colour map."""

    changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        box = QGroupBox("Massive Dirac model: L(-n) → L(n+1)")
        grid = QGridLayout(box)
        self.show_dirac = QCheckBox("Show on the colour map")
        self.velocity = SliderSpin(0.1, 30.0, 0.05, 5.0, " ×10⁵ m/s")
        self.delta = SliderSpin(0.0, 500.0, 0.5, 0.0, " meV")
        self.n_lines = QSpinBox()
        self.n_lines.setRange(1, 40)
        self.n_lines.setValue(5)
        formula = QLabel("E = √(2eħv²Bn + Δ²) + √(2eħv²B(n+1) + Δ²),  n = 0 … N-1")
        formula.setWordWrap(True)
        formula.setEnabled(False)
        grid.addWidget(self.show_dirac, 0, 0, 1, 2)
        grid.addWidget(QLabel("Fermi velocity v"), 1, 0)
        grid.addWidget(self.velocity, 1, 1)
        grid.addWidget(QLabel("Half-gap Δ"), 2, 0)
        grid.addWidget(self.delta, 2, 1)
        grid.addWidget(QLabel("Lines N"), 3, 0)
        grid.addWidget(self.n_lines, 3, 1)
        grid.addWidget(formula, 4, 0, 1, 2)
        grid.setColumnStretch(1, 1)
        layout.addWidget(box)
        layout.addStretch(1)

        self.show_dirac.toggled.connect(self.changed)
        self.velocity.valueChanged.connect(self.changed)
        self.delta.valueChanged.connect(self.changed)
        self.n_lines.valueChanged.connect(self.changed)

    def dirac(self) -> DiracModel | None:
        if not self.show_dirac.isChecked():
            return None
        return DiracModel(self.velocity.value(), self.delta.value(), self.n_lines.value())


class ToolsTab(QWidget):
    PAGES = ("Plot dimensions", "Data corrections", "Models")

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self.page_combo = QComboBox()
        self.page_combo.addItems(self.PAGES)
        self.stack = QStackedWidget()
        self.limits_page = PlotLimitsPage()
        self.corrections_page = CorrectionsPage()
        self.stack.addWidget(self.limits_page)
        self.stack.addWidget(self.corrections_page)
        self.models_page = ModelsPage()
        self.stack.addWidget(self.models_page)
        self.page_combo.currentIndexChanged.connect(self.stack.setCurrentIndex)
        layout.addWidget(self.page_combo)
        layout.addWidget(self.stack, stretch=1)
