"""View section: field, energy and stacked ranges of the plots (temporary), and the shared
parts of the inspector sections.

The range page is the old Tools > Plot dimensions page without the energy cut and the colour
levels (now in the Colour section), plus the stacked offset. It edits ``controller.view``
(:class:`ViewState`); the energy range is typed in the display unit and kept in cm^-1. This
module also shows the inspector sections that belong to the plot on screen
(``window.inspector_views``) and holds the number field and map cache the sections use.
"""

from __future__ import annotations

import dataclasses
import json
import math
import re

from PySide6.QtCore import QLocale, Qt, Signal
from PySide6.QtGui import QFont, QValidator, QWheelEvent
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QButtonGroup,
    QDoubleSpinBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QRadioButton,
    QWidget,
)

from mag_opt_detective.core.spectra import FieldMap
from mag_opt_detective.core.units import Unit, from_cm1
from mag_opt_detective.gui.controller import AppController, ViewState, user_action
from mag_opt_detective.gui.widgets import EnergyEdit, FloatEdit, parse_float

Range = tuple[float, float]

# the plot views each inspector section belongs to (sections not listed show everywhere)
SECTION_VIEWS = {
    "view": ("map", "stacked", "reference"),
    "colour": ("map", "reference"),
    "traces": ("stacked",),
    "overlays": ("map",),
}
_PARTIAL_NUMBER = re.compile(r"[+-]?(\d+\.?\d*|\.\d*)?([eE][+-]?\d*)?")


# ---------------------------------------------------------------------- shared helpers
class NumberSpin(QDoubleSpinBox):
    """Number field (C locale, no arrows) that shows six significant digits of any magnitude
    and accepts scientific notation; the wheel changes it only while it has the focus."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setLocale(QLocale.c())
        self.setDecimals(30)  # keeps tiny values (derivative levels) instead of rounding to 0
        self.setRange(-1e300, 1e300)
        self.setKeyboardTracking(False)
        self.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

    def textFromValue(self, value: float) -> str:
        return f"{value:.6g}"

    def valueFromText(self, text: str) -> float:
        value = parse_float(text)
        return self.value() if value is None else value

    def validate(self, text: str, pos: int):
        if parse_float(text) is not None:
            return QValidator.State.Acceptable, text, pos
        if _PARTIAL_NUMBER.fullmatch(text.strip()):
            return QValidator.State.Intermediate, text, pos
        return QValidator.State.Invalid, text, pos

    def wheelEvent(self, event: QWheelEvent) -> None:
        if self.hasFocus():
            super().wheelEvent(event)
        else:
            event.ignore()

    def set_quietly(self, value: float, step: float | None = None) -> None:
        """Show *value* (and use *step* for the arrow keys) without emitting valueChanged."""
        self.blockSignals(True)
        try:
            if step is not None and math.isfinite(step) and step > 0:
                self.setSingleStep(step)
            self.setValue(value)
        finally:
            self.blockSignals(False)


class ShownMaps:
    """The maps on the plots ("map" and "stacked" show the same one), in the display unit.

    Computed on demand and kept until the result, the selection or the unit change; None when
    there is nothing to show (or it cannot be computed: the plot area reports that).
    """

    def __init__(self, controller: AppController):
        self.controller = controller
        self._for: tuple = (None, None, None)  # result, selection, unit of the kept maps
        self._maps: dict[str, FieldMap | None] = {}

    def get(self, view: str) -> FieldMap | None:
        c = self.controller
        if c.result is None:
            return None
        result, selection, unit = self._for
        if not (result is c.result and selection == c.selection and unit is c.unit):
            self._for, self._maps = (c.result, c.selection, c.unit), {}
        name = "reference" if view == "reference" else "map"
        if name not in self._maps:
            try:
                self._maps[name] = c.reference_map() if name == "reference" else c.current_map()
            except ValueError:
                self._maps[name] = None
        return self._maps[name]


def update_sections(window) -> None:
    """Show the inspector sections of the plot on screen (``window.inspector_views``)."""
    view = window.plot_area.current_view()
    for name, section in window.inspector.items():
        views = window.inspector_views.get(name)
        section.setVisible(views is None or view in views)


def load_json(value) -> dict | None:
    """A stored JSON object (dict or text), else None."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return None
    return value if isinstance(value, dict) else None


def to_pair(value) -> Range | None:
    """A stored ``[lo, hi]`` with finite lo < hi; ValueError for anything else but null."""
    if value is None:
        return None
    if not (isinstance(value, list | tuple) and len(value) == 2):
        raise ValueError(f"not a range: {value!r}")
    lo, hi = (float(v) for v in value)
    if not (math.isfinite(lo) and math.isfinite(hi) and lo < hi):
        raise ValueError(f"not a range: {value!r}")
    return lo, hi


# ---------------------------------------------------------------------- page
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


class PlotDimensionsPage(QWidget):
    """Field, energy and intensity ranges of the plots and the stacked offset."""

    changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        grid = QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(6)
        row = 0

        self.field_group, (self.field_auto, self.field_custom) = _radio_group(
            self, "Autoscale", "Custom"
        )
        row = self._section(grid, row, "Magnetic field range", self.field_auto, self.field_custom)
        self.field_min, self.field_max = FloatEdit(0, "B min"), FloatEdit(16, "B max")
        row = self._range_row(grid, row, "B (T)", self.field_min, self.field_max)

        self.energy_group, (self.energy_auto, self.energy_custom) = _radio_group(
            self, "Autoscale", "Custom"
        )
        row = self._section(grid, row, "Energy range", self.energy_auto, self.energy_custom)
        self.energy_min = EnergyEdit(0.0, "E min")
        self.energy_max = EnergyEdit(200.0, "E max")
        self.energy_label = QLabel("E")
        row = self._range_row(grid, row, self.energy_label, self.energy_min, self.energy_max)

        self.stacked_group, (self.stacked_auto, self.stacked_custom) = _radio_group(
            self, "Autoscale", "Custom", checked=1
        )
        row = self._section(grid, row, "Stacked plot", self.stacked_auto, self.stacked_custom)
        self.stacked_min, self.stacked_max = FloatEdit(0.9, "Y min"), FloatEdit(3, "Y max")
        row = self._range_row(grid, row, "Y", self.stacked_min, self.stacked_max)
        self.offset = FloatEdit(0.01, "Offset")
        self.offset.setMinimumWidth(40)
        grid.addWidget(QLabel("Offset"), row, 0)
        grid.addWidget(self.offset, row, 1)
        self.offset.editingFinished.connect(self.changed)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(2, 1)

        for group in (self.field_group, self.energy_group, self.stacked_group):
            group.buttonToggled.connect(lambda _b, checked: checked and self.changed.emit())

    @staticmethod
    def _section(grid: QGridLayout, row: int, title: str, *buttons: QRadioButton) -> int:
        grid.addWidget(_title(title), row, 0, 1, 3)
        line = QHBoxLayout()
        for button in buttons:
            line.addWidget(button)
        line.addStretch(1)
        grid.addLayout(line, row + 1, 0, 1, 3)
        return row + 2

    def _range_row(self, grid: QGridLayout, row: int, text, lo: FloatEdit, hi: FloatEdit):
        for edit in (lo, hi):
            edit.setMinimumWidth(40)
        grid.addWidget(text if isinstance(text, QLabel) else QLabel(text), row, 0)
        grid.addWidget(lo, row, 1)
        grid.addWidget(hi, row, 2)
        lo.editingFinished.connect(self.changed)
        hi.editingFinished.connect(self.changed)
        return row + 1

    def set_unit(self, unit: Unit) -> None:
        self.energy_min.set_unit(unit)
        self.energy_max.set_unit(unit)
        self.energy_label.setText(f"E ({unit})")

    @staticmethod
    def _range(lo: FloatEdit, hi: FloatEdit) -> Range:
        a, b = lo.value(), hi.value()
        if a >= b:
            raise ValueError(f"invalid range {a:g} – {b:g}: minimum must be below maximum")
        return a, b

    def _energy_range(self) -> Range:
        """The energy range in the display unit, read from the kept cm^-1 values."""
        lo, hi = self.energy_min.cm1(), self.energy_max.cm1()
        if lo is None or hi is None:
            raise ValueError("energy range: enter both limits")
        unit = self.energy_min.unit()
        a, b = float(from_cm1(lo, unit)), float(from_cm1(hi, unit))
        if a >= b:
            raise ValueError(f"invalid range {a:.6g} – {b:.6g}: minimum must be below maximum")
        return a, b

    def view_state(self, current: ViewState) -> ViewState:
        """The view as typed (the colour levels and map are kept)."""
        offset = self.offset.value()
        if not math.isfinite(offset):
            raise ValueError("the stacked offset must be a number")
        return dataclasses.replace(
            current,
            field_range=(
                self._range(self.field_min, self.field_max)
                if self.field_custom.isChecked()
                else None
            ),
            energy_range=self._energy_range() if self.energy_custom.isChecked() else None,
            stacked_range=(
                self._range(self.stacked_min, self.stacked_max)
                if self.stacked_custom.isChecked()
                else None
            ),
            stacked_offset=offset,
        )

    def show_view(self, view: ViewState) -> None:
        """Show *view* in the fields (without emitting :attr:`changed`)."""
        self.blockSignals(True)
        try:
            for rng, auto, custom, lo, hi in (
                (view.field_range, self.field_auto, self.field_custom, *self._field_edits()),
                (view.stacked_range, self.stacked_auto, self.stacked_custom, *self._y_edits()),
            ):
                (custom if rng is not None else auto).setChecked(True)
                if rng is not None:
                    _set(lo, rng[0])
                    _set(hi, rng[1])
            if view.energy_range is not None:
                self.energy_custom.setChecked(True)
                for edit, value in zip(
                    (self.energy_min, self.energy_max), view.energy_range, strict=True
                ):
                    if not _shows(edit, value):
                        edit.set_value(value)
            else:
                self.energy_auto.setChecked(True)
            _set(self.offset, view.stacked_offset)
        finally:
            self.blockSignals(False)

    def _field_edits(self) -> tuple[FloatEdit, FloatEdit]:
        return self.field_min, self.field_max

    def _y_edits(self) -> tuple[FloatEdit, FloatEdit]:
        return self.stacked_min, self.stacked_max


def _set(edit: FloatEdit, value: float) -> None:
    """Show *value* unless the field already holds it (keeps what was typed)."""
    try:
        if edit.value_or_none() == value:
            return
    except ValueError:
        pass
    edit.set_value(value)


def _shows(edit: EnergyEdit, value: float) -> bool:
    """Whether *edit* keeps *value* (display unit) up to rounding."""
    kept = edit.cm1()
    if kept is None:
        return False
    return math.isclose(float(from_cm1(kept, edit.unit())), value, rel_tol=1e-9, abs_tol=1e-12)


@user_action("Plot dimensions")
def apply_page(window, page: PlotDimensionsPage) -> None:
    c = window.controller
    c.set_view(page.view_state(c.view))


def install(window) -> None:
    c = window.controller
    page = PlotDimensionsPage()
    window.add_inspector_section("view", "View", page)
    window.inspector_views = dict(SECTION_VIEWS)
    window.shown_maps = ShownMaps(c)
    page.set_unit(c.unit)
    page.changed.connect(lambda: apply_page(window, page))
    c.viewChanged.connect(lambda: c.is_restoring() or page.show_view(c.view))

    def on_unit(_old, new) -> None:
        page.set_unit(new)  # the fields keep cm^-1: this only shows them in the new unit
        if not c.is_restoring():  # while restoring the view is not converted; see on_restored
            page.show_view(c.view)

    c.unitChanged.connect(on_unit)

    def on_restored() -> None:
        page.set_unit(c.unit)
        apply_page(window, page)

    c.restored.connect(on_restored)
    apply_page(window, page)
    window.plot_area.tabs.currentChanged.connect(lambda _index: update_sections(window))
    update_sections(window)

    p = window.persistence
    if p is not None:
        for attr in (
            "field_auto", "field_custom", "field_min", "field_max",
            "energy_auto", "energy_custom", "energy_min", "energy_max",
            "stacked_auto", "stacked_custom", "stacked_min", "stacked_max", "offset",
        ):  # fmt: skip
            p.bind(f"view/{attr}", getattr(page, attr))
