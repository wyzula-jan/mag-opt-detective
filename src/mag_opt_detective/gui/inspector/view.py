"""View section: field, energy and colour ranges of the plots (temporary).

This is the old Tools > Plot dimensions page without the energy cut (now in Processing), plus
the stacked offset and the colour map choice. It edits ``controller.view`` (:class:`ViewState`).
The energy range is typed in the display unit and kept in cm^-1; the fixed levels of per-unit
energy derivatives are stored in cm^-1 too, so restoring settings does not depend on the unit.
"""

from __future__ import annotations

import json
import math

from PySide6.QtCore import Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QRadioButton,
    QWidget,
)

from mag_opt_detective.core.pipeline import PlotKind
from mag_opt_detective.core.units import Unit, convert_levels, from_cm1
from mag_opt_detective.gui.controller import AUTO_COLOURS, ViewState, parse_level_key, user_action
from mag_opt_detective.gui.widgets import EnergyEdit, FloatEdit

Range = tuple[float, float]
COLOURS = (AUTO_COLOURS, "magma", "viridis", "inferno", "plasma", "turbo", "grey", "bipolar")


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
    """Field, energy and intensity ranges of the plots, the stacked offset and colours."""

    changed = Signal()

    LEVEL_ROWS = (
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
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(6)
        row = 0

        grid.addWidget(QLabel("Colours"), row, 0)
        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems(COLOURS)
        self.cmap_combo.setToolTip("Colour map. Auto: magma for maps, grey for derivatives")
        grid.addWidget(self.cmap_combo, row, 1, 1, 2)
        row += 1

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

        self.level_group, (self.level_auto, self.level_custom) = _radio_group(
            self, "Autoscale", "Custom", checked=1
        )
        row = self._section(grid, row, "Colour map intensity", self.level_auto, self.level_custom)
        self.level_edits: dict[str, tuple[FloatEdit, FloatEdit]] = {}
        for key, text, lo, hi in self.LEVEL_ROWS:
            edits = FloatEdit(lo, f"{text} min"), FloatEdit(hi, f"{text} max")
            self.level_edits[key] = edits
            row = self._range_row(grid, row, text, *edits)

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

        for group in (self.field_group, self.energy_group, self.level_group, self.stacked_group):
            group.buttonToggled.connect(lambda _b, checked: checked and self.changed.emit())
        self.cmap_combo.currentIndexChanged.connect(self.changed)

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
        """The view as typed; levels that have no row here (per-unit ones) are kept."""
        levels = dict(current.levels)
        levels.update({key: self._range(*edits) for key, edits in self.level_edits.items()})
        offset = self.offset.value()
        if not math.isfinite(offset):
            raise ValueError("the stacked offset must be a number")
        return ViewState(
            field_range=(
                self._range(self.field_min, self.field_max)
                if self.field_custom.isChecked()
                else None
            ),
            energy_range=self._energy_range() if self.energy_custom.isChecked() else None,
            levels=levels,
            custom_levels=self.level_custom.isChecked(),
            stacked_range=(
                self._range(self.stacked_min, self.stacked_max)
                if self.stacked_custom.isChecked()
                else None
            ),
            stacked_offset=offset,
            colormap=self.cmap_combo.currentText(),
        )

    def show_view(self, view: ViewState) -> None:
        """Show *view* in the fields (without emitting :attr:`changed`)."""
        self.blockSignals(True)
        try:
            self.cmap_combo.setCurrentText(view.colormap)
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
            (self.level_custom if view.custom_levels else self.level_auto).setChecked(True)
            for key, (lo, hi) in self.level_edits.items():
                if key in view.levels:
                    _set(lo, view.levels[key][0])
                    _set(hi, view.levels[key][1])
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


class UnitLevels:
    """Settings protocol for the fixed levels of per-unit energy derivatives, kept in cm^-1.

    The stored JSON maps level keys to ``[lo, hi]`` of the derivative per cm^-1; after a
    restore :meth:`apply` converts them into the display unit.
    """

    def __init__(self, controller):
        self.controller = controller
        self.pending: dict[str, tuple[float, float]] | None = None

    def settings_value(self) -> str:
        c = self.controller
        stored = {}
        for key, value in c.view.levels.items():
            order, energy, physical = parse_level_key(key)
            if physical and order:
                stored[key] = list(convert_levels(value, c.unit, Unit.CM1, order, energy, physical))
        return json.dumps(stored)

    def set_settings_value(self, value) -> bool:
        try:
            data = json.loads(value) if isinstance(value, str) else value
        except ValueError:
            return False
        if not isinstance(data, dict):
            return False
        pending = {}
        for key, pair in data.items():
            if not (isinstance(key, str) and key.startswith("der") and key.endswith("_unit")):
                return False
            if not (isinstance(pair, list) and len(pair) == 2):
                return False
            lo, hi = (float(v) for v in pair)
            if not (math.isfinite(lo) and math.isfinite(hi) and lo < hi):
                return False
            pending[key] = (lo, hi)
        self.pending = pending
        return True

    def apply(self) -> None:
        """Put the restored levels into the view (replacing per-unit ones), in the display unit."""
        if self.pending is None:
            return
        c = self.controller
        levels = {k: v for k, v in c.view.levels.items() if not parse_level_key(k)[2]}
        for key, value in self.pending.items():
            order, energy, physical = parse_level_key(key)
            levels[key] = convert_levels(value, Unit.CM1, c.unit, order, energy, physical)
        self.pending = None
        c.set_view(levels=levels)


@user_action("Plot dimensions")
def apply_page(window, page: PlotDimensionsPage) -> None:
    c = window.controller
    c.set_view(page.view_state(c.view))


def install(window) -> None:
    c = window.controller
    page = PlotDimensionsPage()
    window.add_inspector_section("view", "View", page)
    page.set_unit(c.unit)
    page.changed.connect(lambda: apply_page(window, page))
    c.viewChanged.connect(lambda: page.show_view(c.view))

    def on_unit(_old, new) -> None:
        page.set_unit(new)  # the fields keep cm^-1: this only shows them in the new unit
        if not c.is_restoring():  # while restoring the view is not converted; see on_restored
            page.show_view(c.view)

    c.unitChanged.connect(on_unit)
    unit_levels = UnitLevels(c)

    def on_restored() -> None:
        page.set_unit(c.unit)
        apply_page(window, page)
        unit_levels.apply()

    c.restored.connect(on_restored)
    apply_page(window, page)

    p = window.persistence
    if p is not None:
        p.bind("view/colours", page.cmap_combo)
        for attr in (
            "field_auto", "field_custom", "field_min", "field_max",
            "energy_auto", "energy_custom", "energy_min", "energy_max",
            "level_auto", "level_custom",
            "stacked_auto", "stacked_custom", "stacked_min", "stacked_max", "offset",
        ):  # fmt: skip
            p.bind(f"view/{attr}", getattr(page, attr))
        for key, (lo, hi) in page.level_edits.items():
            p.bind(f"view/levels_{key}_min", lo)
            p.bind(f"view/levels_{key}_max", hi)
        p.bind("view/unit_levels_cm1", unit_levels)
