"""Processing panel: the energy window and the baseline region, and the Process action.

Both ranges are typed in the display unit and kept in cm^-1 (:class:`EnergyEdit`). While the
panel is open, the map shows them as guides (overlay layer "guides"): dashed lines at the
window's ends and a translucent band over the baseline region.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QLabel, QScrollArea, QStackedWidget, QWidget

from mag_opt_detective.core.units import Range, Unit, from_cm1
from mag_opt_detective.gui.controller import fmt, user_action
from mag_opt_detective.gui.panels.common import (
    Note,
    SwitchRow,
    UnitField,
    block,
    hint,
    panel_layout,
)
from mag_opt_detective.gui.widgets import EnergyEdit

APPLIED = "Applied the next time you process."
CHANGED = "Changed since the last run. Process again to apply."
GUIDES = "guides"
VIEW_ENERGY_LINK = "view-energy"

# guides are drawn over the colour map, so they use data colours (light on any map)
GUIDE_PEN = pg.mkPen((255, 255, 255, 230), width=1.3, style=Qt.PenStyle.DashLine)
GUIDE_SHADOW = pg.mkPen((0, 0, 0, 110), width=3)
BAND_BRUSH = pg.mkBrush(255, 255, 255, 40)
BAND_PEN = pg.mkPen((255, 255, 255, 170), width=1)
LABEL_COLOUR = (255, 255, 255, 240)
LABEL_FILL = pg.mkBrush(20, 16, 24, 140)


class ProcessingPanel(QWidget):
    """Energy window (drops data outside it) and baseline region (shifted to average 1)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.cut_row = SwitchRow(
            "Energy window", "Drops data outside it, e.g. noisy detector edges"
        )
        self.cut_on = self.cut_row.switch
        self.cut_lo = EnergyEdit(name="Energy window from")
        self.cut_hi = EnergyEdit(name="Energy window to")
        self.baseline_row = SwitchRow("Baseline", "Shifts each spectrum so this region averages 1")
        self.baseline_on = self.baseline_row.switch
        self.baseline_lo = EnergyEdit(name="Baseline from")
        self.baseline_hi = EnergyEdit(name="Baseline to")
        self.fields = {edit: UnitField(edit) for edit in self.edits()}
        for edit in (self.cut_lo, self.baseline_lo):
            edit.setPlaceholderText("from")
        for edit in (self.cut_hi, self.baseline_hi):
            edit.setPlaceholderText("to")
        self.unit_labels = [self.fields[edit].unit_label for edit in self.edits()]

        self.view_energy_link = hint(
            f"Only want to zoom? Use <a href='{VIEW_ENERGY_LINK}' style='text-decoration:none'>"
            "View &rsaquo; Energy</a> next to the plot. It keeps the data."
        )
        self.view_energy_link.setTextFormat(Qt.TextFormat.RichText)
        self.view_energy_link.setTextInteractionFlags(
            Qt.TextInteractionFlag.LinksAccessibleByMouse
            | Qt.TextInteractionFlag.LinksAccessibleByKeyboard
        )
        self.cut_note = Note()
        self.baseline_note = Note()
        self.guides: MapGuides | None = None  # set when installed in a window

        layout = panel_layout(self)
        layout.addWidget(
            block(
                self.cut_row,
                self._range_row(self.cut_lo, self.cut_hi),
                self.cut_note,
                self.view_energy_link,
            )
        )
        layout.addWidget(
            block(
                self.baseline_row,
                self._range_row(self.baseline_lo, self.baseline_hi),
                self.baseline_note,
                hint("While this panel is open, the map shows the window and the region."),
            )
        )
        layout.addStretch(1)
        for switch in (self.cut_on, self.baseline_on):
            switch.toggled.connect(self.check)
        for edit in self.edits():
            edit.valueChanged.connect(self.check)
        self.check()

    def _range_row(self, lo: EnergyEdit, hi: EnergyEdit) -> QWidget:
        row = QWidget()
        line = QHBoxLayout(row)
        line.setContentsMargins(0, 0, 0, 0)
        line.setSpacing(6)
        line.addWidget(self.fields[lo], 1)
        dash = QLabel("–")
        dash.setProperty("kit", "muted")
        line.addWidget(dash)
        line.addWidget(self.fields[hi], 1)
        return row

    def edits(self) -> tuple[EnergyEdit, ...]:
        return (self.cut_lo, self.cut_hi, self.baseline_lo, self.baseline_hi)

    def set_unit(self, unit) -> None:
        for edit in self.edits():
            edit.set_unit(unit)
            self.fields[edit].set_unit(str(unit))
        self.check()

    def energy_cut(self) -> Range | None:
        """The window in cm^-1, or None when it is off or both ends are empty."""
        rng = (self.cut_lo.cm1(), self.cut_hi.cm1())
        return rng if self.cut_on.isChecked() and rng != (None, None) else None

    def baseline(self) -> Range | None:
        """The baseline region in cm^-1 (an end may be None), or None when it is off."""
        if not self.baseline_on.isChecked():
            return None
        return self.baseline_lo.cm1(), self.baseline_hi.cm1()

    def show_ranges(self, energy_cut: Range | None, baseline: Range | None) -> None:
        """Show the two ranges (cm^-1); None switches one off and keeps its typed ends."""
        for switch, lo, hi, rng in (
            (self.cut_on, self.cut_lo, self.cut_hi, energy_cut),
            (self.baseline_on, self.baseline_lo, self.baseline_hi, baseline),
        ):
            if rng is None:
                switch.setChecked(False)
                continue
            for edit, value in zip((lo, hi), rng, strict=True):
                if edit.cm1() != value:
                    edit.set_cm1(value)
            switch.setChecked(True)

    def check(self) -> None:
        """Enable the fields of the ranges in use and flag reversed or incomplete ones."""
        unit = self.cut_lo.unit()
        for on, lo, hi, note, what, both in (
            (self.cut_on, self.cut_lo, self.cut_hi, self.cut_note, "Energy window", False),
            (self.baseline_on, self.baseline_lo, self.baseline_hi, self.baseline_note,
             "Baseline", True),
        ):  # fmt: skip
            used = on.isChecked()
            for edit in (lo, hi):
                self.fields[edit].setEnabled(used)
            a, b = lo.cm1(), hi.cm1()
            bad_lo = bad_hi = False
            text = ""
            if used and a is not None and b is not None and a >= b:
                bad_lo = bad_hi = True
                shown = from_cm1(np.array([a, b]), unit)
                text = (
                    f"{what}: the first value must be below the second; it is "
                    f"{fmt(shown[0])} – {fmt(shown[1])} {unit}."
                )
            elif used and both and (a is None or b is None) and (a, b) != (None, None):
                bad_lo, bad_hi = a is None, b is None
                text = f"{what}: enter both limits."
            self.fields[lo].set_invalid(bad_lo)
            self.fields[hi].set_invalid(bad_hi)
            note.set_text(text, "err")
            note.setVisible(bool(text))


class MapGuides:
    """The energy window (dashed lines) and the baseline region (a band) on the map.

    The lines are the map's overlay layer "guides"; the band and the two labels sit below
    the overlay layers (above the image) and are shown and hidden with the layer.
    """

    BAND_Z, LABEL_Z = 9.5, 9.6  # overlay layers start at z = 10

    def __init__(self, plot):
        self.layer = plot.layer(GUIDES)
        self.band = pg.LinearRegionItem(
            orientation="horizontal", movable=False, brush=BAND_BRUSH, pen=BAND_PEN
        )
        self.window_label = pg.TextItem(
            "Energy window", color=LABEL_COLOUR, anchor=(1, 1), fill=LABEL_FILL
        )
        self.band_label = pg.TextItem(
            "Baseline region", color=LABEL_COLOUR, anchor=(0, 1), fill=LABEL_FILL
        )
        self.band.setZValue(self.BAND_Z)
        for item in (self.window_label, self.band_label):
            item.setZValue(self.LABEL_Z)
        for item in (self.band, self.window_label, self.band_label):
            plot.plot.addItem(item, ignoreBounds=True)
        self._visible = False
        self._has_window = self._has_band = False
        self.set_visible(False)

    def set_visible(self, visible: bool) -> None:
        self._visible = visible
        self.layer.set_visible(visible)
        self.window_label.setVisible(visible and self._has_window)
        self.band.setVisible(visible and self._has_band)
        self.band_label.setVisible(visible and self._has_band)

    def is_visible(self) -> bool:
        return self._visible

    def show(self, field: tuple[float, float], window: Range | None, region) -> None:
        """Draw *window* (an end may be None) and *region* (display unit) across *field*."""
        ends = [] if window is None else [e for e in window if e is not None]
        if ends:
            b = np.array(field, dtype=float)
            curves = [(b, np.array([e, e])) for e in ends]
            self.layer.set_curves(curves, GUIDE_PEN, shadow_pen=GUIDE_SHADOW)
            self.window_label.setPos(field[1], max(ends))
        else:
            self.layer.clear()
        if region is not None:
            self.band.setRegion(region)
            self.band_label.setPos(field[0], max(region))
        self._has_window, self._has_band = bool(ends), region is not None
        self.set_visible(self._visible)

    def band_region(self) -> tuple[float, float] | None:
        """The band shown, if any (display unit)."""
        return tuple(self.band.getRegion()) if self.band.isVisible() else None


def guide_ranges(energy_cut: Range | None, baseline: Range | None, unit: Unit):
    """The window and the baseline region (cm^-1) in *unit*; None when not drawable."""
    window = None
    if energy_cut is not None:
        lo, hi = (None if v is None else float(from_cm1(v, unit)) for v in energy_cut)
        if lo is None or hi is None or lo < hi:
            window = (lo, hi)
    region = None
    if baseline is not None and None not in baseline and baseline[0] < baseline[1]:
        region = tuple(float(v) for v in from_cm1(np.array(baseline), unit))
    return window, region


@user_action("Process")
def process(window) -> None:
    window.infobar.dismiss()
    window.controller.process()


def show_view_energy(window) -> None:
    """Open the inspector at the View section's energy range (View > Energy)."""
    area = window.plot_area
    if area.current_view() == "reference":
        area.set_current_view("map")
    window.inspector_panel.set_open(True)
    section = window.inspector.get("view")
    if section is None:
        return
    section.set_expanded(True)
    page = section.body_layout().itemAt(0).widget()
    focus = getattr(page, "focus_energy", None)
    target = None
    if callable(focus):
        target = focus()
    elif (edit := getattr(page, "energy_min", None)) is not None:
        edit.setFocus(Qt.FocusReason.OtherFocusReason)
        target = edit
    scroll = window.inspector_panel.content()
    if isinstance(scroll, QScrollArea):
        scroll.ensureWidgetVisible(target if isinstance(target, QWidget) else section)


def install(window) -> None:
    c = window.controller
    panel = ProcessingPanel()
    page = window.add_panel(
        "processing", "Processing", "sliders-horizontal", "Processing options", panel,
        APPLIED, rail_text="Process",
    )  # fmt: skip

    pulling = False

    def push() -> None:
        if not pulling:
            c.set_processing(energy_cut=panel.energy_cut(), baseline=panel.baseline())

    def pull() -> None:
        """Show ranges set through the controller (e.g. by another area)."""
        nonlocal pulling
        state = c.processing
        if (panel.energy_cut(), panel.baseline()) == (state.energy_cut, state.baseline):
            return
        pulling = True
        try:
            panel.show_ranges(state.energy_cut, state.baseline)
        finally:
            pulling = False

    for switch in (panel.cut_on, panel.baseline_on):
        switch.toggled.connect(push)
    for edit in panel.edits():
        edit.valueChanged.connect(push)

    panel.set_unit(c.unit)
    c.unitChanged.connect(lambda _old, new: panel.set_unit(new))
    c.restored.connect(lambda: panel.set_unit(c.unit))

    def on_changed(changed: bool) -> None:
        page.set_subtitle(CHANGED if changed else APPLIED)

    c.changedSinceProcess.connect(on_changed)
    push()
    c.processingChanged.connect(pull)
    panel.view_energy_link.linkActivated.connect(lambda _link: show_view_energy(window))

    # guides on the map while the panel is open
    guides = MapGuides(window.plots.map)
    panel.guides = guides
    stack = page.parentWidget()

    def panel_open() -> bool:
        current = stack.currentWidget() if isinstance(stack, QStackedWidget) else page
        return current is page and window.side_panel.is_open()

    def draw_guides(*_args) -> None:
        visible = panel_open() and c.result is not None
        guides.set_visible(visible)
        if not visible:
            return
        field = c.result.ratio.field
        p = c.processing
        window_range, region = guide_ranges(p.energy_cut, p.baseline, c.unit)
        guides.show((float(field.min()), float(field.max())), window_range, region)

    if isinstance(stack, QStackedWidget):
        stack.currentChanged.connect(draw_guides)
    window.side_panel.openChanged.connect(draw_guides)
    c.processingChanged.connect(draw_guides)
    c.resultChanged.connect(draw_guides)
    c.unitChanged.connect(draw_guides)
    draw_guides()

    window.commands["process"].triggered.connect(lambda: process(window))

    p = window.persistence
    if p is not None:
        p.bind("processing/cut_on", panel.cut_on)
        p.bind("processing/cut_lo_cm1", panel.cut_lo)
        p.bind("processing/cut_hi_cm1", panel.cut_hi)
        p.bind("processing/baseline_on", panel.baseline_on)
        p.bind("processing/baseline_lo_cm1", panel.baseline_lo)
        p.bind("processing/baseline_hi_cm1", panel.baseline_hi)
