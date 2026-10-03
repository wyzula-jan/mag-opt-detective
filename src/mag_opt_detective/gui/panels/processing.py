"""Processing panel: the energy window and the baseline region, and the Process action.

Both ranges are typed in the display unit and kept in cm^-1 (:class:`EnergyEdit`). While the
panel is open, the map shows the window as guides (overlay layer "guides": dashed lines at its
ends), and the map and the stacked plot show the baseline region as a band to drag by its edges
or body (:class:`BaselineRegion`); *Edit on plot* keeps the band there when the panel is closed.
The band and the fields follow each other. With *Live* on, a new region is applied to the maps
of the last Process at once (:meth:`AppController.apply_baseline`), at most about ten times a
second while it is dragged (:class:`LiveApply`); otherwise it waits for the next Process.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QObject, Qt, QTimer
from PySide6.QtWidgets import QHBoxLayout, QLabel, QScrollArea, QStackedWidget, QWidget

from mag_opt_detective.core.units import Range, Unit, from_cm1
from mag_opt_detective.gui.controller import LIBRARY_KEEPS_BASELINE, fmt, user_action
from mag_opt_detective.gui.display import unit_text
from mag_opt_detective.gui.kit import Switch
from mag_opt_detective.gui.panels.common import (
    Note,
    SwitchRow,
    UnitField,
    block,
    hint,
    panel_layout,
)
from mag_opt_detective.gui.plots.baseline_region import (
    HORIZONTAL,
    VERTICAL,
    BaselineRegion,
    data_style,
    theme_style,
)
from mag_opt_detective.gui.plots.colors import PlotColors
from mag_opt_detective.gui.widgets import EnergyEdit

logger = logging.getLogger("mag_opt_detective")

APPLIED = "Applied the next time you process."
CHANGED = "Changed since the last run. Process again to apply."
GUIDES = "guides"
VIEW_ENERGY_LINK = "view-energy"
BASELINE_HINT = (
    "While this panel is open, the map shows the window and the plots show the region: drag "
    "it or its edges there."
)
LIVE_TIP = "Apply the baseline at once while you drag or type the region, without processing again"
EDIT_TIP = "Keep the region on the map and the stacked plot while this panel is closed"
LIVE_LIBRARY = f"Live waits for a processed map: {LIBRARY_KEEPS_BASELINE}."
LOG_DELAY = 600  # ms: a live change is logged once the region rests this long

# guides are drawn over the colour map, so they use data colours (light on any map)
GUIDE_PEN = pg.mkPen((255, 255, 255, 230), width=1.3, style=Qt.PenStyle.DashLine)
GUIDE_SHADOW = pg.mkPen((0, 0, 0, 110), width=3)
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
        self.baseline_live = Switch("Live")
        self.baseline_live.setAccessibleName("Live baseline")
        self.baseline_live.setToolTip(LIVE_TIP)
        self.edit_on_plot = Switch("Edit on plot")
        self.edit_on_plot.setToolTip(EDIT_TIP)
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
        self.live_note = Note()  # why Live cannot apply the region (set when installed)
        self.live_note.hide()
        self.guides: MapGuides | None = None  # set when installed in a window
        self.regions: dict[str, BaselineRegion] = {}  # the region on "map" and "stacked"
        self.live: LiveApply | None = None

        toggles = QWidget()
        row = QHBoxLayout(toggles)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(16)
        row.addWidget(self.baseline_live)
        row.addWidget(self.edit_on_plot)
        row.addStretch(1)

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
                toggles,
                self.live_note,
                hint(BASELINE_HINT),
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
            self.fields[edit].set_unit(unit_text(unit))
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
                    f"{fmt(shown[0])} – {fmt(shown[1])} {unit_text(unit)}."
                )
            elif used and both and (a is None or b is None) and (a, b) != (None, None):
                bad_lo, bad_hi = a is None, b is None
                text = f"{what}: enter both limits."
            self.fields[lo].set_invalid(bad_lo)
            self.fields[hi].set_invalid(bad_hi)
            note.set_text(text, "err")
            note.setVisible(bool(text))


class MapGuides:
    """The energy window on the map: dashed lines at its ends (the overlay layer "guides")
    and a label (below the overlay layers, above the image, shown and hidden with the layer).

    :meth:`band_region` gives the baseline region drawn on the map (*region*), if shown.
    """

    LABEL_Z = 9.6  # overlay layers start at z = 10

    def __init__(self, plot, region: BaselineRegion | None = None):
        self.layer = plot.layer(GUIDES)
        self.region = region
        self.window_label = pg.TextItem(
            "Energy window", color=LABEL_COLOUR, anchor=(1, 1), fill=LABEL_FILL
        )
        self.window_label.setZValue(self.LABEL_Z)
        plot.plot.addItem(self.window_label, ignoreBounds=True)
        self._visible = False
        self._has_window = False
        self.set_visible(False)

    def set_visible(self, visible: bool) -> None:
        self._visible = visible
        self.layer.set_visible(visible)
        self.window_label.setVisible(visible and self._has_window)

    def is_visible(self) -> bool:
        return self._visible

    def show(self, field: tuple[float, float], window: Range | None) -> None:
        """Draw *window* (display unit; an end may be None) across *field*."""
        ends = [] if window is None else [e for e in window if e is not None]
        if ends:
            b = np.array(field, dtype=float)
            curves = [(b, np.array([e, e])) for e in ends]
            self.layer.set_curves(curves, GUIDE_PEN, shadow_pen=GUIDE_SHADOW)
            self.window_label.setPos(field[1], max(ends))
        else:
            self.layer.clear()
        self._has_window = bool(ends)
        self.set_visible(self._visible)

    def band_region(self) -> tuple[float, float] | None:
        """The baseline region shown on the map, if any (display unit)."""
        region = self.region
        return region.region() if region is not None and region.isVisible() else None


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


def region_limits(
    energy: np.ndarray, energy_cut: Range | None, unit: Unit
) -> tuple[float, float, float]:
    """Where the baseline region may be dragged, in *unit*: over the data's *energy* (cm^-1),
    inside the energy window (cm^-1) when it is set; and its smallest width, the widest step
    between energies (so it always holds data)."""
    shown = from_cm1(np.asarray(energy, dtype=float), unit)
    lo, hi = float(np.min(shown)), float(np.max(shown))
    window, _region = guide_ranges(energy_cut, None, unit)
    if window is not None:
        lo = lo if window[0] is None else max(lo, window[0])
        hi = hi if window[1] is None else min(hi, window[1])
    width = float(np.abs(np.diff(shown)).max()) if shown.size > 1 else 0.0
    return lo, hi, width


def default_region(lo: float, hi: float, width: float) -> tuple[float, float]:
    """A first baseline region in reach [*lo*, *hi*]: its lowest tenth, in round numbers, at
    least *width* wide."""
    digits = -math.floor(math.log10((hi - lo) / 20))
    step = 10.0**-digits
    a = round(math.ceil(lo / step) * step, digits)
    b = round(max(lo + (hi - lo) / 10, a + max(width, step)), digits)
    return a, min(max(b, a + width), hi)


class LiveApply(QObject):
    """Runs *apply* for a stream of changes (the baseline region dragged or typed).

    The first change runs it at once; later ones at most every INTERVAL ms, counted from the
    end of the last run, so the window keeps time to follow the mouse. While the region is
    dragged and a run took longer than SLOW seconds, it waits until the drag ends
    (:meth:`flush`); typed changes then wait until the typing pauses (SETTLE ms).
    """

    INTERVAL = 100  # ms
    SETTLE = 300  # ms
    SLOW = 0.1  # s

    def __init__(self, apply: Callable[[], object], parent: QObject | None = None):
        super().__init__(parent)
        self._apply = apply
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self.flush)
        self._pending = False
        self._done = -math.inf  # time.perf_counter() at the end of the last run
        self.slow = False  # the last run took longer than SLOW
        self.runs = 0  # how many times *apply* ran
        self.last_duration = 0.0  # s, of the last run

    def is_pending(self) -> bool:
        return self._pending

    def request(self, dragging: bool = False) -> None:
        """A change: run now, or as soon as the rules above allow."""
        self._pending = True
        if self.slow:
            if dragging:
                self._timer.stop()
            else:
                self._timer.start(self.SETTLE)
            return
        if self._timer.isActive():
            return
        wait = self.INTERVAL - (time.perf_counter() - self._done) * 1000
        if wait <= 0:
            self.flush()
        else:
            self._timer.start(math.ceil(wait))

    def flush(self) -> None:
        """Run now if a change is waiting (the drag ended, a switch was turned)."""
        self._timer.stop()
        if not self._pending:
            return
        self._pending = False
        start = time.perf_counter()
        try:
            self._apply()
        finally:
            self._done = time.perf_counter()
            self.last_duration = self._done - start
            self.slow = self.last_duration > self.SLOW
            self.runs += 1

    def cancel(self) -> None:
        self._timer.stop()
        self._pending = False


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


def stacked_style(window):
    """The region's style over the stacked spectra in the window's theme."""
    colors = PlotColors.from_mapping(window.theme.plot_colors())
    return theme_style(colors.q("accent"), colors.q("background"))


@user_action("Live baseline")
def apply_live(window, show_problem: Callable[[str], None]) -> bool:
    """Apply the baseline region at once; a region that cannot be applied is explained by
    *show_problem* (next to the switch, not in the error bar)."""
    try:
        changed = window.controller.apply_baseline()
    except ValueError as exc:
        text = str(exc)
        show_problem(f"Live: {text[:1].upper()}{text[1:]}.")
        return False
    show_problem("")
    return changed


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

    def set_baseline(lo: float, hi: float) -> None:
        """Both ends of the baseline region at once (display unit), e.g. from the plot."""
        nonlocal pulling
        pulling = True
        try:
            panel.baseline_lo.set_value(lo)
            panel.baseline_hi.set_value(hi)
        finally:
            pulling = False
        push()

    def start_region(on: bool) -> None:
        """Switched on with both ends empty and a map shown: start with a region over the
        lowest tenth of the energies in reach, to drag from there."""
        ends = (panel.baseline_lo.cm1(), panel.baseline_hi.cm1())
        if not on or pulling or c.result is None or c.is_restoring() or ends != (None, None):
            return
        lo, hi, width = region_limits(c.result.ratio.energy, c.processing.energy_cut, c.unit)
        if lo < hi:
            set_baseline(*default_region(lo, hi, width))

    panel.baseline_on.toggled.connect(start_region)  # before push: one change
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

    # the window's guides on the map and the baseline region on the map and the stacked plot
    regions = {
        "map": BaselineRegion(HORIZONTAL, data_style()),
        "stacked": BaselineRegion(VERTICAL, stacked_style(window), z=5.0),  # above the traces
    }
    for view, region in regions.items():
        region.hide()
        window.plots[view].plot.addItem(region, ignoreBounds=True)
    guides = MapGuides(window.plots.map, regions["map"])
    panel.guides, panel.regions = guides, regions
    stack = page.parentWidget()

    def panel_open() -> bool:
        current = stack.currentWidget() if isinstance(stack, QStackedWidget) else page
        return current is page and window.side_panel.is_open()

    def draw_regions() -> None:
        """Show the baseline region where it can be dragged, or hide it."""
        p = c.processing
        _window, region = guide_ranges(None, p.baseline, c.unit)
        wanted = c.result is not None and region is not None
        wanted = wanted and (panel_open() or panel.edit_on_plot.isChecked())
        if wanted:
            lo, hi, width = region_limits(c.result.ratio.energy, p.energy_cut, c.unit)
            wanted = lo < hi and region[0] < hi and region[1] > lo  # some of it in reach
        for item in regions.values():
            if wanted:
                item.set_limits(lo, hi, min(width, hi - lo))
                item.set_region(*region)
            item.setVisible(wanted)

    def draw_guides(*_args) -> None:
        visible = panel_open() and c.result is not None
        guides.set_visible(visible)
        if visible:
            field = c.result.ratio.field
            window_range, _region = guide_ranges(c.processing.energy_cut, None, c.unit)
            guides.show((float(field.min()), float(field.max())), window_range)
        draw_regions()

    def from_plot(lo: float, hi: float) -> None:
        """The region was dragged (display unit): both fields change at once."""
        if panel.baseline_on.isChecked():
            set_baseline(lo, hi)

    # Live: apply the region at once (coalesced while it is dragged)
    live_problem = ""

    def show_live_note(problem: str | None = None) -> None:
        nonlocal live_problem
        if problem is not None:
            live_problem = problem
        on = panel.baseline_live.isChecked()
        text, level = "", "muted"
        if on and c.result is not None and c.result_source == "library":
            text, level = LIVE_LIBRARY, "info"
        elif on and live_problem and panel.baseline_note.isHidden():  # the fields say it first
            text, level = live_problem, "warn"
        panel.live_note.set_text(text, level)
        panel.live_note.setVisible(bool(text))

    log_timer = QTimer(panel)
    log_timer.setSingleShot(True)
    log_timer.setInterval(LOG_DELAY)

    def log_live() -> None:
        region = c.result.baseline_region if c.result is not None else None
        if region is None:
            logger.info("Baseline correction removed (Live).")
            return
        lo, hi = from_cm1(np.array(region), c.unit)
        logger.info("Baseline corrected in range %.6g – %.6g %s (Live).", lo, hi, c.unit)

    log_timer.timeout.connect(log_live)
    applying = False

    def run_live() -> None:
        nonlocal applying
        applying = True
        try:
            changed = apply_live(window, show_live_note)
        finally:
            applying = False
        if changed:
            log_timer.start()  # once the region rests: one line in the log, not one per step

    live = LiveApply(run_live, panel)
    panel.live = live

    def live_wanted() -> bool:
        return (
            panel.baseline_live.isChecked()
            and c.can_apply_baseline()
            and not c.is_restoring()
            and c.processing.baseline != c.result.baseline_region
        )

    def follow_settings() -> None:
        if live_wanted():
            live.request(dragging=any(region.is_dragging() for region in regions.values()))
        elif not panel.baseline_live.isChecked() or not c.can_apply_baseline():
            live.cancel()

    def apply_now() -> None:
        follow_settings()
        live.flush()

    def on_live(on: bool) -> None:
        if on:
            c.set_live_baseline(True)
            apply_now()
        else:
            live.flush()  # what was changed with Live on is applied
            c.set_live_baseline(False)
        show_live_note()

    def on_result() -> None:
        if not applying:  # a Process (or a library map) logs what it shows itself
            log_timer.stop()
        show_live_note("")
        follow_settings()

    def on_restored() -> None:
        c.set_live_baseline(panel.baseline_live.isChecked())
        apply_now()  # what was restored, now that nothing is restored any more

    c.processingChanged.connect(follow_settings)
    c.processingChanged.connect(lambda: show_live_note())
    panel.baseline_on.toggled.connect(lambda _on: live.flush())  # switched: applied at once
    panel.baseline_live.toggled.connect(on_live)
    c.resultChanged.connect(on_result)
    c.restored.connect(on_restored)

    for region in regions.values():
        region.edited.connect(from_plot)
        region.editFinished.connect(from_plot)
        region.editFinished.connect(lambda *_args: live.flush())

    def on_theme() -> None:
        regions["stacked"].set_style(stacked_style(window))

    window.themeChanged.connect(on_theme)
    if isinstance(stack, QStackedWidget):
        stack.currentChanged.connect(draw_guides)
    window.side_panel.openChanged.connect(draw_guides)
    panel.edit_on_plot.toggled.connect(draw_guides)
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
        p.bind("processing/baseline_live", panel.baseline_live)
        p.bind("processing/baseline_edit_on_plot", panel.edit_on_plot)
