"""Processing panel: the energy window and the baseline region, and the Process action.

Both ranges are typed in the display unit and kept in cm^-1 (:class:`EnergyEdit`). While the
panel is open, the map shows the window as guides (overlay layer "guides": dashed lines at its
ends), and the map and the stacked plot show the baseline region as a band to drag by its edges
or body with the Pan tool (:class:`BaselineRegion`); *Edit on plot* keeps the band there when the
panel is closed. The band and the fields follow each other. With *Live* on, a new region is
applied to the maps of the last Process at once (:meth:`AppController.apply_baseline`), paced
while it is dragged so that the window stays responsive (:class:`LiveApply`); otherwise it waits
for the next Process. A library map keeps the baseline it was plotted with: the band only shows
there.

The status bar shows the baseline of the map on screen (:class:`BaselineChip`); a click on it
opens this panel at the baseline (:func:`show_baseline`).
"""

from __future__ import annotations

import collections
import logging
import math
import statistics
import time
from collections.abc import Callable

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QObject, Qt, QTimer
from PySide6.QtWidgets import QHBoxLayout, QLabel, QScrollArea, QStackedWidget, QWidget

from mag_opt_detective.core.units import Range, Unit, from_cm1
from mag_opt_detective.gui.baseline_chip import BaselineChip, baseline_mark
from mag_opt_detective.gui.controller import fmt, user_action
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
LIVE_APPLIED = "The baseline applies at once (Live); the energy window the next time you process."
CHANGED = "Changed since the last run. Process again to apply."
GUIDES = "guides"
VIEW_ENERGY_LINK = "view-energy"
BASELINE_HINT = (
    "While this panel is open, the map shows the window and the plots show the region: drag "
    "it or its edges there with Pan (V)."
)
LIVE_TIP = "Apply the baseline at once while you drag or type the region, without processing again"
EDIT_TIP = "Keep the region on the map and the stacked plot while this panel is closed"
LIBRARY_NOTE = (
    "The library map shown keeps the baseline it was plotted with: the region applies when you "
    "process or plot a map again."
)
NAVIGATE = "navigate"  # the plot tool that drags the band (Pan); the others leave it alone
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
        self.edit_on_plot.setAccessibleName("Edit baseline on plot")
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
        self.live_note = Note()  # why the region is not applied (set when installed)
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
        self.baseline_block = block(
            self.baseline_row,
            self._range_row(self.baseline_lo, self.baseline_hi),
            self.baseline_note,
            toggles,
            self.live_note,
            hint(BASELINE_HINT),
        )
        layout.addWidget(self.baseline_block)
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

    A run costs more than *apply* itself: what it changed is drawn after it returns (on the
    stacked plot that takes longer than the run). So its cost is counted until the window is
    idle again, found with zero timers: the first that comes back within IDLE seconds (at most
    SETTLE_MAX seconds after the start). A run started meanwhile (:meth:`flush`) takes over the
    measurement. The first change runs at once. Later ones wait, from that point, INTERVAL ms or
    twice the typical cost (:meth:`cost`, at most SLOW) if that is longer, so the window keeps
    at least two thirds of the time to follow the mouse. When two of the last three costs are
    above SLOW seconds it is slow: changes made while the region is dragged then wait until the
    drag ends (:meth:`flush`), and typed ones until the typing pauses (SETTLE ms). It is fast
    again when two of the last three are below FAST seconds. So a single stall (another program,
    garbage collection) changes nothing, and the vote keeps a map from flipping between drags.

    *apply* returns False when it had nothing to do; such a run does not count. *clock*
    (:attr:`clock`) gives the time in seconds; tests may pass or set their own.
    """

    INTERVAL = 100  # ms
    SETTLE = 300  # ms
    SLOW = 0.12  # s (the 7726 x 64 sweep costs about 0.05 on the map, 0.07-0.1 stacked)
    FAST = 0.12  # s (the two-of-three vote alone keeps the mode from flipping)
    IDLE = 0.02  # s: a zero timer back this soon finds the window idle
    SETTLE_MAX = 1.0  # s: the longest a run's cost is measured
    HISTORY = 3  # costs the slow and fast decisions (and the typical cost) look at

    def __init__(
        self,
        apply: Callable[[], object],
        parent: QObject | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ):
        super().__init__(parent)
        self._apply = apply
        self.clock = clock
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setTimerType(Qt.TimerType.PreciseTimer)  # (a coarse one may fire 5 % early)
        self._timer.timeout.connect(self.flush)
        self._pending = False
        self._dragging = False  # the waiting change was made while dragging
        self._busy = False  # the last run's cost is still being measured
        self._run = 0  # number of the last run: its zero timers carry it, older ones stop
        self._start = 0.0  # clock() at the start of the last run
        self._probe_at = 0.0  # clock() when its last zero timer was started
        self._done = -math.inf  # clock() when the window was idle after the last run
        self._costs: collections.deque[float] = collections.deque(maxlen=self.HISTORY)
        self.slow = False  # see the class docstring
        self.runs = 0  # how many times *apply* did something
        self.last_duration = 0.0  # s, the cost of the last run measured

    def is_pending(self) -> bool:
        return self._pending

    def is_busy(self) -> bool:
        """The last run's cost is still being measured (what it changed is being drawn)."""
        return self._busy

    def costs(self) -> tuple[float, ...]:
        """The last measured costs (s), oldest first."""
        return tuple(self._costs)

    def cost(self) -> float:
        """The typical cost (s): the lower median of the last ones (0 before any)."""
        return statistics.median_low(self._costs) if self._costs else 0.0

    def gap(self) -> float:
        """How long (ms) a change waits after the last run's cost while not slow (a lone first
        stall counts as SLOW, so the next update does not wait seconds)."""
        return max(self.INTERVAL, 2000 * min(self.cost(), self.SLOW))

    def request(self, dragging: bool = False) -> None:
        """A change: run now, or as soon as the rules above allow."""
        self._pending = True
        self._dragging = dragging
        if not self._busy:
            self._schedule()

    def _schedule(self) -> None:
        if self.slow:
            if self._dragging:
                self._timer.stop()
            else:
                self._timer.start(self.SETTLE)
            return
        if self._timer.isActive():
            return
        wait = self.gap() - (self.clock() - self._done) * 1000
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
        start = self.clock()
        if self._apply() is False:
            return
        self.runs += 1
        self._run += 1
        self._start = start
        self._busy = True
        self._probe(self._run)

    def _probe(self, run: int) -> None:
        self._probe_at = self.clock()
        QTimer.singleShot(0, self, lambda: self._probed(run))

    def _probed(self, run: int) -> None:
        if run != self._run:  # a later run took over the measurement
            return
        now = self.clock()
        if now - self._probe_at > self.IDLE and now - self._start < self.SETTLE_MAX:
            self._probe(run)  # the window was busy (drawing what the run changed): look again
            return
        self._busy = False
        self._done = now
        self.last_duration = now - self._start
        self._costs.append(self.last_duration)
        most = self.HISTORY // 2 + 1  # two of three
        if self.slow:
            self.slow = sum(c < self.FAST for c in self._costs) < most
        else:
            self.slow = sum(c > self.SLOW for c in self._costs) >= most
        if self._pending:
            self._schedule()

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


def show_baseline(window) -> None:
    """Open this panel at the baseline (a click on the baseline chip): its first field, or its
    switch while the baseline is off."""
    panel: ProcessingPanel = window.panels["processing"]
    window.show_panel("processing")
    target = panel.baseline_lo if panel.baseline_on.isChecked() else panel.baseline_on
    target.setFocus(Qt.FocusReason.OtherFocusReason)
    window.panel_pages["processing"].scroll.ensureWidgetVisible(panel.baseline_block)


@user_action("Live baseline")
def settle_live(window) -> bool:
    """Apply what was changed with Live on before Live is turned off (see
    :meth:`AppController.settle_baseline`); errors go to the error bar."""
    return window.controller.settle_baseline()


def readable(text: str) -> str:
    """A controller message with the energy unit written as in the window (cm⁻¹)."""
    return text.replace(str(Unit.CM1), unit_text(Unit.CM1))


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
        text = readable(str(exc))
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

    def show_subtitle(*_args) -> None:
        if c.changed_since_process():
            page.set_subtitle(CHANGED)
        elif panel.baseline_live.isChecked() and c.can_apply_baseline():
            page.set_subtitle(LIVE_APPLIED)
        else:
            page.set_subtitle(APPLIED)

    c.changedSinceProcess.connect(show_subtitle)
    c.resultChanged.connect(show_subtitle)
    panel.baseline_live.toggled.connect(show_subtitle)
    push()
    c.processingChanged.connect(pull)
    panel.view_energy_link.linkActivated.connect(lambda _link: show_view_energy(window))

    # the window's guides on the map and the baseline region on the map and the stacked plot
    regions = {
        "map": BaselineRegion(HORIZONTAL, data_style()),
        "stacked": BaselineRegion(VERTICAL, stacked_style(window), label="Baseline"),
    }
    regions["map"].add_to(window.plots.map.plot, z=9.5, fill_z=9.45)  # below the overlays
    regions["stacked"].add_to(window.plots.stacked.plot, z=5.0, fill_z=-1.0)  # fill below traces
    for region in regions.values():
        region.hide()
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
        if panel.baseline_on.isChecked() and c.result is not None and c.result_source == "library":
            text, level = LIBRARY_NOTE, "info"
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

    def run_live() -> bool:
        """Apply the region if it still needs it; whether the map was drawn again."""
        nonlocal applying
        if not live_wanted():  # e.g. applied by the controller (settle_baseline) or a Process
            return False
        applying = True
        try:
            changed = apply_live(window, show_live_note)
        finally:
            applying = False
        if changed:
            log_timer.start()  # once the region rests: one line in the log, not one per step
        return changed is not False

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
        else:  # off, nothing to apply it to, or applied already
            live.cancel()
            show_live_note("")

    def apply_now() -> None:
        follow_settings()
        live.flush()

    def on_live(on: bool) -> None:
        nonlocal applying
        if on:
            c.set_live_baseline(True)
            apply_now()
        else:  # what was changed with Live on is applied (Live is still on in the controller)
            applying = True
            try:
                changed = settle_live(window)
            finally:
                applying = False
            live.cancel()
            if changed:
                log_timer.start()
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

    def sync_movable(*_args) -> None:
        """The band is dragged with the Pan tool; other tools (zoom, pick, auto-pick) and
        library maps only show it, and drags go to the plot."""
        movable = window.tools.active() == NAVIGATE and c.result_source != "library"
        for region in regions.values():
            region.setMovable(movable)

    window.tools.toolChanged.connect(sync_movable)
    c.resultChanged.connect(sync_movable)
    sync_movable()

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

    # the baseline of the map shown, in the status bar beside its state (after on_live above,
    # which tells the controller about Live first)
    chip = BaselineChip()
    window.baseline_chip = chip
    window.add_status_chip(chip)
    chip.clicked.connect(lambda: show_baseline(window))

    def show_mark(*_args) -> None:
        # while the band is dragged the chip keeps its width, so the summary beside it stays
        chip.set_mark(baseline_mark(c))
        dragged = [region for region in regions.values() if region.is_dragging()]
        step = dragged[0].step() if dragged else 0.0
        decimals = max(0, -round(math.log10(step))) if step > 0 else 0  # (dragged values')
        chip.set_held(bool(dragged), decimals)

    for signal in (c.resultChanged, c.processingChanged, c.unitChanged, c.restored):
        signal.connect(show_mark)
    panel.baseline_live.toggled.connect(show_mark)
    for region in regions.values():  # a drag ended (after Live applied what it left)
        region.editFinished.connect(show_mark)
    window.themeChanged.connect(chip.update)
    show_mark()

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
