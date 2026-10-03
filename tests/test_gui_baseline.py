"""The baseline region on the plots and the Live baseline (P4-08)."""

import itertools
import logging
import math
import time

import numpy as np
import pyqtgraph as pg
import pytest
from PySide6.QtCore import QCoreApplication, QEvent, QPoint, QPointF, QSettings, Qt, QTimer
from PySide6.QtGui import QColor
from PySide6.QtTest import QTest

import golden
import gui_helpers
from gui_helpers import process, select, set_unit, shown_image
from helpers import sweep_name, write_text
from mag_opt_detective.core.pipeline import PlotKind, ReferenceMode
from mag_opt_detective.core.processing import Axis
from mag_opt_detective.core.units import CM1_PER_UNIT, Unit, from_cm1
from mag_opt_detective.gui.controller import AppController, SweepFiles
from mag_opt_detective.gui.main_window import MainWindow
from mag_opt_detective.gui.panels import library
from mag_opt_detective.gui.panels.processing import (
    APPLIED,
    CHANGED,
    LIBRARY_NOTE,
    LIVE_APPLIED,
    LOG_DELAY,
    LiveApply,
    default_region,
    region_limits,
)
from mag_opt_detective.gui.plots.baseline_region import (
    VERTICAL,
    BaselineRegion,
    data_style,
    theme_style,
)

window, errors = gui_helpers.window, gui_helpers.errors  # shared fixtures

LEFT, PLAIN = Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier


@pytest.fixture
def lines(tmp_path) -> SweepFiles:
    """A sweep whose ratios change along energy (a line moving on a sloped background), so
    every baseline region gives other maps; energies 400 - 1200 cm-1."""
    zero, field = golden.write_sweep(tmp_path)
    return SweepFiles(tuple(map(str, zero)), tuple(map(str, field)))


@pytest.fixture
def reference(tmp_path) -> SweepFiles:
    """A reference sweep on the energies of :func:`lines`, with features of its own."""
    folder = tmp_path / "reference"
    folder.mkdir()
    x = np.linspace(400.0, 1200.0, 81)
    base = 1.0 + 0.2 * np.cos(x / 90.0)
    zero = [
        write_text(folder / "Ref_a00p000T_a00p000T.txt", x, base),
        write_text(folder / "Ref_a00p000T_a02p000T.txt", x, 1.05 * base),
    ]
    field = [
        write_text(folder / sweep_name(b), x, base * (1 + 0.03 * b * x / 1200))
        for b in golden.FIELDS
    ]
    return SweepFiles(tuple(map(str, zero)), tuple(map(str, field)))


@pytest.fixture
def processed(window, lines):
    """The sweep processed with the baseline region 450 - 550 cm-1 and the panel open."""
    c = window.controller
    c.set_processing(sample_files=lines, baseline=(450.0, 550.0))
    process(window)
    assert c.result is not None and c.result.baseline_region == (450.0, 550.0)
    window.show_panel("processing")
    return window


FIELDS = ("sample_files", "reference_files", "reference_mode", "energy_cut", "baseline")


def fresh_result(window):
    """The result of a full Process with the window's current settings (another controller)."""
    other = AppController()
    other.set_processing(**{f: getattr(window.controller.processing, f) for f in FIELDS})
    return other.process()


def drag_like(region, lo: float, hi: float) -> None:
    """Move *region* as a drag does: its change signals report each step and the end."""
    region.setRegion((lo, hi))


def viewport_pos(plot, x: float, y: float) -> QPoint:
    return plot.view.mapFromScene(plot.plot.vb.mapViewToScene(QPointF(x, y)))


def move_pause_ms() -> int:
    """pyqtgraph drops a mouse move within 1/mouseRateLimit s of the last one it took."""
    rate = pg.getConfigOption("mouseRateLimit")
    return 2 * math.ceil(1000 / rate) if rate > 0 else 0


def mouse_drag(qtbot, plot, points: list[tuple[float, float]], release: bool = True) -> None:
    """A left-drag through *points* (plot coordinates), slow enough for pyqtgraph."""
    viewport = plot.view.viewport()
    pixels = [viewport_pos(plot, x, y) for x, y in points]
    QTest.mouseMove(viewport, pixels[0])
    QTest.mousePress(viewport, LEFT, PLAIN, pixels[0])
    for pixel in pixels[1:]:
        qtbot.wait(move_pause_ms())
        QTest.mouseMove(viewport, pixel)
    if release:
        QTest.mouseRelease(viewport, LEFT, PLAIN, pixels[-1])


# ---------------------------------------------------------------------- the region
def test_the_region_item_keeps_its_limits_and_reports_only_drags(qtbot):
    plot = pg.PlotWidget()
    qtbot.addWidget(plot)
    region = BaselineRegion(VERTICAL, data_style())
    plot.addItem(region, ignoreBounds=True)
    edits, ends = [], []
    region.edited.connect(lambda lo, hi: edits.append((lo, hi)))
    region.editFinished.connect(lambda lo, hi: ends.append((lo, hi)))
    region.set_limits(0.0, 100.0, 5.0)
    region.set_region(10, 20)
    assert region.region() == (10, 20) and not edits  # typed: nothing reported
    region.set_region(-50, 300)
    assert region.region() == (0, 100) and not edits
    region.setRegion((30, 40))  # as a drag reports it
    assert edits == [(30, 40)] and ends == [(30, 40)]
    region.lines[0].setValue(80)  # an edge stops short of the other ...
    assert region.region() == (35, 40) and edits[-1] == (35, 40)
    region.lines[1].setValue(200)  # ... and at the limit
    assert region.region() == (35, 100)
    region.set_limits(None, 60.0, 1.0)
    assert region.region() == (35, 60)
    region.lines[0].setValue(-1e6)
    assert region.region() == (-1e6, 60)
    assert not region.is_dragging() and region.step() == 0  # not on screen: no rounding
    region.set_style(theme_style(QColor("#7a2a8c"), QColor("#ffffff")))
    assert region.lines[0].pen.color().name() == "#7a2a8c"


def test_the_region_shows_with_the_panel_or_edit_on_plot(processed):
    w = processed
    panel = w.panels["processing"]
    regions = panel.regions
    assert all(r.isVisible() for r in regions.values())
    assert regions["map"].region() == pytest.approx((450, 550))
    assert regions["stacked"].region() == pytest.approx((450, 550))
    w.show_panel("sample")
    assert not any(r.isVisible() for r in regions.values())
    panel.edit_on_plot.setChecked(True)  # kept on the plots with the panel closed
    assert all(r.isVisible() for r in regions.values())
    w.side_panel.set_open(False, animate=False)
    assert all(r.isVisible() for r in regions.values())
    panel.baseline_on.setChecked(False)  # no baseline: no region
    assert not any(r.isVisible() for r in regions.values())
    panel.baseline_on.setChecked(True)
    panel.edit_on_plot.setChecked(False)
    assert not any(r.isVisible() for r in regions.values())
    w.show_panel("processing")
    assert all(r.isVisible() for r in regions.values())
    panel.baseline_hi.setText("300")  # reversed: nothing to drag (the fields say why)
    assert not any(r.isVisible() for r in regions.values())
    assert not panel.baseline_note.isHidden()


def test_the_window_guides_stay_and_the_band_is_the_region(processed):
    w = processed
    panel = w.panels["processing"]
    panel.cut_on.setChecked(True)
    panel.cut_lo.setText("420")
    panel.cut_hi.setText("1100")
    guides = w.plots.map.layer("guides")
    assert guides.is_visible()
    assert sorted(float(y[0]) for _x, y in guides.curve_data()) == pytest.approx([420, 1100])
    assert panel.guides.band_region() == pytest.approx((450, 550))
    assert panel.regions["map"].limits() == pytest.approx((420, 1100))  # inside the window


@pytest.mark.parametrize("unit", ["cm-1", "meV", "THz"])
def test_dragging_the_region_types_the_fields_in_the_unit(processed, unit):
    w, c = processed, processed.controller
    panel = w.panels["processing"]
    set_unit(w, unit)
    k = CM1_PER_UNIT[Unit(unit)]
    regions = panel.regions
    assert regions["map"].region() == pytest.approx((450 / k, 550 / k))
    lo, hi = round(600 / k, 2), round(700 / k, 2)
    drag_like(regions["map"], lo, hi)
    assert (float(panel.baseline_lo.text()), float(panel.baseline_hi.text())) == (lo, hi)
    assert c.processing.baseline == pytest.approx((lo * k, hi * k))
    assert regions["stacked"].region() == pytest.approx((lo, hi))  # the other plot follows
    drag_like(regions["stacked"], lo, round(900 / k, 2))
    assert float(panel.baseline_hi.text()) == round(900 / k, 2)
    assert regions["map"].region() == pytest.approx((lo, round(900 / k, 2)))
    panel.baseline_lo.setText(str(round(500 / k, 2)))  # typing moves the region
    assert regions["map"].region()[0] == pytest.approx(round(500 / k, 2))
    assert regions["stacked"].region()[0] == pytest.approx(round(500 / k, 2))


def test_a_unit_switch_keeps_the_region_where_it_is(processed):
    w, c = processed, processed.controller
    panel = w.panels["processing"]
    drag_like(panel.regions["map"], 612.5, 733.25)
    stored = c.processing.baseline
    for unit in ("meV", "THz", "cm-1", "meV"):
        set_unit(w, unit)
        k = CM1_PER_UNIT[Unit(unit)]
        assert panel.regions["map"].region() == pytest.approx((612.5 / k, 733.25 / k))
        assert panel.regions["stacked"].region() == pytest.approx((612.5 / k, 733.25 / k))
    assert c.processing.baseline == stored  # exactly: nothing written back


def test_the_region_is_clamped_and_never_inverted(processed, qtbot):
    w, c = processed, processed.controller
    panel = w.panels["processing"]
    region = panel.regions["map"]
    lo, hi = region.limits()
    assert (lo, hi) == pytest.approx((400, 1200))  # the data
    upper, lower = region.lines[1], region.lines[0]
    upper.setValue(5000)  # an edge stops at the data's end ...
    assert region.region() == pytest.approx((450, 1200))
    lower.setValue(3000)  # ... and before the other edge
    a, b = region.region()
    assert a < b and b - a == pytest.approx(10.0)  # one step of the energies
    assert c.processing.baseline == pytest.approx((a, b)) and panel.baseline_note.isHidden()
    lower.setValue(-100)
    assert region.region() == pytest.approx((400, 1200))

    w.resize(1400, 900)
    w.show()
    qtbot.waitExposed(w)
    drag_like(region, 600, 700)
    plot = w.plots.map
    mouse_drag(qtbot, plot, [(1.2, 650), (1.2, 700), (1.2, 900), (1.2, 1190)])  # the band
    a, b = c.processing.baseline
    assert b == pytest.approx(1200) and b - a == pytest.approx(100, abs=1)  # its width kept
    before = plot.plot.vb.viewRange()
    mouse_drag(qtbot, plot, [(1.2, a), (1.2, a - 50), (1.2, a - 150)])  # its lower edge
    assert c.processing.baseline[0] == pytest.approx(a - 150, abs=5)
    assert c.processing.baseline[1] == b
    assert plot.plot.vb.viewRange() == before  # the region took the drags, not the view


def test_switching_the_baseline_on_starts_a_region(window, lines):
    c = window.controller
    c.set_processing(sample_files=lines)
    panel = window.panels["processing"]
    panel.baseline_on.setChecked(True)  # nothing processed: nothing to start from
    assert c.processing.baseline == (None, None)
    panel.baseline_on.setChecked(False)
    process(window)
    set_unit(window, "meV")
    panel.baseline_on.setChecked(True)
    assert (panel.baseline_lo.text(), panel.baseline_hi.text()) == ("50", "60")
    assert default_region(400, 1200, 10) == (400, 480)
    assert default_region(2.99792, 29.9792, 0.3) == (3, 6)
    assert default_region(10.0, 10.5, 0.2) == (10.0, 10.2)  # at least one step of the data
    panel.baseline_on.setChecked(False)
    panel.baseline_lo.setText("70")
    panel.baseline_on.setChecked(True)  # typed ends are kept
    assert (panel.baseline_lo.text(), panel.baseline_hi.text()) == ("70", "60")


def test_region_limits():
    energy = np.array([100.0, 110.0, 130.0, 160.0])
    assert region_limits(energy, None, Unit.CM1) == (100, 160, 30)
    assert region_limits(energy, (105.0, None), Unit.CM1) == (105, 160, 30)
    assert region_limits(energy, (120.0, 140.0), Unit.CM1) == (120, 140, 30)
    lo, hi, width = region_limits(energy, None, Unit.MEV)
    assert (lo, hi, width) == pytest.approx((100 / 8.0656, 160 / 8.0656, 30 / 8.0656))


def test_region_limits_leave_out_a_gap():
    """The empty samples of a map merged by energy do not make the region's width the gap."""
    energy = np.array([100.0, 102.0, 104.0, 106.0, 200.0, 300.0, 302.0, 304.0])
    values = np.ones((energy.size, 3))
    values[[4]] = np.nan  # the gap 106 - 300, marked by an empty sample
    assert region_limits(energy, None, Unit.CM1, values) == (100, 304, 2)
    assert region_limits(energy, None, Unit.CM1) == (100, 304, 100)  # (without the values)


# ---------------------------------------------------------------------- Live off
def test_without_live_the_region_waits_for_process(processed):
    w, c = processed, processed.controller
    panel = w.panels["processing"]
    assert not panel.baseline_live.isChecked()  # off by default
    before = shown_image(w).copy()
    drag_like(panel.regions["map"], 900, 1100)
    assert c.changed_since_process() and w.toolbar.process_button.dot
    np.testing.assert_array_equal(shown_image(w), before)
    assert c.result.baseline_region == (450, 550)
    process(w)
    assert not c.changed_since_process()
    np.testing.assert_allclose(shown_image(w), fresh_result(w).ratio.values)


# ---------------------------------------------------------------------- Live on
@pytest.mark.parametrize(
    ("kind", "order", "axis"),
    [
        (PlotKind.RATIO, 0, "E"),
        (PlotKind.AVERAGE, 0, "E"),
        (PlotKind.STEP, 0, "E"),
        (PlotKind.RATIO, 1, "E"),
        (PlotKind.STEP, 2, "B"),
    ],
)
def test_live_applies_the_region_without_processing(processed, monkeypatch, kind, order, axis):
    w, c = processed, processed.controller
    panel = w.panels["processing"]
    select(w, kind=kind, order=order, axis=axis)
    calls = []
    monkeypatch.setattr(c, "process", lambda: calls.append(1))
    panel.baseline_live.setChecked(True)
    stamp = w.state_text()
    drag_like(panel.regions["map"], 900, 1100)
    panel.live.flush()
    assert c.result.baseline_region == (900, 1100)
    fresh = fresh_result(w)
    expected = fresh.get(kind, order, Axis.FIELD if axis == "B" else Axis.ENERGY)
    np.testing.assert_allclose(shown_image(w), expected.values)
    np.testing.assert_allclose(c.current_map().values, expected.values)
    assert not c.changed_since_process() and not w.toolbar.process_button.dot
    assert w.state_text() == stamp and "Processed" in stamp  # the time of the Process

    panel.baseline_hi.setText("1150")  # typed
    panel.live.flush()
    assert c.result.baseline_region == (900, 1150)
    expected = fresh_result(w).get(kind, order, Axis.FIELD if axis == "B" else Axis.ENERGY)
    np.testing.assert_allclose(shown_image(w), expected.values)
    assert not calls  # never processed again


def test_live_keeps_points_view_and_levels(processed):
    w, c = processed, processed.controller
    panel = w.panels["processing"]
    c.record_point(1.0, 700.0)
    c.set_ranges(field_range=(0.6, 1.8), energy_range=(500.0, 1000.0))
    c.set_levels("Ratio", 0.95, 1.05)
    select(w, kind=PlotKind.DATA)
    data = shown_image(w).copy()
    select(w, kind=PlotKind.RATIO)
    panel.baseline_live.setChecked(True)
    drag_like(panel.regions["stacked"], 1000, 1150)
    panel.live.flush()
    assert c.points.column("LL 1")[c.points.nearest_row(1.0)] == pytest.approx(700.0)
    assert c.view.field_range == (0.6, 1.8) and c.view.energy_range == (500.0, 1000.0)
    assert w.plots.map.levels() == pytest.approx((0.95, 1.05))
    select(w, kind=PlotKind.DATA)
    np.testing.assert_array_equal(shown_image(w), data)  # Data is never normalised


def test_live_coalesces_a_drag(processed, qtbot):
    w, c = processed, processed.controller
    panel = w.panels["processing"]
    w.resize(1400, 900)
    w.show()
    qtbot.waitExposed(w)
    panel.baseline_live.setChecked(True)
    live = panel.live
    runs = live.runs
    plot = w.plots.map
    steps = [(1.2, 550.0 + 10 * k) for k in range(31)]  # the upper edge, up by 300 cm-1
    start = time.perf_counter()
    mouse_drag(qtbot, plot, steps, release=False)
    elapsed = (time.perf_counter() - start) * 1000
    during = live.runs - runs
    assert 1 <= during <= elapsed / live.INTERVAL + 1  # at once, then at most ~10 a second
    QTest.mouseRelease(plot.view.viewport(), LEFT, PLAIN, viewport_pos(plot, *steps[-1]))
    assert not live.is_pending()  # applied when let go
    assert c.result.baseline_region == c.processing.baseline
    assert c.processing.baseline[1] == pytest.approx(850, abs=5)
    np.testing.assert_allclose(shown_image(w), fresh_result(w).ratio.values)


class FakeClock:
    """The time for LiveApply, moved by the test (and by the runs it makes take *cost*)."""

    def __init__(self):
        self.now = 100.0
        self.cost = 0.0
        self.calls = 0

    def __call__(self) -> float:
        return self.now

    def run(self):
        self.calls += 1
        self.now += self.cost


def settled(qtbot, live) -> None:
    """Wait until *live* has measured its last run (the event loop came back idle)."""
    qtbot.waitUntil(lambda: not live.is_busy(), timeout=3000)


def test_live_apply_coalesces_a_burst(qtbot):
    clock = FakeClock()
    live = LiveApply(clock.run, clock=clock)
    for _ in range(20):  # at once, then once the interval is over
        live.request()
    assert clock.calls == 1 and live.is_pending() and live.is_busy()
    qtbot.waitUntil(lambda: clock.calls == 2, timeout=2000)
    settled(qtbot, live)
    assert not live.is_pending() and not live.slow and live.runs == 2
    live.flush()  # nothing waits
    assert clock.calls == 2
    live.request()
    live.cancel()
    qtbot.wait(2 * LiveApply.INTERVAL)
    assert clock.calls == 2 and not live.is_pending()


def run_costing(qtbot, live, clock, cost: float, dragging: bool = False) -> None:
    """One change, long after the last run, that runs now and costs *cost* (s), measured."""
    clock.cost = cost
    clock.now += 10.0
    live.request(dragging)
    live.flush()
    settled(qtbot, live)


def test_live_apply_is_slow_with_hysteresis(qtbot):
    slow, fast = LiveApply.SLOW, LiveApply.FAST
    clock = FakeClock()
    live = LiveApply(clock.run, clock=clock)
    run_costing(qtbot, live, clock, 1.2 * slow)  # one run longer than SLOW changes nothing ...
    assert not live.slow
    run_costing(qtbot, live, clock, 1.2 * slow)  # ... two make it slow
    assert live.slow and live.gap() == pytest.approx(2000 * slow)
    live.request(dragging=True)  # a drag then waits for its end
    qtbot.wait(2 * LiveApply.INTERVAL)
    assert clock.calls == 2 and live.is_pending()
    clock.cost = fast / 2
    live.flush()
    settled(qtbot, live)
    assert clock.calls == 3 and live.slow  # one quick run does not make it fast ...
    live.request()  # (typed: once the typing pauses)
    live.request()
    assert clock.calls == 3
    qtbot.waitUntil(lambda: clock.calls == 4, timeout=3000)
    settled(qtbot, live)
    assert not live.slow  # ... a second one does
    assert live.costs() == pytest.approx((1.2 * slow, fast / 2, fast / 2))


def test_a_first_stall_does_not_hold_the_next_update(qtbot):
    clock = FakeClock()
    live = LiveApply(clock.run, clock=clock)
    run_costing(qtbot, live, clock, 20 * LiveApply.SLOW)  # the very first cost is a stall
    assert not live.slow and live.gap() == pytest.approx(2000 * LiveApply.SLOW)
    live.request()  # waits twice SLOW, not twice the stall
    assert clock.calls == 1 and live.is_pending()
    qtbot.waitUntil(lambda: clock.calls == 2, timeout=int(4000 * LiveApply.SLOW) + 200)


def test_live_apply_ignores_a_single_stall(qtbot):
    slow, fast = LiveApply.SLOW, LiveApply.FAST
    clock = FakeClock()
    live = LiveApply(clock.run, clock=clock)
    for _ in range(3):
        run_costing(qtbot, live, clock, fast / 2)
    run_costing(qtbot, live, clock, 3 * slow)  # one stall (another program, a collection)
    assert not live.slow and live.cost() == pytest.approx(fast / 2)
    assert live.gap() == pytest.approx(max(LiveApply.INTERVAL, 1000 * fast))
    run_costing(qtbot, live, clock, 3 * slow)  # two in a row: slow
    assert live.slow
    run_costing(qtbot, live, clock, fast / 2, dragging=True)  # one quick run ...
    assert live.slow
    run_costing(qtbot, live, clock, fast / 2, dragging=True)  # ... and a second: fast again
    assert not live.slow


def test_a_run_during_a_measurement_takes_it_over(qtbot):
    """A run started while the last one is measured is measured from its own start, with the
    drawing it leaves (an older zero timer does not end the measurement early)."""
    clock = FakeClock()

    def run():
        clock.run()
        if clock.calls == 2:
            QTimer.singleShot(0, draw)  # (posted by the run, as a redraw is)

    def draw():
        measuring.append(live.is_busy())
        clock.now += 2 * LiveApply.IDLE

    measuring = []
    live = LiveApply(run, clock=clock)
    clock.cost = LiveApply.IDLE / 2
    live.request()  # runs; its cost is then measured
    assert live.is_busy()
    live.request()
    live.flush()  # e.g. the drag ended: runs now, while the first one is measured
    assert clock.calls == 2 and live.is_busy()
    settled(qtbot, live)
    qtbot.wait(50)  # every zero timer has come back
    assert measuring == [True]  # still measured while the second run is drawn
    assert live.costs() == pytest.approx((2.5 * LiveApply.IDLE,))  # measured once


def test_live_apply_counts_the_drawing_a_run_leaves(qtbot):
    """What a run changed is drawn after it returns: that time counts as its cost."""
    clock = FakeClock()

    def run():
        clock.run()
        QTimer.singleShot(0, draw)  # (posted by the run, as a redraw is)

    def draw():
        clock.now += 2 * LiveApply.IDLE

    live = LiveApply(run, clock=clock)
    clock.cost = LiveApply.IDLE / 2
    live.request()
    assert live.is_busy()
    settled(qtbot, live)
    assert live.last_duration == pytest.approx(2.5 * LiveApply.IDLE)
    live.request()  # waits twice the cost after the drawing, not after the run
    assert clock.calls == 1 and live.is_pending()


def test_live_apply_counts_only_runs_that_did_something(qtbot):
    clock = FakeClock()

    def nothing():
        clock.now += 1.0
        return False

    live = LiveApply(nothing, clock=clock)
    live.request()
    assert live.runs == 0 and not live.slow and not live.is_pending() and not live.is_busy()
    live.request()  # not held back by a run that did nothing
    assert live.runs == 0


@pytest.mark.parametrize("share", [0.0, 0.4, 0.9, 1.5])  # of SLOW: a run and its drawing
def test_live_paces_a_drag_whatever_a_run_costs(processed, qtbot, monkeypatch, share):
    """While the band is dragged, apply_baseline runs at most once per INTERVAL or per its
    cost plus twice that cost (at most twice SLOW) of idle time, the drawing it leaves for later
    included (that drawing takes twice as long as the run, as on the stacked plot).

    The drag, the runs and the waits between them take real time, but LiveApply's clock moves
    only with the simulated work: the real re-apply and painting, and other programs on a
    loaded machine, cost nothing there, so they cannot make it slow."""
    w, c = processed, processed.controller
    panel = w.panels["processing"]
    live = panel.live
    w.resize(1400, 900)
    w.show()
    qtbot.waitExposed(w)
    cost = share * LiveApply.SLOW
    clock = FakeClock()
    live.clock = clock  # (the panel goes with the window)
    starts = []
    original = c.apply_baseline

    def work(seconds: float) -> None:
        """Keep the window busy for *seconds*, on the wall and on LiveApply's clock."""
        time.sleep(seconds)
        clock.now += seconds

    def costly():
        starts.append(time.perf_counter())
        work(cost / 3)
        QTimer.singleShot(0, lambda: work(2 * cost / 3))  # what the run leaves to draw
        return original()

    monkeypatch.setattr(c, "apply_baseline", costly)
    panel.baseline_live.setChecked(True)
    plot = w.plots.map
    steps = [(1.2, 550.0 + 10 * k) for k in range(31)]  # the upper edge, up by 300 cm-1
    start = time.perf_counter()
    mouse_drag(qtbot, plot, steps, release=False)
    elapsed = time.perf_counter() - start
    costs = live.costs()  # (those measured so far: each a run and all of its drawing)
    assert costs and costs == pytest.approx((cost,) * len(costs))
    idle = max(LiveApply.INTERVAL / 1000, 2 * min(cost, LiveApply.SLOW))
    spacing = 0.98 * (cost + idle)  # (timer precision)
    assert 1 <= len(starts) <= 1 + elapsed / spacing
    assert all(b - a >= spacing for a, b in itertools.pairwise(starts))
    QTest.mouseRelease(plot.view.viewport(), LEFT, PLAIN, viewport_pos(plot, *steps[-1]))
    assert c.result.baseline_region == c.processing.baseline  # applied when let go
    assert c.processing.baseline[1] == pytest.approx(850, abs=5)
    settled(qtbot, live)
    assert live.slow == (cost > LiveApply.SLOW)


def test_live_logs_the_region_once_it_rests(processed, qtbot, caplog):
    w, c = processed, processed.controller
    panel = w.panels["processing"]
    panel.baseline_live.setChecked(True)
    caplog.set_level(logging.INFO, logger="mag_opt_detective")
    caplog.clear()

    def logged() -> list[str]:
        return [r.getMessage() for r in caplog.records if "Live" in r.getMessage()]

    for lo in (600, 610, 620):
        drag_like(panel.regions["map"], lo, lo + 100)
        panel.live.flush()
    assert not logged()
    qtbot.waitUntil(lambda: bool(logged()), timeout=5000)
    assert logged() == ["Baseline corrected in range 620 – 720 cm-1 (Live)."]
    drag_like(panel.regions["map"], 700, 800)
    panel.live.flush()
    process(w)  # logs what it applied itself
    qtbot.wait(2 * LOG_DELAY)
    assert len(logged()) == 1 and c.result.baseline_region == (700, 800)


def test_live_switches_the_baseline_off_and_on_at_once(processed):
    w, c = processed, processed.controller
    panel = w.panels["processing"]
    panel.baseline_live.setChecked(True)
    panel.baseline_on.setChecked(False)
    assert c.result.baseline_region is None
    np.testing.assert_allclose(shown_image(w), fresh_result(w).ratio.values)
    assert not c.changed_since_process()
    panel.baseline_on.setChecked(True)
    assert c.result.baseline_region == (450, 550)
    np.testing.assert_allclose(shown_image(w), fresh_result(w).ratio.values)
    panel.baseline_live.setChecked(False)
    panel.baseline_on.setChecked(False)  # Live off: waits for Process again
    assert c.result.baseline_region == (450, 550) and c.changed_since_process()


def test_live_turned_on_applies_what_changed(processed):
    w, c = processed, processed.controller
    panel = w.panels["processing"]
    drag_like(panel.regions["map"], 700, 800)
    assert c.changed_since_process()
    panel.baseline_live.setChecked(True)
    assert c.result.baseline_region == (700, 800) and not c.changed_since_process()


def test_live_explains_a_region_without_data(processed):
    w, c = processed, processed.controller
    panel = w.panels["processing"]
    panel.baseline_live.setChecked(True)
    panel.baseline_lo.setText("1300")
    panel.baseline_hi.setText("1400")
    panel.live.flush()
    assert c.result.baseline_region == (450, 550)  # kept
    assert not panel.live_note.isHidden()
    assert "1300 – 1400 cm⁻¹ contains no data" in panel.live_note.text()
    assert c.changed_since_process()  # not applied, so it waits for Process
    assert not any(r.isVisible() for r in panel.regions.values())
    panel.baseline_lo.setText("1000")
    panel.live.flush()
    assert panel.live_note.isHidden() and c.result.baseline_region == (1000, 1400)


def test_library_maps_are_not_baselined_live(processed, errors):
    w, c = processed, processed.controller
    panel = w.panels["processing"]
    library.add_processed(w)
    library.plot_entry(w, c.library[0].key)
    assert c.result_source == "library"
    shown = shown_image(w).copy()
    assert not panel.live_note.isHidden() and panel.live_note.text() == LIBRARY_NOTE
    assert all(r.isVisible() and not r.movable for r in panel.regions.values())  # only shown
    panel.baseline_live.setChecked(True)
    assert panel.live_note.text() == LIBRARY_NOTE
    drag_like(panel.regions["map"], 900, 1100)  # (typed would do the same)
    panel.live.flush()
    np.testing.assert_array_equal(shown_image(w), shown)
    assert c.changed_since_process()
    with pytest.raises(ValueError, match="library map"):
        c.apply_baseline()
    process(w)
    assert panel.live_note.isHidden() and not c.changed_since_process()
    assert c.result.baseline_region == (900, 1100) and not errors
    assert all(r.movable for r in panel.regions.values())


def test_live_never_shows_the_missing_reference_note(window, lines, errors):
    c = window.controller
    c.set_processing(sample_files=lines, baseline=(450.0, 550.0))
    window.panels["reference"].set_reference_mode(ReferenceMode.SEPARATE)  # without files
    notes = []
    c.referenceMissing.connect(notes.append)
    process(window)
    assert len(notes) == 1
    window.show_panel("processing")
    panel = window.panels["processing"]
    panel.baseline_live.setChecked(True)
    for lo in (500, 600, 700):
        drag_like(panel.regions["map"], lo, lo + 100)
        panel.live.flush()
    panel.baseline_on.setChecked(False)
    assert len(notes) == 1 and c.result.baseline_region is None and not errors


# ---------------------------------------------------------------------- settings
def test_live_and_edit_on_plot_are_remembered(qtbot, tmp_path, lines, errors):
    ini = str(tmp_path / "settings.ini")

    def make():
        w = MainWindow(settings=QSettings(ini, QSettings.Format.IniFormat))
        qtbot.addWidget(w)
        return w

    w = make()
    panel = w.panels["processing"]
    panel.baseline_on.setChecked(True)
    panel.baseline_lo.setText("450")
    panel.baseline_hi.setText("550")
    panel.baseline_live.setChecked(True)
    panel.edit_on_plot.setChecked(True)
    w.save_settings()
    w.close()

    w2 = make()
    c, panel = w2.controller, w2.panels["processing"]
    assert panel.baseline_live.isChecked() and panel.edit_on_plot.isChecked()
    assert c.live_baseline() and c.result is None  # nothing to apply it to yet
    assert c.processing.baseline == (450.0, 550.0)
    c.set_processing(sample_files=lines)
    process(w2)
    assert c.result.baseline_region == (450.0, 550.0) and not c.changed_since_process()
    assert panel.regions["map"].isVisible()  # Edit on plot: with any panel
    drag_like(panel.regions["map"], 600, 700)
    panel.live.flush()
    assert c.result.baseline_region == (600, 700) and not c.changed_since_process()
    w2.close()
    assert not errors


def test_restored_settings_are_applied_once_after_the_restore(processed):
    w, c = processed, processed.controller
    panel = w.panels["processing"]
    panel.baseline_live.setChecked(True)
    applied = []
    c.resultChanged.connect(lambda: applied.append(c.result.baseline_region))
    with c.restoring():  # as View > Reset settings or a restore restores the fields
        panel.baseline_lo.set_cm1(700.0)
        panel.baseline_hi.set_cm1(800.0)
        assert not applied and not panel.live.is_pending()
    assert applied == [(700.0, 800.0)] and not c.changed_since_process()


# ---------------------------------------------------------------------- controller
def test_apply_baseline_needs_a_processed_map(qapp, lines):
    c = AppController()
    c.set_processing(baseline=(450.0, 550.0))
    with pytest.raises(ValueError, match="no map is processed"):
        c.apply_baseline()
    c.set_processing(sample_files=lines)
    c.process()
    assert not c.apply_baseline()  # nothing changed
    c.set_processing(baseline=(600.0, 700.0))
    assert c.changed_since_process()
    c.set_live_baseline(True)
    assert not c.changed_since_process()  # counts as applied ...
    assert c.result.baseline_region == (450.0, 550.0)  # ... once apply_baseline runs
    assert c.apply_baseline() and c.result.baseline_region == (600.0, 700.0)
    c.set_live_baseline(False)
    assert not c.changed_since_process()
    c.set_processing(baseline=(600.0, None))
    with pytest.raises(ValueError, match="enter both limits"):
        c.apply_baseline()
    c.set_unit("meV")
    c.set_processing(baseline=(5000.0, 6000.0))
    with pytest.raises(ValueError, match=r"619\.917 – 743\.9 meV contains no data"):
        c.apply_baseline()


def test_live_baseline_on_real_data(window, data_dir, errors):
    files = sorted((data_dir / "Data_Macro").glob("*.txt"))
    c = window.controller
    zero = tuple(str(f) for f in files if "_a00p000T_a" in f.name)
    c.set_processing(
        sample_files=SweepFiles(zero, tuple(str(f) for f in files if str(f) not in zero))
    )
    process(window)
    energy = c.result.ratio.energy
    lo, hi = float(energy[energy.size // 10]), float(energy[energy.size // 5])
    window.show_panel("processing")
    panel = window.panels["processing"]
    panel.baseline_live.setChecked(True)
    panel.baseline_on.setChecked(True)
    drag_like(panel.regions["map"], *from_cm1(np.array([lo, hi]), c.unit))
    panel.live.flush()
    np.testing.assert_allclose(shown_image(window), fresh_result(window).ratio.values)
    assert not errors


# ---------------------------------------------------------------------- review round
def test_the_band_moves_only_with_the_pan_tool(processed, qtbot):
    w, c = processed, processed.controller
    regions = w.panels["processing"].regions
    assert w.tools.active() == "navigate"
    assert all(r.movable and r.hasCursor() for r in regions.values())
    others = [name for name in w.tools.names() if name != "navigate"]
    assert {"zoom", "pick", "autopick"} <= set(others)
    for name in others:
        assert w.tools.set_active(name)
        assert not any(r.movable or r.hasCursor() for r in regions.values()), name
        assert all(r.isVisible() for r in regions.values())  # still shown
    w.tools.set_active("navigate")
    assert all(r.movable for r in regions.values())

    w.resize(1400, 900)
    w.show()
    qtbot.waitExposed(w)
    w.tools.set_active("zoom")
    plot = w.plots.map
    before = plot.plot.vb.viewRange()
    mouse_drag(qtbot, plot, [(0.8, 500), (1.0, 600), (1.6, 900)])  # starts inside the band
    assert c.processing.baseline == (450.0, 550.0)  # the band stayed ...
    assert plot.plot.vb.viewRange() != before  # ... and the box zoomed
    w.tools.set_active("navigate")
    mouse_drag(qtbot, plot, [(0.8, 500), (0.8, 550), (0.8, 600)])  # Pan: the band moves
    assert c.processing.baseline[0] == pytest.approx(550, abs=5)


def test_the_stacked_fill_lies_below_the_traces(processed, qtbot):
    w = processed
    region = w.panels["processing"].regions["stacked"]
    curves = w.plots.stacked.curves()
    assert region.fill.zValue() < min(curve.zValue() for curve in curves) < region.zValue()
    assert region.fill.getRegion() == pytest.approx((450, 550))
    assert region.fill.isVisible()
    w.show_panel("sample")
    assert not region.isVisible() and not region.fill.isVisible()
    w.show_panel("processing")
    assert region.fill.isVisible()

    w.resize(1400, 900)
    w.show()
    qtbot.waitExposed(w)
    w.plot_area.set_current_view("stacked")
    qtbot.wait(50)
    drag_like(region, 500, 900)
    assert region.fill.getRegion() == pytest.approx(region.region())
    assert region.label.format == "Baseline" and region.label.isVisibleTo(region)
    drag_like(region, 500, 510)  # narrower than its label: hidden
    assert not region.label.isVisibleTo(region)
    drag_like(region, 500, 900)
    assert region.label.isVisibleTo(region)


def test_the_subtitle_says_when_the_baseline_applies(processed):
    w, c = processed, processed.controller
    panel = w.panels["processing"]
    subtitle = w.panel_pages["processing"].subtitle
    assert subtitle.text() == APPLIED
    panel.baseline_live.setChecked(True)
    assert subtitle.text() == LIVE_APPLIED
    panel.cut_on.setChecked(True)
    panel.cut_lo.setText("450")
    assert subtitle.text() == CHANGED
    panel.cut_on.setChecked(False)
    assert subtitle.text() == LIVE_APPLIED
    library.add_processed(w)
    library.plot_entry(w, c.library[0].key)  # Live waits for a processed map
    assert subtitle.text() == APPLIED
    process(w)
    assert subtitle.text() == LIVE_APPLIED
    panel.baseline_live.setChecked(False)
    assert subtitle.text() == APPLIED


def test_a_library_save_or_plot_applies_a_waiting_region(processed, qtbot, errors):
    w, c = processed, processed.controller
    panel = w.panels["processing"]
    panel.baseline_live.setChecked(True)
    drag_like(panel.regions["map"], 600, 700)
    assert c.result.baseline_region == (600, 700)
    panel.live.slow = True  # typed changes wait until the typing pauses
    panel.baseline_hi.setText("1000")
    assert panel.live.is_pending() and c.result.baseline_region == (600, 700)
    assert not c.changed_since_process()  # it counts as applied ...
    library.add_processed(w)  # ... so the map saved has it
    np.testing.assert_allclose(c.library[-1].fmap.values, fresh_result(w).ratio.values)
    assert c.result.baseline_region == (600, 1000)

    panel.baseline_lo.setText("700")
    assert panel.live.is_pending()
    library.plot_entry(w, c.library[0].key)  # the processed map leaves with its region
    assert c.result_source == "library" and not c.changed_since_process()
    qtbot.wait(2 * LiveApply.SETTLE)  # the waiting change has nothing left to do
    assert not panel.live.is_pending() and not errors


def test_live_equals_a_fresh_process_with_a_separate_reference(window, lines, reference):
    c = window.controller
    c.set_processing(
        sample_files=lines,
        reference_files=reference,
        reference_mode=ReferenceMode.SEPARATE,
        baseline=(450.0, 550.0),
    )
    process(window)
    window.show_panel("processing")
    panel = window.panels["processing"]
    panel.baseline_live.setChecked(True)
    drag_like(panel.regions["map"], 800, 950)
    panel.live.flush()
    fresh = fresh_result(window)
    assert c.result.baseline_region == fresh.baseline_region == (800, 950)
    for kind, order in itertools.product(PlotKind, (0, 1)):
        np.testing.assert_array_equal(
            c.result.get(kind, order).values, fresh.get(kind, order).values
        )
    np.testing.assert_array_equal(c.result.reference_ratio.values, fresh.reference_ratio.values)
    np.testing.assert_allclose(shown_image(window), fresh.ratio.values)
    np.testing.assert_allclose(window.plots.reference.image.image, fresh.reference_ratio.values)


def test_live_equals_a_fresh_process_inside_the_energy_window(processed):
    w, c = processed, processed.controller
    panel = w.panels["processing"]
    panel.cut_on.setChecked(True)
    panel.cut_lo.setText("450")
    panel.cut_hi.setText("1100")
    process(w)
    panel.baseline_live.setChecked(True)
    drag_like(panel.regions["stacked"], 1000, 1200)  # stops at the window's end
    panel.live.flush()
    assert c.processing.baseline == (1000, 1100) == c.result.baseline_region
    fresh = fresh_result(w)
    for kind, order in itertools.product(PlotKind, (0, 2)):
        np.testing.assert_array_equal(
            c.result.get(kind, order).values, fresh.get(kind, order).values
        )
    np.testing.assert_allclose(shown_image(w), fresh.ratio.values)


def test_live_per_unit_derivatives_after_a_unit_switch(processed):
    w, c = processed, processed.controller
    panel = w.panels["processing"]
    set_unit(w, "meV")
    select(w, order=1, per_unit=True)
    panel.baseline_live.setChecked(True)
    drag_like(panel.regions["map"], 100.0, 120.0)  # meV
    panel.live.flush()
    assert c.result.baseline_region == pytest.approx(
        (100 * CM1_PER_UNIT[Unit.MEV], 120 * CM1_PER_UNIT[Unit.MEV])
    )
    for unit in (Unit.MEV, Unit.THZ, Unit.CM1):
        set_unit(w, unit)
        expected = fresh_result(w).get(PlotKind.RATIO, 1, Axis.ENERGY, physical=True, unit=unit)
        np.testing.assert_allclose(shown_image(w), expected.values)


def test_process_during_a_drag(processed, qtbot, errors):
    w, c = processed, processed.controller
    panel = w.panels["processing"]
    w.resize(1400, 900)
    w.show()
    qtbot.waitExposed(w)
    panel.baseline_live.setChecked(True)
    plot = w.plots.map
    viewport = plot.view.viewport()
    pixels = [viewport_pos(plot, 1.2, energy) for energy in (550, 600, 650, 700)]
    QTest.mouseMove(viewport, pixels[0])
    QTest.mousePress(viewport, LEFT, PLAIN, pixels[0])
    for pixel in pixels[1:3]:
        qtbot.wait(move_pause_ms())
        QTest.mouseMove(viewport, pixel)
    process(w)  # Ctrl+Return while the edge is held
    qtbot.wait(move_pause_ms())
    QTest.mouseMove(viewport, pixels[3])
    QTest.mouseRelease(viewport, LEFT, PLAIN, pixels[3])
    assert c.processing.baseline[1] == pytest.approx(700, abs=5)
    assert c.result.baseline_region == c.processing.baseline
    assert not c.changed_since_process() and not errors
    np.testing.assert_allclose(shown_image(w), fresh_result(w).ratio.values)


def test_a_waiting_apply_after_the_window_closes(qtbot, lines, errors):
    w = MainWindow()
    c, panel = w.controller, w.panels["processing"]
    c.set_processing(sample_files=lines, baseline=(450.0, 550.0))
    process(w)
    panel.baseline_live.setChecked(True)
    panel.live.slow = True  # changes wait until the typing pauses
    panel.baseline_hi.setText("650")
    assert panel.live.is_pending()
    w.close()
    qtbot.wait(2 * LiveApply.SETTLE)  # applied to the closed window
    assert c.result.baseline_region == (450.0, 650.0)
    panel.live.slow = True
    panel.baseline_hi.setText("700")
    assert panel.live.is_pending()
    w.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    del w, c, panel
    qtbot.wait(2 * LiveApply.SETTLE)  # its timer went with the window: nothing runs
    assert not errors


@pytest.mark.parametrize("grip", ["edge", "band"])
def test_a_tool_switch_during_a_drag_ends_it(processed, qtbot, grip):
    w, c = processed, processed.controller
    panel = w.panels["processing"]
    region = panel.regions["map"]
    w.resize(1400, 900)
    w.show()
    qtbot.waitExposed(w)
    panel.baseline_live.setChecked(True)
    panel.live.slow = True  # changes made while dragging wait for the release
    plot = w.plots.map
    viewport = plot.view.viewport()
    start = 550 if grip == "edge" else 500  # the upper edge, or inside the band
    pixels = [viewport_pos(plot, 1.2, start + 50 * k) for k in range(4)]
    QTest.mouseMove(viewport, pixels[0])
    QTest.mousePress(viewport, LEFT, PLAIN, pixels[0])
    for pixel in pixels[1:3]:
        qtbot.wait(move_pause_ms())
        QTest.mouseMove(viewport, pixel)
    assert region.is_dragging() and panel.live.is_pending()
    w.tools.set_active("zoom")  # Z before letting go: the drag ends there
    assert not region.is_dragging() and not panel.live.is_pending()
    moved = c.processing.baseline
    assert moved[1] == pytest.approx(650, abs=5)
    assert c.result.baseline_region == moved and not c.changed_since_process()
    qtbot.wait(move_pause_ms())
    QTest.mouseMove(viewport, pixels[3])
    QTest.mouseRelease(viewport, LEFT, PLAIN, pixels[3])
    assert c.processing.baseline == moved  # the rest of that drag is ignored
    panel.live.slow = True
    panel.baseline_hi.setText("1000")  # typed changes apply as usual (once typing pauses)
    qtbot.waitUntil(lambda: c.result.baseline_region == c.processing.baseline, timeout=3000)
    assert c.processing.baseline[1] == 1000 and not c.changed_since_process()


def test_turning_live_off_applies_a_waiting_change(processed, qtbot, caplog):
    w, c = processed, processed.controller
    panel = w.panels["processing"]
    panel.baseline_live.setChecked(True)
    caplog.set_level(logging.INFO, logger="mag_opt_detective")
    panel.live.slow = True  # typed changes wait until the typing pauses
    panel.baseline_hi.setText("1100")
    panel.baseline_lo.setText("900")
    assert panel.live.is_pending() and c.result.baseline_region == (450, 550)
    panel.baseline_live.setChecked(False)
    assert not panel.live.is_pending()
    assert c.result.baseline_region == (900, 1100) and not c.changed_since_process()
    np.testing.assert_allclose(shown_image(w), fresh_result(w).ratio.values)
    qtbot.waitUntil(
        lambda: any("900 – 1100 cm-1 (Live)" in r.getMessage() for r in caplog.records),
        timeout=3000,
    )


def test_reset_settings_with_live_on_applies_nothing(qtbot, tmp_path, lines, errors):
    w = MainWindow(settings=QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat))
    qtbot.addWidget(w)
    c, panel = w.controller, w.panels["processing"]
    c.set_processing(sample_files=lines, baseline=(700.0, 900.0))
    process(w)
    panel.baseline_live.setChecked(True)
    assert c.result.baseline_region == (700.0, 900.0) and not c.changed_since_process()
    during = []
    c.resultChanged.connect(lambda: during.append(c.is_restoring()))
    w.reset_settings()  # the baseline goes off before Live does
    assert not panel.baseline_live.isChecked() and not panel.baseline_on.isChecked()
    assert True not in during  # nothing applied while restoring
    assert c.result.baseline_region == (700.0, 900.0)  # the processed map stays ...
    assert c.changed_since_process()  # ... and its baseline is now a change to process
    assert not errors
    w.close()


def test_baseline_is_live(processed):
    w, c = processed, processed.controller
    panel = w.panels["processing"]
    assert not c.baseline_is_live()  # Live off
    panel.baseline_live.setChecked(True)
    assert c.baseline_is_live()
    c.set_processing(baseline=(1300.0, 1400.0))  # no data there: not applied
    assert not c.baseline_is_live() and c.result.baseline_region == (450.0, 550.0)
    c.set_processing(baseline=None)
    assert c.baseline_is_live() and c.result.baseline_region is None
    library.add_processed(w)
    library.plot_entry(w, c.library[0].key)
    assert not c.baseline_is_live()  # a library map keeps its baseline


def test_processed_options_follow_process_and_live(processed):
    w, c = processed, processed.controller
    assert c.processed_options() == c.processing
    c.set_processing(energy_cut=(500.0, None), baseline=(600.0, 700.0))
    used = c.processed_options()
    assert used.energy_cut is None and used.baseline == (450.0, 550.0)
    w.panels["processing"].baseline_live.setChecked(True)  # applies the region, not the window
    assert c.processed_options().baseline == (600.0, 700.0)
    assert c.processed_options().energy_cut is None
    process(w)
    assert c.processed_options() == c.processing
    assert AppController().processed_options() is None
