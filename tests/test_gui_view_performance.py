"""Panning and zooming the plots: the plot moves at once, the View fields follow at most every
GESTURE_MS, and the view state (with the other plots) takes the ranges once per gesture."""

import json
import math
import time

import numpy as np
import pyqtgraph as pg
import pytest
from PySide6.QtCore import QPoint, QPointF, QSettings, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

import gui_helpers
from gui_helpers import inspector_page, load_sweep, process, set_unit
from mag_opt_detective.core.spectra import FieldMap
from mag_opt_detective.gui import plot_panel
from mag_opt_detective.gui.inspector.view import GESTURE_MS
from mag_opt_detective.gui.main_window import MainWindow
from mag_opt_detective.gui.plots import StackedPlot

window, errors = gui_helpers.window, gui_helpers.errors  # shared fixtures

LEFT, PLAIN = Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier
MEV = 8.0656
DATA_E = (100.0, 1000.0)  # the energies of the sweep fixture


@pytest.fixture
def shown(window, sweep, qtbot):
    window.resize(1400, 900)
    window.show()
    qtbot.waitExposed(window)
    load_sweep(window, sweep)
    process(window)
    return window


class Calls:
    """The view state writes (rangesChanged) and the ranges shown by a View range control."""

    def __init__(self, window, monkeypatch, control):
        self.states = 0
        self.shown: list[tuple[float, float]] = []
        window.controller.rangesChanged.connect(self._on_state)
        set_range = control.set_range

        def spy(lo: float, hi: float) -> None:
            self.shown.append((lo, hi))
            set_range(lo, hi)

        monkeypatch.setattr(control, "set_range", spy)

    def _on_state(self) -> None:
        self.states += 1


def plot_range(window, view: str = "map") -> list[tuple[float, float]]:
    return [(float(lo), float(hi)) for lo, hi in window.plots[view].plot.vb.viewRange()]


def step(window, view: str, dx: float = 0.0, dy: float = 0.0, scale=None) -> None:
    """One pan (or zoom) step on plot *view*, as pyqtgraph makes it on a mouse move."""
    vb = window.plots[view].plot.vb
    if scale is not None:
        vb.scaleBy(x=scale[0], y=scale[1])
    else:
        vb.translateBy(x=dx or None, y=dy or None)
    vb.sigRangeChangedManually.emit(vb.state["mouseEnabled"])


def wheel(window, view: str) -> None:
    plot = window.plots[view]
    centre = plot.view.mapFromScene(plot.plot.vb.sceneBoundingRect().center())
    viewport = plot.view.viewport()
    event = QWheelEvent(
        QPointF(centre), QPointF(viewport.mapToGlobal(centre)), QPoint(0, 0), QPoint(0, 120),
        Qt.MouseButton.NoButton, PLAIN, Qt.ScrollPhase.NoScrollPhase, False,
    )  # fmt: skip
    QApplication.sendEvent(viewport, event)


def move_pause_ms() -> int:
    """pyqtgraph drops mouse moves within 1/mouseRateLimit s of the last one."""
    rate = pg.getConfigOption("mouseRateLimit")
    return 2 * math.ceil(1000 / rate) if rate > 0 else 0


def most_follows(elapsed_s: float) -> float:
    """How often the fields may follow a gesture of *elapsed_s* (timers can fire 5 % early)."""
    return elapsed_s * 1000 / (0.9 * GESTURE_MS) + 1


# ---------------------------------------------------------------------- gestures
def test_range_steps_go_into_the_state_once_with_the_final_ranges(shown, monkeypatch):
    w, c = shown, shown.controller
    page = inspector_page(w, "view")
    calls = Calls(w, monkeypatch, page.energy)
    before = plot_range(w, "stacked")
    for _ in range(50):  # a drag of 50 moves, without time for the timers
        step(w, "map", dx=0.01, dy=4.0)
    moved = plot_range(w)
    assert moved[0] == pytest.approx((1.0, 2.5)) and moved[1] == pytest.approx((300.0, 1200.0))
    assert w.view_ranges.in_gesture()
    assert (calls.states, calls.shown) == (0, [])  # neither the state nor the fields
    assert (c.view.field_range, c.view.energy_range) == (None, None)
    assert plot_range(w, "stacked") == before  # the other plots wait for the end

    w.view_ranges.finish()  # what the release of the mouse button does
    assert not w.view_ranges.in_gesture()
    assert calls.states == 1
    assert calls.shown == [pytest.approx(moved[1])]  # the fields: once, with the final range
    assert c.view.field_range == pytest.approx(moved[0])
    assert c.view.energy_range == pytest.approx(moved[1])
    assert plot_range(w, "stacked")[0] == pytest.approx(moved[1])  # shared
    assert plot_range(w) == pytest.approx(moved)
    assert not page.energy.is_auto() and page.energy.range() == pytest.approx(moved[1])
    assert not page.field.is_auto() and page.field.range() == pytest.approx(moved[0])
    w.view_ranges.finish()  # nothing left to do
    assert calls.states == 1


def test_a_mouse_drag_updates_the_fields_while_held_and_the_state_on_release(
    shown, qtbot, monkeypatch
):
    w, c = shown, shown.controller
    page = inspector_page(w, "view")
    calls = Calls(w, monkeypatch, page.energy)
    before = plot_range(w)
    plot = w.plots.map
    viewport = plot.view.viewport()
    start = plot.view.mapFromScene(plot.plot.vb.sceneBoundingRect().center())
    end = start + QPoint(80, -60)
    QTest.mousePress(viewport, LEFT, PLAIN, start)
    began = time.perf_counter()
    for k in range(1, 21):
        qtbot.wait(move_pause_ms())
        QTest.mouseMove(viewport, start + QPoint(4 * k, -3 * k))
    qtbot.wait(3 * GESTURE_MS)  # held still: the drag goes on
    elapsed = time.perf_counter() - began
    moved = plot_range(w)
    assert moved[0][0] < before[0][0] and moved[1][0] < before[1][0]  # dragged right and up
    assert w.view_ranges.in_gesture() and calls.states == 0
    assert (c.view.field_range, c.view.energy_range) == (None, None)
    assert 1 <= len(calls.shown) <= most_follows(elapsed)  # the fields follow, not every move
    assert calls.shown[-1] == pytest.approx(moved[1])

    QTest.mouseRelease(viewport, LEFT, PLAIN, end)
    qtbot.waitUntil(lambda: not w.view_ranges.in_gesture())
    assert plot_range(w) == pytest.approx(moved)
    assert calls.states == 1
    assert c.view.field_range == pytest.approx(moved[0])
    assert c.view.energy_range == pytest.approx(moved[1])
    assert calls.shown[-1] == pytest.approx(moved[1])
    assert plot_range(w, "stacked")[0] == pytest.approx(moved[1])


def test_fields_follow_a_long_gesture_at_most_every_gesture_ms(shown, qtbot, monkeypatch):
    w = shown
    calls = Calls(w, monkeypatch, inspector_page(w, "view").energy)
    viewport = w.plots.map.view.viewport()
    QTest.mousePress(viewport, Qt.MouseButton.MiddleButton, PLAIN, QPoint(5, 5))  # held
    began = time.perf_counter()
    for _ in range(50):
        step(w, "map", dy=2.0)
        qtbot.wait(5)
    elapsed = time.perf_counter() - began
    assert calls.states == 0
    assert 1 <= len(calls.shown) <= most_follows(elapsed)
    shown_before = len(calls.shown)
    QTest.mouseRelease(viewport, Qt.MouseButton.MiddleButton, PLAIN, QPoint(5, 5))
    qtbot.waitUntil(lambda: not w.view_ranges.in_gesture())
    assert calls.states == 1
    assert len(calls.shown) <= shown_before + 2  # at most one late follow, then the end
    assert calls.shown[-1] == pytest.approx(w.controller.view.energy_range)
    assert w.controller.view.energy_range == pytest.approx((200.0, 1100.0))


def test_wheel_zoom_goes_into_the_state_after_a_pause(shown, qtbot, monkeypatch):
    w, c = shown, shown.controller
    calls = Calls(w, monkeypatch, inspector_page(w, "view").energy)
    for _ in range(3):
        wheel(w, "map")
    zoomed = plot_range(w)
    lo, hi = zoomed[1]
    assert DATA_E[0] < lo < hi < DATA_E[1]  # the plot zooms at once
    assert w.view_ranges.in_gesture() and calls.states == 0
    qtbot.waitUntil(lambda: not w.view_ranges.in_gesture(), timeout=5000)
    assert calls.states == 1
    assert c.view.field_range == pytest.approx(zoomed[0])
    assert c.view.energy_range == pytest.approx(zoomed[1])
    assert calls.shown[-1] == pytest.approx(zoomed[1])


def test_changes_during_a_gesture(shown, qtbot):
    w, c = shown, shown.controller
    step(w, "map", dy=50.0)
    plot_panel.fit_to_data(w)  # Fit wins over the drag in progress
    assert not w.view_ranges.in_gesture()
    assert (c.view.field_range, c.view.energy_range) == (None, None)
    assert plot_range(w)[1] == DATA_E

    step(w, "map", dy=50.0)
    c.set_view(colormap="grey")  # a redraw draws the state's ranges: the drag goes in first
    assert not w.view_ranges.in_gesture()
    assert c.view.energy_range == pytest.approx((150.0, 1050.0))
    assert plot_range(w)[1] == pytest.approx((150.0, 1050.0))

    step(w, "map", dy=50.0)
    set_unit(w, "meV")  # the drag goes in converted, as the state's ranges are
    assert not w.view_ranges.in_gesture()
    assert c.view.energy_range == pytest.approx((200.0 / MEV, 1100.0 / MEV))
    assert plot_range(w)[1] == pytest.approx((200.0 / MEV, 1100.0 / MEV))
    assert inspector_page(w, "view").energy.range() == pytest.approx((200.0 / MEV, 1100.0 / MEV))

    step(w, "stacked", dy=0.5)  # the stacked intensity, also with the other tab on screen
    lo, hi = plot_range(w, "stacked")[1]
    inspector_page(w, "view").field.lo_spin.setValue(1.0)  # an edit goes after the drag
    assert c.view.stacked_range == pytest.approx((lo, hi))
    assert c.view.field_range == (1.0, 2.0)


def test_a_unit_switch_converts_a_zoom_in_progress(shown):
    w, c = shown, shown.controller
    page = inspector_page(w, "view")
    for _ in range(2):  # wheel steps, then the unit within GESTURE_MS
        wheel(w, "map")
    field, energy = plot_range(w)
    set_unit(w, "meV")
    assert not w.view_ranges.in_gesture()
    assert c.view.field_range == pytest.approx(field)
    assert c.view.energy_range == pytest.approx((energy[0] / MEV, energy[1] / MEV))
    assert plot_range(w) == [pytest.approx(field), pytest.approx(c.view.energy_range)]
    assert not page.field.is_auto() and not page.energy.is_auto()
    assert page.energy.range() == pytest.approx(c.view.energy_range)
    set_unit(w, "cm-1")
    assert c.view.energy_range == pytest.approx(energy)

    gui_helpers.select(w, order=1, per_unit=True)  # intensities per cm-1 scale with the unit
    step(w, "stacked", dy=0.002)
    lo, hi = plot_range(w, "stacked")[1]
    set_unit(w, "meV")
    assert c.view.stacked_range == pytest.approx((lo * MEV, hi * MEV))
    assert plot_range(w, "stacked")[1] == pytest.approx((lo * MEV, hi * MEV))


def test_a_drag_in_progress_is_saved(qtbot, tmp_path, sweep):
    ini = str(tmp_path / "settings.ini")
    w = MainWindow(settings=QSettings(ini, QSettings.Format.IniFormat))
    qtbot.addWidget(w)
    load_sweep(w, sweep)
    process(w)
    step(w, "map", dy=50.0)
    w.close()
    ranges = json.loads(QSettings(ini, QSettings.Format.IniFormat).value("v2/view/ranges"))
    assert ranges["energy_cm1"] == pytest.approx([150.0, 1050.0])


# ---------------------------------------------------------------------- range sliders
def test_a_range_slider_keeps_its_extent_while_dragged(shown, qtbot):
    w, c = shown, shown.controller
    control = inspector_page(w, "view").energy
    slider = control.slider
    assert slider.isVisible() and slider.width() > 100
    c.set_ranges(energy_range=(500.0, 1300.0))  # partly beyond the data
    assert slider.extent() == (100.0, 1300.0)  # the data and the range
    assert plot_range(w, "stacked")[0] == (500.0, 1300.0)
    per_px = 1200.0 / (slider.x_for(1300.0) - slider.x_for(100.0))
    y = slider.height() // 2
    x0 = round(slider.x_for(900.0))  # on the bar between the handles
    dx = math.ceil(110.0 / per_px)  # three moves bring the range back inside the data
    qtbot.mousePress(slider, LEFT, pos=QPoint(x0, y))
    for k in range(1, 4):  # the extent would shrink after each
        qtbot.mouseMove(slider, QPoint(x0 - dx * k, y))
        assert slider.extent() == (100.0, 1300.0)  # kept while dragged
        lo, hi = control.range()
        assert lo == pytest.approx(500.0 - dx * k * per_px, abs=0.5 * per_px)  # under the cursor
        assert hi - lo == pytest.approx(800.0)
        assert c.view.energy_range == pytest.approx((lo, hi))  # the plot follows at once
        assert plot_range(w)[1] == pytest.approx((lo, hi))
        assert plot_range(w, "stacked")[0] == (500.0, 1300.0)  # off screen: on the release
    qtbot.mouseRelease(slider, LEFT, pos=QPoint(x0 - 3 * dx, y))
    qtbot.waitUntil(lambda: slider.extent() != (100.0, 1300.0))
    lo, hi = control.range()
    assert hi < 1000.0 and slider.extent() == (100.0, 1000.0)  # the data again
    assert plot_range(w, "stacked")[0] == pytest.approx((lo, hi))

    c.set_ranges(energy_range=(500.0, 1300.0))
    qtbot.keyPress(slider, Qt.Key.Key_Left, Qt.KeyboardModifier.ShiftModifier)  # 1 % left
    assert control.range() == pytest.approx((488.0, 1288.0))
    assert slider.extent() == (100.0, 1300.0)  # kept until the key is released
    qtbot.keyRelease(slider, Qt.Key.Key_Left, Qt.KeyboardModifier.ShiftModifier)
    qtbot.waitUntil(lambda: slider.extent() == pytest.approx((100.0, 1288.0)))


# ---------------------------------------------------------------------- stacked redraw
def test_a_stacked_redraw_reuses_its_traces(qtbot):
    energy = np.linspace(100.0, 1000.0, 50)
    first = FieldMap(energy, np.array([0.5, 1.0, 1.5]), np.ones((50, 3)))
    second = FieldMap(energy, np.array([1.0, 2.0, 4.0]), np.cos(energy / 90.0)[:, None] + [0, 1, 2])
    stacked = StackedPlot()
    qtbot.addWidget(stacked)
    stacked.set_map(first, 0.5)
    curves = stacked.curves()
    redrawn = []
    stacked.tracesChanged.connect(lambda: redrawn.append(True))
    stacked.set_map(second, 2.0)  # a new result of the same shape: the same items
    assert all(a is b for a, b in zip(stacked.curves(), curves, strict=True))
    assert stacked.plot.listDataItems() == curves and redrawn == [True]
    for k, curve in enumerate(curves):
        np.testing.assert_allclose(curve.getData()[1], second.values[:, k] + 2.0 * k)

    stacked.set_trace_options(color_by_field=True, cmap="grey")  # new pens, the same items
    fresh = StackedPlot()
    qtbot.addWidget(fresh)
    fresh.set_trace_options(color_by_field=True, cmap="grey")
    fresh.set_map(second, 2.0)
    assert all(a is b for a, b in zip(stacked.curves(), curves, strict=True))
    for ours, theirs in zip(stacked.curves(), fresh.curves(), strict=True):
        assert pg.mkPen(ours.opts["pen"]).color() == pg.mkPen(theirs.opts["pen"]).color()
        np.testing.assert_array_equal(ours.getData()[1], theirs.getData()[1])

    stacked.set_trace_options(every=2)  # another number of traces: built again
    assert len(stacked.curves()) == 2 and not set(stacked.curves()) & set(curves)
    assert stacked.plot.listDataItems() == stacked.curves()
    np.testing.assert_allclose(stacked.curves()[1].getData()[1], second.values[:, 2] + 2.0)
    stacked.clear_map()
    assert stacked.curves() == [] and stacked.plot.listDataItems() == []
