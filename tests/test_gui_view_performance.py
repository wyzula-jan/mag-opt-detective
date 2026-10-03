"""Panning and zooming the plots: the plot moves at once, the View fields follow at most every
GESTURE_MS, and the view state (with the other plots) takes the ranges once per gesture."""

import json
import math
import time

import pyqtgraph as pg
import pytest
from PySide6.QtCore import QPoint, QPointF, QSettings, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

import gui_helpers
from gui_helpers import inspector_page, load_sweep, process, set_unit
from mag_opt_detective.gui import plot_panel
from mag_opt_detective.gui.inspector.view import GESTURE_MS
from mag_opt_detective.gui.main_window import MainWindow

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
    set_unit(w, "meV")  # a step in the old unit is dropped
    assert not w.view_ranges.in_gesture()
    assert c.view.energy_range == pytest.approx((150.0 / MEV, 1050.0 / MEV))
    assert plot_range(w)[1] == pytest.approx((150.0 / MEV, 1050.0 / MEV))

    step(w, "stacked", dy=0.5)  # the stacked intensity, also with the other tab on screen
    lo, hi = plot_range(w, "stacked")[1]
    inspector_page(w, "view").field.lo_spin.setValue(1.0)  # an edit goes after the drag
    assert c.view.stacked_range == pytest.approx((lo, hi))
    assert c.view.field_range == (1.0, 2.0)


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
