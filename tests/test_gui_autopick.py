"""The Auto-pick tool: Track and Detect on the map, the preview, Accept/Discard, options, units."""

import math

import numpy as np
import pyqtgraph as pg
import pytest
from PySide6.QtCore import QPoint, QPointF, QSettings, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

import gui_helpers
from gui_helpers import click_map, process, select, set_unit
from helpers import sweep_name, write_text
from mag_opt_detective.gui.controller import SweepFiles
from mag_opt_detective.gui.main_window import MainWindow
from mag_opt_detective.gui.tools import autopick
from mag_opt_detective.gui.tools.autopick import (
    Candidate,
    auto_window,
    default_choice,
    prominence_scale,
)

window, errors = gui_helpers.window, gui_helpers.errors  # shared fixtures

MEV = 8.0656
ENERGY = np.linspace(200.0, 1200.0, 501)  # cm-1
STEP = 2.0  # cm-1, the energy step
FIELDS = np.arange(0.5, 8.01, 0.25)


def line1(b):
    return 400.0 + 40.0 * b


def line2(b):
    return 900.0 + 15.0 * b


@pytest.fixture
def lines_sweep(tmp_path):
    """A sweep whose R(B)/R(0) has two dips: 0.08 deep along line1, 0.05 along line2."""
    base = 1.0 + 0.2 * np.sin(ENERGY / 90.0)
    rng = np.random.default_rng(7)
    zero = [
        write_text(tmp_path / "Lines_a00p000T_a00p000T.txt", ENERGY, base),
        write_text(tmp_path / "Lines_a00p000T_a08p000T.txt", ENERGY, base),
    ]
    field = []
    for b in FIELDS:
        dips = 0.08 * np.exp(-(((ENERGY - line1(b)) / 9.0) ** 2))
        dips += 0.05 * np.exp(-(((ENERGY - line2(b)) / 9.0) ** 2))
        noise = 1.0 + 0.002 * rng.standard_normal(ENERGY.size)
        values = (1.0 + 0.01 * b) * base * (1.0 - dips) * noise
        field.append(
            write_text(tmp_path / sweep_name(b).replace("Sample", "Lines"), ENERGY, values)
        )
    return {"zero": zero, "field": field}


@pytest.fixture
def picked(window, lines_sweep):
    """A processed sweep, the Auto-pick tool on, looking for minima."""
    window.controller.set_processing(
        sample_files=SweepFiles(tuple(lines_sweep["zero"]), tuple(lines_sweep["field"]))
    )
    process(window)
    assert window.controller.result is not None
    window.tools.set_active("autopick")
    window.autopick.bar.feature.set_value("min")
    return window


@pytest.fixture
def shown(picked, qtbot):
    picked.resize(1400, 900)
    picked.show()
    qtbot.waitExposed(picked)
    return picked


def preview(window) -> list[tuple[np.ndarray, np.ndarray]]:
    """(x, y) of the marker groups on the map's preview layer."""
    return [
        (np.asarray(x), np.asarray(y)) for x, y in window.plots.map.layer("preview").point_data()
    ]


def lines_drawn(window) -> list[tuple[np.ndarray, np.ndarray]]:
    """The lines found as drawn: the others (joined, NaN between them), then the chosen one."""
    return window.plots.map.layer("preview").curve_data()


def segments(y: np.ndarray) -> int:
    return int(np.count_nonzero(np.isnan(y))) + 1 if y.size else 0


def status(window) -> str:
    return window.autopick.bar.status_text()


def counting(monkeypatch) -> list[int]:
    """Count the searches the tool runs."""
    calls: list[int] = []
    original = autopick.search

    def spy(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(autopick, "search", spy)
    return calls


# ---------------------------------------------------------------------- the tool
def test_the_tool_sits_with_pan_zoom_and_pick(shown):
    w, tools = shown, shown.tools
    tools.set_active("navigate")
    assert tools.names() == ["navigate", "zoom", "pick", "autopick"]
    tool = tools.tool("autopick")
    assert tool.views == ("map",)
    assert tool.button.toolTip() == "Auto-pick (W)"
    bar = w.autopick.bar
    assert bar.isHidden()
    w.plots.map.view.setFocus()
    QTest.keyClick(w.plots.map.view, Qt.Key.Key_W)
    assert tools.active() == "autopick" and tool.button.isChecked()
    assert bar.isVisible()
    QTest.keyClick(w.plots.map.view, Qt.Key.Key_W)
    assert tools.active() == "navigate" and bar.isHidden()
    w.plot_area.set_current_view("stacked")
    assert not tool.button.isEnabled()
    assert not tools.set_active("autopick")


def test_the_options_strip_moves_the_map_down_instead_of_covering_it(shown, qtbot):
    w, bar, area = shown, shown.autopick.bar, shown.plot_area
    stack = area.stack
    qtbot.waitUntil(lambda: stack.geometry().top() > bar.geometry().bottom())
    assert bar.geometry().top() == 0 and bar.width() == area.plot_box.width()
    assert stack.geometry().bottom() == area.plot_box.height() - 1  # the plot gave the room
    w.report_error("Process", "something went wrong")  # the error bar floats below the strip
    qtbot.waitUntil(lambda: not w.infobar.isHidden())
    assert w.infobar.geometry().top() >= stack.geometry().top()
    w.resize(1100, 800)  # narrow: the controls wrap, the strip grows and the map shrinks more
    qtbot.waitUntil(lambda: stack.geometry().top() > bar.geometry().bottom())
    assert bar.controls.height() > bar.mode.height() * 2
    w.tools.set_active("navigate")
    qtbot.waitUntil(lambda: stack.geometry().top() == 0)


def test_nothing_happens_without_a_processed_map(window):
    tool = window.autopick
    window.tools.set_active("autopick")
    assert "process a sweep" in status(window)
    tool.track_at(1.0, 300.0)
    click_map(window, 1.0, 300.0)
    tool.detect_in((0.5, 2.0), (200.0, 800.0))
    assert tool.candidates == [] and preview(window) == []
    assert "process a sweep" in status(window)
    assert not tool.bar.accept_button.isEnabled()


# ---------------------------------------------------------------------- track
def test_track_previews_a_line_and_accept_is_one_undo_step(picked, errors):
    w, c, tool = picked, picked.controller, picked.autopick
    c.record_point(2.0, 700.0)  # replaced by the line's point at 2 T
    undo_before = w.points_undo.count()
    assert click_map(w, 4.0, line1(4.0) + 6.0)
    assert len(tool.candidates) == 1
    found = tool.candidates[0]
    np.testing.assert_allclose(found.field, FIELDS)
    np.testing.assert_allclose(found.energy, line1(FIELDS), atol=STEP)
    markers = preview(w)
    np.testing.assert_allclose(markers[1][0], FIELDS)  # halo, diamonds, then the click
    np.testing.assert_allclose(markers[1][1], found.energy)
    np.testing.assert_allclose(markers[-1][1], [line1(4.0) + 6.0])
    assert len(lines_drawn(w)) == 1
    assert status(w) == "31 points found for LL 1. Replaces 1 of its points."
    assert c.points.points("LL 1")[0].tolist() == [2.0]  # not in the table yet
    assert w.points_undo.count() == undo_before

    assert tool.accept() == 31
    b, e = c.points.points("LL 1")
    np.testing.assert_allclose(b, FIELDS)
    np.testing.assert_allclose(e, found.energy)
    assert w.points_undo.count() == undo_before + 1
    assert w.points_undo.undoText() == "Auto-pick 31 points on LL 1"
    assert preview(w) == [] and tool.target is None
    assert status(w).startswith("Added 31 points to LL 1")
    w.points_undo.undo()
    assert c.points.points("LL 1")[0].tolist() == [2.0]
    assert c.points.column("LL 1")[c.points.nearest_row(2.0)] == pytest.approx(700.0)
    w.points_undo.redo()
    assert c.points.points("LL 1")[0].size == 31
    assert not errors


def test_track_says_why_a_click_finds_nothing(picked):
    w, tool = picked, picked.autopick
    click_map(w, 4.0, 1100.0)  # nothing there
    assert tool.candidates == [] and not tool.bar.accept_button.isEnabled()
    assert status(w).startswith("No minimum within ±20 cm⁻¹ of the click at 4 T.")
    assert tool.bar.status.level() == "warn"
    assert len(preview(w)) == 2  # the click stays marked
    assert tool.bar.prominence_edit.placeholderText().startswith("auto 0.")  # to lower it


def test_track_rising_inflections_of_smoothed_spectra(picked):
    w, tool, bar = picked, picked.autopick, picked.autopick.bar
    bar.feature.set_value("rising")
    bar.smooth.set_value("9")
    flank = 9.0 / np.sqrt(2)  # the rising inflection of a dip exp(-(x/9)^2), above its centre
    click_map(w, 4.0, line1(4.0) + flank)
    assert len(tool.candidates) == 1
    np.testing.assert_allclose(tool.candidates[0].field, FIELDS)
    np.testing.assert_allclose(tool.candidates[0].energy, line1(FIELDS) + flank, atol=STEP)
    assert bar.prominence_field.unit_label.text() == "/cm⁻¹"


# ---------------------------------------------------------------------- detect
def test_detect_finds_the_lines_in_a_box_and_a_click_chooses_one(picked):
    w, c, tool = picked, picked.controller, picked.autopick
    tool.bar.mode.set_value("detect")
    tool.detect_in((1.0, 7.0), (300.0, 1100.0))
    inside = FIELDS[(FIELDS >= 1.0) & (FIELDS <= 7.0)]
    assert len(tool.candidates) == 2
    for candidate, line in zip(tool.candidates, (line1, line2), strict=True):
        np.testing.assert_allclose(candidate.field, inside)
        np.testing.assert_allclose(candidate.energy, line(inside), atol=STEP)
    assert tool.chosen == 0  # as long as the other, but deeper
    others, chosen = lines_drawn(w)
    assert segments(others[1]) == 1
    np.testing.assert_allclose(others[1], tool.candidates[1].energy)
    np.testing.assert_allclose(chosen[1], tool.candidates[0].energy)
    assert status(w).startswith("2 lines found: 25 points for LL 1 on the one with diamonds")
    assert click_map(w, 5.0, line2(5.0) + 2.0)  # a click on the other line chooses it
    assert tool.chosen == 1
    np.testing.assert_allclose(preview(w)[1][1], tool.candidates[1].energy)
    assert not tool.choose_at(5.0, 700.0)  # far from both
    assert tool.chosen == 1
    tool.accept()
    np.testing.assert_allclose(c.points.points("LL 1")[1], line2(inside), atol=STEP)

    tool.detect_in((1.0, 7.0), (300.0, 800.0))
    assert len(tool.candidates) == 1
    tool.detect_in((1.0, 7.0), (1100.0, 1200.0))
    assert tool.candidates == []
    assert status(w).startswith("No minima found in the box.")


def test_detect_chooses_the_line_that_continues_the_curve(picked):
    w, c, tool = picked, picked.controller, picked.autopick
    c.record_points([7.5, 8.0], line2(np.array([7.5, 8.0])), unit="cm-1")
    tool.bar.mode.set_value("detect")
    tool.detect_in((1.0, 8.0), (300.0, 1100.0))
    assert len(tool.candidates) == 2 and tool.chosen == 1  # line2, though line1 is deeper
    assert status(w).endswith("Replaces 2 of its points.")
    c.add_curve("LL 2")  # a curve without points: the longest (and deepest) line
    tool.detect_in((1.0, 8.0), (300.0, 1100.0))
    assert tool.chosen == 0
    c.set_curve("LL 1")  # points far from every line found count for nothing
    c.record_points([7.5, 8.0], [1150.0, 1150.0], unit="cm-1")
    tool.detect_in((1.0, 8.0), (300.0, 1100.0))
    assert tool.chosen == 0


def _viewport_pos(window, b: float, energy: float) -> QPoint:
    plot = window.plots.map
    scene = plot.plot.vb.mapViewToScene(QPointF(b, energy))
    return plot.view.mapFromScene(scene)


def _move_pause_ms() -> int:
    """How long a drag on the plot scene waits before its move, as a real drag does: pyqtgraph
    drops a mouse move within 1/mouseRateLimit s of the last one it took, and a press and
    release without a move is a click."""
    rate = pg.getConfigOption("mouseRateLimit")
    return 2 * math.ceil(1000 / rate) if rate > 0 else 0


def test_a_box_dragged_on_the_map_detects_without_panning(shown, qtbot):
    w, tool = shown, shown.autopick
    viewport = w.plots.map.view.viewport()
    vb = w.plots.map.plot.vb
    tool.bar.mode.set_value("detect")
    before = vb.viewRange()
    start, end = _viewport_pos(w, 1.0, 300.0), _viewport_pos(w, 7.0, 1100.0)
    QTest.mousePress(viewport, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, start)
    QTest.mouseMove(viewport, (start + end) / 2)
    assert tool.region.outline_visible() and tool.region.roi is None  # the rubber band
    QTest.mouseMove(viewport, end)
    QTest.mouseRelease(viewport, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, end)
    assert vb.viewRange() == before
    (b0, b1), (e0, e1) = tool.target.box
    assert b0 == pytest.approx(1.0, abs=0.1) and b1 == pytest.approx(7.0, abs=0.1)
    assert e0 == pytest.approx(300.0, abs=10.0) and e1 == pytest.approx(1100.0, abs=10.0)
    assert tool.target.region is None  # an upright box: searched without a mask
    assert len(tool.candidates) == 2 and tool.chosen == 0
    rect = tool.region.outline_path().boundingRect()  # the box stays, as an editable region
    assert rect.left() == pytest.approx(b0) and rect.bottom() == pytest.approx(e1)
    assert tool.region.shape == "rect" and len(tool.region.roi.getHandles()) == 4
    click = _viewport_pos(w, 5.0, line2(5.0))  # a click without a drag chooses a line
    QTest.mouseClick(viewport, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, click)
    assert tool.chosen == 1 and tool.target.box == ((b0, b1), (e0, e1))
    tool.bar.mode.set_value("track")  # Track leaves left-drags to panning
    QTest.mousePress(viewport, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, start)
    qtbot.wait(_move_pause_ms())
    QTest.mouseMove(viewport, end)
    QTest.mouseRelease(viewport, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, end)
    assert tool.target is None and vb.viewRange() != before


# ---------------------------------------------------------------------- regions
LEFT, PLAIN = Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier
ELLIPSE_BOX = [(0.75, 450.0), (7.75, 950.0)]  # line2 crosses this box but not its ellipse


def in_ellipse(field, energy) -> np.ndarray:
    """The points inside the ellipse in :data:`ELLIPSE_BOX`."""
    return ((field - 4.25) / 3.5) ** 2 + ((energy - 700.0) / 250.0) ** 2 < 1.0


def inside(polygon, field, energy) -> np.ndarray:
    from matplotlib.path import Path

    return Path(np.asarray(polygon)).contains_points(np.column_stack([field, energy]))


def drag(qtbot, viewport, points: list[QPoint]) -> None:
    """A left-drag through *points* (viewport pixels), slow enough for pyqtgraph."""
    QTest.mousePress(viewport, LEFT, PLAIN, points[0])
    for point in points[1:]:
        qtbot.wait(_move_pause_ms())
        QTest.mouseMove(viewport, point)
    QTest.mouseRelease(viewport, LEFT, PLAIN, points[-1])


def test_an_ellipse_finds_only_the_lines_inside_it(picked):
    w, tool, bar = picked, picked.autopick, picked.autopick.bar
    assert bar.shape_group.isHidden()  # Track: no shapes
    bar.mode.set_value("detect")
    assert not bar.shape_group.isHidden()
    assert bar.shape.options() == ["rect", "rotated", "ellipse", "polygon"]
    bar.shape.set_value("ellipse")
    assert status(w).startswith("Drag an ellipse around the lines")
    assert tool.draw_region(ELLIPSE_BOX)
    assert tool.region.shape == "ellipse" and tool.target.region is not None
    (b0, b1), (e0, e1) = tool.target.box
    assert (b0, b1, e0, e1) == pytest.approx((0.75, 7.75, 450.0, 950.0), rel=1e-6)
    assert len(tool.candidates) == 1
    found = tool.candidates[0]
    np.testing.assert_allclose(found.energy, line1(found.field), atol=STEP)
    assert in_ellipse(found.field, found.energy).all() and len(found) > 15
    assert status(w).startswith("21 points found for LL 1.")
    np.testing.assert_allclose(lines_drawn(w)[-1][1], found.energy)
    tool.detect_in((0.75, 7.75), (450.0, 950.0))  # its bounds as a box: both lines
    assert tool.region.shape == "rect" and tool.target.region is None
    assert len(tool.candidates) == 2
    bar.shape.set_value("polygon")  # the next drag draws another shape; the box stays
    assert tool.region.shape == "rect" and len(tool.candidates) == 2
    assert not tool.draw_region([(2.0, 600.0), (3.0, 600.0)])  # a loop without an area
    assert status(w).startswith("Draw a loop around the lines")
    assert tool.region.shape == "rect" and tool.target.box[0] == (0.75, 7.75)
    tool.discard()
    assert tool.region.roi is None and not tool.region.outline_visible()
    assert status(w).startswith("Draw a loop around the lines on the map")
    tool.draw_region([(1.0, 900.0), (8.0, 1010.0), (8.0, 1100.0), (1.0, 1000.0)])
    assert tool.region.shape == "polygon" and len(tool.candidates) == 1
    np.testing.assert_allclose(tool.candidates[0].energy, line2(tool.candidates[0].field), atol=2)


def test_a_rotated_box_turns_and_searches_again(picked, monkeypatch):
    w, tool = picked, picked.autopick
    tool.bar.mode.set_value("detect")
    tool.bar.shape.set_value("rotated")
    tool.draw_region([(1.0, 380.0), (7.5, 760.0)])
    roi = tool.region.roi
    kinds = sorted(h["type"] for h in roi.handles)
    assert kinds == ["r", "s", "s", "s", "s"]  # four corners and the round handle
    before = tool.target
    calls = counting(monkeypatch)
    roi.setAngle(25.0, centerLocal=roi.size() / 2)  # what the round handle does
    assert len(calls) == 1 and tool.target != before
    polygon = np.array(tool.target.region)
    sides = np.diff(np.vstack([polygon, polygon[:1]]), axis=0)
    assert np.all(np.abs(sides) > 1e-6)  # no side is upright or level any more
    assert tool.candidates
    for candidate in tool.candidates:
        assert inside(polygon, candidate.field, candidate.energy).all()
    undo_before = w.points_undo.count()
    n = tool.accept()
    assert n > 0 and w.points_undo.count() == undo_before + 1
    assert tool.region.roi is None and not tool.region.outline_visible()


def test_the_region_moves_and_reshapes_by_drags_and_searches_while_it_changes(
    shown, qtbot, monkeypatch
):
    w, tool = shown, shown.autopick
    view, viewport, vb = w.plots.map.view, w.plots.map.view.viewport(), w.plots.map.plot.vb
    tool.bar.mode.set_value("detect")
    tool.detect_in((1.0, 4.0), (300.0, 1100.0))
    assert len(tool.candidates) == 2
    before, shown_range = tool.target, vb.viewRange()
    calls = counting(monkeypatch)
    start, end = _viewport_pos(w, 2.5, 600.0), _viewport_pos(w, 4.5, 600.0)
    QTest.mousePress(viewport, LEFT, PLAIN, start)
    qtbot.wait(_move_pause_ms())
    QTest.mouseMove(viewport, _viewport_pos(w, 3.5, 600.0))
    assert tool.region.editing
    qtbot.waitUntil(lambda: bool(calls), timeout=5000)  # searched while it is dragged
    qtbot.wait(_move_pause_ms())
    QTest.mouseMove(viewport, end)
    QTest.mouseRelease(viewport, LEFT, PLAIN, end)
    assert not tool.region.editing and vb.viewRange() == shown_range  # moved, not panned
    (b0, b1), energies = tool.target.box
    assert b0 == pytest.approx(3.0, abs=0.1) and b1 == pytest.approx(6.0, abs=0.1)
    assert energies == pytest.approx(before.box[1], abs=1e-6)
    fields = tool.shown_map().field
    np.testing.assert_allclose(tool.candidates[0].field, fields[(fields >= b0) & (fields <= b1)])

    corner = max(tool.region.roi.getHandles(), key=lambda h: (h.scenePos().x(), -h.scenePos().y()))
    grab = view.mapFromScene(corner.scenePos())  # the top right corner
    drag(qtbot, viewport, [grab, grab + QPoint(0, 15), grab + QPoint(0, 40)])
    (b0_new, b1_new), (e0_new, e1_new) = tool.target.box
    assert (b0_new, b1_new) == pytest.approx((b0, b1), abs=1e-6)
    assert e0_new == pytest.approx(energies[0], abs=1e-6) and e1_new < energies[1] - 50.0

    click = _viewport_pos(w, 5.0, line2(5.0))  # a click inside the region chooses a line
    assert tool.chosen == 0
    QTest.mouseClick(viewport, LEFT, PLAIN, click)
    assert tool.chosen == 1 and tool.target.box[0] == (b0_new, b1_new)
    QTest.mouseClick(viewport, LEFT, PLAIN, _viewport_pos(w, 7.5, 300.0))  # outside: nothing
    assert tool.chosen == 1

    elsewhere = [_viewport_pos(w, 6.5, 700.0), _viewport_pos(w, 7.0, 900.0)]
    drag(qtbot, viewport, [elsewhere[0], (elsewhere[0] + elsewhere[1]) / 2, elsewhere[1]])
    (b0, b1), _energies = tool.target.box  # a drag beside the region draws a new one
    assert b0 == pytest.approx(6.5, abs=0.1) and b1 == pytest.approx(7.0, abs=0.1)
    assert vb.viewRange() == shown_range

    wheel_at = view.mapFromScene(tool.region.roi.sceneBoundingRect().center())
    event = QWheelEvent(
        QPointF(wheel_at), QPointF(viewport.mapToGlobal(wheel_at)), QPoint(0, 0), QPoint(0, 120),
        Qt.MouseButton.NoButton, PLAIN, Qt.ScrollPhase.NoScrollPhase, False,
    )  # fmt: skip
    target = tool.target
    QApplication.sendEvent(viewport, event)  # the wheel zooms, also over the region
    assert vb.viewRange() != shown_range and tool.target == target


def test_the_round_handle_turns_the_box_as_drawn_on_screen(shown, qtbot, monkeypatch):
    w, tool = shown, shown.autopick
    view, viewport = w.plots.map.view, w.plots.map.view.viewport()
    tool.bar.mode.set_value("detect")
    tool.bar.shape.set_value("rotated")
    tool.draw_region([(2.0, 500.0), (6.0, 800.0)])
    roi = tool.region.roi
    turn = next(h["item"] for h in roi.handles if h["type"] == "r")
    centre = view.mapFromScene(roi.sceneBoundingRect().center())
    grab = view.mapFromScene(turn.scenePos())
    arm = grab.x() - centre.x()  # px from the centre to the round handle (on the right)
    assert arm > 50 and abs(grab.y() - centre.y()) <= 1
    calls = counting(monkeypatch)
    drag(qtbot, viewport, [grab, grab - QPoint(0, arm // 2), grab - QPoint(0, arm)])
    assert roi.angle() == pytest.approx(45.0, abs=2.0)  # turned on screen as the mouse went
    assert calls and tool.target.region is not None
    width = roi.size()[0]
    side = view.mapFromScene(roi.mapToScene(QPointF(width, 0))) - view.mapFromScene(
        roi.mapToScene(QPointF(0, 0))
    )  # the bottom side of the box, on screen
    assert math.degrees(math.atan2(-side.y(), side.x())) == pytest.approx(45.0, abs=3.0)


def test_a_loop_drawn_with_the_mouse_becomes_a_polygon(shown, qtbot):
    w, tool = shown, shown.autopick
    viewport = w.plots.map.view.viewport()
    tool.bar.mode.set_value("detect")
    tool.bar.shape.set_value("polygon")
    turns = np.linspace(0.0, 2.0 * np.pi, 80)
    loop = [_viewport_pos(w, 4.25 + 3.5 * np.cos(a), 700.0 + 250.0 * np.sin(a)) for a in turns]
    QTest.mousePress(viewport, LEFT, PLAIN, loop[0])
    for point in loop[1:]:
        QTest.mouseMove(viewport, point)
    assert tool.region.outline_visible() and tool.region.roi is None  # the loop so far
    QTest.mouseRelease(viewport, LEFT, PLAIN, loop[-1])
    roi = tool.region.roi
    assert tool.region.shape == "polygon" and 8 <= len(roi.getHandles()) <= 40
    assert len(tool.candidates) == 1  # line1 only, as in the ellipse
    found = tool.candidates[0]
    np.testing.assert_allclose(found.energy, line1(found.field), atol=STEP)


def test_a_region_on_a_map_with_uneven_fields(picked):
    """The mask is made on the map's own fields: a library map with gaps between them."""
    c, tool = picked.controller, picked.autopick
    ratio = c.result.ratio
    keep = [0, 1, 2, 6, 7, 8, 9, 14, 15, 20, 21, 22, 23, 28, 29, 30]
    uneven = ratio.replace(field=ratio.field[keep], values=ratio.values[:, keep])
    c.plot_entry(c.add_map(uneven, "uneven"))
    tool.bar.mode.set_value("detect")
    tool.bar.shape.set_value("ellipse")
    tool.draw_region(ELLIPSE_BOX)
    assert len(tool.candidates) == 1
    found = tool.candidates[0]
    fields = FIELDS[keep]
    surely = fields[((fields - 4.25) / 3.5) ** 2 + ((line1(fields) - 700.0) / 250.0) ** 2 < 0.8]
    assert set(surely) <= set(found.field) <= set(fields)
    assert in_ellipse(found.field, found.energy).all()
    np.testing.assert_allclose(found.energy, line1(found.field), atol=STEP)


def test_unit_switch_keeps_the_region_where_it_is(picked, monkeypatch):
    w, tool = picked, picked.autopick
    tool.bar.mode.set_value("detect")
    tool.bar.shape.set_value("ellipse")
    tool.draw_region(ELLIPSE_BOX)
    target, outline, found = tool.target, tool.region.polygon(), tool.candidates[0]
    calls = counting(monkeypatch)
    set_unit(w, "meV")
    assert calls == [] and tool.target == target
    np.testing.assert_allclose(tool.region.polygon(), outline / [1.0, MEV], rtol=1e-9)
    path = tool.region.outline_path().boundingRect()
    assert path.bottom() == pytest.approx(950.0 / MEV, rel=1e-6)
    tool.region_edited()  # searched again from the region: the same place in cm-1
    np.testing.assert_allclose(np.array(tool.target.region), np.array(target.region), rtol=1e-9)
    np.testing.assert_allclose(tool.candidates[0].energy, found.energy)
    tool.bar.mode.set_value("track")  # another mode drops the region
    assert tool.region.roi is None and tool.target is None


def test_region_shape_is_remembered(qtbot, tmp_path):
    ini = str(tmp_path / "settings.ini")
    first = MainWindow(settings=QSettings(ini, QSettings.Format.IniFormat))
    qtbot.addWidget(first)
    first.autopick.bar.shape.set_value("ellipse")
    first.save_settings()
    first.close()
    second = MainWindow(settings=QSettings(ini, QSettings.Format.IniFormat))
    qtbot.addWidget(second)
    assert second.autopick.bar.shape.value() == "ellipse"


# ---------------------------------------------------------------------- options
def test_options_search_again_for_the_same_click(picked, monkeypatch):
    w, tool, bar = picked, picked.autopick, picked.autopick.bar
    calls = counting(monkeypatch)
    click_map(w, 4.0, line1(4.0))
    assert len(calls) == 1 and len(tool.candidates[0]) == 31
    bar.smooth.set_value("9")
    assert len(calls) == 2 and len(tool.candidates[0]) == 31
    bar.feature.set_value("max")
    assert len(calls) == 3
    assert tool.candidates == [] and status(w).startswith("No maximum within")
    bar.feature.set_value("min")
    assert len(calls) == 4 and len(tool.candidates[0]) == 31
    bar.prominence_edit.setText("0.5")  # typing waits for a pause (or the end of editing)
    assert len(calls) == 4 and tool.pending()
    tool.flush()
    assert len(calls) == 5 and tool.candidates == []
    bar.prominence_edit.setText("-1")
    tool.flush()
    assert bar.prominence_field.is_invalid() and status(w).startswith("Prominence:")
    bar.prominence_edit.clear()
    tool.flush()
    assert len(tool.candidates[0]) == 31 and not bar.prominence_field.is_invalid()
    assert bar.prominence_edit.placeholderText().startswith("auto ")
    bar.window_edit.setText("12")  # the line moves 10 cm-1 per field
    tool.flush()
    assert len(tool.candidates[0]) == 31
    bar.window_edit.setText("4")
    tool.flush()
    assert len(tool.candidates[0]) == 1  # too narrow for the first step from the click
    bar.window_edit.setText("x")
    tool.flush()
    assert bar.window_field.is_invalid() and tool.candidates == []
    bar.window_edit.clear()
    tool.flush()
    count = len(calls)
    bar.prominence_edit.setText("0.01")
    tool.flush()
    select(w, kind="Data")  # another map: searched again, with an automatic prominence
    assert len(calls) == count + 2 and bar.prominence_edit.text() == ""
    assert tool.target is not None
    bar.mode.set_value("detect")  # another mode drops the click
    assert tool.target is None and tool.candidates == [] and preview(w) == []


def test_discard_escape_and_other_tools_clear_the_preview(shown):
    w, tool, tools = shown, shown.autopick, shown.tools
    click_map(w, 4.0, line1(4.0))
    assert preview(w)
    tool.bar.discard_button.click()
    assert tool.candidates == [] and tool.target is None and preview(w) == []
    assert status(w).startswith("Click a line on the map")
    assert tools.active() == "autopick"

    click_map(w, 4.0, line1(4.0))
    w.plots.map.view.setFocus()
    QTest.keyClick(w.plots.map.view, Qt.Key.Key_Escape)
    assert tools.active() == "navigate" and preview(w) == [] and tool.bar.isHidden()

    tools.set_active("autopick")
    click_map(w, 4.0, line1(4.0))
    tools.set_active("pick")
    assert preview(w) == [] and tool.target is None
    tools.set_active("autopick")
    tool.bar.mode.set_value("detect")
    tool.detect_in((1.0, 7.0), (300.0, 1100.0))
    assert tool.region.outline_visible() and tool.region.roi is not None
    w.plot_area.set_current_view("stacked")
    assert tools.active() == "navigate"
    assert preview(w) == [] and not tool.region.outline_visible() and tool.region.roi is None
    w.plot_area.set_current_view("map")
    tools.set_active("autopick")
    tool.detect_in((1.0, 7.0), (300.0, 1100.0))
    assert tool.region.roi is not None
    w.plots.map.view.setFocus()
    QTest.keyClick(w.plots.map.view, Qt.Key.Key_Escape)  # Esc drops the region too
    assert tools.active() == "navigate" and tool.region.roi is None


def test_a_new_result_drops_the_preview_and_the_status_follows_the_curve(picked):
    w, c, tool = picked, picked.controller, picked.autopick
    click_map(w, 4.0, line1(4.0))
    c.add_curve("LL 2")
    assert status(w) == "31 points found for LL 2."
    process(w)
    assert tool.target is None and preview(w) == []


# ---------------------------------------------------------------------- units
def test_unit_switch_converts_the_preview_without_searching(picked, monkeypatch):
    w, c, tool, bar = picked, picked.controller, picked.autopick, picked.autopick.bar
    click_map(w, 4.0, line1(4.0))
    found = tool.candidates[0].energy.copy()
    bar.window_edit.setText("30")
    tool.flush()
    calls = counting(monkeypatch)
    set_unit(w, "meV")
    tool.flush()
    assert calls == [] and not tool.pending()
    np.testing.assert_allclose(tool.candidates[0].energy, found)
    np.testing.assert_allclose(preview(w)[1][1], found / MEV)
    np.testing.assert_allclose(lines_drawn(w)[0][1], found / MEV)
    assert preview(w)[-1][1][0] == pytest.approx(line1(4.0) / MEV)
    assert float(bar.window_edit.text()) == pytest.approx(30.0 / MEV, rel=1e-5)
    assert bar.window_field.unit_label.text() == "meV"
    bar.feature.set_value("rising")  # a slope prominence is per energy unit
    bar.prominence_edit.setText("0.001")
    tool.flush()
    set_unit(w, "cm-1")
    assert float(bar.prominence_edit.text()) == pytest.approx(0.001 / MEV, rel=1e-5)
    assert bar.prominence_field.unit_label.text() == "/cm⁻¹"
    bar.feature.set_value("min")  # the prominence of values again: automatic
    assert bar.prominence_edit.text() == ""
    set_unit(w, "THz")
    tool.accept()
    np.testing.assert_allclose(c.points.points("LL 1")[1], found)


def test_prominence_of_per_unit_derivatives_scales_with_the_unit(picked):
    w, c = picked, picked.controller
    select(w, order=1, per_unit=True)
    s = c.selection
    assert prominence_scale(s, "meV", "min") == pytest.approx(MEV)
    assert prominence_scale(s, "meV", "rising") == pytest.approx(MEV**2)
    select(w, per_unit=False)
    assert prominence_scale(c.selection, "meV", "min") == 1.0


def test_options_are_remembered(qtbot, tmp_path):
    ini = str(tmp_path / "settings.ini")
    first = MainWindow(settings=QSettings(ini, QSettings.Format.IniFormat))
    qtbot.addWidget(first)
    bar = first.autopick.bar
    bar.mode.set_value("detect")
    bar.feature.set_value("rising")
    bar.smooth.set_value("9")
    set_unit(first, "meV")
    bar.window_edit.setText("5")
    first.save_settings()
    first.close()
    second = MainWindow(settings=QSettings(ini, QSettings.Format.IniFormat))
    qtbot.addWidget(second)
    bar = second.autopick.bar
    assert (bar.mode.value(), bar.feature.value(), bar.smooth.value()) == ("detect", "rising", "9")
    assert bar.window_edit.cm1() == pytest.approx(5 * MEV)
    assert bar.window_edit.text() == "5"
    assert bar.prominence_edit.text() == ""  # always automatic in a new session


# ---------------------------------------------------------------------- helpers
def test_default_choice_and_window():
    def candidate(field, energy, strength):
        n = len(field)
        return Candidate(np.array(field, float), np.full(n, energy), np.full(n, strength))

    short, long_weak, long_strong = (
        candidate([1, 2, 3], 100.0, 9.0),
        candidate([1, 2, 3, 4], 200.0, 1.0),
        candidate([1, 2, 3, 4], 300.0, 2.0),
    )
    assert default_choice([]) == -1
    assert default_choice([short, long_weak, long_strong]) == 2
    previous = candidate([2, 3], 205.0, 1.0)
    assert default_choice([short, long_weak, long_strong], previous) == 1
    assert default_choice([short, long_weak, long_strong], previous, reach=1.0) == 2
    elsewhere = candidate([7, 8], 205.0, 1.0)
    assert default_choice([short, long_weak], elsewhere) == 1
    x, y = autopick._joined([short, long_weak], "meV")  # drawn as one curve with a gap
    assert segments(y) == 2 and x.size == 8
    np.testing.assert_allclose(y[:3], 100.0 / MEV)

    from mag_opt_detective.core.spectra import FieldMap

    fmap = FieldMap(
        energy=np.linspace(350.0, 3200.0, 11), field=np.ones(1), values=np.ones((11, 1))
    )
    assert auto_window(fmap) == 57.0


def test_accept_on_a_finer_library_map_stores_every_point(picked, errors):
    """A table made on a coarser map gets the rows of the finer one: Accept stores (and
    reports) every point on its own field."""
    w, c, tool = picked, picked.controller, picked.autopick
    fine = c.result.ratio
    c.set_new_table(True)
    c.from_map(fine.replace(field=fine.field[::4], values=fine.values[:, ::4]))
    assert c.points.field.size == 8
    c.plot_entry(c.add_map(fine, "fine"))
    assert click_map(w, 4.0, line1(4.0) + 6.0)
    assert tool.accept() == 31
    assert status(w).startswith("Added 31 points to LL 1")
    b, e = c.points.points("LL 1")
    np.testing.assert_allclose(b, FIELDS)
    np.testing.assert_allclose(e, line1(FIELDS), atol=STEP)
    assert not errors
