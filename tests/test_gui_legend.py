"""The plot legend: the toolbar button per plot, the rows (picked curves and models) as they
change, its place (dragged, kept through zooms and resizes, remembered) and presses on it while
the plot tools are active."""

import itertools
import json
import math

import numpy as np
import pyqtgraph as pg
import pytest
from PySide6.QtCore import QPoint, QPointF, QSettings, Qt
from PySide6.QtTest import QTest

import gui_helpers
from gui_helpers import inspector_page, load_sweep, process, set_unit
from mag_opt_detective.gui.inspector import model_state as ms
from mag_opt_detective.gui.main_window import MainWindow
from mag_opt_detective.gui.plots.legend import MARGIN
from mag_opt_detective.gui.points_view import curve_color

window, errors = gui_helpers.window, gui_helpers.errors  # shared fixtures

LEFT, PLAIN = Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier


@pytest.fixture
def processed(window, sweep):
    load_sweep(window, sweep)
    process(window)
    return window


@pytest.fixture
def shown(processed, qtbot):
    """The processed window on screen, with points on two curves and the map's legend on."""
    w = processed
    w.resize(1400, 900)
    w.show()
    qtbot.waitExposed(w)
    c = w.controller
    c.record_points([0.5, 1.0], [300.0, 320.0], unit="cm-1")
    c.add_curve("LL 2")
    c.record_points([1.5], [600.0], unit="cm-1")
    w.plot_area.legend_button.click()
    assert w.legends["map"].isVisible()
    return w


def legend(window, view="map"):
    return window.legends[view]


def models_of(window):
    return inspector_page(window, "models").models


def emphasis(window, view="map") -> list[bool]:
    return [entry.emphasis for entry in legend(window, view).entries()]


def viewport_point(window, view_point: QPointF, view="map") -> QPoint:
    """The viewport pixel of a point in the view box of plot *view*."""
    plot = window.plots[view]
    return plot.view.mapFromScene(plot.plot.vb.mapToScene(view_point))


def legend_centre(window, view="map") -> QPoint:
    return viewport_point(window, legend(window, view).rect_in_view().center(), view)


def data_point(window, b: float, energy: float) -> QPoint:
    plot = window.plots.map
    return plot.view.mapFromScene(plot.plot.vb.mapViewToScene(QPointF(b, energy)))


def pause() -> int:
    """Long enough between a press and a move for pyqtgraph (see test_gui_autopick)."""
    rate = pg.getConfigOption("mouseRateLimit")
    return 2 * math.ceil(1000 / rate) if rate > 0 else 0


def drag(qtbot, window, points: list[QPoint], view="map") -> None:
    viewport = window.plots[view].view.viewport()
    QTest.mousePress(viewport, LEFT, PLAIN, points[0])
    for point in points[1:]:
        qtbot.wait(pause())
        QTest.mouseMove(viewport, point)
    QTest.mouseRelease(viewport, LEFT, PLAIN, points[-1])


def drawn_rect(window, view="map"):
    """Where the legend is drawn, in view box pixels (from its item, not its own sums)."""
    item = legend(window, view)
    return window.plots[view].plot.vb.mapRectFromScene(item.sceneBoundingRect())


def assert_near(a: QPointF, b: QPointF, tolerance: float = 0.5) -> None:
    assert (a.x(), a.y()) == pytest.approx((b.x(), b.y()), abs=tolerance)


def assert_drawn_where_placed(window, view="map") -> None:
    drawn, placed = drawn_rect(window, view), legend(window, view).rect_in_view()
    for a, b in zip(drawn.getCoords(), placed.getCoords(), strict=True):
        assert a == pytest.approx(b, abs=0.5)


# ---------------------------------------------------------------------- toggle
def test_the_toolbar_button_switches_the_legend_of_each_plot(processed):
    w, area = processed, processed.plot_area
    button = area.legend_button
    w.controller.record_points([1.0], [300.0], unit="cm-1")
    assert button.isEnabled() and not button.isChecked()  # off by default
    assert not legend(w).is_shown() and not legend(w).isVisible()
    assert legend(w).texts() == ["LL 1"]  # listed, but not shown

    button.click()
    assert legend(w).is_shown() and legend(w).isVisible()
    area.set_current_view("stacked")
    assert button.isEnabled() and not button.isChecked()  # the stacked plot has its own
    button.click()
    assert legend(w, "stacked").is_shown() and legend(w, "stacked").texts() == ["LL 1"]
    area.set_current_view("reference")
    assert not button.isEnabled() and not button.isChecked()
    area.set_current_view("map")
    assert button.isChecked()
    button.click()
    assert not legend(w).is_shown() and not legend(w).isVisible()
    assert legend(w, "stacked").is_shown()


def test_the_plot_toolbar_with_the_legend_button_fits_1100_px(window, qtbot):
    window.resize(1100, 800)
    window.show()
    qtbot.waitExposed(window)
    assert window.side_panel.is_open() and window.inspector_panel.is_open()
    assert window.minimumSizeHint().width() <= 1100 and window.width() == 1100
    tabs = window.plot_area.tabs
    assert all(tabs.tabRect(i).width() < tabs.tabSizeHint(i).width() for i in range(3))
    window.resize(1400, 900)  # with room the tabs keep their padding
    qtbot.waitUntil(lambda: window.width() == 1400)
    assert all(tabs.tabRect(i).width() == tabs.tabSizeHint(i).width() for i in range(3))


def test_the_legend_and_its_place_are_remembered_per_plot(qtbot, tmp_path, sweep, errors):
    ini = str(tmp_path / "settings.ini")
    first = MainWindow(settings=QSettings(ini, QSettings.Format.IniFormat))
    qtbot.addWidget(first)
    first.plot_area.legend_button.click()
    legend(first).set_position(1.0, 0.25)
    first.close()
    stored = json.loads(QSettings(ini, QSettings.Format.IniFormat).value("v2/plot/legend_map"))
    assert stored == {"shown": True, "x": 1.0, "y": 0.25}

    second = MainWindow(settings=QSettings(ini, QSettings.Format.IniFormat))
    qtbot.addWidget(second)
    assert legend(second).is_shown() and legend(second).position() == (1.0, 0.25)
    assert not legend(second, "stacked").is_shown()
    assert second.plot_area.legend_button.isChecked()
    second.plot_area.set_current_view("stacked")
    assert not second.plot_area.legend_button.isChecked()
    second.reset_settings()
    assert not legend(second).is_shown() and legend(second).position() == (0.0, 0.0)
    assert not errors


def test_stored_legend_settings_are_checked(qapp):
    from mag_opt_detective.gui.plots.legend import PlotLegend

    plot = pg.PlotItem()
    item = PlotLegend(plot)
    for bad in ("nonsense", "[]", '{"shown": 1, "x": 0, "y": 0}', '{"shown": true, "x": "a"}'):
        assert not item.set_settings_value(bad)
    assert not item.is_shown()
    assert item.set_settings_value('{"shown": true, "x": 3, "y": -1}')
    assert item.is_shown() and item.position() == (1.0, 0.0)  # kept inside the plot


# ---------------------------------------------------------------------- rows
def test_the_rows_follow_the_picked_curves(processed):
    w, c = processed, processed.controller
    panel = w.panels["points"]
    c.record_points([0.5, 1.0], [300.0, 320.0], unit="cm-1")
    assert legend(w).texts() == ["LL 1"] and emphasis(w) == [True]
    c.add_curve("CR")  # without points: nothing drawn, nothing listed
    assert legend(w).texts() == ["LL 1"] and emphasis(w) == [False]
    c.record_points([2.0], [500.0], unit="cm-1")
    assert legend(w).texts() == ["LL 1", "CR"] and emphasis(w) == [False, True]
    other, current = legend(w).entries()
    assert current.marker and current.brush.color() == curve_color(1)  # filled, as the chip
    assert other.brush is None and other.pens[-1].color() == curve_color(0)  # an open ring
    chips = panel.chips.chips()
    assert [chip.color for chip in chips] == [curve_color(0), curve_color(1)]

    c.rename_curve("CR 2")
    assert legend(w).texts() == ["LL 1", "CR 2"]
    c.drop_curve("LL 1")
    assert legend(w).texts() == ["CR 2"]
    w.commands["undo"].trigger()
    assert legend(w).texts() == ["LL 1", "CR 2"]
    c.set_curve("LL 1")
    assert emphasis(w) == [True, False]

    panel.markers.set_value("current")  # only the current curve is drawn
    assert legend(w).texts() == ["LL 1"]
    legend(w).set_shown(True)
    panel.markers.set_value("hidden")  # nothing to list: the legend hides
    assert legend(w).texts() == [] and legend(w).is_shown() and not legend(w).isVisible()
    panel.markers.set_value("all")
    assert legend(w).isVisible()

    set_unit(w, "meV")
    assert legend(w).texts() == ["LL 1", "CR 2"]
    process(w)
    assert legend(w).texts() == ["LL 1", "CR 2"]
    w.plot_area.set_current_view("stacked")  # the same curves, no models (none drawn there)
    assert legend(w, "stacked").texts() == ["LL 1", "CR 2"]


def test_the_stacked_legend_lists_only_the_curves_drawn_there(processed):
    w, c = processed, processed.controller
    c.record_points([0.5, 1.5], [300.0, 320.0], unit="cm-1")  # LL 1: the 1st and 3rd traces
    c.add_curve("odd")
    c.record_points([1.0, 2.0], [500.0, 520.0], unit="cm-1")  # the 2nd and 4th traces only
    w.plot_area.set_current_view("stacked")
    assert legend(w, "stacked").texts() == ["LL 1", "odd"]
    c.set_view(stacked_every=2)  # traces 0.5 and 1.5 T: "odd" has no marker there
    assert legend(w, "stacked").texts() == ["LL 1"]
    assert legend(w).texts() == ["LL 1", "odd"]  # the map draws both
    w.panels["points"].markers.set_value("current")  # only "odd", which draws nothing here
    assert legend(w, "stacked").texts() == []
    c.set_view(stacked_every=1)
    assert legend(w, "stacked").texts() == ["odd"]


def test_the_rows_follow_the_models_shown(processed):
    w, c = processed, processed.controller
    models = models_of(w)
    c.record_points([1.0], [300.0], unit="cm-1")
    dirac = models.entries[0]
    assert legend(w).texts() == ["LL 1"]  # the Dirac model starts hidden
    models.set_visible(dirac, True)
    assert legend(w).texts() == ["LL 1", "Massive Dirac · 5 transitions"]
    zeeman = models.add("zeeman")
    assert legend(w).texts()[-1] == "Zeeman / magnon"  # one branch
    ms.add_branch(zeeman)
    ms.add_branch(zeeman)
    models.edited(zeeman, structure=True)
    texts = ["LL 1", "Massive Dirac · 5 transitions", "Zeeman / magnon · 3 branches"]
    assert legend(w).texts() == texts
    assert [e.group for e in legend(w).entries()] == ["points", "models", "models"]
    line = legend(w).entries()[2]
    assert not line.marker and line.pens[-1].style() == Qt.PenStyle.DashLine
    assert line.pens[-1].color().name() == zeeman.color

    dirac.model.n_lines = 2  # a parameter change
    models.edited(dirac)
    assert legend(w).texts()[1] == "Massive Dirac · 2 transitions"
    set_unit(w, "THz")
    assert legend(w).texts()[1:] == ["Massive Dirac · 2 transitions", texts[2]]
    models.set_visible(dirac, False)
    assert legend(w).texts() == ["LL 1", texts[2]]
    process(w)
    assert legend(w).texts() == ["LL 1", texts[2]]
    models.remove(zeeman)
    assert legend(w).texts() == ["LL 1"]
    w.plot_area.set_current_view("stacked")
    assert legend(w, "stacked").texts() == ["LL 1"]


# ---------------------------------------------------------------------- place
def test_a_drag_moves_the_legend_and_its_place_survives_zoom_and_resize(shown, qtbot):
    w = shown
    vb = w.plots.map.plot.vb
    w.tools.set_active("pick")
    points = w.controller.points.points("LL 2")[0].size
    before = vb.viewRange()
    assert legend(w).position() == (0.0, 0.0)
    assert_drawn_where_placed(w)
    rect = legend(w).rect_in_view()
    assert rect.left() == pytest.approx(MARGIN) and rect.top() == pytest.approx(MARGIN)

    start = legend_centre(w)
    with qtbot.waitSignal(legend(w).moved):
        drag(qtbot, w, [start, start + QPoint(120, 80)])
    x, y = legend(w).position()
    assert 0 < x < 1 and 0 < y < 1
    assert_near(legend(w).rect_in_view().topLeft(), rect.topLeft() + QPointF(120, 80))
    assert vb.viewRange() == before  # no pan
    assert w.controller.points.points("LL 2")[0].size == points  # no point picked

    start = legend_centre(w)
    drag(qtbot, w, [start, start + QPoint(3000, 3000)])  # beyond the corner: kept inside
    assert legend(w).position() == (1.0, 1.0)
    assert_drawn_where_placed(w)
    corner = vb.rect().bottomRight() - QPointF(MARGIN, MARGIN)
    assert_near(legend(w).rect_in_view().bottomRight(), corner)

    vb.setRange(xRange=(0.8, 1.2), yRange=(400.0, 500.0), padding=0)  # zoom in
    assert_drawn_where_placed(w)
    assert_near(legend(w).rect_in_view().bottomRight(), corner)
    w.resize(1100, 800)
    qtbot.waitUntil(lambda: vb.rect().bottomRight() != corner + QPointF(MARGIN, MARGIN))
    qtbot.wait(50)
    corner = vb.rect().bottomRight() - QPointF(MARGIN, MARGIN)
    assert legend(w).position() == (1.0, 1.0)
    assert_near(legend(w).rect_in_view().bottomRight(), corner)
    assert_drawn_where_placed(w)
    assert json.loads(legend(w).settings_value())["x"] == 1.0


def test_a_click_on_the_legend_is_no_move(shown, qtbot):
    w = shown
    moves = []
    legend(w).moved.connect(lambda: moves.append(legend(w).position()))
    QTest.mouseClick(w.plots.map.view.viewport(), LEFT, PLAIN, legend_centre(w))
    assert moves == [] and legend(w).position() == (0.0, 0.0)
    start = legend_centre(w)
    drag(qtbot, w, [start, start + QPoint(40, 30)])
    assert len(moves) == 1 and moves[0] != (0.0, 0.0)


def test_overlay_layers_stay_below_the_legend_in_their_order(shown):
    w = shown
    plot = w.plots.map
    x = np.array([1.0, 1.5])
    for k in range(30):  # far more layers than models and previews make
        layer = plot.layer(f"extra {k}")
        layer.set_curves([(x, x)], pg.mkPen("w"), shadow_pen=pg.mkPen("k"))
        layer.set_points(x, x)
        for _group in range(25):
            layer.add_points(x, x)
    spans = [[item.zValue() for item in layer.items()] for layer in plot.layers()]
    assert max(max(span, default=0.0) for span in spans) < 19.0 < legend(w).zValue()
    first = [span for span in spans[:18] if span]  # the 18 lowest layers have their own z
    for below, above in itertools.pairwise(first):
        assert max(below) < min(above)  # each layer whole above the one before
    lowest = [min(span) for span in spans if span]
    assert lowest == sorted(lowest)


def test_presses_on_the_legend_reach_no_tool_and_clicks_elsewhere_do(shown, qtbot):
    w, c = shown, shown.controller
    tool = w.autopick
    w.tools.set_active("pick")
    count = c.points.points("LL 2")[0].size
    QTest.mouseClick(w.plots.map.view.viewport(), LEFT, PLAIN, legend_centre(w))
    assert c.points.points("LL 2")[0].size == count  # nothing picked under the legend
    QTest.mouseClick(w.plots.map.view.viewport(), LEFT, PLAIN, data_point(w, 1.8, 700.0))
    assert c.points.points("LL 2")[0].size == count + 1  # a click elsewhere picks

    w.tools.set_active("autopick")
    tool.bar.mode.set_value("detect")
    start = legend_centre(w)
    with qtbot.waitSignal(legend(w).moved):
        drag(qtbot, w, [start, start + QPoint(60, 40), start + QPoint(150, 100)])
    assert legend(w).position() != (0.0, 0.0)  # the legend moved
    assert tool.region.roi is None and not tool.region.outline_visible()  # no region drawn
    assert tool.target is None

    drag(qtbot, w, [data_point(w, 0.8, 250.0), data_point(w, 1.7, 800.0)])  # elsewhere
    assert tool.region.roi is not None and tool.target is not None  # a region drawn

    tool.bar.mode.set_value("track")
    QTest.mouseClick(w.plots.map.view.viewport(), LEFT, PLAIN, legend_centre(w))
    assert tool.target is None  # no track from under the legend
    QTest.mouseClick(w.plots.map.view.viewport(), LEFT, PLAIN, data_point(w, 1.0, 400.0))
    assert tool.target is not None and tool.target.seed[0] == pytest.approx(1.0, abs=0.05)


def test_the_region_drawn_over_the_legend_takes_presses_there(shown, qtbot):
    w = shown
    tool = w.autopick
    w.tools.set_active("autopick")
    tool.bar.mode.set_value("detect")
    vb = w.plots.map.plot.vb
    rect = legend(w).rect_in_view()
    corners = [vb.mapToView(rect.topLeft() - QPointF(5, 5)), vb.mapToView(rect.center())]
    tool.draw_region(corners, "rect")
    roi = tool.region.roi
    before = roi.pos()
    inside = viewport_point(w, rect.center() - QPointF(rect.width() / 4, rect.height() / 4))
    drag(qtbot, w, [inside, inside + QPoint(30, 30)])
    assert roi.pos() != before  # the region moved
    assert legend(w).position() == (0.0, 0.0)  # the legend did not


def test_detect_leaves_presses_on_other_movable_items_to_them(shown, qtbot):
    """As on the legend, a press on a region the user can drag (e.g. a baseline region) moves
    that region and draws no auto-pick region; one on a fixed region draws as before."""
    w = shown
    tool = w.autopick
    w.tools.set_active("autopick")
    tool.bar.mode.set_value("detect")
    band = pg.LinearRegionItem((300.0, 500.0), orientation="horizontal")
    band.setZValue(9.5)
    w.plots.map.plot.addItem(band, ignoreBounds=True)
    start = data_point(w, 1.2, 400.0)
    drag(qtbot, w, [start, start + QPoint(0, 20), start + QPoint(0, 40)])
    assert band.getRegion()[0] < 300.0  # the band moved down
    assert tool.region.roi is None and tool.target is None

    band.setMovable(False)  # as the Processing panel's guide band
    lo, hi = band.getRegion()
    drag(qtbot, w, [start, start + QPoint(40, 20), start + QPoint(80, 40)])
    assert band.getRegion() == (lo, hi)
    assert tool.region.roi is not None and tool.target is not None  # a region drawn


def test_the_legend_follows_the_theme(processed):
    w = processed
    item = legend(w)
    w.theme.set_scheme("light")
    light = item.colors()
    w.theme.set_scheme("dark")
    dark = item.colors()
    assert light.text != dark.text and light.panel != dark.panel
    assert 0 < dark.panel.alpha() < 255  # the map shows through a little
    assert np.isclose(dark.panel.alphaF(), 0.88, atol=0.01)
