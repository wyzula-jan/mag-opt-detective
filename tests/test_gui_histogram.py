"""The histograms keep their value range while only the colour levels change, unless the
auto-scale button of the plot toolbar is on; new data fits them again."""

import math

import numpy as np
import pyqtgraph as pg
import pytest
from PySide6.QtCore import QPoint, QPointF, QSettings, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QToolButton

import gui_helpers
from gui_helpers import inspector_page, load_sweep, process, select, set_unit
from mag_opt_detective.core.spectra import FieldMap
from mag_opt_detective.gui.main_window import MainWindow
from mag_opt_detective.gui.plots import ColorMapPlot
from mag_opt_detective.gui.plots.colorscale import fit_range, same_values, tails, widened

window, errors = gui_helpers.window, gui_helpers.errors  # shared fixtures

LEFT, PLAIN = Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier
FOLLOW_SPAN = 1.2 * 1.2  # auto-scale: the levels +- 10 %, and pyqtgraph pads 10 % more


# ---------------------------------------------------------------------- helpers
def test_range_helpers():
    assert fit_range((0.9, 1.1), None) == pytest.approx((0.884, 1.116))
    assert fit_range((0.9, 1.1), (0.95, 1.2)) == pytest.approx((0.876, 1.224))
    assert fit_range((1.0, 1.0), None) == (0.5, 1.5)
    assert widened((0.0, 1.0), (0.2, 0.8)) == (0.0, 1.0)  # shown: unchanged
    assert widened((0.0, 1.0), (0.2, 1.5)) == pytest.approx((0.0, 1.575))  # one side only
    assert widened((0.0, 1.0), (-1.0, 0.5)) == pytest.approx((-1.1, 1.0))
    ramp = np.arange(1000.0)
    expected = np.percentile(ramp, [0.5, 99.5])
    assert tails(np.r_[ramp, np.nan, np.inf]) == pytest.approx(tuple(expected))
    assert tails(np.full(3, np.nan)) is None
    a = np.array([1.0, np.nan])
    assert same_values(a, a.copy()) and not same_values(a, np.array([1.0, 2.0]))
    assert same_values(None, None) and not same_values(a, None)
    assert not same_values(a, np.array([[1.0, np.nan]]))


# ---------------------------------------------------------------------- one map plot
def ramp_map(scale: float = 1.0) -> FieldMap:
    energy = np.linspace(100.0, 500.0, 40)
    field = np.linspace(0.0, 2.2, 12)
    return FieldMap(energy, field, scale * np.add.outer(energy / 500.0, field / 10.0))


@pytest.fixture
def plot(qtbot):
    plot = ColorMapPlot()
    qtbot.addWidget(plot)
    plot.resize(640, 420)
    plot.show()
    qtbot.waitExposed(plot)
    return plot


def value_range(plot) -> tuple[float, float]:
    return tuple(plot.hist.getHistogramRange())


def fitted(plot) -> tuple[float, float]:
    """The range a fit gives: the bulk of the drawn values and the levels."""
    return fit_range(plot.levels(), tails(plot.image.image))


def test_the_histogram_keeps_still_while_only_the_levels_change(plot, qtbot):
    assert not plot.scale_follows_levels() and not plot.scale.follows_levels()
    fmap = ramp_map()
    plot.set_map(fmap, levels=(0.4, 0.9))
    start = value_range(plot)
    assert start == pytest.approx(fitted(plot))  # fitted to the new data
    assert not plot.hist.vb.autoRangeEnabled()[1]  # the region would drag the view along

    with qtbot.waitSignal(plot.levelsEdited):
        plot.hist.region.setRegion((0.5, 0.7))  # a drag on the histogram ...
    plot.set_map(fmap, levels=(0.5, 0.7))  # ... and the redraw it brings
    assert value_range(plot) == start
    plot.set_levels(0.45, 0.8)  # dragged in the inspector
    assert value_range(plot) == start
    copy = FieldMap(fmap.energy, fmap.field, fmap.values.copy())  # a derivative, computed again
    plot.set_map(copy, levels=None)  # Auto levels of the same values
    assert value_range(plot) == start
    assert plot.hist.region.getRegion() == pytest.approx(plot.levels())


def test_levels_beyond_the_histogram_widen_it_just_enough(plot):
    plot.set_map(ramp_map(), levels=(0.4, 0.9))
    start = value_range(plot)
    plot.set_levels(0.5, 3.0)
    assert value_range(plot) == pytest.approx(widened(start, (0.5, 3.0)))
    lo, hi = value_range(plot)
    assert lo == pytest.approx(start[0]) and 3.0 < hi < 3.0 + 0.1 * (hi - lo)
    plot.set_levels(0.5, 0.6)  # back inside: stays wide
    assert value_range(plot) == pytest.approx((lo, hi))


def test_new_values_fit_the_histogram_again(plot):
    plot.set_map(ramp_map(), levels=(0.4, 0.9))
    plot.set_levels(0.5, 3.0)
    plot.set_map(ramp_map(10.0), levels=(4.0, 9.0))
    assert value_range(plot) == pytest.approx(fitted(plot))
    plot.clear_map()
    plot.set_map(ramp_map(10.0), levels=(4.0, 6.0))  # after no map: new data
    assert value_range(plot) == pytest.approx(fitted(plot))


def test_auto_scale_fits_the_histogram_to_the_levels(plot):
    plot.set_map(ramp_map(), levels=(0.4, 0.9))
    plot.set_scale_follows_levels(True)
    assert plot.scale.follows_levels()
    for levels in [(0.4, 0.9), (0.5, 0.6)]:  # switched on, then a level change
        plot.set_levels(*levels)
        lo, hi = value_range(plot)
        assert (lo + hi) / 2 == pytest.approx(sum(levels) / 2)
        assert hi - lo == pytest.approx(FOLLOW_SPAN * (levels[1] - levels[0]))
    plot.set_map(ramp_map(), levels=None)  # Auto levels: the whole histogram (as before)
    assert plot.hist.vb.autoRangeEnabled()[1]

    plot.set_scale_follows_levels(False)  # stays where it is
    plot.hist.vb.updateAutoRange()
    shown = value_range(plot)
    assert not plot.hist.vb.autoRangeEnabled()[1]
    plot.set_levels(0.5, 0.6)
    assert value_range(plot) == pytest.approx(shown)


@pytest.mark.parametrize("follow", [False, True])
def test_a_new_scale_style_keeps_the_choice(plot, follow):
    plot.set_map(ramp_map(), levels=(0.4, 0.9))
    plot.set_scale_follows_levels(follow)
    plot.set_scale_style("bar")
    assert plot.scale.follows_levels() is follow  # the bar's axis always follows anyway
    plot.set_scale_style("histogram")
    assert plot.scale.follows_levels() is follow
    lo, hi = value_range(plot)
    if follow:
        assert hi - lo == pytest.approx(FOLLOW_SPAN * 0.5)
    else:  # a new histogram fits the data it is attached to
        assert (lo, hi) == pytest.approx(fitted(plot))


def double_click(view: pg.GraphicsView, vb: pg.ViewBox) -> None:
    """A double-click on the middle of *vb*, as the window system sends it (QTest's
    mouseDClick alone has no presses and releases, which pyqtgraph needs)."""
    centre, viewport = view.mapFromScene(vb.sceneBoundingRect().center()), view.viewport()
    QTest.mousePress(viewport, LEFT, PLAIN, centre)
    QTest.mouseRelease(viewport, LEFT, PLAIN, centre)
    QTest.mouseDClick(viewport, LEFT, PLAIN, centre)
    QTest.mouseRelease(viewport, LEFT, PLAIN, centre)


def test_double_click_fits_the_histogram(plot):
    plot.set_map(ramp_map(), levels=(0.4, 0.9))
    plot.hist.setHistogramRange(5.0, 6.0)  # as if zoomed away with the wheel
    double_click(plot.scale.widget, plot.hist.vb)
    assert value_range(plot) == pytest.approx(fitted(plot))
    assert plot.levels() == pytest.approx((0.4, 0.9))  # the levels stay


# ---------------------------------------------------------------------- the window
@pytest.fixture
def shown(window, sweep, qtbot):
    window.resize(1400, 900)
    window.show()
    qtbot.waitExposed(window)
    load_sweep(window, sweep)
    process(window)
    colour_page(window).hist_section.set_expanded(True, animate=False)
    return window


def colour_page(window):
    return inspector_page(window, "colour")


def classic_range(window, view: str = "map") -> tuple[float, float]:
    return tuple(window.plots[view].hist.getHistogramRange())


def inspector_range(window) -> tuple[float, float]:
    return colour_page(window).histogram.value_range()


def ranges(window) -> tuple:
    return classic_range(window), inspector_range(window)


def fits(window) -> tuple:
    """The ranges a fit gives for the map on screen."""
    levels, image = window.plots.map.levels(), window.plots.map.image.image
    shown = window.shown_maps.get("map").values
    return fit_range(levels, tails(image)), fit_range(levels, tails(shown))


def move_pause_ms() -> int:
    """pyqtgraph drops mouse moves within 1/mouseRateLimit s of the last one."""
    rate = pg.getConfigOption("mouseRateLimit")
    return 2 * math.ceil(1000 / rate) if rate > 0 else 0


def drag_level(qtbot, window, line: int, value: float) -> None:
    """Drag level *line* (0 lower, 1 upper) of the map's classic histogram to *value*."""
    hist, view = window.plots.map.hist, window.plots.map.scale.widget
    x = hist.vb.viewRect().center().x()

    def pos(y: float) -> QPoint:
        return view.mapFromScene(hist.vb.mapViewToScene(QPointF(x, y)))

    start, end = pos(hist.region.getRegion()[line]), pos(value)
    viewport = view.viewport()
    QTest.mousePress(viewport, LEFT, PLAIN, start)
    for point in (start + (end - start) / 2, end):
        qtbot.wait(move_pause_ms())
        QTest.mouseMove(viewport, point)
    QTest.mouseRelease(viewport, LEFT, PLAIN, end)


def test_a_level_drag_leaves_both_histograms_still(shown, qtbot):
    w, c = shown, shown.controller
    page = colour_page(w)
    assert not w.plot_area.auto_scale_button.isChecked()  # still by default
    start = ranges(w)
    for shown_range, fit in zip(start, fits(w), strict=True):
        assert shown_range == pytest.approx(fit)
    lo, hi = c.view.levels["Ratio"]
    drag_level(qtbot, w, 1, hi - 0.4 * (hi - lo))  # the upper level, with the mouse
    new_lo, new_hi = c.view.levels["Ratio"]
    assert new_lo == pytest.approx(lo) and new_hi < hi - 0.2 * (hi - lo)
    assert w.plots.map.hist.region.getRegion() == pytest.approx((new_lo, new_hi))
    assert ranges(w) == start  # the region moved, the histograms did not

    page.histogram.region.setRegion((lo + 0.01, hi - 0.01))  # dragged in the inspector
    assert c.view.levels["Ratio"] == pytest.approx((lo + 0.01, hi - 0.01))
    page.fields.lo.setValue(lo + 0.02)  # typed
    for mode in ("auto", "fixed"):  # the same map, other levels
        page.mode.set_value(mode)
    assert c.view.level_mode("Ratio") == "fixed"
    assert ranges(w) == start


def test_with_auto_scale_both_histograms_follow_the_levels(shown, qtbot):
    w, c = shown, shown.controller
    w.plot_area.auto_scale_button.click()
    lo, hi = c.view.levels["Ratio"]
    drag_level(qtbot, w, 0, lo + 0.4 * (hi - lo))  # the lower level
    levels = c.view.levels["Ratio"]
    assert levels[0] > lo + 0.2 * (hi - lo)
    r0, r1 = classic_range(w)
    assert (r0 + r1) / 2 == pytest.approx(sum(levels) / 2)  # the levels in the middle
    assert r1 - r0 == pytest.approx(FOLLOW_SPAN * (levels[1] - levels[0]))
    values = w.shown_maps.get("map").values
    assert inspector_range(w) == pytest.approx(fit_range(levels, tails(values)))


@pytest.mark.parametrize("auto_scale", [False, True])
def test_new_data_fits_both_histograms(shown, auto_scale):
    w = shown
    if auto_scale:
        w.plot_area.auto_scale_button.click()
    for change in (
        lambda: select(w, order=1),  # a derivative (Symmetric levels)
        lambda: select(w, kind="Data", order=0),  # another kind (Auto levels)
        lambda: select(w, kind="Ratio", order=1, per_unit=True),  # per cm-1 ...
        lambda: set_unit(w, "meV"),  # ... and per meV: other values
    ):
        before = ranges(w)
        change()
        assert ranges(w) != before
        classic, inspector = fits(w)
        assert inspector_range(w) == pytest.approx(inspector)
        if not auto_scale:
            assert classic_range(w) == pytest.approx(classic)
        elif w.controller.current_levels() is not None:  # kept levels: around them
            levels = w.plots.map.levels()
            r0, r1 = classic_range(w)
            assert r1 - r0 == pytest.approx(FOLLOW_SPAN * (levels[1] - levels[0]))
        else:  # Auto levels: the whole histogram
            assert w.plots.map.hist.vb.autoRangeEnabled()[1]


def test_levels_typed_beyond_the_histograms_widen_them_just_enough(shown):
    w = shown
    page = colour_page(w)
    classic, inspector = ranges(w)
    lo = page.fields.levels()[0]
    page.fields.hi.setValue(classic[1] + 1.0)
    levels = (lo, classic[1] + 1.0)
    assert classic_range(w) == pytest.approx(widened(classic, levels))
    assert inspector_range(w) == pytest.approx(widened(inspector, levels))
    wide = ranges(w)
    page.fields.hi.setValue(lo + 0.1)  # back inside: they keep still
    assert ranges(w) == wide


def test_double_click_fits_each_histogram(shown):
    w = shown
    page = colour_page(w)
    lo = page.fields.levels()[0]
    page.fields.hi.setValue(5.0)
    page.fields.hi.setValue(lo + 0.1)  # both histograms stay wide
    assert ranges(w) != fits(w)
    double_click(w.plots.map.scale.widget, w.plots.map.hist.vb)
    assert classic_range(w) == pytest.approx(fits(w)[0])
    assert inspector_range(w) != pytest.approx(fits(w)[1])  # each on its own
    double_click(page.histogram.view, page.histogram.plot.vb)
    assert inspector_range(w) == pytest.approx(fits(w)[1])
    assert w.controller.view.levels["Ratio"] == pytest.approx((lo, lo + 0.1))


def fit_button(window) -> QToolButton:
    head = window.plot_area.tabs.parentWidget()
    return next(b for b in head.findChildren(QToolButton) if b.accessibleName() == "Fit to data")


@pytest.mark.parametrize("auto_scale", [False, True])
def test_fit_to_data_fits_the_histograms_of_the_plot_on_screen(shown, auto_scale):
    w = shown
    if auto_scale:
        w.plot_area.auto_scale_button.click()
    page = colour_page(w)
    lo = page.fields.levels()[0]
    page.fields.hi.setValue(5.0)
    page.fields.hi.setValue(lo + 0.1)  # still: both stay wide
    w.plot_area.set_current_view("stacked")
    w.plots.map.hist.setHistogramRange(5.0, 6.0)  # as if zoomed away with the wheel
    before = ranges(w)
    fit_button(w).click()  # fits the stacked plot, which has no histogram
    assert ranges(w) == before

    w.plot_area.set_current_view("map")
    w.plots.map.hist.setHistogramRange(5.0, 6.0)
    assert ranges(w) != fits(w)
    fit_button(w).click()
    for shown_range, fit in zip(ranges(w), fits(w), strict=True):
        assert shown_range == pytest.approx(fit)
    assert w.controller.view.levels["Ratio"] == pytest.approx((lo, lo + 0.1))


def test_the_histograms_say_how_to_fit_them(window):
    w = window
    assert w.plots.map.hist.toolTip() == "Double-click to fit, scroll to zoom"
    assert colour_page(w).histogram.plot.toolTip() == "Double-click to fit"
    w.plot_area.scale_style_button.click()  # slim bars, then new histograms
    w.plot_area.scale_style_button.click()
    assert w.plots.reference.hist.toolTip() == "Double-click to fit, scroll to zoom"


def test_the_auto_scale_button_switches_every_histogram(window):
    w, area = window, window.plot_area
    button = area.auto_scale_button
    assert button.isCheckable() and not button.isChecked()
    assert button.accessibleName() == "Auto-scale the histograms to the colour levels"
    maps = [w.plots.map, w.plots.reference]
    histogram = colour_page(w).histogram

    def following() -> list[bool]:
        return [*(m.scale_follows_levels() and m.scale.follows_levels() for m in maps),
                histogram.follows_levels()]  # fmt: skip

    assert following() == [False, False, False]
    button.click()
    assert following() == [True, True, True]
    area.scale_style_button.click()  # slim bars, then new histograms: they follow too
    area.scale_style_button.click()
    assert following() == [True, True, True]
    button.click()
    assert following() == [False, False, False]


def test_the_auto_scale_choice_is_remembered(qtbot, tmp_path, errors):
    ini = str(tmp_path / "settings.ini")
    first = MainWindow(settings=QSettings(ini, QSettings.Format.IniFormat))
    qtbot.addWidget(first)
    assert not first.plot_area.auto_scale_button.isChecked()
    first.plot_area.auto_scale_button.click()
    first.close()
    stored = QSettings(ini, QSettings.Format.IniFormat).value("v2/plot/histogram_auto_scale")
    assert str(stored).lower() == "true"

    second = MainWindow(settings=QSettings(ini, QSettings.Format.IniFormat))
    qtbot.addWidget(second)
    assert second.plot_area.auto_scale_button.isChecked()
    assert second.plots.map.scale.follows_levels()
    assert colour_page(second).histogram.follows_levels()
    second.persistence.reset()  # View > Reset settings: still again
    assert not second.plot_area.auto_scale_button.isChecked()
    assert not second.plots.reference.scale.follows_levels()
    second.close()


def test_the_plot_toolbar_with_the_auto_scale_button_fits_1100_px(window, qtbot):
    window.resize(1100, 800)
    window.show()
    qtbot.waitExposed(window)
    assert window.side_panel.is_open() and window.inspector_panel.is_open()
    assert window.minimumSizeHint().width() <= 1100 and window.width() == 1100
    area = window.plot_area
    head = area.tabs.parentWidget()
    tabs_end = area.tabs.geometry().right()
    tools = [
        b for b in head.findChildren(QToolButton) if b.isVisible() and not area.tabs.isAncestorOf(b)
    ]
    assert area.auto_scale_button in tools
    for button in tools:  # right of the tabs, all inside the head
        rect = button.geometry().translated(button.parentWidget().mapTo(head, QPoint(0, 0)))
        assert rect.left() > tabs_end + 4 and head.rect().contains(rect)
