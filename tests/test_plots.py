import numpy as np
import pyqtgraph as pg
import pytest
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QColor, QImage, QWheelEvent
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtTest import QTest

from mag_opt_detective.core.colormaps import lut
from mag_opt_detective.core.spectra import FieldMap
from mag_opt_detective.gui.plots import (
    IMAGE_SUFFIXES,
    SCALE_STYLES,
    BarScale,
    ColorMapPlot,
    HistogramScale,
    PlotColors,
    StackedPlot,
    lookup_table,
    pixel_rect,
    robust_levels,
)
from mag_opt_detective.gui.plots.colors import qcolor

LIGHT = PlotColors(
    background="#ffffff",
    foreground="#3b3443",
    grid="#3b344317",
    crosshair="#3b344366",
    accent="#7a2a8c",
    well="#e9e6ed",
)


LEFT = Qt.MouseButton.LeftButton
NO_KEYS = Qt.KeyboardModifier.NoModifier


def ramp_map(n_energy=40, n_field=12) -> FieldMap:
    energy = np.linspace(100.0, 500.0, n_energy)
    field = np.linspace(0.0, 2.2, n_field)
    values = np.add.outer(energy / 500.0, field / 10.0)
    return FieldMap(energy, field, values)


@pytest.fixture
def make_plot(qtbot):
    def make(style="histogram", size=(640, 420)) -> ColorMapPlot:
        plot = ColorMapPlot(scale_style=style)
        qtbot.addWidget(plot)
        plot.resize(*size)
        plot.show()
        qtbot.waitExposed(plot)
        return plot

    return make


def test_package_keeps_old_names():
    assert ".svg" in IMAGE_SUFFIXES
    assert robust_levels(np.array([np.nan, 1.0, 1.0])) == (0.5, 1.5)
    assert pixel_rect(np.array([0.0, 1.0]), np.array([10.0, 20.0])).width() == 2.0
    assert SCALE_STYLES == ("histogram", "bar")


def test_plot_colors_from_mapping():
    colors = PlotColors.from_mapping({**LIGHT.__dict__, "unknown": "#000"})
    assert colors == LIGHT
    assert PlotColors.from_mapping({"accent": "#123456"}).background == "#000000"
    assert qcolor("#11223344").getRgb() == (0x11, 0x22, 0x33, 0x44)  # CSS order
    with pytest.raises(ValueError):
        PlotColors(well="not a colour")


@pytest.mark.parametrize("start, other", [("histogram", "bar"), ("bar", "histogram")])
def test_style_switch_keeps_levels_and_colour_map(make_plot, qtbot, start, other):
    plot = make_plot(start)
    plot.set_map(ramp_map(), levels=(0.4, 0.9), cmap="viridis")
    old = plot.scale
    with qtbot.assertNotEmitted(plot.levelsEdited):
        plot.set_scale_style(other)
    assert plot.scale_style() == other
    assert plot.scale is not old
    assert plot.levels() == pytest.approx((0.4, 0.9))
    assert plot.colormap() == "viridis"
    assert plot.image.levels == pytest.approx([0.4, 0.9])
    assert old.image() is None  # detached
    assert (plot.hist is not None) == (other == "histogram")
    assert plot.scale_container.layout().count() == 1
    if other == "bar":
        np.testing.assert_array_equal(plot.image.lut, lookup_table("viridis"))
    else:
        assert plot.image.lut.__self__ is plot.hist


def test_switching_to_the_same_style_does_nothing(make_plot):
    plot = make_plot("bar")
    scale = plot.scale
    plot.set_scale_style("bar")
    assert plot.scale is scale
    with pytest.raises(ValueError):
        plot.set_scale_style("ribbon")


def test_detached_histogram_no_longer_drives_the_image(qtbot):
    scale = HistogramScale()
    qtbot.addWidget(scale.widget)
    image = pg.ImageItem(axisOrder="row-major")
    scale.set_colormap("magma")
    scale.set_levels(0.0, 1.0)
    scale.attach(image)
    image.setImage(np.linspace(0, 1, 100).reshape(10, 10), autoLevels=False, levels=(0, 1))
    attached_hist = scale.hist.plot.getData()[1].copy()
    assert attached_hist.sum() > 0  # the histogram follows the image while attached

    scale.detach()
    assert scale.image() is None
    assert not callable(image.lut)  # a static copy, not the histogram's bound method
    image.setImage(np.full((10, 10), 7.0), autoLevels=False, levels=(6, 8))
    np.testing.assert_array_equal(scale.hist.plot.getData()[1], attached_hist)

    custom = np.zeros((256, 3), dtype=np.uint8)
    image.setLookupTable(custom)
    scale.set_colormap("bipolar")
    assert image.lut is custom
    scale.hist.region.setRegion((3.0, 4.0))
    assert image.levels == pytest.approx([6, 8])


def test_histogram_emits_only_for_user_changes(make_plot, qtbot):
    plot = make_plot("histogram")
    with qtbot.assertNotEmitted(plot.levelsEdited):
        plot.set_map(ramp_map())
        plot.set_map(ramp_map(), levels=(0.3, 0.8), cmap="grey")
        plot.set_levels(0.2, 0.7)
        plot.scale.set_levels(0.25, 0.75)
    with qtbot.waitSignal(plot.levelsEdited) as blocker:
        plot.hist.region.setRegion((0.35, 0.65))  # as if the user dragged it
    assert blocker.args == pytest.approx([0.35, 0.65])
    assert plot.image.levels == pytest.approx([0.35, 0.65])


def test_no_signal_without_a_map(make_plot, qtbot):
    plot = make_plot("histogram")
    with qtbot.assertNotEmitted(plot.levelsEdited):
        plot.hist.region.setRegion((0.1, 0.2))


def drag(widget, y0: float, y1: float, x: int = 12) -> None:
    QTest.mousePress(widget, LEFT, NO_KEYS, QPoint(x, y0))
    QTest.mouseMove(widget, QPoint(x, round((y0 + y1) / 2)))
    QTest.mouseMove(widget, QPoint(x, y1))


def release(widget, y: float, x: int = 12) -> None:
    QTest.mouseRelease(widget, LEFT, NO_KEYS, QPoint(x, y))


def test_bar_handle_drag_is_live_and_emits_on_release(make_plot, qtbot):
    plot = make_plot("bar")
    plot.set_map(ramp_map(), levels=(0.4, 0.8))
    bar = plot.scale.bar
    value_range = bar.value_range()
    y_hi = round(bar.y_for(0.8))
    with qtbot.assertNotEmitted(plot.levelsEdited):
        drag(bar, y_hi, y_hi - 30)
        assert bar.value_range() == value_range  # frozen while dragging
    lo, hi = bar.levels()
    assert lo == 0.4
    assert hi > 0.85
    assert plot.image.levels == pytest.approx([lo, hi])  # live
    with qtbot.waitSignal(plot.levelsEdited) as blocker:
        release(bar, y_hi - 30)
    assert blocker.args == pytest.approx([lo, hi])
    assert bar.value_range() != value_range


def test_bar_drags_lo_handle_and_middle(make_plot, qtbot):
    plot = make_plot("bar")
    plot.set_map(ramp_map(), levels=(0.4, 0.8))
    bar = plot.scale.bar
    y_lo = round(bar.y_for(0.4))
    drag(bar, y_lo, y_lo + 20)
    with qtbot.waitSignal(plot.levelsEdited) as blocker:
        release(bar, y_lo + 20)
    lo, hi = blocker.args
    assert lo < 0.4
    assert hi == pytest.approx(0.8)

    y_mid = round(bar.y_for((lo + hi) / 2))
    drag(bar, y_mid, y_mid + 15)
    with qtbot.waitSignal(plot.levelsEdited) as blocker:
        release(bar, y_mid + 15)
    new_lo, new_hi = blocker.args
    assert new_lo < lo
    assert new_hi - new_lo == pytest.approx(hi - lo)


def test_bar_handle_cannot_cross(make_plot, qtbot):
    plot = make_plot("bar")
    plot.set_map(ramp_map(), levels=(0.4, 0.8))
    bar = plot.scale.bar
    y_hi = round(bar.y_for(0.8))
    drag(bar, y_hi, bar.height() - 2)
    release(bar, bar.height() - 2)
    lo, hi = bar.levels()
    assert hi > lo == 0.4


def test_bar_click_without_move_does_not_emit(make_plot, qtbot):
    plot = make_plot("bar")
    plot.set_map(ramp_map(), levels=(0.4, 0.8))
    bar = plot.scale.bar
    y = round(bar.y_for(0.8))
    with qtbot.assertNotEmitted(plot.levelsEdited):
        QTest.mouseClick(bar, LEFT, NO_KEYS, QPoint(12, y))


def wheel(widget, delta: int) -> None:
    pos = QPointF(10, widget.height() / 2)
    event = QWheelEvent(
        pos,
        widget.mapToGlobal(pos),
        QPoint(0, 0),
        QPoint(0, delta),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    widget.wheelEvent(event)


def test_bar_wheel_scales_the_span_and_emits_once(make_plot, qtbot):
    plot = make_plot("bar")
    plot.set_map(ramp_map(), levels=(0.4, 0.8))
    bar = plot.scale.bar
    emitted = []
    plot.levelsEdited.connect(lambda lo, hi: emitted.append((lo, hi)))
    wheel(bar, 120)
    wheel(bar, 120)
    lo, hi = bar.levels()
    assert hi - lo < 0.4
    assert (lo + hi) / 2 == pytest.approx(0.6)
    assert plot.image.levels == pytest.approx([lo, hi])
    qtbot.waitUntil(lambda: bool(emitted))
    qtbot.wait(2 * bar.WHEEL_DELAY_MS)
    assert emitted == [pytest.approx((lo, hi))]


def test_bar_wheel_edit_is_cancelled_by_set_levels(make_plot, qtbot):
    plot = make_plot("bar")
    plot.set_map(ramp_map(), levels=(0.4, 0.8))
    wheel(plot.scale.bar, -120)
    with qtbot.assertNotEmitted(plot.levelsEdited, wait=2 * plot.scale.bar.WHEEL_DELAY_MS):
        plot.set_levels(0.1, 0.2)
    assert plot.levels() == (0.1, 0.2)


@pytest.mark.parametrize("cls", [HistogramScale, BarScale])
def test_scale_set_levels_never_emits(qtbot, cls):
    scale = cls()
    qtbot.addWidget(scale.widget)
    scale.attach(pg.ImageItem())
    with qtbot.assertNotEmitted(scale.levelsEdited):
        scale.set_levels(1.0, 2.0)
        scale.set_levels(-1.0, 3.0, auto_range=True)
    assert scale.levels() == (-1.0, 3.0)


def rendered_colours(plot: ColorMapPlot) -> list[tuple[int, int, int]]:
    plot.image.render()
    qimage = plot.image.qimage
    return [qimage.pixelColor(col, 0).getRgb()[:3] for col in range(qimage.width())]


@pytest.mark.parametrize("style", SCALE_STYLES)
@pytest.mark.parametrize(
    "name, expected",
    [
        ("grey", [(0, 0, 0), (128, 128, 128), (255, 255, 255)]),
        ("bipolar", [(0, 255, 255), (0, 0, 0), (255, 255, 0)]),
    ],
)
def test_grey_and_bipolar_render(make_plot, style, name, expected):
    plot = make_plot(style)
    fmap = FieldMap(np.array([1.0]), np.array([0.0, 1.0, 2.0]), np.array([[0.0, 0.5, 1.0]]))
    plot.set_map(fmap, levels=(0.0, 1.0), cmap=name)
    colours = rendered_colours(plot)
    for got, want in zip(colours, expected, strict=True):
        assert np.abs(np.subtract(got, want)).max() <= 2


def similar(a: QColor, b: QColor, tol: int = 8) -> bool:
    return np.abs(np.subtract(a.getRgb(), b.getRgb())).max() <= tol


@pytest.mark.parametrize("name, top, bottom", [("bipolar", "#ffff00", "#00ffff")])
def test_bar_paints_clamped_colours(make_plot, name, top, bottom):
    plot = make_plot("bar")
    plot.set_map(ramp_map(), levels=(0.4, 0.8), cmap=name)
    bar = plot.scale.bar
    image = bar.grab().toImage()
    rect = bar.bar_rect()
    x = round(rect.center().x())
    assert similar(image.pixelColor(x, round(rect.top()) + 3), QColor(top))
    assert similar(image.pixelColor(x, round(rect.bottom()) - 3), QColor(bottom))
    assert similar(image.pixelColor(x, round(bar.y_for(0.6))), QColor(0, 0, 0))


@pytest.mark.parametrize("style", SCALE_STYLES)
def test_apply_theme(make_plot, style):
    plot = make_plot(style)
    plot.set_map(ramp_map())
    assert plot.view.backgroundBrush().color() == QColor(0, 0, 0)
    plot.apply_theme(LIGHT)
    assert plot.colors() == LIGHT
    assert plot.view.backgroundBrush().color() == QColor("#ffffff")
    assert plot.plot.vb.background.brush().color() == QColor("#e9e6ed")
    assert plot.plot.getAxis("left").textPen().color() == QColor("#3b3443")
    assert plot._vline.pen.color() == qcolor("#3b344366")
    corner = plot.scale_container.grab().toImage().pixelColor(0, 0)
    assert corner == QColor("#ffffff")
    scale_image = plot.scale.widget.grab().toImage()
    assert scale_image.pixelColor(scale_image.width() - 1, 0) == QColor("#ffffff")


def test_stacked_apply_theme(qtbot):
    stacked = StackedPlot()
    qtbot.addWidget(stacked)
    stacked.apply_theme(LIGHT)
    assert stacked.view.backgroundBrush().color() == QColor("#ffffff")
    assert stacked.plot.getAxis("bottom").pen().color() == QColor("#3b3443")


def test_overlay_layers(make_plot):
    plot = make_plot()
    plot.set_map(ramp_map())
    names = [layer.name for layer in plot.layers()]
    assert names == ["points", "models"]
    preview = plot.layer("preview")
    assert plot.layer("preview") is preview
    assert [layer.name for layer in plot.layers()] == ["points", "models", "preview"]

    far = np.array([100.0, 200.0])
    preview.set_curves([(far, far)], pg.mkPen("y"), shadow_pen=pg.mkPen("k", width=3))
    preview.set_points(far, far, symbol="t", size=6, pen=pg.mkPen("g"))
    plot.set_model_curves(np.array([0.0, 1.0]), np.array([[150.0, 200.0]]))
    plot.set_points((np.array([1.0]), np.array([300.0])), [(far, far, "c")])
    z = [min(i.zValue() for i in layer.items()) for layer in plot.layers()]
    assert z == sorted(z)
    for layer in plot.layers():
        for item in layer.items():
            assert item not in plot.plot.vb.addedItems  # never auto-ranged
    assert plot.plot.vb.childrenBounds()[0][1] < 100  # bounds come from the image only

    assert len(preview.curve_data()) == 1
    assert len(preview.point_data()) == 1
    preview.set_visible(False)
    assert not preview.is_visible()
    assert preview.curve_data() == []
    assert not any(item.isVisible() for item in preview.items())
    preview.set_visible(True)
    assert len(preview.curve_data()) == 1
    preview.clear()
    assert preview.curve_data() == []
    assert preview.point_data() == []

    points = plot.layer("points").point_data()
    assert len(points) == 2  # others first, then the current curve on top
    np.testing.assert_allclose(points[1][1], [300.0])
    plot.set_points(None)
    assert plot.layer("points").point_data() == []
    assert len(plot.model_curve_data()) == 1
    plot.set_model_curves(np.array([]), None)
    assert plot.model_curve_data() == []


def test_cursor_moved(make_plot, qtbot):
    plot = make_plot()
    fmap = ramp_map()
    plot.set_map(fmap)
    b, e = fmap.field[3], fmap.energy[5]
    pos = plot.plot.vb.mapViewToScene(QPointF(b, e))
    with qtbot.waitSignal(plot.cursorMoved) as blocker:
        plot._on_mouse_moved((pos,))
    x, y, value = blocker.args
    assert (x, y) == pytest.approx((b, e), abs=1e-6)
    assert value == pytest.approx(fmap.values[5, 3])
    assert "value =" in plot.label.text

    plot.set_map(fmap, x_range=(-5, 5))
    pos = plot.plot.vb.mapViewToScene(QPointF(-4, e))
    with qtbot.waitSignal(plot.cursorMoved) as blocker:
        plot._on_mouse_moved((pos,))
    assert blocker.args[2] is None


def stacked_with(qtbot, fmap, offset=1.0) -> StackedPlot:
    stacked = StackedPlot()
    qtbot.addWidget(stacked)
    stacked.set_map(fmap, offset)
    return stacked


def test_stacked_defaults_unchanged(qtbot):
    fmap = ramp_map(n_field=5)
    stacked = stacked_with(qtbot, fmap, 0.5)
    assert len(stacked._curves) == 5
    for j, curve in enumerate(stacked._curves):
        assert pg.mkPen(curve.opts["pen"]).color() == pg.intColor(j, hues=5)
        np.testing.assert_allclose(curve.getData()[1], fmap.values[:, j] + 0.5 * j)


def test_trace_options_and_trace_y(qtbot):
    fmap = ramp_map(n_field=6)
    stacked = stacked_with(qtbot, fmap, 1.0)
    e = 0.5 * (fmap.energy[3] + fmap.energy[4])
    expected = 0.5 * (fmap.values[3, 2] + fmap.values[4, 2]) + 2.0
    assert stacked.trace_y(2, e) == pytest.approx(expected)
    assert stacked.trace_y(2, fmap.energy[0] - 1) is None
    assert stacked.trace_y(9, e) is None

    stacked.set_trace_options(every=2)
    assert len(stacked._curves) == 3
    np.testing.assert_array_equal(stacked.shown_fields(), [0, 2, 4])
    assert stacked.trace_y(1, e) is None
    assert stacked.trace_y(2, e) == pytest.approx(expected - 1.0)  # second trace shown

    stacked.set_trace_options(every=1, color_by_field=True, cmap="grey")
    table = lut("grey")
    first = pg.mkPen(stacked._curves[0].opts["pen"]).color()
    last = pg.mkPen(stacked._curves[-1].opts["pen"]).color()
    assert first.red() == table[round(0.1 * 255)][0]
    assert last.red() == table[round(0.82 * 255)][0]
    with pytest.raises(ValueError):
        stacked.set_trace_options(every=0)


def test_trace_at(qtbot):
    fmap = ramp_map(n_field=6)
    stacked = stacked_with(qtbot, fmap, 1.0)
    e = fmap.energy[10]
    for j in range(6):
        assert stacked.trace_at(e, stacked.trace_y(j, e) + 0.2) == j
    stacked.set_trace_options(every=3)
    assert stacked.trace_at(e, stacked.trace_y(3, e) - 0.1) == 3
    assert stacked.trace_at(fmap.energy[-1] + 10, 0.0) is None
    stacked.clear_map()
    assert stacked.trace_at(e, 0.0) is None
    assert stacked.trace_y(0, e) is None


def test_trace_y_with_descending_energy_and_nan(qtbot):
    fmap = ramp_map(n_field=3)
    values = fmap.values[::-1].copy()
    values[5, 1] = np.nan
    flipped = FieldMap(fmap.energy[::-1], fmap.field, values)
    stacked = stacked_with(qtbot, flipped, 0.0)
    assert stacked.trace_y(0, fmap.energy[7]) == pytest.approx(fmap.values[7, 0])
    assert stacked.trace_y(1, flipped.energy[5]) is None
    assert stacked.trace_at(flipped.energy[5], fmap.values[-6, 1]) in (0, 2)


@pytest.mark.parametrize("descending", [False, True])
def test_trace_y_interpolates_between_the_bracketing_energies(qtbot, descending):
    energy = np.linspace(100.0, 500.0, 40)
    values = np.sin(np.add.outer(energy / 37.0, np.arange(4.0)))
    fmap = FieldMap(energy, np.arange(4.0), values)
    if descending:
        fmap = FieldMap(energy[::-1], fmap.field, values[::-1].copy())
    stacked = stacked_with(qtbot, fmap, 1.0)
    stacked.set_trace_options(every=2)  # traces 0 and 2
    e = 0.75 * energy[3] + 0.25 * energy[4]
    assert stacked.trace_y(2, e) == pytest.approx(0.75 * values[3, 2] + 0.25 * values[4, 2] + 1)
    for k in (0, -1):  # the ends of the axis
        assert stacked.trace_y(0, energy[k]) == pytest.approx(values[k, 0])
        assert stacked.trace_y(2, energy[k]) == pytest.approx(values[k, 2] + 1)


@pytest.mark.parametrize("style", SCALE_STYLES)
def test_export_png_contains_the_scale(make_plot, tmp_path, style):
    plot = make_plot(style)
    plot.set_map(ramp_map())
    plot_width = plot.view.ci.sceneBoundingRect().width()
    scale_width = plot.scale.widget.width()
    out = tmp_path / "map.png"
    plot.export_image(out, scale=2.0)
    image = QImage(str(out))
    assert image.width() == int((plot_width + scale_width) * 2)
    assert image.width() > plot_width * 2
    assert plot._vline.isVisible()

    plot.set_scale_visible(False)
    assert not plot.scale_visible()
    plot.export_image(out, scale=2.0)
    assert QImage(str(out)).width() == int(plot_width * 2)


@pytest.mark.parametrize("style", SCALE_STYLES)
def test_export_svg_contains_the_scale(make_plot, tmp_path, style):
    plot = make_plot(style)
    plot.set_map(ramp_map(), cmap="bipolar")
    plot._on_mouse_moved((plot.plot.vb.sceneBoundingRect().center(),))
    label = plot.label.text
    out = tmp_path / "map.svg"
    plot.export_image(out)
    box = QSvgRenderer(str(out)).viewBoxF()
    expected = plot.view.ci.sceneBoundingRect().width() + plot.scale.widget.width()
    assert box.width() == pytest.approx(expected, abs=1)
    assert box.width() > plot.view.ci.sceneBoundingRect().width()
    assert label and plot.label.text == label  # restored after the export
    assert "clip-path" in out.read_text()  # SVG 1.1 keeps the image inside the axes


def test_export_rejects_unknown_suffix(make_plot, tmp_path):
    with pytest.raises(ValueError, match="unsupported"):
        make_plot().export_image(tmp_path / "map.pdf")
