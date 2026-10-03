"""The View, Colour and Traces sections of the inspector, and how they follow the plots."""

import json

import numpy as np
import pytest
from PySide6.QtCore import QPoint, QPointF, QSettings, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtWidgets import QApplication

import gui_helpers
from gui_helpers import inspector_page, load_sweep, process, select, set_unit
from mag_opt_detective.core.colormaps import lut
from mag_opt_detective.core.pipeline import PlotKind, ReferenceMode
from mag_opt_detective.core.processing import Axis
from mag_opt_detective.core.units import Unit
from mag_opt_detective.gui import plot_panel
from mag_opt_detective.gui.controller import level_key
from mag_opt_detective.gui.main_window import MainWindow
from mag_opt_detective.gui.plots import BarScale, robust_levels

window, errors = gui_helpers.window, gui_helpers.errors  # shared fixtures

MEV = 8.0656
THZ = 33.35641


@pytest.fixture
def shown(window, qtbot):
    window.resize(1400, 900)
    window.show()
    qtbot.waitExposed(window)
    return window


@pytest.fixture
def processed(window, sweep):
    load_sweep(window, sweep)
    process(window)
    return window


def view_page(window):
    return inspector_page(window, "view")


def colour_page(window):
    return inspector_page(window, "colour")


def traces_page(window):
    return inspector_page(window, "traces")


def shown_range(window, view: str):
    """(x, y) range of plot *view* as drawn."""
    return [tuple(axis) for axis in window.plots[view].plot.vb.viewRange()]


def pan(window, view: str, dx: float = 0.0, dy: float = 0.0, scale=None) -> None:
    """Move plot *view* as pyqtgraph does on a drag (or zoom with *scale*), then report it."""
    vb = window.plots[view].plot.vb
    if scale is not None:
        vb.scaleBy(x=scale[0], y=scale[1])
    else:
        vb.translateBy(x=dx or None, y=dy or None)
    vb.sigRangeChangedManually.emit(vb.state["mouseEnabled"])


def map_values(window) -> np.ndarray:
    return window.controller.current_map().values


def wheel(window, view: str) -> None:
    """One wheel tick (zoom in) over the middle of plot *view*, as Qt delivers it."""
    plot = window.plots[view]
    centre = plot.view.mapFromScene(plot.plot.vb.sceneBoundingRect().center())
    viewport = plot.view.viewport()
    event = QWheelEvent(
        QPointF(centre), QPointF(viewport.mapToGlobal(centre)), QPoint(0, 0), QPoint(0, 120),
        Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.NoScrollPhase,
        False,
    )  # fmt: skip
    QApplication.sendEvent(viewport, event)


# ---------------------------------------------------------------------- view ranges
def test_ranges_start_on_auto_and_fit_the_data(processed, sweep, errors):
    w = processed
    page = view_page(w)
    assert all(control.is_auto() for control in page.controls().values())
    fields, energy = sweep["fields"], sweep["x"]
    assert shown_range(w, "map") == [(fields[0], fields[-1]), (energy[0], energy[-1])]
    assert page.field.range() == (fields[0], fields[-1])
    assert page.field.note.text() == "Data 0.5 – 2 T"
    assert page.energy.note.text() == "Data 100 – 1000 cm⁻¹ · shared with Stacked"
    stacked = shown_range(w, "stacked")
    assert stacked[0] == (energy[0], energy[-1])
    values = map_values(w) + 0.01 * np.arange(fields.size)
    span = values.max() - values.min()
    assert stacked[1] == pytest.approx((values.min() - 0.04 * span, values.max() + 0.04 * span))
    assert not errors


def test_range_controls_set_the_plot_ranges(processed):
    w, c = processed, processed.controller
    page = view_page(w)
    page.field.lo_spin.setValue(1.0)
    assert c.view.field_range == (1.0, 2.0) and not page.field.is_auto()
    assert shown_range(w, "map")[0] == pytest.approx((1.0, 2.0))
    page.energy.slider.valuesChanged.emit(300.0, 700.0)  # a slider drag
    assert c.view.energy_range == (300.0, 700.0)
    assert shown_range(w, "map")[1] == pytest.approx((300.0, 700.0))
    assert shown_range(w, "stacked")[0] == pytest.approx((300.0, 700.0))  # shared
    assert shown_range(w, "map")[0] == pytest.approx((1.0, 2.0))  # untouched
    page.field.mode.set_value("auto")
    assert c.view.field_range is None and page.field.is_auto()
    assert shown_range(w, "map")[0] == pytest.approx((0.5, 2.0))
    page.energy.lo_spin.setValue(800.0)  # above the upper limit: refused inline
    assert page.energy.error() and c.view.energy_range == (300.0, 700.0)


def test_pan_and_zoom_fix_the_axes_they_move(processed):
    w, c = processed, processed.controller
    page = view_page(w)
    pan(w, "map", dy=50.0)  # drag along the energy axis only
    assert c.view.field_range is None and page.field.is_auto()
    assert c.view.energy_range == pytest.approx((150.0, 1050.0))
    assert not page.energy.is_auto() and page.energy.range() == pytest.approx((150.0, 1050.0))
    assert shown_range(w, "stacked")[0] == pytest.approx((150.0, 1050.0))
    pan(w, "map", scale=(0.5, None))  # zoom the field axis
    assert c.view.field_range == pytest.approx((0.875, 1.625))
    assert page.field.range() == pytest.approx((0.875, 1.625))
    assert not page.field.is_auto()

    pan(w, "stacked", dx=-50.0)  # the stacked plot moves the shared energy range
    assert c.view.energy_range == pytest.approx((100.0, 1000.0))
    assert shown_range(w, "map")[1] == pytest.approx((100.0, 1000.0))
    assert c.view.stacked_range is None  # its intensity axis did not move
    pan(w, "stacked", dy=0.1)
    lo, hi = c.view.stacked_range
    w.plot_area.set_current_view("stacked")
    assert not page.intensity.is_auto() and page.intensity.range() == pytest.approx((lo, hi))


def test_auto_and_fit_go_back_to_the_data(processed, sweep):
    w, c = processed, processed.controller
    page = view_page(w)
    pan(w, "map", dx=0.2, dy=100.0)
    page.energy.mode.set_value("auto")
    assert c.view.energy_range is None and c.view.field_range is not None
    assert shown_range(w, "map")[1] == (100.0, 1000.0)
    pan(w, "map", scale=(0.5, 0.5))
    plot_panel.fit_to_data(w)  # the Fit to data tool (A) on the map
    assert (c.view.field_range, c.view.energy_range) == (None, None)
    assert shown_range(w, "map") == [(0.5, 2.0), (100.0, 1000.0)]
    assert page.field.is_auto() and page.energy.is_auto()
    pan(w, "stacked", dx=10.0, dy=0.5)
    w.plot_area.set_current_view("stacked")
    plot_panel.fit_to_data(w)
    assert c.view.energy_range is None and c.view.stacked_range is None
    w.plots.stacked.plot.vb.menu.viewAll.trigger()  # the plot menu's View All fits as well
    assert c.view.stacked_range is None


class Click:
    """A mouse click as pyqtgraph's scene reports it (sigMouseClicked)."""

    def __init__(self, pos: QPointF, double: bool = True):
        self._pos, self._double = pos, double

    def double(self) -> bool:
        return self._double

    def button(self):
        return Qt.MouseButton.LeftButton

    def scenePos(self) -> QPointF:
        return self._pos

    def modifiers(self):
        return Qt.KeyboardModifier.NoModifier


def test_wheel_and_double_click_on_the_plot(shown, sweep):
    w, c = shown, shown.controller
    load_sweep(w, sweep)
    process(w)
    plot = w.plots.map
    vb = plot.plot.vb
    wheel(w, "map")
    assert c.view.field_range is not None and c.view.energy_range is not None
    lo, hi = c.view.energy_range
    assert 100.0 < lo < hi < 1000.0
    assert not view_page(w).energy.is_auto()
    plot.plot.scene().sigMouseClicked.emit(Click(vb.sceneBoundingRect().center(), double=False))
    assert c.view.energy_range is not None  # a single click does not fit
    plot.plot.scene().sigMouseClicked.emit(Click(QPointF(1, 1)))  # outside the data area
    assert c.view.energy_range is not None
    plot.plot.scene().sigMouseClicked.emit(Click(vb.sceneBoundingRect().center()))
    assert c.view.field_range is None and c.view.energy_range is None
    assert shown_range(w, "map") == [(0.5, 2.0), (100.0, 1000.0)]


def test_double_click_fits_only_while_panning_or_zooming(processed):
    w, c = processed, processed.controller
    vb = w.plots.map.plot.vb
    w.tools.set_active("pick")
    c.set_ranges(field_range=(1.0, 1.5))
    w.plots.map.plot.scene().sigMouseClicked.emit(Click(vb.sceneBoundingRect().center()))
    assert c.view.field_range == (1.0, 1.5)  # the clicks belong to Pick
    for tool in ("zoom", "navigate"):
        w.tools.set_active(tool)
        c.set_ranges(field_range=(1.0, 1.5))
        w.plots.map.plot.scene().sigMouseClicked.emit(Click(vb.sceneBoundingRect().center()))
        assert c.view.field_range is None


def test_pan_or_zoom_on_an_empty_plot_keeps_the_ranges(shown, sweep, qtbot):
    w, c = shown, shown.controller
    pan(w, "map", dx=0.3, dy=0.2)  # nothing processed yet
    w.plots.map.plot.scene().sigMouseClicked.emit(Click(QPointF(200, 200)))
    assert (c.view.field_range, c.view.energy_range) == (None, None)
    load_sweep(w, sweep)
    process(w)
    assert shown_range(w, "map") == [(0.5, 2.0), (100.0, 1000.0)]

    w.plot_area.set_current_view("reference")  # no reference mode: an empty plot
    qtbot.waitExposed(w.plots.reference.view)
    wheel(w, "reference")
    pan(w, "reference", dx=5.0, dy=-3.0)
    assert (c.view.field_range, c.view.energy_range) == (None, None)
    assert view_page(w).field.is_auto() and view_page(w).energy.is_auto()
    w.plot_area.set_current_view("map")
    assert shown_range(w, "map") == [(0.5, 2.0), (100.0, 1000.0)]
    process(w)
    assert shown_range(w, "map") == [(0.5, 2.0), (100.0, 1000.0)]


def test_manual_ranges_survive_redraws(processed, sweep):
    w, c = processed, processed.controller
    pan(w, "map", dx=0.25, dy=-40.0)
    field, energy = c.view.field_range, c.view.energy_range
    w.plots.map.hist.region.setRegion((0.9, 1.05))  # a level drag redraws the map
    assert shown_range(w, "map") == [pytest.approx(field), pytest.approx(energy)]
    select(w, kind=PlotKind.DATA, order=1)
    assert shown_range(w, "map") == [pytest.approx(field), pytest.approx(energy)]
    process(w)  # a new result keeps a fixed range
    assert shown_range(w, "map") == [pytest.approx(field), pytest.approx(energy)]
    set_unit(w, "meV")
    assert shown_range(w, "map")[0] == pytest.approx(field)
    assert shown_range(w, "map")[1] == pytest.approx(tuple(e / MEV for e in energy))
    assert shown_range(w, "stacked")[0] == pytest.approx(tuple(e / MEV for e in energy))


def test_unit_switch_converts_ranges_but_not_the_data(processed):
    w, c = processed, processed.controller
    result = c.result
    page = view_page(w)
    c.set_ranges(field_range=(1.0, 1.5), energy_range=(200.0, 800.0), stacked_range=(0.5, 2.0))
    set_unit(w, "THz")
    assert c.result is result
    assert c.view.field_range == (1.0, 1.5)
    assert c.view.energy_range == pytest.approx((200 / THZ, 800 / THZ))
    assert c.view.stacked_range == (0.5, 2.0)  # a ratio does not depend on the unit
    assert page.energy.range() == pytest.approx((200 / THZ, 800 / THZ))
    assert page.energy.unit() == "THz"
    assert page.energy.note.text() == "Data 2.9979 – 29.979 THz · shared with Stacked"
    set_unit(w, "cm-1")
    assert c.view.energy_range == pytest.approx((200.0, 800.0))

    select(w, order=1, per_unit=True)  # a derivative per cm-1: intensities scale with the unit
    c.set_ranges(stacked_range=(-0.01, 0.01))
    c.set_view(stacked_offset=0.002)
    set_unit(w, "meV")
    assert c.view.stacked_range == pytest.approx((-0.01 * MEV, 0.01 * MEV))
    assert c.view.stacked_offset == pytest.approx(0.002 * MEV)
    assert shown_range(w, "stacked")[1] == pytest.approx((-0.01 * MEV, 0.01 * MEV))
    select(w, axis="B")  # d/dB per T does not
    set_unit(w, "THz")
    assert c.view.stacked_range == pytest.approx((-0.01 * MEV, 0.01 * MEV))


def test_sections_follow_the_plot_on_screen(window):
    page = view_page(window)
    visible = {
        "map": ["view", "colour", "overlays"],
        "stacked": ["view", "traces"],
        "reference": ["view", "colour"],
    }
    for view, names in visible.items():
        window.plot_area.set_current_view(view)
        assert [n for n, s in window.inspector.items() if not s.isHidden()] == names
        assert page.field.isHidden() == (view == "stacked")
        assert page.intensity.isHidden() == (view != "stacked")


# ---------------------------------------------------------------------- colour
def test_colour_map_swatches(processed):
    w, c = processed, processed.controller
    picker = colour_page(w).picker
    assert picker.value() == "Auto" and picker.swatches["Auto"].isChecked()
    assert w.plots.map.colormap() == "magma"  # Auto: magma for maps ...
    select(w, order=1)
    assert w.plots.map.colormap() == "grey"  # ... and grey for derivatives
    picker.swatches["inferno"].click()
    assert c.view.colormap == "inferno"
    assert w.plots.map.colormap() == "inferno"
    select(w, order=0)
    assert w.plots.map.colormap() == "inferno"
    c.set_view(colormap="turbo")  # the swatches follow the controller
    assert picker.value() == "turbo" and not picker.swatches["inferno"].isChecked()
    picker.swatches["Auto"].click()
    assert c.view.colormap == "Auto" and w.plots.map.colormap() == "magma"


def test_level_modes(processed):
    w, c = processed, processed.controller
    page = colour_page(w)
    plot = w.plots.map
    assert page.mode.value() == "fixed" and plot.levels() == (0.9, 1.1)  # ratio default
    assert page.fields.levels() == (0.9, 1.1)
    page.mode.button("auto").click()
    auto = robust_levels(map_values(w))
    assert c.current_levels() is None and plot.levels() == pytest.approx(auto)
    assert page.fields.levels() == pytest.approx(auto)
    assert "1st to 99th percentile" in page.fields.note.text()
    page.mode.button("fixed").click()  # keeps what Auto showed
    assert c.view.levels["Ratio"] == pytest.approx(auto) and plot.levels() == pytest.approx(auto)
    page.mode.button("sym").click()  # centred on 1 for ratios
    lo, hi = c.view.levels["Ratio"]
    assert (lo + hi) / 2 == pytest.approx(1.0)
    assert hi - 1 == pytest.approx(max(abs(auto[0] - 1), abs(auto[1] - 1)))
    assert plot.levels() == pytest.approx((lo, hi))
    assert page.fields.note.text() == "Centred on 1. Remembered for R(B)/R(0)."
    assert page.mode.button("sym").isEnabled()

    select(w, kind=PlotKind.DATA)  # raw data: no centre, so no symmetric levels
    assert not page.mode.button("sym").isEnabled() and page.mode.value() == "auto"
    assert plot.levels() == pytest.approx(robust_levels(map_values(w)))

    select(w, order=2)  # derivatives start symmetric about 0
    assert page.mode.value() == "sym" and page.mode.button("sym").isEnabled()
    lo, hi = plot.levels()
    assert lo == pytest.approx(-hi) and hi > 0
    assert page.fields.note.text().startswith("Centred on 0.")


def test_symmetric_levels_stay_symmetric_when_one_end_is_dragged(processed):
    w, c = processed, processed.controller
    select(w, order=1)
    key = c.selection.level_key
    lo, hi = w.plots.map.levels()
    w.plots.map.hist.region.setRegion((lo, hi * 2))  # the upper end, on the plot's scale
    assert c.view.level_mode(key) == "sym"
    assert c.view.levels[key] == pytest.approx((-2 * hi, 2 * hi))
    assert w.plots.map.levels() == pytest.approx((-2 * hi, 2 * hi))
    w.plots.map.hist.region.setRegion((-hi, 3 * hi))  # both ends: a shift fixes them
    assert c.view.level_mode(key) == "fixed"
    assert c.view.levels[key] == pytest.approx((-hi, 3 * hi))
    colour_page(w).fields.lo.setValue(-hi)  # typed levels are fixed too
    assert c.view.level_mode(key) == "fixed"


def test_levels_are_remembered_per_kind_order_axis_and_unit(processed):
    w, c = processed, processed.controller
    page = colour_page(w)
    keys = {}
    for kind, order, axis, per_unit, levels in (
        (PlotKind.RATIO, 1, "E", False, (-0.1, 0.2)),
        (PlotKind.RATIO, 1, "B", False, (-0.3, 0.4)),
        (PlotKind.RATIO, 1, "E", True, (-0.5, 0.6)),
        (PlotKind.DATA, 1, "E", False, (-0.7, 0.8)),
        (PlotKind.RATIO, 2, "E", False, (-0.9, 1.0)),
    ):
        select(w, kind=kind, order=order, axis=axis, per_unit=per_unit)
        page.fields.lo.setValue(levels[0])
        page.fields.hi.setValue(levels[1])
        keys[c.selection.level_key] = levels
    assert len(keys) == 5
    assert page.label.text() == "R(B)/R(0) · 2nd derivative d/dE per point"
    assert page.fields.note.text().startswith("Remembered for R(B)/R(0) · 2nd derivative")
    for key, levels in keys.items():
        k = c.view.levels[key]
        assert k == pytest.approx(levels) and c.view.level_mode(key) == "fixed"
    select(w, kind=PlotKind.RATIO, order=1, axis="B", per_unit=False)
    assert w.plots.map.levels() == pytest.approx((-0.3, 0.4))
    assert page.fields.levels() == pytest.approx((-0.3, 0.4))
    select(w, per_unit=True)
    assert page.label.text() == "R(B)/R(0) · 1st derivative d/dB per T"
    assert c.view.level_mode(c.selection.level_key) == "sym"  # not set yet: the default


def test_unit_switch_converts_only_per_unit_energy_levels(processed):
    w, c = processed, processed.controller
    page = colour_page(w)
    select(w, order=1, per_unit=True)
    page.fields.lo.setValue(-0.002)
    page.fields.hi.setValue(0.003)
    c.set_levels(level_key(PlotKind.RATIO, 1, Axis.FIELD, True), -0.1, 0.1)
    c.set_levels(level_key(PlotKind.RATIO, 1), -0.01, 0.02)
    set_unit(w, "meV")
    assert c.view.levels["Ratio_der1_E_unit"] == pytest.approx((-0.002 * MEV, 0.003 * MEV))
    assert page.fields.levels() == pytest.approx((-0.002 * MEV, 0.003 * MEV))
    assert w.plots.map.levels() == pytest.approx((-0.002 * MEV, 0.003 * MEV))
    assert c.view.levels["Ratio_der1_B_unit"] == (-0.1, 0.1)
    assert c.view.levels["Ratio_der1_E"] == (-0.01, 0.02)
    assert c.view.levels["Ratio"] == (0.9, 1.1)


def test_inspector_histogram_and_the_plot_scale_stay_in_step(processed):
    w, c = processed, processed.controller
    page = colour_page(w)
    hist = page.histogram
    style = w.plot_area.scale_style_button
    assert not page.hist_section.is_expanded()  # the classic histogram is on the plot
    style.setChecked(True)
    assert page.hist_section.is_expanded()  # the slim bar: the inspector shows the histogram
    assert hist.region.getRegion() == pytest.approx((0.9, 1.1))
    strip = hist.strip.rect()
    assert strip.left() < 0.9 < 1.1 < strip.right()  # the histogram spans more than the levels
    stops = hist.strip.brush().gradient()
    assert (stops.start().x(), stops.finalStop().x()) == pytest.approx((0.9, 1.1))
    style.setChecked(False)
    assert not page.hist_section.is_expanded()

    for style_bar in (False, True):
        style.setChecked(style_bar)
        page.hist_section.set_expanded(True, animate=False)
        hist.region.setRegion((0.92, 1.04))  # dragged in the inspector
        assert c.view.levels["Ratio"] == pytest.approx((0.92, 1.04))
        assert w.plots.map.levels() == pytest.approx((0.92, 1.04))
        assert w.plots.map.scale.levels() == pytest.approx((0.92, 1.04))
        assert page.fields.levels() == pytest.approx((0.92, 1.04))
        scale = w.plots.map.scale
        if isinstance(scale, BarScale):  # dragged on the plot's slim bar
            scale.bar.set_levels(0.95, 1.08)
            scale.bar.levelsChosen.emit(0.95, 1.08)
        else:
            w.plots.map.hist.region.setRegion((0.95, 1.08))
        assert c.view.levels["Ratio"] == pytest.approx((0.95, 1.08))
        assert hist.region.getRegion() == pytest.approx((0.95, 1.08))
        assert page.fields.levels() == pytest.approx((0.95, 1.08))
        c.set_levels("Ratio", 0.9, 1.1)


def test_histogram_drag_recolours_the_map_live(processed):
    w, c = processed, processed.controller
    page = colour_page(w)
    page.hist_section.set_expanded(True, animate=False)
    hist = page.histogram
    hist.region.lines[1].setValue(1.2)  # moving, not let go yet
    assert w.plots.map.levels() == pytest.approx((0.9, 1.2))
    assert page.fields.levels() == pytest.approx((0.9, 1.2))
    assert c.view.levels["Ratio"] == (0.9, 1.1)
    hist.region.lineMoveFinished()
    assert c.view.levels["Ratio"] == pytest.approx((0.9, 1.2))

    select(w, order=1)  # symmetric: the other end follows
    lo, _hi = hist.region.getRegion()
    hist.region.lines[0].setValue(lo * 3)
    assert hist.region.getRegion() == pytest.approx((3 * lo, -3 * lo))
    hist.region.lineMoveFinished()
    assert c.view.levels[c.selection.level_key] == pytest.approx((3 * lo, -3 * lo))
    assert c.view.level_mode(c.selection.level_key) == "sym"
    assert hist.centre.isVisible()


def test_reference_tab_edits_the_reference_levels(window, sweep, errors):
    c = window.controller
    load_sweep(window, sweep)
    c.set_processing(reference_mode=ReferenceMode.SELF)
    process(window)
    window.plot_area.set_current_view("reference")
    page = colour_page(window)
    window.plot_area.ref_data.setChecked(True)  # reference data: no symmetric levels
    assert page.label.text() == "Data"
    assert not page.mode.button("sym").isEnabled()
    page.mode.button("fixed").click()
    page.fields.hi.setValue(5.0)
    assert c.view.levels["Data"][1] == 5.0
    assert window.plots.reference.levels()[1] == 5.0
    pan(window, "reference", dx=0.5)  # the reference shares the field and energy ranges
    assert c.view.field_range == pytest.approx((1.0, 2.5))
    assert shown_range(window, "map")[0] == pytest.approx((1.0, 2.5))
    assert not errors


def test_typed_levels_are_checked(processed, qtbot):
    w, c = processed, processed.controller
    fields = colour_page(w).fields
    fields.lo.setValue(1.5)  # above the upper level
    assert fields.error() and fields.note.text() == fields.error()
    assert fields.lo.property("invalid") is True
    assert c.view.levels["Ratio"] == (0.9, 1.1)
    fields.lo.setValue(0.8)
    assert not fields.error() and c.view.levels["Ratio"] == (0.8, 1.1)
    fields.lo.selectAll()
    qtbot.keyClicks(fields.lo, "2.5e-3x")  # scientific notation; the letter is refused
    qtbot.keyClick(fields.lo, Qt.Key.Key_Return)
    assert c.view.levels["Ratio"] == (0.0025, 1.1)
    assert fields.lo.text() == "0.0025"
    fields.lo.setValue(1.25e-7)  # tiny levels keep their digits
    assert fields.lo.value() == 1.25e-7 and fields.lo.text() == "1.25e-07"


# ---------------------------------------------------------------------- traces
def test_traces_options(processed, sweep, qtbot):
    w, c = processed, processed.controller
    page = traces_page(w)
    stacked = w.plots.stacked
    page.offset.spin.setValue(0.05)
    assert c.view.stacked_offset == 0.05
    first, second = stacked.trace_y(0, 500.0), stacked.trace_y(1, 500.0)
    values = map_values(w)
    row = int(np.argmin(np.abs(sweep["x"] - 500.0)))
    assert second - first == pytest.approx(values[row, 1] - values[row, 0] + 0.05)
    page.offset.slider.setValue(page.offset.slider.maximum())  # live, merged
    qtbot.waitUntil(lambda: c.view.stacked_offset == pytest.approx(page.offset.maximum()))
    assert page.offset.spin.value() == pytest.approx(page.offset.maximum())

    page.every.button("2").click()
    assert c.view.stacked_every == 2
    assert list(stacked.shown_fields()) == [0, 2]
    assert c.view.stacked_by_field and page.by_field.isChecked()  # coloured by field
    pen = stacked._curves[0].opts["pen"]
    expected = lut("viridis")[round(0.1 * 255)]  # Auto colours the fields with viridis
    assert pen.color().getRgb()[:3] == tuple(int(v) for v in expected)
    assert page.legend.labels() == ("0.5 T", "2 T") and not page.legend.isHidden()
    colour_page(w).picker.swatches["plasma"].click()  # the selected map colours the traces
    pen = stacked._curves[0].opts["pen"]
    assert pen.color().getRgb()[:3] == tuple(int(v) for v in lut("plasma")[round(0.1 * 255)])
    page.by_field.click()
    assert not c.view.stacked_by_field and page.legend.isHidden()


def test_intensity_auto_follows_the_offset(processed):
    w = processed
    traces_page(w).offset.spin.setValue(0.5)
    values = map_values(w) + 0.5 * np.arange(4)
    pad = 0.04 * (values.max() - values.min())
    expected = (values.min() - pad, values.max() + pad)
    assert shown_range(w, "stacked")[1] == pytest.approx(expected)
    w.plot_area.set_current_view("stacked")
    assert view_page(w).intensity.range() == pytest.approx(expected)
    assert view_page(w).intensity.is_auto()
    traces_page(w).every.button("2").click()  # every 2nd: two traces, one offset apart
    values = map_values(w)[:, ::2] + 0.5 * np.arange(2)
    pad = 0.04 * (values.max() - values.min())
    assert shown_range(w, "stacked")[1] == pytest.approx((values.min() - pad, values.max() + pad))
    c = w.controller
    c.set_ranges(energy_range=(100.0, 300.0))  # only the energies shown count
    rows = w.controller.current_map().energy <= 300.0
    values = map_values(w)[rows][:, ::2] + 0.5 * np.arange(2)
    pad = 0.04 * (values.max() - values.min())
    assert shown_range(w, "stacked")[1] == pytest.approx((values.min() - pad, values.max() + pad))


# ---------------------------------------------------------------------- settings
def test_settings_round_trip(qtbot, tmp_path, sweep):
    ini = str(tmp_path / "settings.ini")
    w = MainWindow(settings=QSettings(ini, QSettings.Format.IniFormat))
    qtbot.addWidget(w)
    c = w.controller
    load_sweep(w, sweep)
    process(w)
    set_unit(w, "meV")
    select(w, order=1, per_unit=True)
    c.set_ranges(field_range=(1.0, 1.5), energy_range=(20.0, 80.0), stacked_range=(-0.2, 0.3))
    colour_page(w).fields.lo.setValue(-0.04)
    colour_page(w).picker.swatches["viridis"].click()
    traces = traces_page(w)
    traces.offset.spin.setValue(0.016)
    traces.every.button("4").click()
    traces.by_field.click()
    w.close()
    stored = QSettings(ini, QSettings.Format.IniFormat)
    ranges = json.loads(stored.value("v2/view/ranges"))
    assert ranges["energy_cm1"] == pytest.approx([20 * MEV, 80 * MEV])
    assert ranges["intensity"] == pytest.approx([-0.2 / MEV, 0.3 / MEV])  # per cm-1
    levels = json.loads(stored.value("v2/view/levels"))
    key = "Ratio_der1_E_unit"
    assert levels[key]["mode"] == "fixed" and levels[key]["levels"][0] == pytest.approx(-0.04 / MEV)
    assert json.loads(stored.value("v2/view/traces"))["offset"] == pytest.approx(0.016 / MEV)

    for unit in ("meV", "THz"):  # restored in the unit saved, or after a live switch
        w2 = MainWindow(settings=QSettings(ini, QSettings.Format.IniFormat))
        qtbot.addWidget(w2)
        c2 = w2.controller
        if unit == "THz":
            set_unit(w2, "THz")
            w2.restore_settings()
        v = c2.view
        assert c2.unit is Unit.MEV and c2.selection.level_key == key
        assert v.field_range == (1.0, 1.5)
        assert v.energy_range == pytest.approx((20.0, 80.0))
        assert v.stacked_range == pytest.approx((-0.2, 0.3))
        assert v.levels[key][0] == pytest.approx(-0.04) and v.level_mode(key) == "fixed"
        assert v.colormap == "viridis" and colour_page(w2).picker.value() == "viridis"
        assert v.stacked_offset == pytest.approx(0.016)
        assert (v.stacked_every, v.stacked_by_field) == (4, False)
        assert traces_page(w2).every.value() == "4" and not traces_page(w2).by_field.isChecked()
        assert view_page(w2).energy.range() == pytest.approx((20.0, 80.0))
        w2.close()


def test_reset_brings_back_the_default_view(qtbot, tmp_path):
    w = MainWindow(settings=QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat))
    qtbot.addWidget(w)
    c = w.controller
    default = c.view
    c.set_ranges(energy_range=(10.0, 20.0))
    c.set_levels("Ratio", 0.5, 1.5)
    c.set_view(colormap="grey", stacked_every=2)
    w.reset_settings()
    assert c.view == default
