"""The View, Colour and Traces sections of the inspector, and how they follow the plots."""

import numpy as np
import pytest
from PySide6.QtCore import Qt

import gui_helpers
from gui_helpers import inspector_page, load_sweep, process, select, set_unit
from mag_opt_detective.core.colormaps import lut
from mag_opt_detective.core.pipeline import PlotKind, ReferenceMode
from mag_opt_detective.core.processing import Axis
from mag_opt_detective.gui.controller import level_key
from mag_opt_detective.gui.plots import BarScale, robust_levels

window, errors = gui_helpers.window, gui_helpers.errors  # shared fixtures

MEV = 8.0656


@pytest.fixture
def processed(window, sweep):
    load_sweep(window, sweep)
    process(window)
    return window


def colour_page(window):
    return inspector_page(window, "colour")


def traces_page(window):
    return inspector_page(window, "traces")


def map_values(window) -> np.ndarray:
    return window.controller.current_map().values


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


def test_sections_follow_the_plot_on_screen(window):
    visible = {
        "map": ["view", "colour", "overlays"],
        "stacked": ["view", "traces"],
        "reference": ["view", "colour"],
    }
    for view, names in visible.items():
        window.plot_area.set_current_view(view)
        assert [n for n, s in window.inspector.items() if not s.isHidden()] == names


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
