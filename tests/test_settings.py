import json

import pytest
from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QGuiApplication

from mag_opt_detective.core.pipeline import PlotKind, ReferenceMode
from mag_opt_detective.core.processing import Axis
from mag_opt_detective.core.units import Unit
from mag_opt_detective.gui.controller import PlotSelection, ViewState
from mag_opt_detective.gui.main_window import MainWindow
from mag_opt_detective.gui.plots import BarScale
from mag_opt_detective.gui.settings import PREFIX

MEV = 8.0656


@pytest.fixture
def ini(tmp_path):
    return str(tmp_path / "settings.ini")


def make_window(qtbot, ini, show=False):
    window = MainWindow(settings=QSettings(ini, QSettings.Format.IniFormat))
    qtbot.addWidget(window)
    if show:
        window.resize(1400, 900)
        window.show()
        qtbot.waitExposed(window)
    return window


def view_page(window):
    return window.inspector["view"].body_layout().itemAt(0).widget()


def colour_page(window):
    return window.inspector["colour"].body_layout().itemAt(0).widget()


def test_settings_round_trip(qtbot, ini):
    w = make_window(qtbot, ini, show=True)
    w.toolbar.unit.set_value("meV")
    sample = w.panels["sample"]
    sample.field_source.set_value("custom")
    sample.field_range.start.setText("0.5")
    reference = w.panels["reference"]
    reference.set_reference_mode(ReferenceMode.SEPARATE)
    reference.sg_window.setValue(15)
    w.controller.set_ranges(energy_range=(0.0, 120.0))  # as if zoomed (meV)
    colour_page(w).fields.lo.setValue(0.95)
    colour_page(w).picker.swatches["viridis"].click()
    processing = w.panels["processing"]
    processing.cut_on.setChecked(True)
    processing.cut_lo.setText("55")
    processing.cut_hi.setText("140")
    w.panels["points"].column_name.setText("CR1")
    w.toolbar.order.set_value("2")
    w.plot_area.scale_style_button.setChecked(True)
    w.show_panel("library")
    w.side_panel.set_open(False, animate=False)
    w.log_panel.set_open(True, animate=False)
    w.plot_area.map_scale.set_open(False, animate=False)
    w.inspector["models"].set_expanded(False, animate=False)
    w.commands["export_suffix"].setChecked(False)
    w.close()

    w2 = make_window(qtbot, ini, show=True)
    c = w2.controller
    assert c.unit is Unit.MEV and w2.toolbar.unit.value() == "meV"
    assert w2.panels["sample"].custom_field()
    assert not w2.panels["sample"].field_range.isHidden()
    assert w2.panels["sample"].field_range.start.text() == "0.5"
    assert w2.panels["reference"].reference_mode() is ReferenceMode.SEPARATE
    assert c.processing.reference_mode is ReferenceMode.SEPARATE
    assert w2.panels["reference"].sg_window.value() == 15
    assert c.view.energy_range == pytest.approx((0.0, 120.0))  # meV
    assert view_page(w2).energy.range() == pytest.approx((0.0, 120.0))
    assert not view_page(w2).energy.is_auto()
    assert c.view.levels["Ratio"] == (0.95, 1.1)
    assert c.view.colormap == "viridis"
    assert c.processing.energy_cut == pytest.approx((55 * MEV, 140 * MEV))
    assert w2.panels["processing"].cut_lo.text() == "55"
    assert c.curve == "CR1"
    assert c.selection.order == 2
    assert isinstance(w2.plots.reference.scale, BarScale)
    assert not w2.commands["export_suffix"].isChecked()
    assert not w2.inspector["models"].is_expanded()
    assert w2.current_panel() == "library"
    assert w2.geometry_restored
    qtbot.waitUntil(lambda: not w2.side_panel.is_open())  # splitter layouts
    assert w2.log_panel.is_open()
    assert not w2.plot_area.map_scale.is_open() and w2.plot_area.reference_scale.is_open()
    assert not w2.plots.map.scale_visible()
    assert w2.inspector_panel.is_open()


def test_restoring_does_not_depend_on_the_order(qtbot, ini):
    w = make_window(qtbot, ini)
    w.toolbar.unit.set_value("meV")
    w.controller.set_ranges(energy_range=(10.0, 120.0))
    w.controller.set_levels("Ratio_der1_E_unit", -0.5, 0.5)  # per meV
    w.panels["processing"].baseline_on.setChecked(True)
    w.panels["processing"].baseline_lo.setText("60")
    w.panels["processing"].baseline_hi.setText("70")
    w.save_settings()
    stored = json.loads(QSettings(ini, QSettings.Format.IniFormat).value("v2/view/levels"))
    levels = stored["Ratio_der1_E_unit"]["levels"]
    assert levels == pytest.approx([-0.5 / MEV, 0.5 / MEV])  # kept per cm-1
    w.close()

    for reverse in (False, True):
        w2 = make_window(qtbot, ini)
        c = w2.controller
        w2.toolbar.unit.set_value("THz")  # a live switch before restoring again
        p = w2.persistence
        if reverse:
            p._widgets = dict(reversed(list(p._widgets.items())))
        w2.restore_settings()
        assert c.unit is Unit.MEV
        assert c.view.energy_range == pytest.approx((10.0, 120.0))
        assert c.view.levels["Ratio_der1_E_unit"] == pytest.approx((-0.5, 0.5))
        assert c.processing.baseline == pytest.approx((60 * MEV, 70 * MEV))
        assert w2.panels["processing"].baseline_lo.text() == "60"
        assert view_page(w2).energy.range() == pytest.approx((10.0, 120.0))
        w2.close()


def test_state_set_through_the_controller_is_remembered(qtbot, ini):
    w = make_window(qtbot, ini)
    c = w.controller
    c.set_unit("THz")
    c.set_selection(kind=PlotKind.DATA, order=1, axis=Axis.FIELD, physical=True)
    c.set_processing(energy_cut=(200.0, 800.0), reference_mode=ReferenceMode.SELF)
    w.close()
    c2 = make_window(qtbot, ini).controller
    assert c2.unit is Unit.THZ
    assert c2.selection == PlotSelection(PlotKind.DATA, 1, Axis.FIELD, physical=True)
    assert c2.processing.energy_cut == (200.0, 800.0)
    assert c2.processing.reference_mode is ReferenceMode.SELF


def test_invalid_values_fall_back_to_defaults(qtbot, ini):
    raw = QSettings(ini, QSettings.Format.IniFormat)
    raw.setValue("v2/view/unit", "eV")
    raw.setValue("v2/reference/sg_window", "abc")
    raw.setValue("v2/view/ranges", "{bad")
    raw.setValue("v2/processing/cut_lo_cm1", "x")
    raw.setValue("v2/view/levels", '{"Ratio": {"mode": "fixed", "levels": [2, 1]}}')
    raw.setValue("v2/view/traces", '{"offset": -1, "every": 2, "by_field": false}')
    raw.setValue("v2/view/colormap", "rainbow")
    raw.setValue("v2/window/panel", "nowhere")
    raw.setValue("v2/window/appearance", "purple")
    raw.setValue("v2/window/geometry", "garbage")
    raw.setValue("v2/splitters/body", "garbage")
    raw.sync()
    w = make_window(qtbot, ini)
    assert w.controller.unit is Unit.CM1
    assert w.panels["reference"].sg_window.value() == 11
    assert w.controller.view == ViewState()
    assert w.panels["processing"].cut_lo.text() == ""
    assert w.current_panel() == "sample"
    assert w.theme.scheme() == "system"
    assert not w.geometry_restored


def test_version_1_values_are_ignored(qtbot, ini):
    raw = QSettings(ini, QSettings.Format.IniFormat)
    raw.setValue("v1/data/unit", "meV")
    raw.setValue("v1/plot/colours", "viridis")
    raw.sync()
    w = make_window(qtbot, ini)
    assert w.controller.unit is Unit.CM1
    assert w.controller.view.colormap == "Auto"


@pytest.mark.parametrize(
    ("ratio", "data", "kind"), [("false", "true", "Data"), ("true", "false", "Ratio")]
)
def test_the_reference_map_choice_of_two_buttons_is_kept(qtbot, ini, ratio, data, kind):
    """Earlier versions stored the reference map choice as two buttons."""
    raw = QSettings(ini, QSettings.Format.IniFormat)
    raw.setValue(f"{PREFIX}/plot/reference_ratio", ratio)
    raw.setValue(f"{PREFIX}/plot/reference_data", data)
    raw.sync()
    w = make_window(qtbot, ini)
    assert w.controller.selection.reference_kind is PlotKind(kind)
    assert w.plot_area.ref_kind.value() == kind
    w.close()
    stored = QSettings(ini, QSettings.Format.IniFormat)
    assert stored.value(f"{PREFIX}/plot/reference_kind") == kind
    assert stored.value(f"{PREFIX}/plot/reference_data") is None


def test_reset_settings(qtbot, ini):
    w = make_window(qtbot, ini, show=True)
    w.toolbar.unit.set_value("THz")
    w.log_panel.set_open(True, animate=False)
    w.side_panel.set_open(False, animate=False)
    w.save_settings()
    w.reset_settings()
    assert w.toolbar.unit.value() == "cm-1" and w.controller.unit is Unit.CM1
    assert QSettings(ini, QSettings.Format.IniFormat).value("v2/view/unit") is None
    assert w.side_panel.is_open() and not w.log_panel.is_open()


def test_appearance_is_remembered(qtbot, ini):
    w = make_window(qtbot, ini)
    try:
        w.set_appearance("dark")
        w.close()
        w2 = make_window(qtbot, ini)
        assert w2.theme.scheme() == "dark"
        assert w2.plots.map.colors().background == "#121015"
    finally:
        for window in (w, locals().get("w2")):
            if window is not None:
                window.theme.set_scheme("system")
        QGuiApplication.styleHints().setColorScheme(Qt.ColorScheme.Unknown)


def test_window_without_settings_stores_nothing(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    assert w.persistence is None
    w.close()  # must not fail
