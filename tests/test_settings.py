import pytest
from PySide6.QtCore import QSettings

from mag_opt_detective.gui.main_window import MainWindow


@pytest.fixture
def ini(tmp_path):
    return str(tmp_path / "settings.ini")


def make_window(qtbot, ini):
    window = MainWindow(settings=QSettings(ini, QSettings.Format.IniFormat))
    qtbot.addWidget(window)
    return window


def test_settings_round_trip(qtbot, ini):
    w = make_window(qtbot, ini)
    dp = w.data_panel
    dp.unit.setCurrentText("meV")
    dp.field_source.setCurrentIndex(1)
    dp.sample.field_range.start.setText("0.5")
    dp.reference.ref_separate.setChecked(True)
    dp.reference.sg_window.setValue(15)
    w.limits_page.energy_custom.setChecked(True)
    w.limits_page.energy_max.setText("120")
    w.limits_page.level_edits["Ratio"][0].setText("0.95")
    w.corrections.column_name.setText("CR1")
    w.plot_panel.cmap_combo.setCurrentText("viridis")
    w.plot_panel.order_buttons[2].setChecked(True)
    w.close()

    w2 = make_window(qtbot, ini)
    dp2 = w2.data_panel
    assert dp2.unit.currentText() == "meV"
    assert dp2.custom_field()
    assert dp2.sample.field_range.isEnabled()
    assert dp2.sample.field_range.start.text() == "0.5"
    assert dp2.reference.ref_separate.isChecked()
    assert not dp2.reference.ref_none.isChecked()
    assert dp2.reference.sg_window.value() == 15
    limits = w2.limits_page.limits()
    assert limits.energy_view == (0.0, 120.0)
    assert limits.levels["Ratio"] == (0.95, 1.1)
    assert w2.corrections.column_name.text() == "CR1"
    assert w2.plot_panel.cmap_combo.currentText() == "viridis"
    assert w2.plot_panel.order() == 2
    assert w2.geometry_restored


def test_invalid_values_fall_back_to_defaults(qtbot, ini):
    raw = QSettings(ini, QSettings.Format.IniFormat)
    raw.setValue("v1/data/unit", "eV")
    raw.setValue("v1/reference/sg_window", "abc")
    raw.setValue("v1/limits/field_min", "not a number")
    raw.setValue("v1/window/geometry", "garbage")
    raw.sync()
    w = make_window(qtbot, ini)
    assert w.data_panel.unit.currentText() == "cm-1"
    assert w.data_panel.reference.sg_window.value() == 11
    assert w.limits_page.field_min.text() == "0"
    assert not w.geometry_restored


def test_reset_settings(qtbot, ini):
    w = make_window(qtbot, ini)
    w.data_panel.unit.setCurrentText("THz")
    w.save_settings()
    w.reset_settings()
    assert w.data_panel.unit.currentText() == "cm-1"
    assert QSettings(ini, QSettings.Format.IniFormat).value("v1/data/unit") is None


def test_window_without_settings_stores_nothing(qtbot, tmp_path):
    w = MainWindow()
    qtbot.addWidget(w)
    assert w.persistence is None
    w.close()  # must not fail
