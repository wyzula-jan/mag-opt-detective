import numpy as np
import pytest
from PySide6.QtCore import QMimeData, QPointF, Qt, QUrl
from PySide6.QtGui import QDropEvent
from PySide6.QtWidgets import QFileDialog, QMessageBox

import golden
from mag_opt_detective.core.pipeline import PlotKind, ReferenceMode
from mag_opt_detective.core.processing import Axis
from mag_opt_detective.core.spectra import load_tsv
from mag_opt_detective.core.units import Unit, convert, to_cm1
from mag_opt_detective.gui.main_window import MainWindow
from mag_opt_detective.gui.widgets import FileListWidget


@pytest.fixture
def errors(monkeypatch):
    """Collect error dialogs instead of blocking on them."""
    messages: list[str] = []
    monkeypatch.setattr(QMessageBox, "warning", lambda _p, title, msg: messages.append(msg))
    return messages


@pytest.fixture
def window(qtbot, errors):
    w = MainWindow()
    qtbot.addWidget(w)
    yield w
    w.close()


def load_sweep(window, sweep):
    window.data_panel.sample.zero_list.set_paths(sweep["zero"])
    window.data_panel.sample.field_list.set_paths(sweep["field"])


def shown_image(window) -> np.ndarray:
    return window.plot_panel.color_map.image.image


def energy_label(window) -> str:
    return window.plot_panel.color_map.plot.getAxis("left").labelText


def save_to(monkeypatch, path) -> None:
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(path), ""))


def open_from(monkeypatch, path) -> None:
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: (str(path), ""))


def test_file_list_accepts_drop(qtbot, sweep):
    widget = FileListWidget()
    qtbot.addWidget(widget)
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(p)) for p in reversed(sweep["field"])])
    event = QDropEvent(
        QPointF(5, 5),
        Qt.DropAction.CopyAction,
        mime,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    widget.dropEvent(event)
    assert widget.paths() == [str(p) for p in sweep["field"]]  # sorted by field
    assert widget.count() == 4


def test_process_and_switch_plots(window, sweep, errors):
    load_sweep(window, sweep)
    window.process_data()
    assert not errors
    result = window.result
    assert result is not None
    np.testing.assert_allclose(shown_image(window), result.ratio.values)

    pp = window.plot_panel
    pp.kind_buttons[PlotKind.DATA].click()
    np.testing.assert_allclose(shown_image(window), result.data.values)
    pp.order_buttons[2].click()
    np.testing.assert_allclose(
        shown_image(window), result.get(PlotKind.DATA, 2, Axis.ENERGY).values
    )
    pp.axis_combo.setCurrentIndex(1)
    np.testing.assert_allclose(shown_image(window), result.get(PlotKind.DATA, 2, Axis.FIELD).values)
    pp.kind_buttons[PlotKind.AVERAGE].click()
    pp.order_buttons[0].click()
    np.testing.assert_allclose(shown_image(window), result.average.values)
    assert len(pp.stacked._curves) == sweep["fields"].size


def test_errors_are_reported_not_raised(window, errors):
    window.process_data()
    assert window.result is None
    assert errors and "no field files" in errors[0]


def test_custom_field_and_reference(window, sweep, errors):
    dp = window.data_panel
    load_sweep(window, sweep)
    dp.field_source.setCurrentIndex(1)
    dp.sample.field_range.start.setText("1")
    dp.sample.field_range.step.setText("1")
    dp.sample.field_range.end.setText("4")
    dp.reference.zero_list.set_paths(sweep["zero"])
    dp.reference.field_list.set_paths(sweep["field"])
    dp.reference.field_range.start.setText("1")
    dp.reference.field_range.step.setText("1")
    dp.reference.field_range.end.setText("4")
    dp.reference.set_reference_mode(ReferenceMode.SEPARATE)
    window.process_data()
    assert not errors
    np.testing.assert_allclose(window.result.ratio.field, [1, 2, 3, 4])
    np.testing.assert_allclose(window.result.ratio.values, 1.0)
    assert window.plot_panel.reference_map.image.image is not None


def test_points_record_remove_export(window, sweep, tmp_path, monkeypatch, errors):
    load_sweep(window, sweep)
    window.data_panel.unit.setCurrentText("meV")
    window.process_data()
    c = window.corrections
    cmap = window.plot_panel.color_map
    c.point_record.setChecked(True)
    cmap.pointClicked.emit(1.04, 37.2)  # meV, as shown
    cmap.pointClicked.emit(1.96, 38.5)
    assert window.point_model.rowCount() == 4
    assert window.point_model.data(window.point_model.index(1, 0)) == "37.2"
    np.testing.assert_allclose(window.points.points("LL 1")[1], [37.2 * 8.0656, 38.5 * 8.0656])
    np.testing.assert_allclose(cmap.current_points.getData()[1], [37.2, 38.5])  # markers in meV
    c.point_remove.setChecked(True)
    cmap.pointClicked.emit(2.0, 0.0)
    b, _ = window.points.points("LL 1")
    np.testing.assert_allclose(b, [1.0])

    path = tmp_path / "points.csv"
    save_to(monkeypatch, path)
    window.export_points()
    assert path.read_text().splitlines()[:3] == ["Energy (meV)\tLL 1", "0.5\t", "1.0\t37.2"]
    open_from(monkeypatch, path)
    window.points = None
    window.load_points()
    assert window.points.names == ["LL 1"]
    np.testing.assert_allclose(window.points.points("LL 1")[1], [37.2 * 8.0656])

    legacy = tmp_path / "Points_V1.csv"  # no unit in the header: read in the shown unit
    legacy.write_text("\tLL 2\n0.5\t40\n1.0\t\n1.5\t\n2.0\t41\n")
    open_from(monkeypatch, legacy)
    window.load_points()
    np.testing.assert_allclose(window.points.points("LL 2")[1], [40 * 8.0656, 41 * 8.0656])
    assert window.point_model.data(window.point_model.index(3, 0)) == "41"
    np.testing.assert_allclose(cmap.current_points.getData()[1], [40, 41])
    assert not errors


def test_points_keep_cm1_when_the_unit_changes(window, sweep, errors):
    load_sweep(window, sweep)
    dp, pp = window.data_panel, window.plot_panel
    dp.unit.setCurrentText("meV")
    window.process_data()
    window.corrections.point_record.setChecked(True)
    pp.color_map.pointClicked.emit(1.0, 40.0)

    dp.unit.setCurrentText("THz")  # applies from the next Process on (no live switch yet)
    pp.order_buttons[1].click()
    assert energy_label(window) == "Energy (meV)"
    np.testing.assert_allclose(pp.color_map.current_points.getData()[1], [40.0])

    window.process_data()  # keeps the table: "New table on next Process" is off now
    assert not errors
    assert energy_label(window) == "Energy (THz)"
    thz = 40.0 * 8.0656 / 33.35641
    np.testing.assert_allclose(window.points.points("LL 1")[1], [40.0 * 8.0656])
    np.testing.assert_allclose(pp.color_map.current_points.getData()[1], [thz])
    assert window.point_model.data(window.point_model.index(1, 0)) == f"{thz:.5g}"


def test_export_slots_and_merge(window, sweep, tmp_path, monkeypatch, errors):
    load_sweep(window, sweep)
    window.data_panel.unit.setCurrentText("meV")
    window.process_data()
    assert window.result.ratio.unit is Unit.CM1  # processed in cm-1, shown in meV
    assert energy_label(window) == "Energy (meV)"
    save_to(monkeypatch, tmp_path / "S1.csv")
    window.export_current()
    exported = tmp_path / "S1_Ratio.csv"
    assert exported.read_text().startswith("Energy (meV)\t0.50T")  # legacy header
    fmap = load_tsv(exported)
    assert fmap.unit is Unit.MEV
    np.testing.assert_allclose(fmap.energy, sweep["x"] / 8.0656)
    np.testing.assert_allclose(fmap.values, window.result.ratio.values, rtol=1e-11)

    open_from(monkeypatch, exported)
    window.load_slot(0)
    window.save_slot(1)
    assert window.slots[0].unit is window.slots[1].unit is Unit.CM1
    np.testing.assert_allclose(window.slots[0].energy, sweep["x"], rtol=1e-11)
    processed = window.data_panel.processed
    processed.set_energy_range(0, None, 60)  # meV, the unit of the panel
    processed.set_energy_range(1, 60, None)
    window.merge_slots()
    assert not errors
    merged = window.result.ratio
    np.testing.assert_allclose(merged.energy[[0, -1]], sweep["x"][[0, -1]], rtol=1e-11)
    np.testing.assert_allclose(np.diff(merged.energy), np.diff(merged.energy)[0])
    assert energy_label(window) == "Energy (meV)"

    processed.full_energy.setChecked(False)
    window.plot_slot(0)
    assert window.result.ratio.energy.max() <= to_cm1(60, Unit.MEV)
    assert window.result.ratio.to_unit(Unit.MEV).energy.max() <= 60
    window.plot_slot(1)
    assert window.result.ratio.to_unit(Unit.MEV).energy.min() >= 60


def test_slots_from_different_units_merge(window, sweep, tmp_path, monkeypatch, errors):
    """Slots keep cm-1, so slots saved while another unit was chosen still combine."""
    load_sweep(window, sweep)
    dp, processed = window.data_panel, window.data_panel.processed
    dp.unit.setCurrentText("meV")
    window.process_data()
    window.save_slot(0)
    save_to(monkeypatch, tmp_path / "S1.csv")
    window.export_current()  # an meV table

    dp.unit.setCurrentText("THz")
    open_from(monkeypatch, tmp_path / "S1_Ratio.csv")
    window.load_slot(1)
    window.process_data()
    window.save_slot(2)
    for slot in (1, 2):
        np.testing.assert_allclose(window.slots[slot].energy, window.slots[0].energy, rtol=1e-11)
    processed.set_used(2, False)
    processed.set_energy_range(0, None, 15)  # THz, about 500 cm-1
    processed.set_energy_range(1, 15, None)
    window.merge_slots()
    assert not errors
    assert window.display_unit() is Unit.THZ
    assert energy_label(window) == "Energy (THz)"
    shown = window.result.get(PlotKind.RATIO, unit=Unit.THZ)
    np.testing.assert_allclose(shown_image(window), shown.values)
    np.testing.assert_allclose(window.result.ratio.energy[[0, -1]], [100, 1000], rtol=1e-11)

    processed.set_used(2, True)
    for slot in (0, 1):
        processed.set_energy_range(slot, None, None)
    window.average_slots()
    window.merge_slots_by_field()
    assert not errors
    np.testing.assert_allclose(window.result.ratio.values, window.slots[2].values, rtol=1e-9)


def test_golden_exports_through_the_window(window, tmp_path, monkeypatch, errors):
    """Cut, baseline and exports typed and written in meV reproduce the golden tables."""
    zero, field = golden.write_sweep(tmp_path)
    window.data_panel.sample.zero_list.set_paths(zero)
    window.data_panel.sample.field_list.set_paths(field)
    window.data_panel.unit.setCurrentText("meV")
    limits, c = window.limits_page, window.corrections
    limits.energy_cut.setChecked(True)
    limits.energy_min.set_value(golden.ENERGY_CUT[0])
    limits.energy_max.set_value(golden.ENERGY_CUT[1])
    c.baseline_on.setChecked(True)
    c.baseline_min.set_value(golden.BASELINE[0])
    c.baseline_max.set_value(golden.BASELINE[1])
    window.process_data()
    pp = window.plot_panel
    save_to(monkeypatch, tmp_path / "G.csv")
    for kind, order, per_unit in (
        (PlotKind.RATIO, 0, False),
        (PlotKind.DATA, 0, False),
        (PlotKind.RATIO, 1, False),
        (PlotKind.RATIO, 1, True),
    ):
        pp.kind_buttons[kind].click()
        pp.order_buttons[order].click()
        pp.per_unit.setChecked(per_unit)
        window.export_current()
    assert not errors
    for name in golden.MAPS:
        golden.assert_matches(load_tsv(tmp_path / f"G_{name}.csv"), name)


def test_real_macro_data(window, data_dir, errors):
    files = sorted((data_dir / "Data_Macro").glob("*.txt"))
    zero = [str(f) for f in files if "_a00p000T_a" in f.name]
    field = [str(f) for f in files if "_a00p000T_a" not in f.name]
    window.data_panel.sample.zero_list.set_paths(zero)
    window.data_panel.sample.field_list.set_paths(field)
    window.data_panel.unit.setCurrentText("meV")
    window.process_data()
    assert not errors
    assert shown_image(window).shape == (7726, 64)


def test_per_unit_derivative(window, sweep, tmp_path, monkeypatch, errors):
    load_sweep(window, sweep)
    window.data_panel.unit.setCurrentText("meV")
    window.process_data()
    pp = window.plot_panel
    pp.kind_buttons[PlotKind.DATA].click()
    pp.order_buttons[1].click()
    pp.per_unit.setChecked(True)
    expected = window.result.get(PlotKind.DATA, 1, Axis.ENERGY, physical=True, unit=Unit.MEV)
    np.testing.assert_allclose(shown_image(window), expected.values)
    per_cm1 = window.result.get(PlotKind.DATA, 1, Axis.ENERGY, physical=True)
    np.testing.assert_allclose(shown_image(window), per_cm1.values * 8.0656, rtol=1e-9)

    save_to(monkeypatch, tmp_path / "S1.csv")
    window.export_current()
    exported = load_tsv(tmp_path / "S1_Data_1stDer_perUnit.csv")
    assert exported.unit is Unit.MEV
    np.testing.assert_allclose(exported.values, expected.values, rtol=1e-11)
    assert not errors


def test_merge_by_field_uses_ticked_slots(window, sweep, errors):
    load_sweep(window, sweep)
    window.process_data()
    processed = window.data_panel.processed
    for slot in (0, 1, 2):
        window.save_slot(slot)
    assert processed.used_slots(window.slots) == [0, 1, 2]
    processed.set_field_range(0, None, 1.0)
    processed.set_field_range(1, 1.5, None)
    processed.set_used(2, False)
    window.merge_slots_by_field()
    assert not errors
    np.testing.assert_allclose(window.result.ratio.field, sweep["fields"])

    for slot in (0, 1):
        processed.set_used(slot, False)
    window.merge_slots_by_field()
    assert errors and "Use column" in errors[-1]


def test_average_slots(window, sweep, errors):
    load_sweep(window, sweep)
    window.process_data()
    window.save_slot(0)
    window.save_slot(1)
    window.average_slots()
    assert not errors
    np.testing.assert_allclose(window.result.ratio.values, window.slots[0].values)


def test_histogram_levels_update_tools(window, sweep, errors):
    load_sweep(window, sweep)
    page = window.limits_page
    page.level_auto.setChecked(True)
    window.process_data()
    window.plot_panel.color_map.hist.region.setRegion((0.95, 1.05))  # as if dragged
    lo, hi = page.level_edits["Ratio"]
    assert (lo.value(), hi.value()) == pytest.approx((0.95, 1.05))
    assert page.level_custom.isChecked()
    assert window.plot_panel.color_map.image.levels == pytest.approx([0.95, 1.05])
    assert not errors


def test_colour_map_choice(window, sweep):
    load_sweep(window, sweep)
    window.process_data()
    pp = window.plot_panel
    assert pp.color_map._cmap == "magma"
    pp.order_buttons[1].click()
    assert pp.color_map._cmap == "grey"
    pp.cmap_combo.setCurrentText("viridis")
    assert pp.color_map._cmap == "viridis"


def test_cursor_shows_value(window, sweep):
    load_sweep(window, sweep)
    window.data_panel.unit.setCurrentText("meV")
    window.process_data()
    cmap = window.plot_panel.color_map
    ratio = window.result.get(PlotKind.RATIO, unit=Unit.MEV)
    text = cmap._cursor_text(ratio.field[1], ratio.energy[3])
    assert f"E = {ratio.energy[3]:.3f} meV" in text
    assert f"value = {ratio.values[3, 1]:.5g}" in text
    assert "value" not in cmap._cursor_text(100.0, ratio.energy[3])


@pytest.mark.parametrize("suffix", [".png", ".svg"])
def test_save_image(window, sweep, tmp_path, monkeypatch, errors, suffix):
    load_sweep(window, sweep)
    window.process_data()
    out = tmp_path / f"map{suffix}"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(out), ""))
    window.save_image()
    assert not errors
    assert out.stat().st_size > 1000


def test_step_ratio_plot_and_export(window, sweep, tmp_path, monkeypatch, errors):
    load_sweep(window, sweep)
    window.process_data()
    window.plot_panel.kind_buttons[PlotKind.STEP].click()
    np.testing.assert_allclose(shown_image(window), window.result.step.values)
    assert window.limits_page.limits().levels_for(PlotKind.STEP) == (0.98, 1.02)
    out = tmp_path / "S1.csv"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(out), ""))
    window.export_current()
    assert (tmp_path / "S1_Ratio_Step.csv").exists()
    assert not errors


def test_dirac_overlay(window, sweep):
    from mag_opt_detective.core.models import dirac_interband

    load_sweep(window, sweep)
    window.data_panel.unit.setCurrentText("THz")
    window.process_data()
    cmap = window.plot_panel.color_map
    assert cmap.model_curve_data() == []
    mp = window.models_page
    mp.n_lines.setValue(3)
    mp.delta.setValue(10.0)
    mp.velocity.setValue(5.0)
    mp.show_dirac.setChecked(True)
    curves = cmap.model_curve_data()
    assert len(curves) == 3
    field, energy = curves[1]
    expected = convert(dirac_interband(field, 5.0, 10.0, 3)[1], Unit.MEV, Unit.THZ)
    np.testing.assert_allclose(energy, expected)
    mp.n_lines.setValue(2)
    assert len(cmap.model_curve_data()) == 2
    mp.show_dirac.setChecked(False)
    assert cmap.model_curve_data() == []
