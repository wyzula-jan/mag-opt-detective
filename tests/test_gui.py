import numpy as np
import pytest
from PySide6.QtCore import QMimeData, QPointF, Qt, QUrl
from PySide6.QtGui import QDropEvent
from PySide6.QtWidgets import QFileDialog, QMessageBox

from mag_opt_detective.core.pipeline import PlotKind, ReferenceMode
from mag_opt_detective.core.processing import Axis
from mag_opt_detective.core.spectra import load_tsv
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
    window.process_data()
    c = window.corrections
    c.point_record.setChecked(True)
    window.plot_panel.color_map.pointClicked.emit(1.04, 300.0)
    window.plot_panel.color_map.pointClicked.emit(1.96, 310.0)
    assert window.point_model.rowCount() == 4
    assert window.point_model.data(window.point_model.index(1, 0)) == "300"
    c.point_remove.setChecked(True)
    window.plot_panel.color_map.pointClicked.emit(2.0, 0.0)
    b, _ = window.points.points("LL 1")
    np.testing.assert_allclose(b, [1.0])

    path = tmp_path / "points.csv"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(path), ""))
    window.export_points()
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: (str(path), ""))
    window.points = None
    window.load_points()
    assert window.points.names == ["LL 1"]
    assert not errors


def test_export_slots_and_merge(window, sweep, tmp_path, monkeypatch, errors):
    load_sweep(window, sweep)
    window.data_panel.unit.setCurrentText("meV")
    window.process_data()
    out = tmp_path / "S1.csv"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(out), ""))
    window.export_current()
    exported = tmp_path / "S1_Ratio.csv"
    assert exported.exists()
    fmap = load_tsv(exported)
    assert str(fmap.unit) == "meV"

    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: (str(exported), ""))
    window.load_slot(0)
    window.save_slot(1)
    processed = window.data_panel.processed
    processed.set_energy_range(0, None, 60)
    processed.set_energy_range(1, 60, None)
    window.merge_slots()
    assert not errors
    merged = window.result.ratio
    np.testing.assert_allclose(merged.energy[[0, -1]], fmap.energy[[0, -1]])
    np.testing.assert_allclose(np.diff(merged.energy), np.diff(merged.energy)[0])

    processed.full_energy.setChecked(False)
    window.plot_slot(0)
    assert window.result.ratio.energy.max() <= 60


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
    window.process_data()
    pp = window.plot_panel
    pp.order_buttons[1].click()
    pp.per_unit.setChecked(True)
    expected = window.result.get(PlotKind.RATIO, 1, Axis.ENERGY, physical=True)
    np.testing.assert_allclose(shown_image(window), expected.values)

    out = tmp_path / "S1.csv"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(out), ""))
    window.export_current()
    assert (tmp_path / "S1_Ratio_1stDer_perUnit.csv").exists()
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
