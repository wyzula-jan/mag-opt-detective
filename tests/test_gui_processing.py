import numpy as np
import pytest
from PySide6.QtCore import QMimeData, QPointF, Qt, QUrl
from PySide6.QtGui import QDropEvent
from PySide6.QtWidgets import QMessageBox, QWidget

import golden
import gui_helpers
from gui_helpers import (
    energy_label,
    hover,
    infobar_text,
    inspector_page,
    load_sweep,
    process,
    save_to,
    select,
    set_unit,
    shown_image,
)
from mag_opt_detective.core.pipeline import PlotKind, ReferenceMode
from mag_opt_detective.core.processing import Axis
from mag_opt_detective.core.spectra import load_tsv
from mag_opt_detective.core.units import Unit
from mag_opt_detective.gui.controller import FieldRange
from mag_opt_detective.gui.panels import library
from mag_opt_detective.gui.widgets import FileListWidget

window, errors = gui_helpers.window, gui_helpers.errors  # shared fixtures


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


def test_file_lists_follow_the_controller(window, sweep):
    load_sweep(window, sweep)
    files = window.controller.processing.sample_files
    assert files.field == tuple(str(p) for p in sweep["field"])
    assert len(files.zero) == 2
    window.controller.set_processing(sample_files=type(files)(files.zero[:1], files.field[:2]))
    tab = window.panels["sample"].measurement
    assert tab.field_list.count() == 2 and tab.zero_list.count() == 1


def test_process_and_switch_plots(window, sweep, errors):
    load_sweep(window, sweep)
    process(window)
    assert not errors
    result = window.controller.result
    assert result is not None
    np.testing.assert_allclose(shown_image(window), result.ratio.values)

    select(window, kind=PlotKind.DATA)
    np.testing.assert_allclose(shown_image(window), result.data.values)
    select(window, order=2)
    np.testing.assert_allclose(
        shown_image(window), result.get(PlotKind.DATA, 2, Axis.ENERGY).values
    )
    select(window, axis="B")
    np.testing.assert_allclose(shown_image(window), result.get(PlotKind.DATA, 2, Axis.FIELD).values)
    select(window, kind=PlotKind.AVERAGE, order=0)
    np.testing.assert_allclose(shown_image(window), result.average.values)
    assert len(window.plots.stacked.curves()) == sweep["fields"].size
    assert window.plot_area.description.text() == "R(B)/R(B-AVR)"


def test_shortcuts_select_the_plot(window, sweep):
    load_sweep(window, sweep)
    process(window)
    tb = window.toolbar
    shortcuts = {a.shortcut().toString(): a for a in QWidget.actions(window)}
    shortcuts["Ctrl+2"].trigger()
    shortcuts["Alt+3"].trigger()
    assert (tb.kind.value(), tb.order.value()) == ("Data", "2")
    assert tb.axis.button("B").isEnabled() and tb.per_unit.isEnabled()
    shortcuts["Alt+1"].trigger()
    assert not tb.axis.button("B").isEnabled() and not tb.per_unit.isEnabled()
    process_keys = [k.toString() for k in window.commands["process"].shortcuts()]
    assert process_keys[0] == "Ctrl+Return" and "Ctrl+F" in process_keys


def test_errors_show_in_the_error_bar(window, monkeypatch):
    dialogs = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: dialogs.append(a))
    window.show_panel("points")
    process(window)
    assert window.controller.result is None
    assert "no field files" in infobar_text(window).lower()
    assert not dialogs  # expected errors never open a dialog
    action = window.infobar.action_button
    assert action.text() == "Open Sample"
    action.click()
    assert window.current_panel() == "sample"
    assert window.infobar.isHidden()


def test_unexpected_errors_open_a_dialog(window, sweep, monkeypatch):
    dialogs = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: dialogs.append(a[1:]))

    def broken():
        raise RuntimeError("boom")

    monkeypatch.setattr(window.controller, "process", broken)
    process(window)
    assert dialogs and "boom" in dialogs[0][1]
    assert window.infobar.isHidden()


def test_custom_field_and_reference(window, sweep, errors):
    load_sweep(window, sweep)
    sample = window.panels["sample"]
    sample.field_source.set_value("custom")
    box = sample.field_range
    assert not box.isHidden()
    box.start.setText("1")
    box.step.setText("1")
    box.end.setText("4")
    reference = window.panels["reference"]
    reference.set_reference_mode(ReferenceMode.SEPARATE)
    assert reference.field_range.isVisibleTo(reference)
    reference.zero_list.set_paths(sweep["zero"])
    reference.field_list.set_paths(sweep["field"])
    reference.field_range.start.setText("1")
    reference.field_range.step.setText("1")
    reference.field_range.end.setText("4")
    process(window)
    assert not errors
    result = window.controller.result
    np.testing.assert_allclose(result.ratio.field, [1, 2, 3, 4])
    np.testing.assert_allclose(result.ratio.values, 1.0)
    assert window.plots.reference.image.image is not None
    window.plot_area.ref_kind.set_value("Data")
    np.testing.assert_allclose(window.plots.reference.image.image, result.reference_data.values)


def test_missing_reference_offers_the_reference_panel(window, sweep):
    load_sweep(window, sweep)
    window.panels["reference"].set_reference_mode(ReferenceMode.SEPARATE)
    window.panels["reference"].zero_list.set_paths(sweep["zero"])  # a sweep without field files
    process(window)
    assert "no field files" in infobar_text(window).lower()
    assert window.infobar.action_button.text() == "Open Reference"
    bar = window.infobar  # the mockup's wording: what failed, then what fixes it
    assert bar.title_label.text() == "Can't process: no field files loaded."
    assert bar.text_label.text() == (
        "Check the reference sweep in the Reference panel, or set Reference to None."
    )
    message = "energy window: the first value must be below the second; it is 3 – 2 meV"
    window.report_error("Process", message, panel="processing")  # one colon in the head
    assert (bar.title_label.text(), bar.text_label.text()) == (
        "Can't process: energy window.",
        "The first value must be below the second; it is 3 – 2 meV. Check the energy window "
        "and the baseline in the Processing panel.",
    )
    window.report_error("Colour range", "invalid colour range 2 – 1")  # no panel: no remedy
    assert (bar.title_label.text(), bar.text_label.text()) == (
        "Can't set the colour range",
        "Invalid colour range 2 – 1",
    )


def test_golden_exports_through_the_window(window, tmp_path, monkeypatch, errors):
    """Window and baseline typed and exported in meV reproduce the golden tables."""
    zero, field = golden.write_sweep(tmp_path)
    tab = window.panels["sample"].measurement
    tab.zero_list.set_paths(zero)
    tab.field_list.set_paths(field)
    set_unit(window, "meV")
    panel = window.panels["processing"]
    panel.cut_on.setChecked(True)
    panel.cut_lo.setText(str(golden.ENERGY_CUT[0]))
    panel.cut_hi.setText(str(golden.ENERGY_CUT[1]))
    panel.baseline_on.setChecked(True)
    panel.baseline_lo.setText(str(golden.BASELINE[0]))
    panel.baseline_hi.setText(str(golden.BASELINE[1]))
    process(window)
    save_to(monkeypatch, tmp_path / "G.csv")
    for kind, order, per_unit in (
        (PlotKind.RATIO, 0, False),
        (PlotKind.DATA, 0, False),
        (PlotKind.RATIO, 1, False),
        (PlotKind.RATIO, 1, True),
    ):
        select(window, kind=kind, order=order, per_unit=per_unit)
        window.commands["export_table"].trigger()
    assert not errors
    for name in golden.MAPS:
        golden.assert_matches(load_tsv(tmp_path / f"G_{name}.csv"), name)


def test_energy_errors_are_in_the_display_unit(window, sweep, errors):
    load_sweep(window, sweep)
    set_unit(window, "meV")
    panel = window.panels["processing"]
    panel.cut_on.setChecked(True)
    panel.cut_lo.setText("100")
    panel.cut_hi.setText("50")
    process(window)
    assert "energy window" in errors[-1] and "100 – 50 meV" in errors[-1]
    assert window.infobar.action_button.text() == "Open Processing"
    panel.cut_lo.setText("200")
    panel.cut_hi.setText("300")
    process(window)
    assert "200 – 300 meV contains no data" in errors[-1]
    assert "12.3983 – 123.983 meV" in errors[-1]  # the data, 100 – 1000 cm-1
    panel.cut_on.setChecked(False)
    panel.baseline_on.setChecked(True)
    panel.baseline_lo.setText("500")
    panel.baseline_hi.setText("600")
    process(window)
    assert "baseline region 500 – 600 meV contains no data" in errors[-1]
    assert window.controller.result is None


def test_real_macro_data(window, data_dir, errors):
    files = sorted((data_dir / "Data_Macro").glob("*.txt"))
    tab = window.panels["sample"].measurement
    tab.zero_list.set_paths([str(f) for f in files if "_a00p000T_a" in f.name])
    tab.field_list.set_paths([str(f) for f in files if "_a00p000T_a" not in f.name])
    set_unit(window, "meV")
    process(window)
    assert not errors
    assert shown_image(window).shape == (7726, 64)
    assert energy_label(window) == "Energy (meV)"


def test_per_unit_derivative(window, sweep, tmp_path, monkeypatch, errors):
    load_sweep(window, sweep)
    set_unit(window, "meV")
    process(window)
    select(window, kind=PlotKind.DATA, order=1, per_unit=True)
    result = window.controller.result
    expected = result.get(PlotKind.DATA, 1, Axis.ENERGY, physical=True, unit=Unit.MEV)
    np.testing.assert_allclose(shown_image(window), expected.values)
    per_cm1 = result.get(PlotKind.DATA, 1, Axis.ENERGY, physical=True)
    np.testing.assert_allclose(shown_image(window), per_cm1.values * 8.0656, rtol=1e-9)
    assert window.plot_area.description.text().endswith("1st derivative d/dE per meV")

    save_to(monkeypatch, tmp_path / "S1.csv")
    window.commands["export_table"].trigger()
    exported = load_tsv(tmp_path / "S1_Data_1stDer_perUnit.csv")
    assert exported.unit is Unit.MEV
    np.testing.assert_allclose(exported.values, expected.values, rtol=1e-11)
    assert not errors


def test_export_without_the_plot_type_in_the_name(window, sweep, tmp_path, monkeypatch):
    load_sweep(window, sweep)
    process(window)
    window.commands["export_suffix"].setChecked(False)
    save_to(monkeypatch, tmp_path / "plain")
    window.commands["export_table"].trigger()
    assert (tmp_path / "plain.csv").exists()


def test_histogram_levels_update_the_view(window, sweep, errors):
    load_sweep(window, sweep)
    page = inspector_page(window, "colour")
    page.mode.set_value("auto")
    process(window)
    window.plots.map.hist.region.setRegion((0.95, 1.05))  # as if dragged
    assert page.fields.levels() == pytest.approx((0.95, 1.05))
    assert page.mode.value() == "fixed"
    assert window.controller.view.levels["Ratio"] == pytest.approx((0.95, 1.05))
    assert window.plots.map.image.levels == pytest.approx([0.95, 1.05])
    assert not errors


def test_colour_map_choice(window, sweep):
    load_sweep(window, sweep)
    process(window)
    plot = window.plots.map
    assert plot.colormap() == "magma"
    select(window, order=1)
    assert plot.colormap() == "grey"
    inspector_page(window, "colour").picker.swatches["viridis"].click()
    assert plot.colormap() == "viridis"
    assert window.controller.view.colormap == "viridis"


def test_cursor_shows_value(window, sweep, qtbot):
    load_sweep(window, sweep)
    set_unit(window, "meV")
    process(window)
    plot = window.plots.map
    ratio = window.controller.result.get(PlotKind.RATIO, unit=Unit.MEV)
    hover(qtbot, plot, ratio.field[1], ratio.energy[3])
    assert f"E = {ratio.energy[3]:.3f} meV" in plot.label.text
    assert f"value = {ratio.values[3, 1]:.5g}" in plot.label.text
    assert plot.value_at(100.0, ratio.energy[3]) is None  # no value outside the map
    plot.cursorMoved.emit(1.0, 50.0, 1.25)  # the value is named by the plot kind
    assert window.cursor_text() == "B = 1 T    E = 50 meV    R(B)/R(0) = 1.25"
    window.plots.stacked.cursorMoved.emit(50.0, 1.5, None)
    assert window.cursor_text() == "E = 50 meV    I = 1.5"
    select(window, order=1)
    set_unit(window, "cm-1")
    plot.cursorMoved.emit(1.25, 403.28, -0.0021)
    assert window.cursor_text() == "B = 1.25 T    E = 403.28 cm⁻¹    1st derivative = -0.0021"


@pytest.mark.parametrize("suffix", [".png", ".svg"])
def test_save_image(window, sweep, tmp_path, monkeypatch, errors, suffix):
    load_sweep(window, sweep)
    process(window)
    out = tmp_path / f"map{suffix}"
    save_to(monkeypatch, out)
    window.commands["export_image"].trigger()
    assert not errors
    assert out.stat().st_size > 1000


def test_step_ratio_plot_and_export(window, sweep, tmp_path, monkeypatch, errors):
    load_sweep(window, sweep)
    process(window)
    select(window, kind=PlotKind.STEP)
    np.testing.assert_allclose(shown_image(window), window.controller.result.step.values)
    assert window.controller.current_levels() == (0.98, 1.02)
    save_to(monkeypatch, tmp_path / "S1.csv")
    window.commands["export_table"].trigger()
    assert (tmp_path / "S1_Ratio_Step.csv").exists()
    assert not errors


def test_changed_since_process(window, sweep):
    c, tb = window.controller, window.toolbar
    load_sweep(window, sweep)
    assert not tb.process_button.dot
    assert "Not processed" in window.state_text()
    process(window)
    assert not tb.process_button.dot
    assert "Processed" in window.state_text()
    assert window.summary_text() == "4 spectra · B 0.5 – 2 T · E 100 – 1000 cm⁻¹"

    window.panels["processing"].baseline_on.setChecked(True)
    assert c.changed_since_process() and tb.process_button.dot
    assert window.rail_button("processing").badge
    assert "Settings changed" in window.state_text()
    window.panels["processing"].baseline_on.setChecked(False)
    assert not tb.process_button.dot  # back to the settings that were used
    set_unit(window, "meV")  # the unit is not a processing setting
    assert not tb.process_button.dot
    assert window.summary_text().endswith("E 12.4 – 124 meV")
    reference = window.panels["reference"]
    reference.smooth.setChecked(True)
    reference.sg_window.setValue(15)
    assert not tb.process_button.dot  # smoothing without a reference changes nothing
    reference.set_reference_mode(ReferenceMode.SELF)
    assert tb.process_button.dot
    process(window)
    assert not tb.process_button.dot and not window.rail_button("processing").badge


def test_library_maps_leave_the_process_state(window, sweep, errors):
    load_sweep(window, sweep)
    process(window)
    c, tb = window.controller, window.toolbar
    library.add_processed(window)
    key = c.library[0].key
    library.plot_entry(window, key)
    assert "Showing a library map" in window.state_text() and not tb.process_button.dot
    window.panels["processing"].baseline_on.setChecked(True)
    window.panels["processing"].baseline_lo.setText("100")
    window.panels["processing"].baseline_hi.setText("200")
    assert tb.process_button.dot
    library.plot_entry(window, key)  # the sweep was not processed again
    assert tb.process_button.dot and "Settings changed" in window.state_text()
    process(window)
    assert not tb.process_button.dot and "Processed" in window.state_text()
    assert c.result_source == "process" and not errors


def test_panels_follow_the_processing_state(window, sweep):
    c = window.controller
    set_unit(window, "meV")
    c.set_processing(
        energy_cut=(200.0, 800.0),
        baseline=(300.0, None),
        reference_mode=ReferenceMode.SELF,
        smooth=True,
        sg_window=15,
        custom_field=True,
        sample_field=FieldRange(1.0, 0.5, 3.0),
    )
    processing = window.panels["processing"]
    assert processing.cut_on.isChecked() and processing.baseline_on.isChecked()
    assert processing.cut_lo.text() == "24.7967" and processing.baseline_hi.text() == ""
    reference = window.panels["reference"]
    assert reference.mode.value() == "self" and reference.smooth.isChecked()
    assert reference.sg_window.value() == 15
    assert window.panel_pages["reference"].subtitle.text().startswith("Corrects the sample with")
    sample = window.panels["sample"]
    assert sample.custom_field() and not sample.field_range.isHidden()
    box = sample.field_range
    assert (box.start.text(), box.step.text(), box.end.text()) == ("1", "0.5", "3")

    processing.baseline_on.setChecked(False)  # an unrelated edit keeps the rest
    assert c.processing.energy_cut == (200.0, 800.0) and c.processing.baseline is None
    assert c.processing.sample_field == FieldRange(1.0, 0.5, 3.0)
    assert c.processing.sg_window == 15
    c.set_processing(energy_cut=None, custom_field=False)
    assert not processing.cut_on.isChecked() and not sample.custom_field()
    processing.cut_on.setChecked(True)  # the typed ends come back
    assert c.processing.energy_cut == pytest.approx((200.0, 800.0))
