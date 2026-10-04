"""The Library panel: maps on the plot together, the Combine preview and the product."""

import numpy as np
import pytest
from PySide6.QtCore import QSettings, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QFileDialog

import gui_helpers
from gui_helpers import (
    click_map,
    design_width,
    energy_label,
    infobar_text,
    load_sweep,
    process,
    save_to,
    set_unit,
    shown_image,
    widest_parts,
)
from mag_opt_detective.core.pipeline import PlotKind
from mag_opt_detective.core.spectra import FieldMap, load_tsv, save_tsv
from mag_opt_detective.core.units import Unit, to_cm1
from mag_opt_detective.gui import plot_panel
from mag_opt_detective.gui.controller import SHOWS_MAPS, SHOWS_PROCESS, SHOWS_PRODUCT
from mag_opt_detective.gui.main_window import MainWindow
from mag_opt_detective.gui.panels import library

window, errors = gui_helpers.window, gui_helpers.errors  # shared fixtures

FIELDS = np.array([0.5, 1.0, 1.5, 2.0])


def rows(window):
    """The Library panel's rows, in library order."""
    panel = window.panels["library"]
    return [panel.rows[entry.key] for entry in window.controller.library]


def synthetic(lo, hi, step, fields=FIELDS, level=1.0):
    """A map of *level* + B / 100 over *lo* - *hi* cm^-1."""
    energy = np.arange(lo, hi + step / 2, step)
    return FieldMap(energy, fields, np.full((energy.size, fields.size), level) + fields / 100)


@pytest.fixture
def two_ranges(window, sweep):
    """The processed sweep in the library, then a FIR (50 - 300) and a MIR map (200 - 1000
    cm^-1); none is ticked."""
    load_sweep(window, sweep)
    process(window)
    c = window.controller
    c.add_processed()
    c.add_map(synthetic(50, 300, 1.0), "FIR")
    c.add_map(synthetic(200, 1000, 4.0, level=1.02), "MIR")
    return window


def tick(window, *names):
    c = window.controller
    for name in names:
        entry = next(e for e in c.library if e.name == name)
        window.panels["library"].rows[entry.key].use.setChecked(True)


# ---------------------------------------------------------------------- old flows
def test_export_load_save_and_merge(window, sweep, tmp_path, monkeypatch, errors):
    load_sweep(window, sweep)
    set_unit(window, "meV")
    process(window)
    c = window.controller
    assert c.result.ratio.unit is Unit.CM1  # processed in cm-1, shown in meV
    assert energy_label(window) == "Energy (meV)"
    save_to(monkeypatch, tmp_path / "S1.csv")
    window.commands["export_table"].trigger()
    exported = tmp_path / "S1_Ratio.csv"
    assert exported.read_text(encoding="utf-8").startswith(
        "Energy (meV)\t0.50T"
    )  # unit, then the fields
    fmap = load_tsv(exported)
    assert fmap.unit is Unit.MEV
    np.testing.assert_allclose(fmap.energy, sweep["x"] / 8.0656)
    np.testing.assert_allclose(fmap.values, c.result.ratio.values, rtol=1e-11)

    library.load_tables(window, [str(exported)])
    library.add_processed(window)
    first, second = c.library
    assert first.fmap.unit is second.fmap.unit is Unit.CM1
    np.testing.assert_allclose(first.fmap.energy, sweep["x"], rtol=1e-11)
    assert first.name == "S1_Ratio.csv" and second.name == "Sample_4p2K_Sam1"
    row0, row1 = rows(window)
    row0.use.setChecked(True)
    row1.use.setChecked(True)
    row0.e_max.setText("60")  # meV, the display unit
    row1.e_min.setText("60")
    library.refresh_preview(window)  # (it follows edits after a pause)
    window.panels["library"].create_button.click()  # by energy, the default
    assert not errors
    merged = c.result.ratio
    np.testing.assert_allclose(merged.energy[[0, -1]], sweep["x"][[0, -1]], rtol=1e-11)
    np.testing.assert_allclose(np.diff(merged.energy), np.diff(merged.energy)[0])
    assert energy_label(window) == "Energy (meV)"

    window.panels["library"].full_energy.setChecked(False)
    library.plot_entry(window, first.key)
    assert c.result.ratio.energy.max() <= to_cm1(60, Unit.MEV)
    assert c.result.ratio.to_unit(Unit.MEV).energy.max() <= 60
    library.plot_entry(window, second.key)
    assert c.result.ratio.to_unit(Unit.MEV).energy.min() >= 60
    assert not errors


def test_maps_from_different_units_merge(window, sweep, tmp_path, monkeypatch, errors):
    """Library maps keep cm-1, so maps saved while another unit was shown still combine."""
    load_sweep(window, sweep)
    c = window.controller
    set_unit(window, "meV")
    process(window)
    library.add_processed(window)
    save_to(monkeypatch, tmp_path / "S1.csv")
    window.commands["export_table"].trigger()  # an meV table

    set_unit(window, "THz")
    library.load_tables(window, [str(tmp_path / "S1_Ratio.csv")])
    process(window)
    library.add_processed(window)
    for entry in c.library[1:]:
        np.testing.assert_allclose(entry.fmap.energy, c.library[0].fmap.energy, rtol=1e-11)
    row0, row1, row2 = rows(window)
    row0.use.setChecked(True)
    row1.use.setChecked(True)
    row0.e_max.setText("15")  # THz, about 500 cm-1
    row1.e_min.setText("15")
    library.make_product(window, "energy")
    assert not errors
    assert c.unit is Unit.THZ
    assert energy_label(window) == "Energy (THz)"
    shown = c.result.get(PlotKind.RATIO, unit=Unit.THZ)
    np.testing.assert_allclose(shown_image(window), shown.values)
    np.testing.assert_allclose(c.result.ratio.energy[[0, -1]], [100, 1000], rtol=1e-11)

    row2.use.setChecked(True)
    for row in (row0, row1):
        row.e_min.setText("")
        row.e_max.setText("")
    library.make_product(window, "average")
    library.make_product(window, "field")
    assert not errors
    np.testing.assert_allclose(c.result.ratio.values, c.library[2].fmap.values, rtol=1e-9)


def test_energy_limits_follow_the_unit(window, sweep, errors):
    load_sweep(window, sweep)
    process(window)
    library.add_processed(window)
    entry = window.controller.library[0]
    (row,) = rows(window)
    set_unit(window, "meV")
    row.e_min.setText("30")
    row.e_max.setText("60")
    lo, hi = entry.energy_cut
    assert (lo, hi) == (30 * 8.0656, 60 * 8.0656)
    set_unit(window, "cm-1")
    assert (row.e_min.text(), row.e_max.text()) == ("241.968", "483.936")
    assert entry.energy_cut == (lo, hi)  # kept in cm-1: no rounding
    set_unit(window, "meV")
    assert (row.e_min.text(), row.e_max.text()) == ("30", "60")

    row.e_min.setText("200")  # meV: beyond the data (12.4 - 124 meV)
    row.e_max.setText("300")
    window.panels["library"].full_energy.setChecked(False)
    library.plot_entry(window, entry.key)
    assert "Sample_4p2K_Sam1: E range 200 – 300 meV contains no data" in errors[-1]
    assert window.infobar.action_button.text() == "Open Library"
    assert not entry.used  # it could not be shown, so it is not ticked


def test_merge_by_field_uses_ticked_maps(window, sweep, errors):
    load_sweep(window, sweep)
    process(window)
    for _ in range(3):
        library.add_processed(window)
    c = window.controller
    assert [e.name for e in c.library] == [
        "Sample_4p2K_Sam1",
        "Sample_4p2K_Sam1 (2)",
        "Sample_4p2K_Sam1 (3)",
    ]
    assert c.ticked() == []  # added, not plotted
    row0, row1, _row2 = rows(window)
    row0.use.setChecked(True)
    row1.use.setChecked(True)
    row0.b_max.setText("1.0")
    row1.b_min.setText("1.5")
    library.make_product(window, "field")
    assert not errors
    np.testing.assert_allclose(c.result.ratio.field, sweep["fields"])

    row0.use.setChecked(False)
    library.refresh_preview(window)
    assert not window.panels["library"].create_button.isEnabled()
    library.make_product(window, "field")
    assert errors and "tick at least two maps" in errors[-1]


def test_average_maps(window, sweep, errors):
    load_sweep(window, sweep)
    process(window)
    library.add_processed(window)
    library.add_processed(window)
    for row in rows(window):
        row.use.setChecked(True)
    window.panels["library"].method.set_value("average")
    window.panels["library"].create_button.click()
    assert not errors
    np.testing.assert_allclose(
        window.controller.result.ratio.values, window.controller.library[0].fmap.values
    )


def test_a_new_result_closes_the_error_bar(window, sweep, errors):
    load_sweep(window, sweep)
    process(window)
    library.add_processed(window)
    library.add_processed(window)
    row0, row1 = rows(window)
    row1.use.setChecked(True)
    library.make_product(window, "average")
    assert "tick at least two maps" in infobar_text(window).lower()
    row0.use.setChecked(True)  # shows it: a new result
    assert window.infobar.isHidden()
    library.make_product(window, "average")
    assert window.infobar.isHidden()
    assert len(errors) == 1


# ---------------------------------------------------------------------- plotting
def test_a_ticked_map_is_shown_and_unticked_it_leaves(two_ranges, errors):
    w, c = two_ranges, two_ranges.controller
    processed = c.result
    assert c.showing() == SHOWS_PROCESS
    tick(w, "FIR")
    assert c.showing() == SHOWS_MAPS and c.result_name() == "FIR"
    np.testing.assert_allclose(c.result.ratio.values, c.library[1].fmap.values)
    np.testing.assert_allclose(shown_image(w), c.library[1].fmap.values)
    assert "Showing a library map" in w.state_text()
    assert c.processed_result() is processed  # kept while it is hidden
    w.panels["library"].rows[c.library[1].key].use.setChecked(False)
    assert c.showing() == SHOWS_PROCESS and c.result is processed  # not processed again
    assert w.plots.map.overlay_maps() == []
    assert not errors


def test_ticked_maps_are_drawn_together_with_an_opacity(two_ranges, qtbot, errors):
    w, c = two_ranges, two_ranges.controller
    panel = w.panels["library"]
    tick(w, "MIR", "FIR")  # FIR ticked last: on top
    assert [e.name for e in c.plotted_entries()] == ["MIR", "FIR"]
    assert c.result_name() == "FIR"
    (below,) = w.plots.map.overlay_maps()
    np.testing.assert_allclose(below.energy, c.library[2].fmap.energy)
    drawn = w.plots.map.plot.vb.childrenBounds()  # what the auto range fits
    assert drawn[1][0] <= 50 and drawn[1][1] >= 1000
    assert w.plots.map.image.opacity() == pytest.approx(0.5)  # above the bottom map
    assert all(image.levels is not None for image in w.plots.map.image.followers)
    np.testing.assert_allclose(w.plots.map.image.followers[1].levels, w.plots.map.image.levels)
    assert panel.opacity.isEnabled()
    panel.opacity.slider.setValue(80)
    assert c.overlay_opacity == pytest.approx(0.8)
    assert w.plots.map.image.opacity() == pytest.approx(0.8)
    assert w.plots.map.drawn_value_at(1.0, 800.0) == pytest.approx(1.02 + 0.01)  # MIR only

    library.plot_entry(w, c.library[2].key)  # MIR on top
    assert [e.name for e in c.plotted_entries()] == ["FIR", "MIR"]
    assert c.result_name() == "MIR" and w.plots.map.overlay_maps()[0].energy.max() == 300

    c.set_selection(order=1)  # the maps below show the same derivative
    np.testing.assert_allclose(w.plots.map.overlay_maps()[0].values, 0.0, atol=1e-12)
    c.set_selection(order=0)

    w.panels["library"].full_energy.setChecked(False)  # cut to the limits
    c.update_entry(c.library[1], energy_cut=(None, 250.0))
    library.apply_limits(w)  # (the panel does it once typing pauses)
    assert w.plots.map.overlay_maps()[0].energy.max() == 250
    assert not errors


def test_the_plotted_maps_are_marked(two_ranges, errors):
    w, c = two_ranges, two_ranges.controller
    panel = w.panels["library"]
    fir, mir = c.library[1], c.library[2]
    tick(w, "MIR")
    assert panel.rows[mir.key].plot_state() == "top"
    assert panel.rows[mir.key].tag.text() == "On plot"
    tick(w, "FIR")
    assert panel.rows[fir.key].plot_state() == "top" and panel.rows[fir.key].tag.text() == "Top"
    assert panel.rows[mir.key].plot_state() == "plotted" and panel.rows[mir.key].tag.isHidden()
    assert panel.rows[c.library[0].key].plot_state() == ""
    head = w.plot_area.description.text()
    assert head == "R(B)/R(0) · FIR over MIR"
    assert panel.plot_chip.text() == "FIR over MIR" and not panel.plot_chip.isHidden()
    w.plot_area.set_current_view("stacked")  # the stacked plot shows the top map only
    assert w.plot_area.description.text() == "R(B)/R(0) · FIR"
    w.plot_area.set_current_view("map")

    process(w)  # the processed sweep: the ticks stay, the rows are not on the plot
    assert c.showing() == SHOWS_PROCESS and fir.used and mir.used
    assert all(row.plot_state() == "" for row in panel.rows.values())
    assert not panel.show_ticked_link.isHidden() and panel.plot_chip.isHidden()
    assert not panel.sweep.tag.isHidden()  # the sweep is on the plot
    panel.show_ticked_link.click()
    assert c.showing() == SHOWS_MAPS and panel.rows[fir.key].plot_state() == "top"
    assert panel.sweep.tag.isHidden() and not panel.sweep.show_link.isHidden()
    panel.sweep.show_link.click()
    assert c.showing() == SHOWS_PROCESS
    assert not errors


def test_a_click_on_a_row_brings_its_map_on_top(two_ranges, qtbot, errors):
    w, c = two_ranges, two_ranges.controller
    panel = w.panels["library"]
    mir = c.library[2]
    qtbot.mouseClick(panel.rows[mir.key].name_label, Qt.MouseButton.LeftButton)
    assert mir.used and c.result_name() == "MIR"  # ticked by the click
    tick(w, "FIR")
    qtbot.mouseClick(panel.rows[mir.key].name_label, Qt.MouseButton.LeftButton)
    assert [e.name for e in c.plotted_entries()] == ["FIR", "MIR"]
    assert not errors


def test_the_window_still_fits_1100_px(two_ranges, qtbot):
    w, c = two_ranges, two_ranges.controller
    width = design_width(qtbot)
    long = "A_very_long_library_map_name_from_a_long_sweep_" * 3
    c.add_map(synthetic(50, 300, 1.0, fields=FIELDS[:2]), long)
    for entry in c.library[1:]:
        c.update_entry(entry, used=True)
    assert long in library.refresh_preview(w).problem  # named in the preview
    assert w.minimumSizeHint().width() <= width, widest_parts(w)
    c.update_entry(c.library[-1], used=False)
    library.make_product(w, "field")
    c.product.name = long
    c.save_product()
    w.resize(width, 800)
    w.show()
    w.show_panel("library")
    assert w.minimumSizeHint().width() <= width, widest_parts(w)
    panel = w.panels["library"]
    assert not panel.show_ticked_link.isHidden()  # the product shows: the ticked maps wait
    QTest.qWait(50)  # laid out
    assert panel.scroll.horizontalScrollBar().maximum() == 0  # nothing reaches sideways


# ---------------------------------------------------------------------- combining
def test_the_preview_shows_the_result_or_why_not(two_ranges, errors):
    w = two_ranges
    panel = w.panels["library"]
    preview = library.refresh_preview(w)
    assert preview.fmap is None and not panel.create_button.isEnabled()
    assert panel.combine.preview.text() == "Tick at least two maps to merge or average them"
    tick(w, "FIR", "MIR")
    preview = library.refresh_preview(w)
    assert panel.combine.title.text() == "COMBINE 2 TICKED MAPS"
    assert panel.create_button.isEnabled() and preview.fmap is not None
    text = panel.combine.preview.text()
    assert text.splitlines() == [
        "Result: 4 fields · 0.5 – 2 T · 50 – 1000 cm⁻¹ · 389 energies",  # every measured one
        "Overlap 200 – 300 cm⁻¹ joined at 250 cm⁻¹",
    ]
    set_unit(w, "meV")
    library.refresh_preview(w)
    assert "joined at 31 meV" in panel.combine.preview.text()
    set_unit(w, "cm-1")

    panel.method.set_value("average")  # the energies both maps have
    assert panel.combine.preview.text().splitlines() == [
        "Result: 4 fields · 0.5 – 2 T · 200 – 300 cm⁻¹ · 101 energies",
        "Cut to the energies all maps share, 200 – 300 cm⁻¹",
    ]
    panel.method.set_value("field")
    assert panel.combine.preview.text().startswith("Result: 4 fields")
    assert "4 fields in more than one map averaged" in panel.combine.preview.text()
    assert not errors


@pytest.mark.parametrize(
    ("method", "maps", "message"),
    [
        (
            "energy",
            [synthetic(50, 300, 1.0), synthetic(400, 900, 1.0, fields=FIELDS[:2])],
            "merging by energy needs the same fields in every map (A 4 fields, 0.5 – 2 T; "
            "B 2 fields, 0.5 – 1 T): cut them to the same fields with B limits, or merge by "
            "field",
        ),
        (
            "average",
            [synthetic(50, 300, 1.0), synthetic(50, 300, 1.0, fields=FIELDS + 1)],
            "averaging needs the same fields in every map",
        ),
        (
            "energy",
            [synthetic(50, 900, 1.0), synthetic(200, 300, 1.0)],
            "B (200 – 300 cm⁻¹) lies inside the energy range of A: give them E limits that "
            "meet, or average repeated measurements",
        ),
        (
            "field",
            [synthetic(50, 300, 1.0), synthetic(400, 900, 1.0, fields=FIELDS + 2)],
            "the maps share no energy range (A 50 – 300 cm⁻¹; B 400 – 900 cm⁻¹)",
        ),
    ],
)
def test_combining_explains_why_it_cannot(window, errors, method, maps, message):
    c = window.controller
    for name, fmap in zip("AB", maps, strict=True):
        c.update_entry(c.add_map(fmap, name), used=True)
    preview = c.combine_preview(method)
    assert preview.fmap is None and message in preview.problem
    library.make_product(window, method)
    assert message in errors[-1] and c.product is None


def test_maps_of_different_kinds_are_not_combined(window, tmp_path, errors):
    c = window.controller
    data = tmp_path / "S1_Data.csv"
    save_tsv(synthetic(50, 300, 1.0), data)
    ratio = tmp_path / "S1_Ratio_1stDer.csv"
    save_tsv(synthetic(300, 900, 1.0), ratio)
    library.load_tables(window, [str(data), str(ratio)])
    assert [e.kind for e in c.library] == ["Data", "R(B)/R(0) · 1st derivative"]
    assert rows(window)[0].meta_label.text().startswith("Data · 4 fields")
    for entry in c.library:
        c.update_entry(entry, used=True)
    problem = c.combine_preview("energy").problem
    assert problem.startswith("combine maps of one kind: S1_Data.csv is Data, S1_Ratio_1stDer")
    assert not errors


# ---------------------------------------------------------------------- the product
def test_the_product_box(two_ranges, tmp_path, monkeypatch, errors):
    w, c = two_ranges, two_ranges.controller
    panel = w.panels["library"]
    box = panel.product_box
    assert box.isHidden()
    tick(w, "FIR", "MIR")
    c.update_entry(c.library[1], energy_cut=(None, 240.0))
    library.refresh_preview(w)
    panel.create_button.click()
    product = c.product
    assert c.showing() == SHOWS_PRODUCT and not box.isHidden()
    assert product.name == "Merged by energy: FIR + MIR"
    assert box.name_edit.text() == product.name
    assert box.method_label.text() == "Merge by energy of 2 maps:"
    assert box.part_lines() == ["• FIR (E ≤ 240 cm⁻¹)", "• MIR"]
    assert box.size_label.text() == "4 fields · 0.5 – 2 T · 50 – 1000 cm⁻¹ · 366 energies"
    assert box.notes.text() == "Overlap 200 – 240 cm⁻¹ joined at 220 cm⁻¹"
    assert not box.tag.isHidden() and not box.plot_button.isEnabled()
    assert w.plot_area.description.text() == "R(B)/R(0) · Merged by energy: FIR + MIR"
    assert all(row.plot_state() == "" for row in panel.rows.values())  # the product shows
    fir_part = c.result.ratio.values[c.result.ratio.energy <= 220]
    np.testing.assert_allclose(fir_part, np.broadcast_to(1 + FIELDS / 100, fir_part.shape))

    box.name_edit.setText("FIR+MIR sample 1")
    box.name_edit.editingFinished.emit()
    assert product.name == "FIR+MIR sample 1" and c.result_name() == "FIR+MIR sample 1"
    assert w.windowTitle().endswith("FIR+MIR sample 1")

    save_to(monkeypatch, tmp_path / "product")
    box.table_button.click()
    table = load_tsv(tmp_path / "product.csv")
    np.testing.assert_allclose(table.values, product.fmap.values)
    note = (tmp_path / "product_provenance.txt").read_text(encoding="utf-8").splitlines()
    assert note[0] == "FIR+MIR sample 1" and note[1].startswith("Merge by energy of 2 maps, ")
    assert note[2:4] == ["- FIR (R(B)/R(0), E ≤ 240 cm⁻¹)", "- MIR (R(B)/R(0))"]
    assert note[-2:] == [
        "Overlap 200 – 240 cm⁻¹ joined at 220 cm⁻¹",
        "4 fields · 0.5 – 2 T · 50 – 1000 cm⁻¹ · 366 energies",
    ]

    box.save_button.click()
    saved = c.library[-1]
    assert saved.name == "FIR+MIR sample 1" and saved.provenance is product.provenance
    assert not saved.used and not box.save_button.isEnabled()
    assert box.save_button.text() == "Saved"
    assert box.save_button.toolTip() == "The product is in the library as FIR+MIR sample 1"
    assert "Merge by energy of 2 maps" in panel.rows[saved.key].name_label.toolTip()

    library.plot_entry(w, c.library[1].key)  # a map: the product leaves the plot
    assert box.tag.isHidden() and box.plot_button.isEnabled()
    box.plot_button.click()
    assert c.showing() == SHOWS_PRODUCT
    box.discard_button.click()
    assert c.product is None and box.isHidden()
    assert c.showing() == SHOWS_MAPS and c.result_name() == "FIR"  # the ticked maps return
    assert not errors


def test_the_product_opens_the_figure_export(two_ranges, monkeypatch, errors):
    w, c = two_ranges, two_ranges.controller
    tick(w, "FIR", "MIR")
    library.make_product(w, "energy")
    library.plot_entry(w, c.library[1].key)
    opened = []
    monkeypatch.setattr(w.commands["export_figure"], "trigger", lambda: opened.append(c.result))
    w.panels["library"].product_box.figure_button.click()
    assert c.showing() == SHOWS_PRODUCT and opened == [c.result]
    state = c.figure_state()
    np.testing.assert_allclose(state.fmap.values, c.product.fmap.values)
    assert not errors


# ---------------------------------------------------------------------- the sweep
def test_the_processed_sweep_goes_into_the_library_in_one_step(window, sweep, errors):
    panel = window.panels["library"]
    c = window.controller
    action = next(a for a in window.actions() if a.text() == "Add processed map to library")
    assert panel.sweep.isHidden() and not action.isEnabled()
    load_sweep(window, sweep)
    c.set_processing(baseline=(200.0, 300.0))
    process(window)
    assert not panel.sweep.isHidden() and action.isEnabled()
    assert panel.sweep.name_label.text() == "Sample_4p2K_Sam1"
    action.trigger()  # in the File menu
    (entry,) = c.library
    assert entry.fmap is c.result.ratio and c.processed_in_library()
    assert not panel.sweep.added.isHidden() and panel.save_button.isHidden()
    assert not action.isEnabled()
    lines = entry.provenance.lines(Unit.CM1)
    assert lines[0].startswith("Processed sweep Sample_4p2K_Sam1, ")
    assert lines[1:] == [
        "4 in-field and 2 zero-field files",
        "Reference: none",
        "Baseline region 200 – 300 cm⁻¹",
    ]
    assert panel.rows[entry.key].name_label.toolTip().startswith("Processed sweep")
    c.set_processing(baseline=None)
    process(window)  # a new map: it can go in again
    assert not panel.save_button.isHidden() and not c.processed_in_library()
    assert not errors


def test_picking_and_exporting_work_on_a_library_map(two_ranges, tmp_path, monkeypatch, errors):
    w, c = two_ranges, two_ranges.controller
    c.add_map(synthetic(50, 300, 1.0, fields=np.array([0.75, 1.25])), "Other fields")
    tick(w, "MIR", "Other fields")
    w.tools.set_active("pick")
    click_map(w, 1.2, 120.0)  # on the top map: its field
    assert c.points.points(c.curve)[0].tolist() == [1.25]
    click_map(w, 1.8, 600.0)  # beyond the top map (on the MIR below): not picked
    assert "outside the map" in errors[-1]
    errors.clear()
    save_to(monkeypatch, tmp_path / "lib.csv")
    w.commands["export_table"].trigger()
    table = load_tsv(tmp_path / "lib_Ratio.csv")
    np.testing.assert_allclose(table.field, [0.75, 1.25])
    assert c.figure_state().fmap.field.tolist() == [0.75, 1.25]
    assert not errors


def test_library_settings_round_trip(qtbot, tmp_path):
    ini = str(tmp_path / "settings.ini")

    def make():
        w = MainWindow(settings=QSettings(ini, QSettings.Format.IniFormat))
        qtbot.addWidget(w)
        return w

    w = make()
    panel = w.panels["library"]
    panel.opacity.slider.setValue(70)
    panel.method.set_value("field")
    panel.full_energy.setChecked(False)
    assert w.controller.overlay_opacity == pytest.approx(0.7)
    assert not w.controller.full_range
    w.close()

    w2 = make()
    panel = w2.panels["library"]
    assert panel.opacity.slider.value() == 70 and w2.controller.overlay_opacity == pytest.approx(
        0.7
    )
    assert panel.method.value() == "field"
    assert not panel.full_energy.isChecked() and not w2.controller.full_range


def test_the_plot_ranges_fit_every_ticked_map(two_ranges, errors):
    """Auto ranges (and Fit to data) cover the maps below the top one; they stay Auto."""
    w, c = two_ranges, two_ranges.controller
    view = w.inspector["view"].body_layout().itemAt(0).widget()
    tick(w, "MIR", "FIR")  # FIR (50 - 300) on top of MIR (200 - 1000 cm^-1)
    assert c.view.energy_range is None and c.view.field_range is None
    assert w.plots.map.plot.vb.viewRange()[1] == pytest.approx([50.0, 1000.0])
    assert view.energy.range() == pytest.approx((50.0, 1000.0))
    c.set_ranges(energy_range=(100.0, 400.0))
    w.plot_area.set_current_view("map")
    plot_panel.fit_to_data(w)  # Fit to data
    assert c.view.energy_range is None
    assert w.plots.map.plot.vb.viewRange()[1] == pytest.approx([50.0, 1000.0])
    w.plot_area.set_current_view("stacked")  # the stacked plot shows the top map only
    assert w.plots.stacked.plot.vb.viewRange()[0] == pytest.approx([50.0, 300.0])
    w.plot_area.set_current_view("map")
    w.panels["library"].rows[c.library[2].key].use.setChecked(False)  # MIR off: FIR alone
    assert w.plots.map.plot.vb.viewRange()[1] == pytest.approx([50.0, 300.0])
    assert not errors


def test_ticked_maps_restore_auto_ranges(qtbot, tmp_path):
    ini = str(tmp_path / "settings.ini")

    def make():
        w = MainWindow(settings=QSettings(ini, QSettings.Format.IniFormat))
        qtbot.addWidget(w)
        return w

    w = make()
    c = w.controller
    for name, fmap in (("FIR", synthetic(50, 300, 1.0)), ("MIR", synthetic(200, 1000, 4.0))):
        c.update_entry(c.add_map(fmap, name), used=True)
    assert c.view.energy_range is None
    w.close()
    w2 = make()
    assert w2.controller.view.energy_range is None and w2.controller.view.field_range is None


# ---------------------------------------------------------------------- fix round
def test_limits_can_be_typed_for_the_map_on_top(two_ranges, qtbot, errors):
    """Typing a limit key by key never fails: '4' (on the way to 450) holds no data, so it is
    only marked; the map is cut once typing pauses."""
    w, c = two_ranges, two_ranges.controller
    panel = w.panels["library"]
    panel.full_energy.setChecked(False)
    tick(w, "MIR", "FIR")
    fir = c.library[1]
    row = panel.rows[fir.key]
    row.set_expanded(True)
    row.e_max.setFocus()
    QTest.keyClick(row.e_max, "4")
    assert fir.energy_cut == (None, 4.0) and row.fields[row.e_max].is_invalid()
    library.apply_limits(w)  # not drawn while the top map's limits cannot be used
    assert c.result.ratio.energy.max() == 300 and not errors
    QTest.keyClicks(row.e_max, "50")
    assert fir.energy_cut == (None, 450.0) and not row.fields[row.e_max].is_invalid()
    assert row.e_max.text() == "450"
    qtbot.waitUntil(lambda: c.result.ratio.energy.max() == 300.0, timeout=100)  # (FIR ends there)
    QTest.keyClicks(row.e_max, "\b\b")  # "4": invalid again
    assert row.fields[row.e_max].is_invalid()
    QTest.keyClicks(row.e_max, "\b250")
    assert fir.energy_cut == (None, 250.0)
    qtbot.waitUntil(lambda: c.result.ratio.energy.max() == 250.0, timeout=2000)
    assert not errors


def test_a_ticked_map_not_drawn_says_why(two_ranges, tmp_path, errors):
    w, c = two_ranges, two_ranges.controller
    panel = w.panels["library"]
    data = c.add_map(synthetic(200, 1000, 4.0, level=1000.0), "Raw", kind="Data")
    tick(w, "Raw", "FIR")  # a Data map under a ratio map would saturate
    row = panel.rows[data.key]
    assert row.problem.text() == "Not drawn: it is Data and the top map R(B)/R(0)"
    assert not row.problem.isHidden() and row.plot_state() == ""
    assert w.plots.map.overlay_maps() == [] and not panel.opacity.isEnabled()
    row.use.setChecked(False)  # FIR alone
    c.set_processing(baseline=(500.0, 600.0))  # applies to the next map shown
    tick(w, "MIR")  # on top, with the region; FIR below has no data there
    assert panel.rows[c.library[1].key].problem.text() == (
        "Not drawn: the baseline region holds none of its data"
    )
    assert not errors


def test_process_and_removals_leave_a_consistent_state(two_ranges, errors):
    w, c = two_ranges, two_ranges.controller
    tick(w, "MIR", "FIR")
    process(w)  # a new sweep: the one hidden before is out of date
    assert c.processed_result() is c.result
    library.show_ticked(w)
    assert c.processed_result() is not c.result and c.processed_result() is not None
    low = c.add_map(synthetic(50, 300, 1.0), "Low")
    c.update_entry(c.library[2], used=False)
    c.update_entry(low, used=True)  # FIR below Low on top
    c.set_processing(baseline=(400.0, 500.0))  # FIR, the next on top, misses it
    library.remove_entry(w, low.key)
    assert "baseline region" in errors[-1]
    assert c.showing() == SHOWS_PROCESS and low not in c.library  # not the removed map
    assert c.library[1].used  # the tick stays
    errors.clear()

    c.set_processing(baseline=None)
    tick(w, "MIR")
    library.plot_entry(w, c.library[1].key)  # FIR on top
    library.make_product(w, "energy")
    c.set_processing(baseline=(400.0, 500.0))  # FIR, on top of the ticked maps, misses it
    w.panels["library"].product_box.discard_button.click()
    assert "baseline region" in errors[-1] and c.product is None
    assert c.showing() == SHOWS_PROCESS


def test_a_baseline_region_in_a_gap_is_refused(window, errors):
    c = window.controller
    for name, fmap in (("A", synthetic(50, 300, 1.0)), ("B", synthetic(500, 900, 2.0))):
        c.update_entry(c.add_map(fmap, name), used=True)
    library.make_product(window, "energy")
    gap = c.product.fmap.energy[np.isnan(c.product.fmap.values[:, 0])]
    np.testing.assert_allclose(gap, [301.0, 498.0])  # a step beyond each side
    library.save_product(window)
    c.set_processing(baseline=(300.5, 499.0))  # only the empty samples of the gap
    library.plot_entry(window, c.library[-1].key)
    assert "baseline region 300.5 – 499 cm-1 contains no data (it lies in a gap" in errors[-1]


def test_the_product_table_is_what_the_plot_shows(two_ranges, tmp_path, monkeypatch, errors):
    w, c = two_ranges, two_ranges.controller
    tick(w, "FIR", "MIR")
    library.make_product(w, "energy")
    c.set_processing(baseline=(100.0, 200.0))
    library.plot_product(w)  # shown (and exported) with the region
    library.plot_entry(w, c.library[1].key)
    library.plot_product(w)
    c.product.name = "FIR+MIR: sample 1"
    asked = []

    def dialog(parent, title, suggested, filter):
        asked.append(suggested)
        return str(tmp_path / "out.csv"), ""

    monkeypatch.setattr(QFileDialog, "getSaveFileName", dialog)
    w.panels["library"].product_box.table_button.click()
    assert asked[0].endswith("FIR+MIR_sample_1.csv")  # the product's name, safe for files
    table = load_tsv(tmp_path / "out.csv")
    np.testing.assert_allclose(table.values, c.result.ratio.values)  # with the baseline
    assert not np.allclose(table.values, c.product.fmap.values)
    note = (tmp_path / "out_provenance.txt").read_text(encoding="utf-8").splitlines()
    assert "Baseline region 100 – 200 cm⁻¹, as plotted" in note
    assert not errors


def test_the_combine_footer_folds_and_the_options_stay_in_view(two_ranges, qtbot):
    w, c = two_ranges, two_ranges.controller
    panel = w.panels["library"]
    combine = panel.combine
    library.refresh_preview(w)
    assert not combine.need.isHidden() and combine.body.isHidden()  # one line only
    tick(w, "FIR", "MIR")
    library.refresh_preview(w)
    assert combine.need.isHidden() and not combine.body.isHidden()
    combine.toggle.click()
    assert combine.body.isHidden() and not combine.is_open()
    for _ in range(4):
        c.add_map(synthetic(50, 300, 1.0), "More")
    w.show_panel("library")
    qtbot.waitUntil(lambda: panel.list_card.y() > 0)
    for option in (panel.opacity, panel.full_energy_row):  # above the list: never below it
        assert option.y() < panel.list_card.y()


def test_the_combine_footer_state_is_remembered(qtbot, tmp_path):
    ini = str(tmp_path / "settings.ini")

    def make():
        w = MainWindow(settings=QSettings(ini, QSettings.Format.IniFormat))
        qtbot.addWidget(w)
        return w

    w = make()
    w.panels["library"].combine.set_open(False)
    w.close()
    assert not make().panels["library"].combine.is_open()
