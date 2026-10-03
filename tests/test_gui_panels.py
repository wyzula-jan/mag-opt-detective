"""The left panels: Sample, Reference, Processing and Library."""

import numpy as np
import pytest
from PySide6.QtCore import QMimeData, QPointF, QSettings, Qt, QUrl
from PySide6.QtGui import QDropEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QFileDialog

import gui_helpers
from gui_helpers import load_sweep, process, save_to, set_unit
from helpers import sweep_name, write_text
from mag_opt_detective.core.pipeline import ReferenceMode
from mag_opt_detective.core.units import Unit
from mag_opt_detective.gui.controller import FieldRange, SweepFiles, common_prefix
from mag_opt_detective.gui.main_window import MainWindow
from mag_opt_detective.gui.panels import library
from mag_opt_detective.gui.panels.common import FileDrops
from mag_opt_detective.gui.panels.files import (
    FileTableModel,
    check_fields,
    is_zero_field,
    split_zero,
)
from mag_opt_detective.gui.panels.processing import APPLIED, CHANGED, VIEW_ENERGY_LINK
from mag_opt_detective.gui.panels.reference import HINTS

window, errors = gui_helpers.window, gui_helpers.errors  # shared fixtures

MEV = 8.0656


def open_many(monkeypatch, paths) -> None:
    monkeypatch.setattr(
        QFileDialog, "getOpenFileNames", lambda *a, **k: ([str(p) for p in paths], "")
    )


def drop(widget, paths) -> None:
    """Drop *paths* on *widget* (through its :class:`FileDrops` handler: offscreen Qt
    delivers no drop events without a drag in progress)."""
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(p)) for p in paths])
    event = QDropEvent(
        QPointF(5, 5),
        Qt.DropAction.CopyAction,
        mime,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    handler = widget.findChild(FileDrops, "", Qt.FindChildOption.FindDirectChildrenOnly)
    assert handler.eventFilter(widget, event) and event.isAccepted()


def gap_sweep(folder, fields):
    """Field files at *fields* (T), named like a real sweep."""
    x = np.linspace(100.0, 1000.0, 91)
    return [write_text(folder / sweep_name(b), x, 1 + 0.01 * b + 0 * x) for b in fields]


# ---------------------------------------------------------------------- file names
def test_zero_field_names():
    assert is_zero_field("S_a00p000T_a16p000T.txt") and is_zero_field("S_a00p000T.0")
    assert not is_zero_field("S_a00p250T.txt") and not is_zero_field("S_a16p000T_a00p000T")
    assert not is_zero_field("notes.txt")
    zero, field = split_zero(
        ["S_a01p000T.txt", "S_a00p000T_a16p000T.txt", "S_a00p500T.txt", "S_a00p000T_a00p000T.txt"]
    )
    assert zero == ["S_a00p000T_a00p000T.txt", "S_a00p000T_a16p000T.txt"]  # before, after
    assert field == ["S_a00p500T.txt", "S_a01p000T.txt"]


def test_common_prefix():
    names = ["Demo_4p2K_Sam1_a00p250T.txt", "Demo_4p2K_Sam1_a00p500T.txt"]
    assert common_prefix(names) == "Demo_4p2K_Sam1_"
    assert common_prefix(names[:1]) == "Demo_4p2K_Sam1_"
    assert common_prefix(["abc.txt", "abd.txt"]) == ""
    assert common_prefix([]) == ""
    assert common_prefix(["A_1.txt", "A_12.txt", "B_1.txt"]) == ""
    assert common_prefix(["Run 2_a.txt", "Run 2_b.txt", "Run 1_c.txt"]) == "Run "


@pytest.mark.parametrize(
    ("fields", "ok", "text"),
    [
        ([0.25, 0.5, 0.75, 1.0], True, "Even 0.25 T steps, no missing fields."),
        ([0.25, 0.5, 1.25, 1.5], False, "Missing 0.75 and 1 T (0.25 T steps)."),
        ([1, 2, 4], False, "Missing 3 T (1 T steps)."),
        ([0.5, 1.0, 1.0, 1.5], False, "Repeated field: 1 T."),
        ([0.25, 0.5, 0.85], False, "Uneven field steps, 0.25 to 0.35 T."),
        ([2.0], True, "One field, 2 T."),
        (
            [0.5, None, 1.5],
            False,
            "1 file name has no field (like …_a01p250T): use a custom range.",
        ),
    ],
)
def test_check_fields(fields, ok, text):
    check = check_fields(fields)
    assert (check.ok, check.text) == (ok, text)


def test_check_fields_lists_a_few_missing_fields():
    check = check_fields([0.25, 0.5, 3.0])
    assert check.text == "Missing 0.75, 1, 1.25, 1.5 T and 5 more (0.25 T steps)."
    assert check_fields([]) is None


def test_file_table_model(tmp_path):
    model = FileTableModel()
    names = ["Demo_S1_a00p500T.txt", "Demo_S1_a00p250T.txt", "Demo_S1_a01p000T.txt"]
    model.set_paths([str(tmp_path / n) for n in names])
    assert model.prefix() == "Demo_S1_"
    assert model.fields() == [0.25, 0.5, 1.0]
    shown = [[model.index(r, c).data() for c in range(2)] for r in range(3)]
    assert shown == [["0.25", "a00p250T.txt"], ["0.5", "a00p500T.txt"], ["1", "a01p000T.txt"]]
    assert model.index(0, 1).data(Qt.ItemDataRole.ToolTipRole) == str(tmp_path / names[1])
    assert model.headerData(0, Qt.Orientation.Horizontal) == "B (T)"
    model.remove_rows([1])
    assert model.fields() == [0.25, 1.0]
    model.set_paths([str(tmp_path / "Demo_S1_a00p250T.txt"), str(tmp_path / "Demo_S1_x.txt")])
    assert model.index(1, 0).data() == "–"


# ---------------------------------------------------------------------- sample
def test_file_table_shows_fields_prefix_and_gaps(window, tmp_path):
    field = gap_sweep(tmp_path, [0.5, 1.0, 2.0, 2.5])  # 1.5 T is missing
    box = window.panels["sample"].measurement
    box.field_list.set_paths(field)
    assert window.controller.processing.sample_files.field == tuple(map(str, field))
    assert box.prefix_label.text() == "Common prefix <b>Sample_4p2K_Sam1_</b>"
    assert box.gap_note.text() == "Missing 1.5 T (0.5 T steps)."
    assert box.gap_note.level() == "warn"
    assert box.field_head.count_label.text() == "4 files"

    table = box.field_list
    table.selectRow(0)
    QTest.keyClick(table, Qt.Key.Key_Delete)
    assert table.file_model.fields() == [1.0, 2.0, 2.5]
    table.selectRow(2)
    QTest.keyClick(table, Qt.Key.Key_Backspace)
    assert window.controller.processing.sample_files.field == tuple(map(str, field[1:3]))
    assert (box.gap_note.text(), box.gap_note.level()) == (
        "Even 1 T steps, no missing fields.",
        "ok",
    )
    table.set_paths(field[:1])  # one file: its name is still split, so the prefix shows
    assert not box.prefix_label.isHidden()
    assert box.prefix_label.text() == "Prefix <b>Sample_4p2K_Sam1_</b>"
    box.clear_button.click()
    assert table.count() == 0 and not box.field_drop.isHidden() and box.gap_note.isHidden()


def test_drops_put_zero_field_files_in_the_zero_list(window, sweep, tmp_path):
    box = window.panels["sample"].measurement
    drop(box.drop_zone, [*sweep["field"], *sweep["zero"]])
    files = window.controller.processing.sample_files
    assert files.zero == tuple(map(str, sweep["zero"]))
    assert files.field == tuple(map(str, sweep["field"]))
    assert box.zero_list.tags() == ["Before", "After"]
    assert box.zero_note.level() == "ok" and "drift is corrected" in box.zero_note.text()
    assert window.panel_pages["sample"].subtitle.text() == "4 spectra · 0.5 – 2 T · text export"

    drop(box, sweep["field"][:2])  # in-field files only: the zero list stays
    assert box.zero_list.count() == 2 and box.field_list.count() == 2
    drop(box.zero_list, sweep["zero"][:1])  # the zero list takes what is dropped on it
    assert box.zero_list.paths() == [str(sweep["zero"][0])]
    assert box.zero_list.tags() == [""] and "not corrected" in box.zero_note.text()

    folder = tmp_path / "sweep"
    folder.mkdir()
    for path in [*sweep["zero"], *sweep["field"]]:
        (folder / path.name).write_bytes(path.read_bytes())
    drop(box, [folder])  # a folder: its files, sorted into both lists
    assert box.zero_list.count() == 2 and box.field_list.count() == 4


def test_open_sweep_and_add_files(window, sweep, tmp_path, monkeypatch):
    box = window.panels["sample"].measurement
    open_many(monkeypatch, [*sweep["zero"], *sweep["field"][:2]])
    window.panels["reference"].measurement.open_sweep_dialog()  # the reference's own lists
    assert window.controller.processing.reference_files.zero == tuple(map(str, sweep["zero"]))
    window.show_panel("points")
    window.toolbar.open_button.click()
    assert window.current_panel() == "sample"
    assert box.zero_list.count() == 2 and box.field_list.count() == 2
    open_many(monkeypatch, sweep["field"][1:])
    box.add_field_button.click()
    assert box.field_paths() == [str(p) for p in sweep["field"]]
    open_many(monkeypatch, sweep["zero"][1:])
    box.add_zero_button.click()  # the zero list's own dialog replaces it
    assert box.zero_paths() == [str(sweep["zero"][1])]
    box.zero_list.remove(str(sweep["zero"][1]))
    assert not box.zero_drop.isHidden() and window.controller.processing.sample_files.zero == ()


def test_custom_field_range_shows_only_when_chosen(window, sweep, errors):
    load_sweep(window, sweep)
    sample = window.panels["sample"]
    c = window.controller
    assert sample.field_range.isHidden() and not sample.names_hint.isHidden()
    sample.field_source.button("custom").click()
    assert c.processing.custom_field
    assert not sample.field_range.isHidden() and sample.names_hint.isHidden()
    assert sample.measurement.gap_note.isHidden()
    note = sample.field_range.note
    assert (note.text(), note.level()) == ("64 values, but there are 4 files.", "warn")
    box = sample.field_range
    box.start.setText("0.5")
    box.step.setText("0.5")
    box.end.setText("2")
    assert (note.text(), note.level()) == ("4 values, matches the 4 files.", "ok")
    assert c.processing.sample_field == FieldRange(0.5, 0.5, 2.0)
    box.step.setText("0")
    assert note.text() == "Step must be a positive number." and box.fields[box.step].is_invalid()
    box.step.setText("1")
    box.end.setText("0")
    assert note.text() == "End must not be below start." and box.fields[box.end].is_invalid()
    assert not box.fields[box.step].is_invalid()
    box.end.setText("")
    assert note.text() == "Enter the start, step and end."

    c.set_processing(custom_field=False, sample_field=FieldRange(1.0, 1.0, 4.0))
    assert sample.field_source.value() == "names" and sample.field_range.isHidden()
    assert (box.start.text(), box.step.text(), box.end.text()) == ("1", "1", "4")
    sample.field_source.set_value("custom")
    process(window)
    assert not errors
    np.testing.assert_allclose(c.result.ratio.field, [1, 2, 3, 4])


# ---------------------------------------------------------------------- reference
def test_reference_modes_and_smoothing(window):
    ref = window.panels["reference"]
    c = window.controller
    assert ref.reference_mode() is ReferenceMode.NONE
    assert ref.files_block.isHidden() and not ref.smooth_block.isEnabled()
    assert ref.mode_hint.text() == HINTS[ReferenceMode.NONE]

    ref.mode.button("separate").click()
    assert c.processing.reference_mode is ReferenceMode.SEPARATE
    assert not ref.files_block.isHidden() and ref.field_block.isHidden()
    assert ref.smooth_block.isEnabled() and not ref.sg_window.isEnabled()
    ref.smooth.setChecked(True)
    assert c.processing.smooth and ref.sg_window.isEnabled() and ref.sg_poly.isEnabled()
    ref.sg_window.setValue(10)
    assert not ref.sg_note.isHidden() and "odd" in ref.sg_note.text()
    ref.sg_window.setValue(15)
    assert ref.sg_note.isHidden() and c.processing.sg_window == 15
    c.set_processing(custom_field=True)
    assert not ref.field_block.isHidden()  # the custom range, as for the sample

    ref.mode.button("self").click()
    assert ref.files_block.isHidden() and ref.smooth_block.isEnabled()
    assert ref.mode_hint.text() == HINTS[ReferenceMode.SELF]
    c.set_processing(reference_mode=ReferenceMode.NONE)
    assert ref.mode.value() == "none" and not ref.smooth_block.isEnabled()
    assert c.processing.smooth  # kept for when a reference is chosen again


def test_reference_mode_labels_shorten_when_narrow(window):
    ref = window.panels["reference"]
    ref.fit_width(1000)
    assert ref.mode.button("separate").text() == "Separate sweep"
    ref.fit_width(200)
    assert ref.mode.button("separate").text() == "Separate"
    assert ref.mode.button("self").text() == "Self"


# ---------------------------------------------------------------------- processing
def test_processing_fields_are_shown_in_the_unit_and_kept_in_cm1(window):
    panel = window.panels["processing"]
    c = window.controller
    set_unit(window, "meV")
    panel.cut_on.setChecked(True)
    panel.cut_lo.setText("20")
    panel.cut_hi.setText("")
    assert c.processing.energy_cut == (20 * MEV, None)  # open-ended
    assert [label.text() for label in panel.unit_labels] == ["meV"] * 4
    set_unit(window, "THz")
    assert panel.cut_lo.text() == "4.83601" and c.processing.energy_cut == (20 * MEV, None)

    panel.cut_lo.setText("30")
    panel.cut_hi.setText("10")
    assert not panel.cut_note.isHidden()
    assert panel.cut_note.text() == (
        "Energy window: the first value must be below the second; it is 30 – 10 THz."
    )
    assert panel.fields[panel.cut_lo].is_invalid() and panel.fields[panel.cut_hi].is_invalid()
    set_unit(window, "cm-1")
    assert "1000.69 – 333.564 cm⁻¹" in panel.cut_note.text()
    panel.cut_on.setChecked(False)
    assert panel.cut_note.isHidden() and not panel.fields[panel.cut_lo].isEnabled()

    panel.baseline_on.setChecked(True)
    panel.baseline_lo.setText("100")
    assert panel.baseline_note.text() == "Baseline: enter both limits."
    assert panel.fields[panel.baseline_hi].is_invalid()
    assert not panel.fields[panel.baseline_lo].is_invalid()


def test_guides_show_only_while_the_panel_is_open(window, sweep):
    load_sweep(window, sweep)
    process(window)
    set_unit(window, "meV")
    panel = window.panels["processing"]
    guides = window.plots.map.layer("guides")
    assert not guides.is_visible()
    window.show_panel("processing")
    assert guides.is_visible() and guides.curve_data() == []
    panel.cut_on.setChecked(True)
    panel.cut_lo.setText("20")
    panel.cut_hi.setText("100")
    ends = sorted(float(y[0]) for _x, y in guides.curve_data())
    assert ends == pytest.approx([20, 100])
    x, _y = guides.curve_data()[0]
    assert tuple(x) == (0.5, 2.0)  # across the map's fields
    panel.baseline_on.setChecked(True)
    panel.baseline_lo.setText("40")
    panel.baseline_hi.setText("50")
    assert panel.guides.band_region() == pytest.approx((40, 50))

    set_unit(window, "cm-1")
    ends = sorted(float(y[0]) for _x, y in guides.curve_data())
    assert ends == pytest.approx([20 * MEV, 100 * MEV])
    assert panel.guides.band_region() == pytest.approx((40 * MEV, 50 * MEV))
    window.show_panel("sample")
    assert not guides.is_visible() and panel.guides.band_region() is None
    window.show_panel("processing")
    assert guides.is_visible()
    window.side_panel.set_open(False, animate=False)
    assert not guides.is_visible()


def test_view_energy_link_opens_the_inspector(window):
    window.inspector_panel.set_open(False, animate=False)
    window.inspector["view"].set_expanded(False, animate=False)
    window.plot_area.set_current_view("reference")
    window.panels["processing"].view_energy_link.linkActivated.emit(VIEW_ENERGY_LINK)
    assert window.inspector_panel.is_open() and window.inspector["view"].is_expanded()
    assert window.plot_area.current_view() == "map"


def test_changed_since_the_last_run(window, sweep):
    subtitle = window.panel_pages["processing"].subtitle
    panel = window.panels["processing"]
    assert subtitle.text() == APPLIED
    load_sweep(window, sweep)
    process(window)
    panel.baseline_on.setChecked(True)
    assert subtitle.text() == CHANGED
    panel.baseline_on.setChecked(False)
    assert subtitle.text() == APPLIED
    window.controller.set_processing(energy_cut=(200.0, None))
    assert subtitle.text() == CHANGED and panel.cut_on.isChecked()
    process(window)
    assert subtitle.text() == APPLIED


# ---------------------------------------------------------------------- library
def test_library_list(window, sweep, tmp_path, monkeypatch, errors):
    panel = window.panels["library"]
    c = window.controller
    assert not panel.empty.isHidden() and panel.list_card.isHidden()
    assert panel.count_label.text() == "No maps in the library."
    assert not any(b.isEnabled() for b in panel.batch_buttons())

    load_sweep(window, sweep)
    process(window)
    panel.save_button.click()
    (entry,) = c.library
    row = panel.rows[entry.key]
    assert row.name_label.text() == "Sample_4p2K_Sam1"
    assert row.meta_label.text() == "R(B)/R(0) · 4 fields · 100 – 1000 cm⁻¹"
    assert panel.count_label.text() == "1 of 1 ticked. Tick at least two to merge or average."
    assert not panel.average_button.isEnabled()
    assert panel.average_button.toolTip().startswith("Tick at least two maps")
    set_unit(window, "meV")
    assert row.meta_label.text() == "R(B)/R(0) · 4 fields · 12.4 – 124 meV"

    save_to(monkeypatch, tmp_path / "S1.csv")
    window.commands["export_table"].trigger()
    open_many(monkeypatch, [tmp_path / "S1_Ratio.csv"])
    panel.load_button.click()
    assert [e.name for e in c.library] == ["Sample_4p2K_Sam1", "S1_Ratio.csv"]
    assert panel.count_label.text() == "2 of 2 ticked"
    assert all(b.isEnabled() for b in panel.batch_buttons())
    panel.rows[c.library[1].key].use.setChecked(False)
    assert not c.library[1].used and not panel.merge_energy_button.isEnabled()
    c.update_entry(c.library[1], used=True)  # set elsewhere: the row follows
    assert panel.rows[c.library[1].key].use.isChecked() and panel.average_button.isEnabled()

    row.plot_button.click()
    assert c.result_source == "library" and "Showing a library map" in window.state_text()
    other = panel.rows[c.library[1].key]
    assert not other.remove_button.isVisibleTo(other)  # in the opened row, with the limits
    other.expand_button.click()
    other.remove_button.click()
    assert [e.name for e in c.library] == ["Sample_4p2K_Sam1"] and len(panel.rows) == 1
    c.remove_entry(entry)
    assert not panel.empty.isHidden()
    drop(panel.empty, [tmp_path / "S1_Ratio.csv"])  # tables dropped on the empty list
    assert [e.name for e in c.library] == ["S1_Ratio.csv"]
    assert not errors


def test_library_cut_limits_are_in_the_display_unit(window, sweep):
    load_sweep(window, sweep)
    process(window)
    library.save_current(window)
    c = window.controller
    entry = c.library[0]
    row = window.panels["library"].rows[entry.key]
    assert row.cut_box.isHidden()
    row.expand_button.click()
    assert not row.cut_box.isHidden()
    assert not any(edit.acceptDrops() for edit in row.fields)  # drops go to the list
    set_unit(window, "meV")
    assert row.fields[row.e_min].unit_label.text() == "meV"
    row.e_min.setText("20")
    assert entry.energy_cut == (20 * MEV, None)
    c.update_entry(entry, energy_cut=(20 * MEV, 40 * MEV))
    assert (row.e_min.text(), row.e_max.text()) == ("20", "40")
    set_unit(window, "THz")
    assert row.e_min.text() == "4.83601" and entry.energy_cut == (20 * MEV, 40 * MEV)
    row.b_min.setText("1")
    assert entry.field_cut == (1.0, None)
    row.b_max.setText("0.5")
    assert row.fields[row.b_min].is_invalid() and row.fields[row.b_max].is_invalid()
    row.b_max.setText("x")
    assert entry.field_cut == (1.0, None) and row.fields[row.b_max].is_invalid()
    library.save_current(window)  # a new row: the open one stays open
    assert not window.panels["library"].rows[entry.key].cut_box.isHidden()


def test_library_labels_break_between_parts(qtbot):
    meta = library.PartsLabel(" · ")
    qtbot.addWidget(meta)
    parts = ["R(B)/R(0)", "64 fields", "350 – 3200 cm-1"]
    meta.set_parts(parts)
    assert meta.text() == "R(B)/R(0) · 64 fields · 350 – 3200 cm-1"
    metrics = meta.fontMetrics()
    assert meta.lines(metrics.horizontalAdvance(meta.text())) == [meta.text()]
    two = metrics.horizontalAdvance("R(B)/R(0) · 64 fields")
    assert meta.lines(two) == ["R(B)/R(0) · 64 fields", "350 – 3200 cm-1"]
    assert meta.lines(10) == parts  # never inside a part
    assert meta.heightForWidth(two) == 2 * meta.heightForWidth(10_000)
    assert library.name_parts("Sample_4p2K_Sam1-FIR map.csv") == [
        "Sample_",
        "4p2K_",
        "Sam1-",
        "FIR ",
        "map.csv",
    ]


def test_library_rows_stay_readable_in_a_narrow_panel(qtbot, window, sweep):
    load_sweep(window, sweep)
    process(window)
    c = window.controller
    c.add_map(c.result.ratio, "Sample_4p2K_Sam1_FIR_Ratio.csv")
    panel = library.LibraryPanel()
    qtbot.addWidget(panel)
    panel.show_entries(c.library)
    panel.resize(255, 700)
    panel.show()
    qtbot.waitExposed(panel)
    row = panel.rows[c.library[0].key]
    name, meta = row.name_label, row.meta_label
    # the name has the line to itself, but for the tick and the expand button
    assert name.width() >= row.width() - row.use.width() - row.expand_button.width() - 30
    for label in (name, meta):
        lines = label.lines()
        assert label.separator.join(lines) == label.text()  # nothing cut off
        assert all(label.fontMetrics().horizontalAdvance(line) <= label.width() for line in lines)
        assert label.height() >= len(lines) * label.fontMetrics().lineSpacing()
    assert name.text() == "Sample_4p2K_Sam1_FIR_Ratio.csv" and len(name.lines()) > 1


def test_a_failed_library_plot_keeps_the_name_of_the_map_shown(window, sweep, errors):
    load_sweep(window, sweep)
    process(window)
    c = window.controller
    library.save_current(window)
    low = c.add_map(c.result.ratio.replace(energy=c.result.ratio.energy * 0.05), "Low")
    library.plot_entry(window, c.library[0].key)
    c.set_processing(baseline=(500.0, 600.0))
    library.plot_entry(window, low.key)  # the baseline region misses this map
    assert errors and "baseline region" in errors[-1]
    assert c.result_name() == "Sample_4p2K_Sam1"
    assert c.save_current_map().name == "Sample_4p2K_Sam1 (2)"


def test_library_tables_can_take_the_custom_field_range(window, sweep, tmp_path, monkeypatch):
    load_sweep(window, sweep)
    process(window)
    save_to(monkeypatch, tmp_path / "S1.csv")
    window.commands["export_table"].trigger()
    c = window.controller
    c.set_processing(sample_field=FieldRange(1.0, 1.0, 4.0))
    window.panels["library"].auto_field.setChecked(False)
    library.load_tables(window, [str(tmp_path / "S1_Ratio.csv")])
    np.testing.assert_allclose(c.library[0].fmap.field, [1, 2, 3, 4])


# ---------------------------------------------------------------------- settings
def test_panel_settings_round_trip(qtbot, tmp_path):
    ini = str(tmp_path / "settings.ini")

    def make():
        w = MainWindow(settings=QSettings(ini, QSettings.Format.IniFormat))
        qtbot.addWidget(w)
        return w

    w = make()
    set_unit(w, "meV")
    sample, ref = w.panels["sample"], w.panels["reference"]
    sample.field_source.set_value("custom")
    sample.field_range.end.setText("4")
    ref.set_reference_mode(ReferenceMode.SELF)
    ref.smooth.setChecked(True)
    ref.sg_window.setValue(21)
    ref.sg_poly.setValue(3)
    ref.field_range.start.setText("0.5")
    processing = w.panels["processing"]
    processing.cut_on.setChecked(True)
    processing.cut_lo.setText("20")
    processing.baseline_on.setChecked(True)
    processing.baseline_lo.setText("40")
    processing.baseline_hi.setText("50")
    lib = w.panels["library"]
    lib.full_energy.setChecked(False)
    lib.auto_field.setChecked(False)
    state = w.controller.processing
    w.close()

    w2 = make()
    c = w2.controller
    assert c.processing == state
    assert c.processing.energy_cut == (20 * MEV, None)
    assert c.processing.reference_field == FieldRange(0.5, 0.25, 16.0)
    assert w2.panels["sample"].custom_field() and not w2.panels["sample"].field_range.isHidden()
    assert w2.panels["reference"].mode.value() == "self"
    assert w2.panels["processing"].cut_lo.text() == "20"
    assert w2.panels["processing"].unit_labels[0].text() == "meV"
    assert not w2.panels["library"].full_energy.isChecked()
    assert not w2.panels["library"].auto_field.isChecked()
    assert c.processing.sample_files == SweepFiles()  # files are not remembered
    assert c.unit is Unit.MEV
