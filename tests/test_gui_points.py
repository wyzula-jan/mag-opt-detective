"""The Points panel, the Pick tool on the map and the stacked plot, the markers and undo."""

import numpy as np
import pyqtgraph as pg
import pytest
from PySide6.QtCore import QPoint, QSettings, Qt
from PySide6.QtGui import QKeySequence
from PySide6.QtTest import QTest

import gui_helpers
from gui_helpers import (
    click_map,
    click_stacked,
    current_marker_energies,
    load_sweep,
    open_from,
    process,
    save_to,
    select,
    set_unit,
)
from mag_opt_detective.core.spectra import FieldMap
from mag_opt_detective.core.units import from_cm1
from mag_opt_detective.gui.controller import AppController, SweepFiles
from mag_opt_detective.gui.main_window import MainWindow
from mag_opt_detective.gui.plots import StackedPlot
from mag_opt_detective.gui.points_view import CURVE_COLORS, PickHint

window, errors = gui_helpers.window, gui_helpers.errors  # shared fixtures

ALT = Qt.KeyboardModifier.AltModifier
CTRL = Qt.KeyboardModifier.ControlModifier
MEV = 8.0656


@pytest.fixture
def processed(window, sweep):
    load_sweep(window, sweep)
    process(window)
    return window


@pytest.fixture
def shown(processed, qtbot):
    processed.resize(1400, 900)
    processed.show()
    qtbot.waitExposed(processed)
    processed.show_panel("points", open=True)
    return processed


def chips(window) -> list[tuple[str, str, bool]]:
    """(name, count, current) of the curve chips."""
    return [
        (chip.text(), chip.count, chip.current) for chip in window.panels["points"].chips.chips()
    ]


def rows(window) -> list[tuple[str, str]]:
    """(B, E) as shown in the points table."""
    model = window.panels["points"].model
    return [
        (model.data(model.index(r, 0)), model.data(model.index(r, 1))) for r in range(len(model))
    ]


def scatters(layer) -> list[pg.ScatterPlotItem]:
    return [
        item for item in layer.items() if isinstance(item, pg.ScatterPlotItem) and item.isVisible()
    ]


# ---------------------------------------------------------------------- panel
def test_curves_as_chips_new_rename_and_delete(processed, errors):
    w = processed
    c, panel = w.controller, w.panels["points"]
    w.tools.set_active("pick")
    click_map(w, 1.0, 300.0)
    click_map(w, 1.5, 400.0)
    assert chips(w) == [("LL 1", "2", True)]
    panel.chips.new_chip.click()
    assert c.curve == "LL 2" and panel.column_name.text() == "LL 2"
    assert chips(w) == [("LL 1", "2", False), ("LL 2", "0", True)]
    click_map(w, 2.0, 500.0)
    assert [chip.color.name() for chip in panel.chips.chips()] == list(CURVE_COLORS[:2])

    panel.column_name.setText("CR")  # typing renames the current curve
    assert c.points.names == ["LL 1", "CR"] and chips(w)[1] == ("CR", "1", True)
    panel.column_name.setText("LL 1")  # taken: said, not applied
    assert not panel.name_note.isHidden() and "exists already" in panel.name_note.text()
    assert c.curve == "CR"
    panel.column_name.editingFinished.emit()
    assert panel.column_name.text() == "CR" and panel.name_note.isHidden()

    panel.chips.chips()[0].click()
    assert c.curve == "LL 1" and panel.column_name.text() == "LL 1"
    panel.delete_button.click()
    assert c.points.names == ["CR"] and chips(w) == [("CR", "1", True)]
    assert not errors


def test_points_table_in_the_display_unit_with_a_remove_button_per_row(shown, errors, qtbot):
    w = shown
    c, panel = w.controller, w.panels["points"]
    set_unit(w, "meV")
    w.tools.set_active("pick")
    for b, e in ((2.0, 41.0), (0.5, 37.2), (1.5, 40.0)):
        click_map(w, b, e)
    assert rows(w) == [("0.5", "37.2"), ("1.5", "40"), ("2", "41")]  # sorted by B
    assert panel.count_label.text() == "3 of 4 fields"
    assert panel.model.headerData(1, Qt.Orientation.Horizontal) == "E (meV)"

    table = panel.table
    qtbot.waitUntil(table.isVisible)
    cell = table.visualRect(panel.model.index(1, panel.model.REMOVE)).center()
    QTest.mouseClick(table.viewport(), Qt.MouseButton.LeftButton, pos=cell)
    assert rows(w) == [("0.5", "37.2"), ("2", "41")]
    np.testing.assert_allclose(c.points.points("LL 1")[0], [0.5, 2.0])
    table.setFocus()
    table.selectAll()
    QTest.keyClick(table, Qt.Key.Key_Delete)
    assert rows(w) == [] and panel.table_stack.currentWidget() is panel.empty_label
    assert w.points_undo.undoText() == "Remove 2 points from LL 1"
    assert panel.count_label.text() == "0 of 4 fields"
    assert not errors


def test_the_table_keeps_its_selection_across_edits(processed):
    w = processed
    c, table = w.controller, w.panels["points"].table
    c.record_points([0.5, 1.5, 2.0], [300.0, 400.0, 500.0], unit="cm-1")
    table.selectRow(1)

    def selected() -> list[float]:
        return [table.model().field_at(i.row()) for i in table.selectionModel().selectedRows()]

    assert selected() == [1.5]
    c.record_point(1.0, 350.0)  # a row above it
    assert selected() == [1.5]
    c.rename_curve("CR")
    assert selected() == [1.5]
    c.add_curve()  # another curve: nothing selected
    c.set_curve("CR")
    assert selected() == []


def test_a_unit_switch_changes_the_table_but_not_the_points(processed):
    w = processed
    c, panel = w.controller, w.panels["points"]
    w.tools.set_active("pick")
    click_map(w, 1.0, 300.0)
    stored = c.points.column("LL 1").copy()
    for unit in ("meV", "THz", "cm-1"):
        set_unit(w, unit)
        shown = float(from_cm1(300.0, unit))
        assert rows(w) == [("1", f"{shown:.5g}")]
        assert panel.model.headerData(1, Qt.Orientation.Horizontal) == f"E ({unit})"
        np.testing.assert_allclose(current_marker_energies(w), [shown])
        np.testing.assert_array_equal(c.points.column("LL 1"), stored)


def test_new_table_on_the_next_process(shown):
    w = shown
    c, panel = w.controller, w.panels["points"]
    assert not panel.new_table.isChecked() and not c.new_table
    c.record_point(1.0, 300.0)
    c.add_curve("CR")
    process(w)
    assert len(c.points.points("LL 1")[0]) == 1  # kept
    row = panel.new_table_row
    QTest.mouseClick(row, Qt.MouseButton.LeftButton, pos=QPoint(5, 5))  # the text toggles too
    assert panel.new_table.isChecked() and c.new_table
    process(w)
    assert c.points.names == ["LL 1", "CR"] and c.curve == "CR"  # the curves stay, empty
    assert all(np.isnan(c.points.column(name)).all() for name in c.points.names)
    assert not panel.new_table.isChecked()
    assert w.points_undo.undoText() == "New point table"
    w.commands["undo"].trigger()
    np.testing.assert_allclose(c.points.points("LL 1")[1], [300.0])


def test_import_export_round_trip_and_legacy_files(processed, tmp_path, monkeypatch, errors):
    w = processed
    c, panel = w.controller, w.panels["points"]
    set_unit(w, "meV")
    w.tools.set_active("pick")
    click_map(w, 1.04, 37.2)  # meV, as shown
    click_map(w, 1.96, 38.5)
    path = tmp_path / "points.csv"
    save_to(monkeypatch, path)
    panel.export_button.click()
    assert path.read_text().splitlines()[:3] == ["Energy (meV)\tLL 1", "0.5\t", "1.0\t37.2"]
    c.drop_curve()
    assert len(c.points.points("LL 1")[0]) == 0
    open_from(monkeypatch, path)
    panel.import_button.click()
    np.testing.assert_allclose(c.points.points("LL 1")[1], [37.2 * MEV, 38.5 * MEV])
    assert w.points_undo.undoText() == "Import points.csv"

    legacy = tmp_path / "Points_V1.csv"  # no unit in the header: read in the display unit
    legacy.write_text("\tLL 2\tCR\n0.5\t40\t\n1.0\t\t\n1.5\t\t44\n2.0\t41\t\n")
    open_from(monkeypatch, legacy)
    panel.import_button.click()
    assert chips(w) == [("LL 2", "2", True), ("CR", "1", False)]
    assert panel.column_name.text() == "LL 2"
    assert rows(w) == [("0.5", "40"), ("2", "41")]
    np.testing.assert_allclose(c.points.points("LL 2")[1], [40 * MEV, 41 * MEV])
    np.testing.assert_allclose(current_marker_energies(w), [40, 41])
    w.commands["undo"].trigger()
    assert c.points.names == ["LL 1"] and len(c.points.points("LL 1")[0]) == 2
    assert not errors


def test_markers_and_curve_name_are_remembered(qtbot, tmp_path):
    ini = str(tmp_path / "settings.ini")
    first = MainWindow(settings=QSettings(ini, QSettings.Format.IniFormat))
    qtbot.addWidget(first)
    first.panels["points"].markers.set_value("current")
    first.panels["points"].column_name.setText("CR1")
    first.close()
    second = MainWindow(settings=QSettings(ini, QSettings.Format.IniFormat))
    qtbot.addWidget(second)
    panel = second.panels["points"]
    assert panel.markers.value() == "current"
    assert second.controller.curve == "CR1" and panel.column_name.text() == "CR1"


# ---------------------------------------------------------------------- pick tool
def test_pick_on_the_map_and_on_the_stacked_plot(processed, errors):
    w = processed
    c, tools = w.controller, w.tools
    w.panels["points"].pick_button.click()
    assert tools.active() == "pick"
    click_map(w, 1.04, 300.0)  # the nearest field
    click_map(w, 1.5, 400.0)
    click_map(w, 1.47, 0.0, ALT)  # far from the point: the clicked field's point goes
    np.testing.assert_allclose(c.points.points("LL 1")[0], [1.0])
    click_map(w, 2.0, 500.0, ALT)  # nothing there
    np.testing.assert_allclose(c.points.points("LL 1")[0], [1.0])

    w.plot_area.set_current_view("stacked")
    assert tools.active() == "pick"
    assert click_stacked(w, 2.0, 500.0)  # on the 2 T trace, at the clicked energy
    b, e = c.points.points("LL 1")
    np.testing.assert_allclose(b, [1.0, 2.0])
    np.testing.assert_allclose(e, [300.0, 500.0])
    click_stacked(w, 1.0, 650.0, ALT)  # Alt on the 1 T trace removes its point
    np.testing.assert_allclose(c.points.points("LL 1")[0], [2.0])
    stacked = w.plots.stacked
    tools.click("stacked", 500.0, stacked.trace_y(3, 500.0) + 1.0)  # far above every trace
    tools.click("stacked", 5000.0, 1.0)  # outside the spectra
    np.testing.assert_allclose(c.points.points("LL 1")[0], [2.0])
    assert not errors


def test_pick_toggle_follows_the_tool_and_shows_a_hint(shown):
    w = shown
    tools, panel, area = w.tools, w.panels["points"], w.plot_area
    hint = w.findChild(PickHint)
    assert hint.isHidden()
    area.set_current_view("reference")
    panel.pick_button.click()  # shows the map first
    assert area.current_view() == "map" and tools.active() == "pick"
    assert panel.pick_button.isChecked() and tools.tool("pick").button.isChecked()
    assert hint.isVisible() and hint.text().startswith("Picking LL 1 · click to record")
    area.set_current_view("stacked")
    assert tools.active() == "pick" and "click a trace to record" in hint.text()
    view = w.plots.stacked.view
    view.setFocus()
    QTest.keyClick(view, Qt.Key.Key_Escape)
    assert tools.active() == "navigate" and not panel.pick_button.isChecked()
    assert hint.isHidden()
    QTest.keyClick(view, Qt.Key.Key_P)  # P works on the stacked tab
    assert tools.active() == "pick" and panel.pick_button.isChecked()
    w.controller.add_curve()
    assert hint.text().startswith("Picking LL 2")
    w.report_error("Pick point", "something went wrong", panel="points")
    assert hint.geometry().top() > w.infobar.geometry().bottom()  # below the error bar
    panel.pick_button.click()
    assert tools.active() == "navigate" and hint.isHidden()
    area.set_current_view("reference")
    assert not tools.tool("pick").button.isEnabled()


def test_tool_registry_routes_clicks_with_modifiers(processed):
    w = processed
    tools, clicks = w.tools, []
    events = []
    tools.register(
        "probe",
        "wand-sparkles",
        "Probe",
        on_activate=lambda: events.append("on"),
        on_deactivate=lambda: events.append("off"),
        on_click=clicks.append,
        views=("map", "stacked"),
    )
    assert tools.names()[:4] == ["navigate", "zoom", "pick", "probe"]
    assert tools.active() == "navigate"
    assert not tools.click("map", 1.0, 2.0)  # navigate takes no clicks
    tools.set_active("probe")
    assert events == ["on"]
    assert tools.click("stacked", 300.0, 1.5, ALT)
    assert tools.click("map", 1.0, 2.0)
    assert not tools.click("reference", 1.0, 2.0)  # not one of its views
    assert [(k.view, k.x, k.y, k.modifiers) for k in clicks] == [
        ("stacked", 300.0, 1.5, ALT),
        ("map", 1.0, 2.0, Qt.KeyboardModifier.NoModifier),
    ]
    w.plot_area.tabs.setCurrentIndex(2)  # Reference: the probe does not work there
    assert tools.active() == "navigate" and events == ["on", "off"]
    assert not tools.tool("probe").button.isEnabled()


def test_tool_modes_and_shortcuts(processed):
    w = processed
    tools, vb = w.tools, w.plots.map.plot.vb
    tools.set_active("zoom")
    assert vb.state["mouseMode"] == pg.ViewBox.RectMode
    tools.toggle("zoom")  # the shortcut again: back to pan and zoom
    assert tools.active() == "navigate"
    assert vb.state["mouseMode"] == pg.ViewBox.PanMode
    tools.toggle("pick")
    assert w.panels["points"].pick_button.isChecked()
    w.panels["points"].pick_button.click()
    assert tools.active() == "navigate"


def test_real_clicks_on_the_map_pick_and_alt_remove(shown):
    w = shown
    c, plot = w.controller, w.plots.map
    w.tools.set_active("pick")
    vb = plot.plot.vb

    def click(b, e, modifier=Qt.KeyboardModifier.NoModifier, dx=0):
        target = plot.view.mapFromScene(vb.mapViewToScene(pg.Point(b, e)))
        pos = QPoint(int(target.x()) + dx, int(target.y()))
        QTest.mouseClick(plot.view.viewport(), Qt.MouseButton.LeftButton, modifier, pos)

    click(1.5, 550.0)
    click(1.0, 300.0)
    b, e = c.points.points("LL 1")
    np.testing.assert_allclose(b, [1.0, 1.5])
    assert abs(e[1] - 550.0) < 20
    click(1.5, e[1], ALT, dx=6)  # a few pixels off the marker
    np.testing.assert_allclose(c.points.points("LL 1")[0], [1.0])


# ---------------------------------------------------------------------- markers
def test_map_markers_current_filled_others_open_in_curve_colours(processed):
    w = processed
    c, panel = w.controller, w.panels["points"]
    c.record_point(1.0, 300.0)
    c.add_curve()
    c.record_points([1.5, 2.0], [400.0, 500.0], unit="cm-1")
    layer = w.plots.map.layer("points")
    assert [len(x) for x, _y in layer.point_data()] == [1, 1, 2]  # outline, LL 1, current LL 2
    _outline, ring, current = scatters(layer)
    assert ring.points()[0].pen().color().name() == CURVE_COLORS[0]
    assert ring.points()[0].brush().style() == Qt.BrushStyle.NoBrush
    assert current.points()[0].brush().color().name() == CURVE_COLORS[1]
    np.testing.assert_allclose(current_marker_energies(w), [400.0, 500.0])
    stacked = w.plots.stacked.layer("points")
    assert stacked.point_data() == []  # drawn when the Stacked tab is shown
    w.plot_area.set_current_view("stacked")
    assert [len(x) for x, _y in stacked.point_data()] == [1, 1, 2]
    panel.markers.set_value("current")
    assert [len(x) for x, _y in layer.point_data()] == [2]
    assert [len(x) for x, _y in stacked.point_data()] == [2]
    panel.markers.set_value("hidden")
    assert layer.point_data() == [] and stacked.point_data() == []
    panel.markers.set_value("all")
    assert [len(x) for x, _y in stacked.point_data()] == [1, 1, 2]
    c.add_curve()  # one ring group per curve, each in one pen
    c.record_point(0.5, 600.0)
    assert [len(x) for x, _y in layer.point_data()] == [3, 1, 2, 1]
    _outline, first, second, _current = scatters(layer)
    assert {p.pen().color().name() for p in second.points()} == {CURVE_COLORS[1]}
    assert first.points()[0].pen().color().name() == CURVE_COLORS[0]


def test_stacked_markers_sit_on_their_traces(processed, monkeypatch):
    w = processed
    c, stacked = w.controller, w.plots.stacked
    w.plot_area.set_current_view("stacked")
    c.record_points([0.5, 1.0, 2.0], [300.0, 400.0, 500.0], unit="cm-1")
    c.add_curve()
    c.record_point(1.5, 700.0)
    layer = stacked.layer("points")

    def check(others: list[tuple[int, float]], current: list[tuple[int, float]]) -> None:
        data = layer.point_data()
        for (x, y), expected in ((data[0], others), (data[-1], current)):
            np.testing.assert_allclose(x, [e for _j, e in expected])
            np.testing.assert_allclose(y, [stacked.trace_y(j, e) for j, e in expected])

    check([(0, 300.0), (1, 400.0), (3, 500.0)], [(2, 700.0)])
    y_before = layer.point_data()[-1][1][0]
    c.set_view(stacked_offset=0.5)  # the markers follow the offset
    check([(0, 300.0), (1, 400.0), (3, 500.0)], [(2, 700.0)])
    assert layer.point_data()[-1][1][0] == pytest.approx(y_before + 2 * (0.5 - 0.01))
    stacked.set_trace_options(every=2)  # traces 0 and 2 shown: the others' markers go
    check([(0, 300.0)], [(2, 700.0)])
    set_unit(w, "meV")
    check([(0, 300.0 / MEV)], [(2, 700.0 / MEV)])
    w.plot_area.set_current_view("map")  # edits on another tab: drawn on coming back
    c.record_point(0.5, 80.0)
    stacked.set_trace_options(every=1)
    w.plot_area.set_current_view("stacked")
    check([(0, 300.0 / MEV), (1, 400.0 / MEV), (3, 500.0 / MEV)], [(0, 80.0), (2, 700.0 / MEV)])

    placed = []  # the curves whose points did not change are not placed again
    trace_y = stacked.trace_y
    monkeypatch.setattr(stacked, "trace_y", lambda j, e: placed.append(j) or trace_y(j, e))
    c.set_curve("LL 1")
    assert placed == []
    c.record_point(1.5, 450.0)
    assert sorted(placed) == [0, 1, 2, 3]


def test_pick_and_markers_on_the_field_step_ratio(processed, errors):
    """R(B)/R(B-ΔB) has no trace for the first field: its trace j is the field j + 1."""
    w = processed
    c, stacked = w.controller, w.plots.stacked
    select(w, kind="Ratio_Step")
    w.plot_area.set_current_view("stacked")
    w.tools.set_active("pick")
    np.testing.assert_allclose(c.current_map().field, [1.0, 1.5, 2.0])
    assert w.tools.click("stacked", 500.0, stacked.trace_y(2, 500.0))  # the 2 T trace
    b, e = c.points.points("LL 1")
    np.testing.assert_allclose(b, [2.0])
    np.testing.assert_allclose(e, [500.0])
    assert click_stacked(w, 1.5, 450.0)
    c.record_points([0.5, 1.0], [300.0, 400.0], unit="cm-1")  # no trace for 0.5 T here
    x, y = stacked.layer("points").point_data()[-1]
    np.testing.assert_allclose(x, [400.0, 450.0, 500.0])
    expected = [stacked.trace_y(j, e) for j, e in ((0, 400.0), (1, 450.0), (2, 500.0))]
    np.testing.assert_allclose(y, expected)
    assert w.tools.click("stacked", 520.0, stacked.trace_y(2, 520.0), ALT)  # Alt on 2 T
    np.testing.assert_allclose(c.points.points("LL 1")[0], [0.5, 1.0, 1.5])
    select(w, order=1, axis="B")  # a field derivative keeps the field axis
    assert w.tools.click("stacked", 600.0, stacked.trace_y(2, 600.0))
    np.testing.assert_allclose(c.points.points("LL 1")[0], [0.5, 1.0, 1.5, 2.0])
    assert not errors


def test_stacked_plot_signals_redrawn_traces(qtbot):
    stacked = StackedPlot()
    qtbot.addWidget(stacked)
    fmap = FieldMap(np.linspace(100.0, 200.0, 11), np.array([0.5, 1.0, 1.5]), np.ones((11, 3)))
    with qtbot.waitSignal(stacked.tracesChanged):
        stacked.set_map(fmap, 1.0)
    with qtbot.waitSignal(stacked.tracesChanged):
        stacked.set_trace_options(every=2)  # markers placed with trace_y must follow
    assert stacked.trace_y(1, 150.0) is None
    with qtbot.waitSignal(stacked.tracesChanged):
        stacked.clear_map()


# ---------------------------------------------------------------------- undo (window)
def test_undo_and_redo_in_the_window_leave_text_fields_alone(shown):
    w = shown
    c, stack = w.controller, w.points_undo
    undo, redo = w.commands["undo"], w.commands["redo"]
    assert undo in w.edit_menu.actions() and redo in w.edit_menu.actions()
    assert not undo.isEnabled() and not redo.isEnabled()
    w.tools.set_active("pick")
    click_map(w, 1.0, 300.0)
    assert undo.isEnabled() and undo.text().endswith("Record point on LL 1")
    view = w.plots.map.view
    view.setFocus()
    QTest.keyClick(view, Qt.Key.Key_Z, CTRL)
    assert len(c.points.points("LL 1")[0]) == 0 and redo.isEnabled()
    QTest.keyClick(view, Qt.Key.Key_Z, CTRL | Qt.KeyboardModifier.ShiftModifier)
    assert len(c.points.points("LL 1")[0]) == 1

    name = w.panels["points"].column_name
    name.setFocus()
    name.end(False)
    QTest.keyClicks(name, "x")
    assert c.curve == "LL 1x" and stack.undoText() == "Rename curve LL 1"
    QTest.keyClick(name, Qt.Key.Key_Z, CTRL)  # the field's own undo, not the points'
    assert name.text() == "LL 1" and c.curve == "LL 1"
    assert stack.count() == 1 and stack.undoText() == "Record point on LL 1"  # typed back
    assert len(c.points.points("LL 1")[0]) == 1
    QTest.keyClicks(name, "2")
    undo.trigger()  # Edit > Undo with the mouse: the field follows
    assert c.curve == "LL 1" and name.text() == "LL 1"
    assert undo.shortcut() == QKeySequence("Ctrl+Z")
    assert redo.shortcut() == QKeySequence("Ctrl+Shift+Z")  # shown in the menu


# ---------------------------------------------------------------------- undo (controller)
@pytest.fixture
def ctl(qapp, sweep):
    c = AppController()
    c.set_processing(sample_files=SweepFiles(tuple(sweep["zero"]), tuple(sweep["field"])))
    c.process()
    return c


def energies(c, name=None):
    return c.points.column(name or c.curve).copy()


def test_every_point_edit_is_undone_and_redone(ctl, tmp_path):
    c, stack = ctl, ctl.points_undo
    assert stack.count() == 0 and c.points.names == ["LL 1"]  # the first table is no step
    c.record_point(1.0, 300.0)
    c.record_point(1.5, 400.0)
    c.remove_point(1.04)
    c.add_curve()
    c.record_point(2.0, 500.0)
    c.rename_curve("CR")
    c.drop_curve("LL 1")
    path = tmp_path / "points.csv"
    path.write_text("Energy (cm-1)\tA\n0.5\t\n1.0\t333\n1.5\t\n2.0\t\n")
    c.load_points(path)
    c.set_new_table(True)
    c.process()
    texts = [stack.text(i) for i in range(stack.count())]
    assert texts == [
        "Record point on LL 1",
        "Record point on LL 1",
        "Remove point from LL 1",
        "New curve LL 2",
        "Record point on LL 2",
        "Rename curve LL 2",
        "Delete curve LL 1",
        "Import points.csv",
        "New point table",
    ]
    assert np.isnan(energies(c)).all() and c.points.names == ["A"]
    states = []
    for _ in texts:
        states.append((c.points.names, c.curve, energies(c)))
        stack.undo()
    assert c.points.names == ["LL 1"] and c.curve == "LL 1" and np.isnan(energies(c)).all()
    for names, curve, values in reversed(states):
        stack.redo()
        assert (c.points.names, c.curve) == (names, curve)
        np.testing.assert_array_equal(energies(c), values)


def test_undo_steps_follow_the_display_unit_and_skip_no_ops(ctl):
    c, stack = ctl, ctl.points_undo
    c.set_unit("meV")
    c.record_point(1.0, 40.0)
    c.remove_point(2.0)  # nothing there: no step
    c.record_point(1.0, 40.0)  # the same value again: no step
    assert stack.count() == 1
    c.set_unit("THz")
    stack.undo()
    stack.redo()
    np.testing.assert_allclose(c.points.points("LL 1")[1], [40.0 * MEV])  # kept in cm-1


def test_typed_renames_merge_into_one_step(ctl):
    c, stack = ctl, ctl.points_undo
    for name in ("C", "CR", "CR1"):
        c.rename_curve(name, merge=("rename", 1))
    assert stack.count() == 1 and c.curve == "CR1"
    assert stack.undoText() == "Rename curve LL 1"  # the name the step brings back
    c.rename_curve("CR", merge=("rename", 2))  # another editing session
    assert stack.count() == 2
    c.rename_curve("CR1", merge=("rename", 2))  # typed back: the step goes away
    assert stack.count() == 1
    stack.undo()
    assert c.curve == "LL 1" and c.points.names == ["LL 1"]
    with pytest.raises(ValueError, match="exists already"):
        c.add_curve("LL 1")
    c.add_curve("LL 2")
    with pytest.raises(ValueError, match="exists already"):
        c.rename_curve("LL 1")
    with pytest.raises(ValueError, match="letters"):
        c.rename_curve("LL\t2")


def test_a_batch_is_one_step_and_one_signal(ctl, qtbot):
    c, stack = ctl, ctl.points_undo
    signals = []
    c.pointsChanged.connect(lambda: signals.append(1))
    fields = c.points.field
    n = c.record_points(fields, [100.0, np.nan, 120.0, 130.0], curve="auto", unit="meV")
    assert n == 3 and stack.count() == 1 and len(signals) == 1
    assert stack.undoText() == "Record 3 points on auto"
    np.testing.assert_allclose(c.points.points("auto")[1], np.array([100, 120, 130]) * MEV)
    with c.point_edit("Tidy up"):
        c.record_point(0.5, 300.0)
        c.remove_point(1.5)
    assert stack.count() == 2 and stack.undoText() == "Tidy up" and len(signals) == 2
    stack.undo()
    stack.undo()
    assert "auto" not in c.points.names and len(c.points.points("LL 1")[0]) == 0


def test_dropping_the_last_curve_leaves_an_empty_one(ctl):
    c = ctl
    c.record_point(1.0, 300.0)
    c.rename_curve("CR")
    c.drop_curve()
    assert c.points.names == ["LL 1"] and c.curve == "LL 1"
    c.points_undo.undo()
    assert c.points.names == ["CR"] and c.curve == "CR"
