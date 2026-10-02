import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest

import gui_helpers
from gui_helpers import (
    click_map,
    current_marker_energies,
    load_sweep,
    open_from,
    process,
    save_to,
    set_unit,
)

window, errors = gui_helpers.window, gui_helpers.errors  # shared fixtures

ALT = Qt.KeyboardModifier.AltModifier


def test_points_record_remove_export(window, sweep, tmp_path, monkeypatch, errors):
    load_sweep(window, sweep)
    set_unit(window, "meV")
    process(window)
    c, panel = window.controller, window.panels["points"]
    model = panel.model
    panel.point_record.setChecked(True)
    assert window.tools.active() == "pick"
    click_map(window, 1.04, 37.2)  # meV, as shown
    click_map(window, 1.96, 38.5)
    assert model.rowCount() == 4
    assert model.data(model.index(1, 0)) == "37.2"
    np.testing.assert_allclose(c.points.points("LL 1")[1], [37.2 * 8.0656, 38.5 * 8.0656])
    np.testing.assert_allclose(current_marker_energies(window), [37.2, 38.5])  # markers in meV
    panel.point_remove.setChecked(True)
    click_map(window, 2.0, 0.0)
    b, _ = c.points.points("LL 1")
    np.testing.assert_allclose(b, [1.0])

    path = tmp_path / "points.csv"
    save_to(monkeypatch, path)
    panel.export_button.click()
    assert path.read_text().splitlines()[:3] == ["Energy (meV)\tLL 1", "0.5\t", "1.0\t37.2"]
    open_from(monkeypatch, path)
    c.points = None
    panel.load_button.click()
    assert c.points.names == ["LL 1"]
    np.testing.assert_allclose(c.points.points("LL 1")[1], [37.2 * 8.0656])

    legacy = tmp_path / "Points_V1.csv"  # no unit in the header: read in the display unit
    legacy.write_text("\tLL 2\n0.5\t40\n1.0\t\n1.5\t\n2.0\t41\n")
    open_from(monkeypatch, legacy)
    panel.load_button.click()
    np.testing.assert_allclose(c.points.points("LL 2")[1], [40 * 8.0656, 41 * 8.0656])
    assert panel.column_name.text() == "LL 2"
    assert model.data(model.index(3, 0)) == "41"
    np.testing.assert_allclose(current_marker_energies(window), [40, 41])
    assert not errors


def test_show_all_and_drop_curve(window, sweep, errors):
    load_sweep(window, sweep)
    process(window)
    c, panel = window.controller, window.panels["points"]
    window.tools.set_active("pick")
    assert panel.point_record.isChecked()
    click_map(window, 0.5, 300.0)
    panel.column_name.setText("LL 2")
    click_map(window, 1.0, 400.0)
    click_map(window, 1.5, 500.0)
    assert c.points.names == ["LL 1", "LL 2"]
    layer = window.plots.map.layer("points")
    assert len(layer.point_data()) == 1  # the current curve only
    panel.show_all_button.setChecked(True)
    assert [len(x) for x, _y in layer.point_data()] == [3, 2]
    panel.drop_button.click()
    assert c.points.names == ["LL 1"]
    assert not errors


def test_new_table_on_next_process(window, sweep):
    load_sweep(window, sweep)
    process(window)
    c, panel = window.controller, window.panels["points"]
    assert not panel.init_table.isChecked()  # a table was started
    window.tools.set_active("pick")
    click_map(window, 1.0, 300.0)
    process(window)
    assert len(c.points.points("LL 1")[0]) == 1  # kept
    panel.init_table.setChecked(True)
    process(window)
    assert len(c.points.points("LL 1")[0]) == 0
    assert not panel.init_table.isChecked()


def test_tool_registry_routes_clicks_with_modifiers(window, sweep):
    load_sweep(window, sweep)
    process(window)
    tools, clicks = window.tools, []
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
    window.plot_area.tabs.setCurrentIndex(2)  # Reference: the probe does not work there
    assert tools.active() == "navigate" and events == ["on", "off"]
    assert not tools.tool("probe").button.isEnabled()


def test_alt_click_removes_a_picked_point(window, sweep, errors):
    load_sweep(window, sweep)
    process(window)
    c = window.controller
    window.tools.set_active("pick")
    click_map(window, 1.0, 300.0)
    click_map(window, 1.5, 400.0)
    click_map(window, 1.04, 0.0, ALT)
    np.testing.assert_allclose(c.points.points("LL 1")[0], [1.5])
    assert not errors


def test_tool_modes_and_shortcuts(window, sweep):
    load_sweep(window, sweep)
    process(window)
    tools, vb = window.tools, window.plots.map.plot.vb
    tools.set_active("zoom")
    assert vb.state["mouseMode"] == pg.ViewBox.RectMode
    tools.toggle("zoom")  # the shortcut again: back to pan and zoom
    assert tools.active() == "navigate"
    assert vb.state["mouseMode"] == pg.ViewBox.PanMode
    tools.toggle("pick")
    assert window.panels["points"].point_record.isChecked()
    window.panels["points"].point_off.setChecked(True)
    assert tools.active() == "navigate"


def test_a_real_click_on_the_map_picks_a_point(window, sweep, qtbot):
    window.resize(1400, 900)
    window.show()
    qtbot.waitExposed(window)
    load_sweep(window, sweep)
    process(window)
    window.tools.set_active("pick")
    plot = window.plots.map
    vb = plot.plot.vb
    target = vb.mapViewToScene(pg.Point(1.5, 550.0))
    pos = plot.view.mapFromScene(target)
    QTest.mouseClick(plot.view.viewport(), Qt.MouseButton.LeftButton, pos=QPoint(pos.x(), pos.y()))
    b, e = window.controller.points.points("LL 1")
    np.testing.assert_allclose(b, [1.5])
    assert abs(e[0] - 550.0) < 20
