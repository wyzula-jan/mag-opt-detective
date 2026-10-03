"""The window frame: rail and slide panels, colour scales, the log drawer and the appearance."""

import logging

import numpy as np
import pytest
import shiboken6
from PySide6.QtCore import QEvent, QPoint, Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QComboBox, QToolButton

import gui_helpers
from gui_helpers import energy_label, load_sweep, process, shown_image
from mag_opt_detective import __version__
from mag_opt_detective.core.pipeline import PlotKind, ReferenceMode
from mag_opt_detective.core.processing import Axis
from mag_opt_detective.core.units import Unit
from mag_opt_detective.gui import display, theme
from mag_opt_detective.gui.console import Badge
from mag_opt_detective.gui.kit import SlidePanel
from mag_opt_detective.gui.main_window import MainWindow
from mag_opt_detective.gui.plots import BarScale, HistogramScale
from mag_opt_detective.gui.theme import Theme, current_tokens

window, errors = gui_helpers.window, gui_helpers.errors  # shared fixtures


@pytest.fixture
def shown(window, qtbot):
    window.resize(1400, 900)
    window.show()
    qtbot.waitExposed(window)
    return window


def view_page(window):
    return window.inspector["view"].body_layout().itemAt(0).widget()


def test_window_exposes_its_areas(window):
    assert list(window.panels) == ["sample", "reference", "processing", "library", "points"]
    assert list(window.inspector) == ["view", "colour", "traces", "models"]
    assert [name for name, _plot in window.plots.items()] == ["map", "stacked", "reference"]
    assert window.plots["map"] is window.plots.map
    for panel in (window.side_panel, window.inspector_panel, window.log_panel):
        assert isinstance(panel, SlidePanel)
    assert window.side_panel.is_open() and window.inspector_panel.is_open()
    assert not window.log_panel.is_open()  # the log drawer starts closed
    assert window.tools.names() == ["navigate", "zoom", "pick", "autopick"]


def test_rail_switches_and_closes_the_side_panel(shown):
    w = shown
    buttons = w._rail_buttons
    assert w.current_panel() == "sample" and buttons["sample"].isChecked()
    buttons["library"].click()
    assert w.current_panel() == "library" and w.side_panel.is_open()
    assert buttons["library"].isChecked() and not buttons["sample"].isChecked()
    buttons["library"].click()  # the active one: close
    assert not w.side_panel.is_open() and not buttons["library"].isChecked()
    buttons["points"].click()
    assert w.side_panel.is_open() and w.current_panel() == "points"


def test_inspector_and_log_toggles(shown, qtbot):
    w = shown
    w.plot_area.inspector_button.click()
    qtbot.waitUntil(lambda: not w.inspector_panel.is_animating())
    assert not w.inspector_panel.is_open()
    assert w.body_splitter.sizes()[2] == 0
    w.plot_area.inspector_button.click()
    assert w.inspector_panel.is_open()
    w.log_button.click()
    assert w.log_panel.is_open() and w.log_button.isChecked()
    qtbot.waitUntil(lambda: not w.log_panel.is_animating())
    assert w.stage_splitter.sizes()[1] == w.log_panel.open_size()
    w.log_drawer.close_button.click()
    assert not w.log_panel.is_open() and not w.log_button.isChecked()


def test_log_badge_counts_unseen_errors(window):
    logger = logging.getLogger("mag_opt_detective")
    process(window)  # no files: an error
    logger.error("second")
    QGuiApplication.processEvents()
    assert window.log_button.unseen() == 2
    assert not window.log_button.badge.isHidden()
    window.log_panel.set_open(True, animate=False)
    assert window.log_button.unseen() == 0 and window.log_button.isChecked()
    assert "no field files" in window.console.toPlainText()
    logger.error("while open")
    QGuiApplication.processEvents()
    assert window.log_button.unseen() == 0
    window.log_panel.set_open(False, animate=False)
    logger.error("after")
    QGuiApplication.processEvents()
    assert window.log_button.unseen() == 1


def test_log_badge_sits_inside_the_button(shown, qtbot):
    """A 16 px pill after the text, inside the button (it was stretched to the bar's height)."""
    button, badge = shown.log_button, shown.log_button.badge
    plain = button.sizeHint().width()
    logging.getLogger("mag_opt_detective").error("an error")
    qtbot.waitUntil(lambda: not badge.isHidden())
    assert badge.parentWidget() is button and badge.height() == Badge.HEIGHT
    assert button.sizeHint().width() >= plain + badge.width()  # room for it after the text
    qtbot.waitUntil(lambda: badge.geometry().right() < button.width())
    assert abs(badge.geometry().center().y() - button.rect().center().y()) <= 1


def test_log_lines_are_coloured_by_level(window):
    logger = logging.getLogger("mag_opt_detective")
    logger.info("fine")
    logger.error("broken <file>")
    QGuiApplication.processEvents()
    text = window.console.toPlainText().splitlines()
    assert text[-1].endswith("broken <file>  [error]")  # the level also in words, for copies
    tokens = current_tokens()
    html = window.console.document().toHtml()
    assert tokens["err"].name() in html and tokens["faint"].name() in html


def test_colour_scale_style_switch_applies_to_every_map(shown, sweep, errors, qtbot):
    w = shown
    load_sweep(w, sweep)
    process(w)
    maps = (w.plots.map, w.plots.reference)
    assert all(isinstance(plot.scale, HistogramScale) for plot in maps)  # classic by default
    assert w.plot_area.map_splitter.sizes()[1] == HistogramScale.WIDTH
    w.plots.map.set_levels(0.95, 1.05)
    w.plot_area.scale_style_button.click()
    assert all(isinstance(plot.scale, BarScale) for plot in maps)
    assert w.plots.map.levels() == pytest.approx((0.95, 1.05))
    width = w.plots.map.scale.widget.width()
    assert w.plot_area.map_scale.maximumWidth() == width < HistogramScale.WIDTH
    qtbot.waitUntil(lambda: w.plot_area.map_splitter.sizes()[1] == width)
    w.plot_area.scale_style_button.click()
    assert all(isinstance(plot.scale, HistogramScale) for plot in maps)
    qtbot.waitUntil(lambda: w.plot_area.map_splitter.sizes()[1] == HistogramScale.WIDTH)
    assert not errors


def test_show_and_hide_all_colour_scales(shown, sweep, qtbot):
    w = shown
    load_sweep(w, sweep)
    process(w)
    area = w.plot_area
    panels = area.scale_panels()
    area.scales_button.click()
    qtbot.waitUntil(lambda: not any(p.is_animating() for p in panels.values()))
    assert not any(p.is_open() for p in panels.values())
    assert not w.plots.map.scale_visible()  # exported images match the screen
    assert len(w.plots.map._export_parts(100)) == 1
    area.scales_button.click()
    assert all(p.is_open() for p in panels.values())
    assert w.plots.map.scale_visible() and len(w.plots.map._export_parts(100)) == 2

    panels["map"].set_open(False, animate=False)  # one plot alone
    assert not area.scales_button.isChecked()
    assert panels["reference"].is_open()


def test_a_click_on_the_handle_toggles_the_scale(shown, sweep, qtbot):
    w = shown
    load_sweep(w, sweep)
    process(w)
    splitter = w.plot_area.map_splitter
    handle = splitter.handle(1)
    panel = w.plot_area.map_scale
    center = QPoint(handle.width() // 2, handle.height() // 2)
    QTest.mouseClick(handle, Qt.MouseButton.LeftButton, pos=center)
    qtbot.waitUntil(lambda: not panel.is_animating())
    assert not panel.is_open() and not w.plots.map.scale_visible()
    QTest.mouseClick(splitter.handle(1), Qt.MouseButton.LeftButton, pos=center)
    qtbot.waitUntil(lambda: not panel.is_animating())
    assert panel.is_open() and w.plots.map.scale_visible()


def test_plots_follow_the_theme(window):
    window.set_appearance("dark")
    try:
        bg = window.plots.map.colors().background
        assert bg == window.theme.plot_colors()["background"] == "#121015"
        window.set_appearance("light")
        assert window.plots.stacked.colors().background == "#ffffff"
        window.cycle_appearance()
        assert window.theme.scheme() == "dark"
        assert window.toolbar.appearance.toolTip().startswith("Appearance: Dark")
    finally:
        window.set_appearance("system")
        QGuiApplication.styleHints().setColorScheme(Qt.ColorScheme.Unknown)


def test_toolbar_follows_the_controller(window, sweep, errors):
    load_sweep(window, sweep)
    process(window)
    c, tb = window.controller, window.toolbar
    c.set_unit("meV")
    assert tb.unit.value() == "meV" and energy_label(window) == "Energy (meV)"
    tb.unit.button("cm-1").click()  # a later click in the toolbar still works
    assert c.unit is Unit.CM1 and energy_label(window) == "Energy (cm⁻¹)"

    c.set_selection(kind=PlotKind.DATA, order=1)
    assert (tb.kind.value(), tb.order.value()) == ("Data", "1")
    assert tb.axis.button("B").isEnabled() and tb.per_unit.isEnabled()
    c.set_selection(axis=Axis.FIELD, physical=True, reference_kind=PlotKind.DATA)
    assert tb.axis.value() == "B" and tb.per_unit.isChecked()
    assert window.plot_area.ref_kind.value() == "Data"
    tb.kind.button("Ratio").click()
    assert c.selection.kind is PlotKind.RATIO and c.selection.axis is Axis.FIELD
    expected = c.result.get(PlotKind.RATIO, 1, Axis.FIELD, physical=True)
    np.testing.assert_allclose(shown_image(window), expected.values)
    c.set_selection(order=0)
    assert tb.order.value() == "0" and not tb.per_unit.isEnabled()
    assert not errors


def test_typed_letters_stay_in_tables_and_combo_boxes(shown, sweep):
    w, tools = shown, shown.tools
    load_sweep(w, sweep)
    w.show_panel("sample")
    table = w.panels["sample"].measurement.field_list
    table.setFocus()
    QTest.keyClick(table, Qt.Key.Key_Z)  # keyboard search, not the Box zoom tool
    assert tools.active() == "navigate"
    w.plots.map.view.setFocus()
    QTest.keyClick(w.plots.map.view, Qt.Key.Key_Z)
    assert tools.active() == "zoom"
    combo = QComboBox(w.panels["sample"])
    combo.addItems(["Venus", "Mars"])
    combo.show()
    combo.setFocus()
    QTest.keyClick(combo, Qt.Key.Key_V)
    assert tools.active() == "zoom"


def test_theme_connections_end_with_the_window(qtbot):
    theme = Theme("light")
    try:
        first = MainWindow(theme=theme)
        first.close()
        shiboken6.delete(first)
        second = MainWindow(theme=theme)
        qtbot.addWidget(second)
        theme.set_scheme("dark")  # must not reach the deleted window
        assert second.plots.map.colors().background == "#121015"
    finally:
        theme.set_scheme("system")
        QGuiApplication.styleHints().setColorScheme(Qt.ColorScheme.Unknown)


def test_empty_states_replace_plots_with_nothing_to_show(shown, sweep, monkeypatch):
    area, c = shown.plot_area, shown.controller
    empty = area.empty
    for view in ("map", "stacked"):  # nothing loaded: open a sweep
        area.set_current_view(view)
        assert not empty.isHidden() and empty.title() == "No sweep loaded"
        assert empty.geometry() == area.stack.geometry()  # it covers the plot and its scale
        assert empty.action_button.text() == "Open sweep…"
    opened = []
    monkeypatch.setattr(
        shown.panels["sample"].measurement, "open_sweep_dialog", lambda: opened.append(1)
    )
    empty.action_button.click()
    assert opened == [1]
    load_sweep(shown, sweep)
    assert empty.title() == "Not processed yet" and empty.action_button.text() == "Process"
    assert not shown.panels["library"].save_button.isEnabled()
    empty.action_button.click()  # processes
    assert c.result is not None and empty.isHidden()
    assert shown.panels["library"].save_button.isEnabled()

    area.set_current_view("reference")  # no reference map: say why, by reference mode
    assert not empty.isHidden() and empty.title() == "No reference in use"
    assert area.ref_kind.isHidden() and area.description.text() == ""
    shown.side_panel.set_open(False, animate=False)
    empty.action_button.click()
    assert shown.current_panel() == "reference" and shown.side_panel.is_open()
    c.set_processing(reference_mode=ReferenceMode.SEPARATE)
    assert empty.title() == "Reference files missing"
    c.set_processing(reference_mode=ReferenceMode.SELF)
    assert empty.title() == "Reference not processed yet"
    empty.action_button.click()
    assert empty.isHidden() and not area.ref_kind.isHidden()
    area.set_current_view("map")
    assert empty.isHidden() and area.ref_kind.isHidden()


def test_toolbar_and_status_bar_details(window, sweep):
    tb, status = window.toolbar, window.statusBar()
    assert tb.per_unit.property("kit") == "chip"  # drawn as a toggle, not as plain text
    assert not tb.per_unit.isEnabled()  # no derivative: dimmed
    assert tb.process_button.key_text() == display.process_key()
    assert tb.export_button.menu() is tb.export_menu and tb.export_button.toolTip()
    for button in (tb.open_button, tb.export_button):  # icon, gap, text (and chevron) fit
        assert button.sizeHint().width() > button.fontMetrics().horizontalAdvance(button.text())
    assert status.contentsMargins().left() == 12
    assert window.windowTitle() == f"Magneto-Optical Detective {__version__}"
    load_sweep(window, sweep)
    process(window)
    assert window.windowTitle() == f"Magneto-Optical Detective {__version__} · Sample_4p2K_Sam1"
    window.controller.set_processing(smooth=True, reference_mode=ReferenceMode.SELF)
    assert f"Settings changed · process again ({display.process_key()})" in window.state_text()


def test_tab_follows_the_layout_of_the_areas(shown, sweep):
    """Toolbar, rail, panel, plot tabs and tools (registry order), plot, inspector, status bar;
    widgets made later (the zero-field remove buttons) join where they are shown."""
    w = shown
    load_sweep(w, sweep)
    process(w)
    QGuiApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)  # rows of earlier lists
    QGuiApplication.processEvents()
    chain = w.update_tab_order()
    tools = [w.tools.tool(name).button for name in w.tools.names()]
    box = w.panels["sample"].measurement
    remove = sorted(box.zero_list.findChildren(QToolButton), key=chain.index)
    assert len(remove) == 2  # made when the files were set, after everything else
    expected = [
        w.toolbar.open_button,
        w.toolbar.process_button,
        w.toolbar.appearance,
        *w._rail_buttons.values(),
        *remove,
        box.add_field_button,
        w.plot_area.tabs,
        *tools,
        w.plot_area.scales_button,
        w.plots.map.view,
        view_page(w).field.lo_spin,
        w.log_button,
    ]
    positions = [chain.index(widget) for widget in expected]
    assert positions == sorted(positions)
    assert [w.tools.tool(n).button for n in ("navigate", "zoom", "pick", "autopick")] == tools[:4]
    walked, widget = [], chain[0]  # Qt's focus chain holds them in this order
    for _ in range(5000):
        if widget in expected and widget not in walked:
            walked.append(widget)
        widget = widget.nextInFocusChain()
    assert walked == expected


def test_keyboard_focus_and_hover_show_on_tabs_and_buttons(shown, qtbot):
    tabs = shown.plot_area.tabs
    qtbot.waitUntil(lambda: shown.plots.map.view.hasFocus())  # no button starts ringed
    shown.toolbar.open_button.setFocus()
    qtbot.waitUntil(lambda: shown.toolbar.open_button.hasFocus())
    plain = tabs.grab().toImage()
    tabs.setFocus(Qt.FocusReason.TabFocusReason)
    qtbot.waitUntil(tabs.hasFocus)
    assert tabs.grab().toImage() != plain  # a ring around the current tab
    QTest.mouseMove(tabs, tabs.tabRect(1).center())
    qtbot.waitUntil(lambda: tabs.hovered == 1)
    buttons = (
        shown.toolbar.open_button,
        shown.toolbar.export_button,
        shown.panels["points"].import_button,
        shown.panels["points"].export_button,
        shown.panels["library"].load_button,
        shown.plot_area.empty.action_button,
    )
    assert all(button.property("kit") == "button" for button in buttons)
    sheet = theme.build_stylesheet(theme.tokens_for(False))
    assert 'QPushButton[kit="button"]:focus' in sheet
