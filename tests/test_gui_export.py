"""The journal figure window (Export > Image…): menus, presets, checks, preview and saving."""

import json
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from PySide6.QtCore import QSettings
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import QFileDialog, QMessageBox

import gui_helpers
from gui_helpers import click_map, infobar_text, load_sweep, process, save_to, set_unit
from mag_opt_detective.core.spectra import FieldMap
from mag_opt_detective.core.units import Unit, from_cm1
from mag_opt_detective.export import (
    APS,
    CUSTOM,
    NATURE,
    FigureStyle,
    StackedOptions,
    TickStyle,
    robust_levels,
)
from mag_opt_detective.gui.controller import AppController
from mag_opt_detective.gui.export_menu import DEFAULTS, ExportSettings
from mag_opt_detective.gui.export_state import (
    POINTS_CURRENT,
    POINTS_NONE,
    FigureContent,
    auto_labels,
    figure_state,
    figure_style,
    file_name,
    print_size,
    safe_name,
    with_format,
)
from mag_opt_detective.gui.main_window import MainWindow

window, errors = gui_helpers.window, gui_helpers.errors  # shared fixtures

MM = 25.4
PT = 72 / MM
MEV = 8.0656
WAIT_MS = 20000


@pytest.fixture
def processed(window, sweep):
    load_sweep(window, sweep)
    process(window)
    return window


@pytest.fixture
def no_dialogs(monkeypatch):
    """Fail on any message box: problems must show inline."""

    def fail(*_args, **_kwargs):
        raise AssertionError("a message box was opened")

    for name in ("warning", "critical", "information"):
        monkeypatch.setattr(QMessageBox, name, fail)


def open_export(window, qtbot):
    """Trigger Export > Image… and wait for the first preview."""
    window.commands["export_figure"].trigger()
    dialog = window.export_dialog
    assert dialog is not None and dialog.isVisible()
    qtbot.waitUntil(
        lambda: dialog.preview.image() is not None and dialog.is_idle(), timeout=WAIT_MS
    )
    return dialog


def redraw(dialog, qtbot) -> None:
    """Draw the preview now (the pause after a change is skipped) and wait for it."""
    with qtbot.waitSignal(dialog.previewUpdated, timeout=WAIT_MS):
        dialog.flush()


def add_model(window, name="model", n=2):
    """A model overlay (as the inspector registers one): *n* flat lines at 300, 400, ... cm-1."""
    b = np.linspace(0.5, 2.0, 20)

    def curves(unit):
        return [(b, from_cm1(np.full(b.shape, 300.0 + 100 * k), unit)) for k in range(n)]

    window.controller.set_overlay(name, curves)


def save_as(dialog, qtbot, monkeypatch, path):
    """Click Save… with the file dialog answering *path*; returns the start name it was given."""
    seen = {}

    def answer(_parent, _title, start, filters):
        seen["start"], seen["filters"] = start, filters
        return str(path), ""

    monkeypatch.setattr(QFileDialog, "getSaveFileName", answer)
    with qtbot.waitSignal(dialog.figureSaved, timeout=WAIT_MS) as blocker:
        dialog.save_button.click()
    return Path(blocker.args[0]), seen


# ---------------------------------------------------------------------- figure state (no window)
@pytest.fixture
def controller(qapp):
    """A controller showing a small cm-1 map, with two curves of points and one model line."""
    c = AppController()
    energy = np.linspace(100.0, 1000.0, 46)
    field = np.array([0.5, 1.0, 1.5, 2.0])
    values = 1 + 0.01 * np.add.outer(np.sin(energy / 50), field)
    c.from_map(FieldMap(energy, field, values, unit="cm-1"))
    c.record_points([1.0, 1.5], [300.0, 320.0], unit="cm-1")
    c.add_curve("LL 2")
    c.record_points([2.0], [500.0], unit="cm-1")
    c.set_overlay("model", lambda unit: [(field, from_cm1(np.full(4, 400.0), unit))])
    return c


def test_figure_state_of_the_controller(controller):
    c = controller
    c.set_view(colormap="viridis")
    c.set_ranges(field_range=(0.6, 1.9), energy_range=(200.0, 900.0))
    c.set_unit("meV")
    state = figure_state(c, FigureContent())
    assert state.kind == "map" and state.fmap.unit is Unit.MEV and state.cmap == "viridis"
    assert state.levels == pytest.approx((0.9, 1.1))  # the fixed R(B)/R(0) default
    assert state.x_range == (0.6, 1.9)
    assert state.y_range == pytest.approx((200 / MEV, 900 / MEV))
    (curve,) = state.curves
    assert curve.style == "model" and curve.label == "model"
    np.testing.assert_allclose(curve.y, 400 / MEV)
    assert [(p.label, p.current) for p in state.points] == [("LL 1", False), ("LL 2", True)]
    np.testing.assert_allclose(state.points[0].y, [300 / MEV, 320 / MEV])

    content = FigureContent(models=False, points=POINTS_CURRENT, x_label="B", y_label=" ")
    state = figure_state(c, content)
    assert state.curves == [] and [p.label for p in state.points] == ["LL 2"]
    assert (state.x_label, state.y_label) == ("B", "")
    assert figure_state(c, FigureContent(points=POINTS_NONE)).points == []

    c.set_view(stacked_offset=0.02, stacked_every=2, stacked_by_field=False)
    state = figure_state(c, FigureContent(kind="stacked"))
    assert state.kind == "stacked" and state.stacked == StackedOptions(0.02, 2, False)
    assert state.x_range == pytest.approx((200 / MEV, 900 / MEV)) and state.y_range is None
    assert state.cmap == "viridis"  # the colour map of the traces
    assert not state.colorbar  # traces are not coloured by field
    assert auto_labels("map", Unit.CM1) == ("Magnetic field (T)", "Energy (cm⁻¹)")
    assert auto_labels("stacked", Unit.THZ) == ("Energy (THz)", "Intensity (a.u.)")
    with pytest.raises(ValueError):
        figure_state(AppController(), FigureContent())  # nothing processed


def test_file_name_from_the_data(controller):
    name = file_name(controller, "map", NATURE, "single", "pdf")
    assert name == "Library_map_Ratio_nature-single.pdf"
    assert file_name(controller, "stacked", NATURE, "1.5 wide", "tif").endswith(
        "_Ratio_stacked_nature-1.5-wide.tif"
    )
    assert file_name(controller, "map", CUSTOM, "", "png").endswith("_Ratio_custom.png")


def test_print_size_rules():
    size = print_size(APS, 86, 65, 8, 0.25, 600, "eps")
    assert size.ok
    assert size.warnings == [f"{APS.name}: lines should be at least 0.5 pt, not 0.25 pt."]
    size = print_size(NATURE, 89, None, 7, 0.5, 450)
    assert not size.ok and size.invalid == {"height"}
    size = print_size(CUSTOM, 120, 90, 30, 1, 600, "svg")
    assert size.font_pt == 24 and size.warnings == ["Custom: text is 4–24 pt; using 24 pt."]


def test_file_names():
    assert with_format("/a/fig", "pdf") == Path("/a/fig.pdf")
    assert with_format("/a/fig.PDF", "pdf") == Path("/a/fig.PDF")
    assert with_format("/a/fig.png", "eps") == Path("/a/fig.eps")
    assert with_format("/a/fig.tiff", "tif") == Path("/a/fig.tiff")
    assert with_format("/a/B_1.5T", "svg") == Path("/a/B_1.5T.svg")
    assert safe_name("Merged by energy: a + b") == "Merged_by_energy_a_+_b"


# ---------------------------------------------------------------------- menus
def test_export_menu_entries(window):
    figure, quick = window.commands["export_figure"], window.commands["export_image"]
    assert figure.text() == "Journal figure…"  # one name for the window, its menu and Help
    assert figure.icon().cacheKey() != quick.icon().cacheKey()
    assert figure.shortcut() == QKeySequence("Ctrl+Shift+E")
    assert quick.text() == "Quick image (PNG/SVG)…" and quick.shortcut().isEmpty()
    actions = window.toolbar.export_menu.actions()
    assert actions.index(window.commands["export_table"]) < actions.index(figure)
    assert actions.index(figure) + 1 == actions.index(quick)
    file_menu = window.menuBar().actions()[0].menu()  # File > Export is the same menu
    assert window.toolbar.export_menu.menuAction() in file_menu.actions()
    texts = [a.text() for a in file_menu.actions() if a.text()]
    assert texts[:4] == [  # sentence case, as the toolbar
        "Open sample sweep…",
        "Load sample zero field…",
        "Open reference sweep…",
        "Load reference zero field…",
    ]


NEW_WINDOW_IMPORTS = """
import sys

import shiboken6


class Excluded:  # the bundle leaves pyplot out (packaging/mag-opt-detective.spec)
    def find_spec(self, name, path=None, target=None):
        if name == "matplotlib.pyplot":
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)


sys.meta_path.insert(0, Excluded())
from PySide6.QtWidgets import QApplication

app = QApplication([])
from mag_opt_detective.gui.main_window import MainWindow

window = MainWindow()
loaded = ("mag_opt_detective.export", "matplotlib.font_manager", "matplotlib.pyplot")
print([name for name in loaded if name in sys.modules])
window.close()
shiboken6.delete(window)
"""


def test_the_export_package_is_loaded_only_for_the_export_window():
    """A new main window (without pyplot, as in the bundle) loads neither the export package nor
    matplotlib's font manager, whose font cache is slow to build the first time."""
    env = {**os.environ, "QT_QPA_PLATFORM": "offscreen"}
    out = subprocess.run(
        [sys.executable, "-c", NEW_WINDOW_IMPORTS],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    assert out.stdout.strip().splitlines()[-1] == "[]"


def test_image_needs_processed_data(window, errors):
    window.commands["export_figure"].trigger()
    assert window.export_dialog is None
    assert "process data first" in infobar_text(window)


def test_closing_the_window_closes_the_export_window(processed, qtbot):
    dialog = open_export(processed, qtbot)
    processed.close()
    assert not dialog.isVisible()


# ---------------------------------------------------------------------- opening
def test_opens_with_the_view_and_unit_on_screen(processed, qtbot, no_dialogs):
    w = processed
    set_unit(w, "meV")
    w.plot_area.set_current_view("stacked")
    dialog = open_export(w, qtbot)
    assert dialog.view.value() == "stacked"
    assert dialog.x_label.placeholderText() == "Energy (meV)"
    assert dialog.y_label.placeholderText() == "Intensity (a.u.)"
    state = dialog.figure_state()
    assert state.kind == "stacked" and state.fmap.unit is Unit.MEV
    image = dialog.preview.image()
    assert image.width() / image.height() == pytest.approx(89 / 70, rel=0.01)
    assert dialog.preview.caption() == "89 × 70 mm · PDF"

    dialog.close()
    w.plot_area.set_current_view("reference")  # not offered: the map
    dialog = open_export(w, qtbot)
    assert dialog.view.value() == "map"
    assert dialog.x_label.placeholderText() == "Magnetic field (T)"
    assert dialog.y_label.placeholderText() == "Energy (meV)"


def test_points_follow_the_markers_of_the_window(processed, qtbot):
    w = processed
    w.tools.set_active("pick")
    click_map(w, 1.0, 300.0)
    markers = w.panels["points"].markers
    markers.set_value("current")
    dialog = open_export(w, qtbot)
    assert dialog.points.isChecked() and dialog.points_scope.value() == "current"
    dialog.close()
    markers.set_value("hidden")
    dialog = open_export(w, qtbot)
    assert not dialog.points.isChecked()
    assert dialog.figure_state().points == []


# ---------------------------------------------------------------------- presets and checks
def test_presets_set_widths_text_and_notes(processed, qtbot):
    dialog = open_export(processed, qtbot)
    nature = dialog.widths["nature"]
    assert dialog.preset.value() == "nature" and dialog.width_stack.currentWidget() is nature
    assert nature.options() == ["single", "1.5 narrow", "1.5 wide", "double"]
    assert [nature.button(k).text() for k in nature.options()] == ["89", "120", "136", "183"]
    size = dialog.print_size()
    assert (size.width_mm, size.height_mm, size.font_pt, size.line_pt, size.dpi) == (
        89,
        70,
        7,
        0.5,
        450,
    )
    assert "Text 5–7 pt" in dialog.notes.text() and dialog.width_caption.text() == "Single column"
    assert dialog.panel_label.button("b").toolTip() == "Panel b: bold, top left"

    dialog.preset.set_value("aps")
    aps = dialog.widths["aps"]
    assert dialog.width_stack.currentWidget() is aps and aps.options() == ["single", "double"]
    size = dialog.print_size()
    assert (size.width_mm, size.height_mm, size.font_pt, size.line_pt, size.dpi) == (
        86,
        65,
        8,
        0.75,
        600,
    )
    assert "not confirmed" in dialog.notes.text()
    assert dialog.width_caption.text() == "Single column · 8.6 cm"
    aps.set_value("double")
    assert dialog.print_size().width_mm == 178
    assert dialog.width_caption.text() == "Double column · 17.8 cm"
    assert dialog.panel_label.button("b").toolTip() == "Panel (b): bold, top left"
    assert dialog.panel_caption.text() == "Panel label · APS style: (a), bold"

    dialog.preset.set_value("custom")
    assert dialog.width_stack.currentWidget() is dialog.custom_width
    dialog.custom_width.edit.setText("100")
    assert dialog.print_size().width_mm == 100
    dialog.preset.set_value("nature")  # back: the journal's defaults again
    assert dialog.print_size().width_mm == 89 and dialog.print_size().font_pt == 7


def test_sizes_are_checked_with_inline_warnings(processed, qtbot, no_dialogs):
    dialog = open_export(processed, qtbot)
    dialog.height_field.edit.setText("300")  # capped at the page depth
    assert dialog.print_size().height_mm == 247
    assert "Nature: height is at most 247 mm; using 247 mm." in dialog.messages.texts("warn")
    assert dialog.preview.caption().startswith("89 × 247 mm")
    dialog.font_field.edit.setText("9")  # kept in the journal's range
    assert dialog.print_size().font_pt == 7
    assert any("5–7 pt; using 7 pt" in text for text in dialog.messages.texts("warn"))
    dialog.line_field.edit.setText("2")
    assert any("at most 1 pt" in text for text in dialog.messages.texts("warn"))
    dialog.format.set_value("png")
    dialog.dpi_field.edit.setText("150")  # raised to the minimum
    assert dialog.print_size().dpi == 300
    assert any("at least 300 dpi" in text for text in dialog.messages.texts("warn"))
    assert dialog.dpi_hint.text() == f"{round(89 / MM * 300)} × {round(247 / MM * 300)} px"
    dialog.format.set_value("svg")
    assert "Nature asks for PDF, EPS, TIFF or PNG files." in dialog.messages.texts("warn")
    assert dialog.save_button.isEnabled()  # warnings do not stop saving
    assert not dialog.messages.texts("err")

    dialog.preset.set_value("custom")
    assert not dialog.messages.texts("warn")
    for text, problem in (("", "Enter the width in mm."), ("2000", "5–1000 mm")):
        dialog.custom_width.edit.setText(text)
        assert not dialog.print_size().ok and dialog.custom_width.is_invalid()
        assert any(problem in message for message in dialog.messages.texts("err"))
        assert not dialog.save_button.isEnabled()
    dialog.custom_width.edit.setText("1000")
    dialog.height_field.edit.setText("1000")
    dialog.dpi_field.edit.setText("2400")
    assert dialog.dpi_field.is_invalid()  # far too many pixels
    assert any("million pixels" in message for message in dialog.messages.texts("err"))
    dialog.dpi_field.edit.setText("300")
    dialog.custom_width.edit.setText("120")
    assert dialog.print_size().ok and not dialog.custom_width.is_invalid()
    assert dialog.save_button.isEnabled() and not dialog.messages.texts("err")


# ---------------------------------------------------------------------- preview
def test_preview_follows_changes_and_the_window(processed, qtbot, no_dialogs):
    w = processed
    result = w.controller.result
    dialog = open_export(w, qtbot)
    first = dialog.preview.image()
    dialog.widths["nature"].set_value("double")
    assert not dialog.is_idle() and dialog.preview.image() is first  # drawn after a pause
    redraw(dialog, qtbot)
    image = dialog.preview.image()
    assert image is not first
    assert image.width() / image.height() == pytest.approx(183 / 70, rel=0.01)
    assert dialog.preview.caption() == "183 × 70 mm · PDF"

    with qtbot.waitSignal(dialog.previewUpdated, timeout=WAIT_MS):
        set_unit(w, "THz")  # a live switch in the window: shown here too, nothing reprocessed
    assert dialog.x_label.placeholderText() == "Magnetic field (T)"
    assert dialog.y_label.placeholderText() == "Energy (THz)"
    assert dialog.figure_state().fmap.unit is Unit.THZ
    assert w.controller.result is result


def test_preview_follows_model_curves_shown_in_the_window(processed, qtbot, no_dialogs):
    """A model shown or edited in the inspector redraws the open export window at once."""
    w = processed
    dialog = open_export(w, qtbot)
    assert not dialog.models.isEnabled()
    with qtbot.waitSignal(dialog.previewUpdated, timeout=WAIT_MS):
        add_model(w, n=2)
        w.controller.notify_overlays()  # as the inspector does
    assert dialog.models.isEnabled()
    assert dialog.models_row.description_label.text() == "2 curves from the window"
    assert len(dialog.figure_state().curves) == 2


def test_a_label_that_cannot_be_drawn_is_reported_inline(processed, qtbot, no_dialogs):
    dialog = open_export(processed, qtbot)
    dialog.x_label.setText("$x^$")  # math text with a missing superscript
    with qtbot.waitSignal(dialog.renderer.failed, timeout=WAIT_MS):
        dialog.flush()
    assert dialog.preview.message().startswith("The preview could not be drawn:")
    dialog.x_label.setText("B (T)")
    redraw(dialog, qtbot)
    assert dialog.preview.message() == ""


def test_figure_state_reflects_the_window(processed, qtbot):
    w = processed
    c = w.controller
    w.tools.set_active("pick")
    click_map(w, 1.0, 300.0)
    click_map(w, 1.5, 320.0)
    c.add_curve("LL 2")
    click_map(w, 2.0, 500.0)
    add_model(w, n=3)
    c.set_view(colormap="viridis")
    c.set_levels(c.selection.level_key, 0.95, 1.05)
    c.set_ranges(field_range=(0.6, 1.9), energy_range=(200.0, 900.0))
    set_unit(w, "meV")
    dialog = open_export(w, qtbot)

    state = dialog.figure_state()
    assert state.kind == "map" and state.fmap.unit is Unit.MEV
    np.testing.assert_allclose(state.fmap.values, c.result.ratio.values)
    np.testing.assert_allclose(state.fmap.energy, c.result.ratio.energy / MEV)
    assert state.cmap == "viridis" and state.levels == pytest.approx((0.95, 1.05))
    assert state.x_range == pytest.approx((0.6, 1.9))
    assert state.y_range == pytest.approx((200 / MEV, 900 / MEV))
    assert len(state.curves) == 3 and {curve.style for curve in state.curves} == {"model"}
    np.testing.assert_allclose(state.curves[0].y, 300 / MEV)
    assert [(p.label, p.current) for p in state.points] == [("LL 1", False), ("LL 2", True)]
    np.testing.assert_allclose(state.points[0].x, [1.0, 1.5])
    np.testing.assert_allclose(state.points[0].y, [300 / MEV, 320 / MEV])
    assert dialog.models_row.description_label.text() == "3 curves from the window"
    assert dialog.points_row.description_label.text() == "3 points on 2 curves"

    dialog.points_scope.set_value("current")
    assert [p.label for p in dialog.figure_state().points] == ["LL 2"]
    dialog.points.setChecked(False)
    dialog.models.setChecked(False)
    dialog.colorbar.setChecked(False)
    dialog.x_label.setText("B (T)")
    dialog.y_label.setText("  ")  # only spaces: no label
    dialog.colorbar_label.setText("ΔT/T")
    state = dialog.figure_state()
    assert state.points == [] and state.curves == [] and not state.colorbar
    assert (state.x_label, state.y_label, state.colorbar_label) == ("B (T)", "", "ΔT/T")
    dialog.x_label.clear()
    assert dialog.figure_state().x_label is None  # automatic


def test_symmetric_and_auto_levels_are_the_ones_drawn(processed, qtbot):
    w = processed
    c = w.controller
    gui_helpers.select(w, order=1)  # a derivative: symmetric about 0 by default
    dialog = open_export(w, qtbot)
    levels = dialog.figure_state().levels
    assert levels == pytest.approx(c.current_levels())
    assert levels[0] == pytest.approx(-levels[1])


def test_stacked_figure_uses_the_trace_options(processed, qtbot):
    w = processed
    c = w.controller
    add_model(w)
    c.set_view(colormap="plasma", stacked_offset=0.05, stacked_every=2, stacked_by_field=False)
    c.set_ranges(energy_range=(200.0, 800.0), stacked_range=(0.9, 1.5))
    w.plot_area.set_current_view("stacked")
    dialog = open_export(w, qtbot)
    state = dialog.figure_state()
    assert state.kind == "stacked" and state.stacked == StackedOptions(0.05, 2, False)
    assert state.x_range == (200.0, 800.0) and state.y_range == (0.9, 1.5)
    assert state.cmap == "plasma" and state.curves == []
    assert not state.colorbar  # traces not coloured by field: no field bar
    assert not dialog.colorbar.isEnabled() and not dialog.models.isEnabled()
    assert not dialog.models.isChecked()  # shown off, as "Not drawn on stacked spectra" says
    assert dialog.colorbar_label.placeholderText() == "Magnetic field (T)"
    c.set_view(stacked_by_field=True)
    assert dialog.figure_state().colorbar and dialog.colorbar.isEnabled()
    dialog.view.set_value("map")  # the choice for the map comes back
    assert dialog.models.isEnabled() and dialog.models.isChecked()
    dialog.models.setChecked(False)
    dialog.view.set_value("stacked")
    dialog.view.set_value("map")
    assert not dialog.models.isChecked()


def test_labels_are_kept_per_view(processed, qtbot):
    dialog = open_export(processed, qtbot)
    dialog.x_label.setText("B")
    dialog.view.set_value("stacked")
    assert dialog.x_label.text() == ""
    dialog.x_label.setText("E")
    dialog.view.set_value("map")
    assert dialog.x_label.text() == "B"


# ---------------------------------------------------------------------- saving
def _pdf_size_mm(path):
    (box,) = re.findall(rb"/MediaBox\s*\[([^\]]*)\]", path.read_bytes())
    x0, y0, x1, y1 = (float(v) for v in box.split())
    return (x1 - x0) / PT, (y1 - y0) / PT


@pytest.mark.parametrize("fmt", ["pdf", "png", "svg"])
def test_save_writes_the_print_size(processed, qtbot, tmp_path, monkeypatch, no_dialogs, fmt):
    w = processed
    add_model(w)
    dialog = open_export(w, qtbot)
    dialog.format.set_value(fmt)
    dialog.dpi_field.edit.setText("300")
    dialog.panel_label.set_value("a")
    path, seen = save_as(dialog, qtbot, monkeypatch, tmp_path / "figure")
    assert path == tmp_path / f"figure.{fmt}" and path.is_file()
    assert Path(seen["start"]).name == f"Sample_4p2K_Sam1_Ratio_nature-single.{fmt}"
    assert f"*.{fmt}" in seen["filters"]
    if fmt == "pdf":
        assert _pdf_size_mm(path) == pytest.approx((89, 70), abs=0.01)
    elif fmt == "png":
        with Image.open(path) as image:
            assert image.size == (round(89 / MM * 300), round(70 / MM * 300))
            assert image.info["dpi"] == pytest.approx((300, 300), abs=0.01)
    else:
        root = ET.parse(path).getroot()
        width, height = (float(root.attrib[k].removesuffix("pt")) for k in ("width", "height"))
        assert (width / PT, height / PT) == pytest.approx((89, 70), abs=0.01)
    assert dialog.messages.texts("ok") == [f"Saved figure.{fmt}"]
    assert w.statusBar().currentMessage() == f"Saved figure figure.{fmt}"


def test_save_keeps_a_typed_name_and_uses_the_format(processed, qtbot, tmp_path, monkeypatch):
    dialog = open_export(processed, qtbot)
    dialog.preset.set_value("aps")
    dialog.widths["aps"].set_value("double")
    dialog.view.set_value("stacked")
    dialog.format.set_value("tif")
    name = file_name(processed.controller, "stacked", dialog.current_preset(), "double", "tif")
    assert name == dialog.default_file_name() == "Sample_4p2K_Sam1_Ratio_stacked_aps-double.tif"
    path, _seen = save_as(dialog, qtbot, monkeypatch, tmp_path / "fig.png")
    assert path == tmp_path / "fig.tif"
    with Image.open(path) as image:
        assert image.format == "TIFF"
        assert image.size == (round(178 / MM * 600), round(65 / MM * 600))


def test_save_error_shows_inline(processed, qtbot, tmp_path, monkeypatch, no_dialogs):
    dialog = open_export(processed, qtbot)
    save_to(monkeypatch, tmp_path / "missing" / "figure.pdf")
    with qtbot.waitSignal(dialog.renderer.failed, timeout=WAIT_MS):
        dialog.save_button.click()
    assert any("Could not save the figure" in text for text in dialog.messages.texts("err"))
    assert dialog.save_button.isEnabled() and processed.infobar.isHidden()


# ---------------------------------------------------------------------- settings
def make_window(qtbot, ini, sweep):
    w = MainWindow(settings=QSettings(ini, QSettings.Format.IniFormat))
    qtbot.addWidget(w)
    load_sweep(w, sweep)
    process(w)
    return w


def test_settings_round_trip(qtbot, tmp_path, sweep, errors):
    ini = str(tmp_path / "settings.ini")
    w = make_window(qtbot, ini, sweep)
    dialog = open_export(w, qtbot)
    dialog.preset.set_value("aps")
    dialog.widths["aps"].set_value("double")
    dialog.height_field.edit.setText("80")
    dialog.font_field.edit.setText("9")
    dialog.line_field.edit.setText("1")
    dialog.format.set_value("png")
    dialog.dpi_field.edit.setText("900")
    dialog.colorbar.setChecked(False)
    dialog.colorbar_label.setText("ΔT/T")
    dialog.panel_label.set_value("b")
    w.close()
    stored = json.loads(QSettings(ini, QSettings.Format.IniFormat).value("v2/export/figure"))
    assert stored["preset"] == "aps" and stored["dpi"] == 900

    w2 = make_window(qtbot, ini, sweep)
    dialog = open_export(w2, qtbot)
    assert dialog.preset.value() == "aps" and dialog.widths["aps"].value() == "double"
    fields = (dialog.height_field, dialog.font_field, dialog.line_field, dialog.dpi_field)
    assert [f.edit.text() for f in fields] == ["80", "9", "1", "900"]
    assert dialog.format.value() == "png" and not dialog.colorbar.isChecked()
    assert dialog.colorbar_label.text() == "ΔT/T" and dialog.panel_label.value() == "b"
    assert dialog.print_size().width_mm == 178
    w2.reset_settings()  # back to the defaults
    assert dialog.preset.value() == "nature" and dialog.format.value() == "pdf"
    assert not errors


def test_export_settings_drop_invalid_values():
    settings = ExportSettings()
    assert settings.set_settings_value(
        json.dumps({"preset": 3, "height": "x", "dpi": 300, "colorbar": False, "extra": 1})
    )
    assert settings.values == {**DEFAULTS, "dpi": 300, "colorbar": False}
    assert not settings.set_settings_value("not json")
    assert not settings.set_settings_value("[1, 2]")


# ---------------------------------------------------------------------- colour range and style


def test_colour_range_of_the_adapter(controller):
    c = controller
    c.set_ranges(field_range=(0.6, 1.4))  # zoomed: Auto still uses the whole map
    window_levels = figure_state(c, FigureContent()).levels
    assert window_levels == pytest.approx((0.9, 1.1))
    auto = figure_state(c, FigureContent(colour_range="auto")).levels
    assert auto == pytest.approx(robust_levels(c.current_map().values))
    fixed = FigureContent(colour_range="fixed", fixed_levels=(0.98, 1.02))
    assert figure_state(c, fixed).levels == (0.98, 1.02)
    unset = FigureContent(colour_range="fixed")  # nothing typed yet: the window's
    assert figure_state(c, unset).levels == pytest.approx((0.9, 1.1))


def test_style_of_the_adapter():
    check = figure_style("top", "in", True, "4", "", True, 5, "1.5")
    assert check.ok
    assert check.style == FigureStyle("top", TickStyle("in", True, 4.0, None, True, 5, 1.5))
    check = figure_style(length="30", width="x")
    assert not check.ok and check.invalid == {"tick_length", "tick_width"}
    assert check.style.ticks == TickStyle()  # refused values are drawn automatic
    assert check.errors == [
        "The tick length must be 0–20 pt.",
        "Enter the tick width in pt, or leave it empty.",
    ]
