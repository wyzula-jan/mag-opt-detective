"""The rail's data states (Sample, Reference), its labels, and Process without reference files."""

import dataclasses

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QGuiApplication, QPalette
from PySide6.QtWidgets import QApplication, QToolButton

import gui_helpers
from gui_helpers import load_sweep, process, set_unit
from mag_opt_detective.core.pipeline import ReferenceMode
from mag_opt_detective.gui import icons, theme
from mag_opt_detective.gui.controller import (
    DATA_CHANGED,
    DATA_CURRENT,
    DATA_EMPTY,
    NO_REFERENCE_FILES,
    AppController,
    FieldRange,
    ProcessingState,
    SweepFiles,
)
from mag_opt_detective.gui.main_window import MainWindow
from mag_opt_detective.gui.theme import DARK, LIGHT, Theme

window, errors = gui_helpers.window, gui_helpers.errors


@pytest.fixture
def controller(qapp):
    return AppController()


@pytest.fixture
def app(qapp):
    """The application, with its look restored after the test applied a theme."""
    palette, sheet, style = QPalette(qapp.palette()), qapp.styleSheet(), qapp.style().name()
    yield qapp
    qapp.setStyleSheet(sheet)
    qapp.setPalette(palette)
    if qapp.style().name() != style:
        QApplication.setStyle(style)
    QGuiApplication.styleHints().setColorScheme(Qt.ColorScheme.Unknown)
    theme._active = None
    icons.clear_cache()


def files(sweep) -> SweepFiles:
    return SweepFiles(tuple(sweep["zero"]), tuple(sweep["field"]))


def states(c: AppController) -> tuple[str, str]:
    return c.data_state("sample"), c.data_state("reference")


def tile(button: QToolButton) -> QColor:
    """The colour at the middle of the icon's top edge: a filled icon's tile, else nothing."""
    return button.icon().pixmap(20, 20).toImage().pixelColor(10, 0)


# ---------------------------------------------------------------------- reference fallback
def test_separate_sweep_without_files_counts_as_no_reference():
    separate = ProcessingState(reference_mode=ReferenceMode.SEPARATE, smooth=True)
    assert separate.reference_used() is ReferenceMode.NONE
    assert separate.effective() == ProcessingState().effective()
    with_files = dataclasses.replace(separate, reference_files=SweepFiles(("z",), ("f",)))
    assert with_files.reference_used() is ReferenceMode.SEPARATE
    assert with_files.effective() != ProcessingState().effective()
    half = dataclasses.replace(separate, reference_files=SweepFiles(("z",), ()))
    assert half.reference_used() is ReferenceMode.SEPARATE  # given files are used (and fail)


def test_process_without_reference_files_shows_the_sample_alone(controller, sweep):
    notes = []
    controller.referenceMissing.connect(notes.append)
    controller.set_processing(sample_files=files(sweep))
    plain = controller.process()
    assert not notes
    controller.set_processing(reference_mode=ReferenceMode.SEPARATE, smooth=True)
    assert not controller.changed_since_process()  # Process would show the same
    result = controller.process()
    assert notes == [NO_REFERENCE_FILES]
    assert result.reference_data is None and result.reference_ratio is None
    assert controller.reference_map() is None
    for name in ("data", "ratio", "average", "step"):
        np.testing.assert_allclose(getattr(result, name).values, getattr(plain, name).values)
    controller.set_processing(reference_files=files(sweep))
    assert controller.changed_since_process()  # a reference now: process again
    result = controller.process()
    assert notes == [NO_REFERENCE_FILES] and result.reference_data is not None
    for mode in (ReferenceMode.SELF, ReferenceMode.NONE):  # these modes are unchanged
        controller.set_processing(reference_mode=mode, reference_files=SweepFiles())
        controller.process()
    assert notes == [NO_REFERENCE_FILES]


def test_reference_files_that_cannot_be_read_still_fail(controller, sweep, tmp_path):
    notes = []
    controller.referenceMissing.connect(notes.append)
    controller.set_processing(sample_files=files(sweep), reference_mode=ReferenceMode.SEPARATE)
    controller.set_processing(reference_files=SweepFiles(tuple(sweep["zero"]), ()))
    with pytest.raises(ValueError, match="no field files") as info:
        controller.process()
    assert info.value.panel == "reference"
    missing = str(tmp_path / "Gone_a01p000T.txt")
    controller.set_processing(reference_files=SweepFiles(tuple(sweep["zero"]), (missing,)))
    with pytest.raises(OSError) as info:
        controller.process()
    assert info.value.panel == "reference"
    assert controller.result is None and not notes


def test_the_window_notes_a_process_without_reference_files(window, sweep, errors):
    load_sweep(window, sweep)
    window.panels["reference"].set_reference_mode(ReferenceMode.SEPARATE)
    window.show_panel("points")
    process(window)
    c, bar = window.controller, window.infobar
    assert not errors and c.result is not None
    assert gui_helpers.shown_image(window) is not None  # the sample is plotted
    assert not bar.isHidden() and bar.level() == "warning"  # a note, not an error
    assert bar.title_label.text() == "Reference sweep has no files – shown without reference."
    assert bar.text_label.text() == "Add them in the Reference panel, or set Reference to None."
    assert "Reference sweep has no files" in window.console.toPlainText()
    assert window.log_button.unseen() == 0  # a warning, not an error
    bar.action_button.click()
    assert window.current_panel() == "reference" and bar.isHidden()
    assert not window.toolbar.process_button.dot
    reference = window.panels["reference"]
    reference.zero_list.set_paths(sweep["zero"])
    reference.field_list.set_paths(sweep["field"])
    assert c.changed_since_process() and window.toolbar.process_button.dot
    process(window)
    assert bar.isHidden() and c.reference_map() is not None and not errors


# ---------------------------------------------------------------------- data states
def test_sample_state_follows_load_process_change_and_clear(controller, sweep):
    c = controller
    seen = []
    c.dataStateChanged.connect(lambda part, state: seen.append((part, state)))
    assert states(c) == (DATA_EMPTY, DATA_EMPTY)
    c.set_processing(sample_files=files(sweep))
    assert states(c) == (DATA_CHANGED, DATA_EMPTY)  # loaded, not processed yet
    c.process()
    assert states(c) == (DATA_CURRENT, DATA_EMPTY)
    assert seen == [("sample", DATA_CHANGED), ("sample", DATA_CURRENT)]
    c.set_unit("meV")  # not a change
    c.set_processing(energy_cut=(200.0, 800.0))  # the Processing panel's, not the sample's
    c.from_map(c.result.ratio)  # a library map describes nothing of the last Process
    assert states(c) == (DATA_CURRENT, DATA_EMPTY) and len(seen) == 2
    c.set_processing(energy_cut=None)
    c.set_processing(sample_files=SweepFiles(tuple(sweep["zero"]), tuple(sweep["field"][:3])))
    assert c.data_state("sample") == DATA_CHANGED
    c.set_processing(sample_files=files(sweep))  # back to the files processed
    assert c.data_state("sample") == DATA_CURRENT
    c.set_processing(custom_field=True, sample_field=FieldRange(1.0, 1.0, 4.0))
    assert c.data_state("sample") == DATA_CHANGED
    c.process()
    c.set_processing(sample_field=FieldRange(1.0, 0.5, 2.5))
    assert c.data_state("sample") == DATA_CHANGED
    c.set_processing(sample_files=SweepFiles())  # cleared
    assert states(c) == (DATA_EMPTY, DATA_EMPTY)
    assert seen[-1] == ("sample", DATA_EMPTY)
    with pytest.raises(ValueError):
        c.data_state("points")


def test_reference_state_follows_its_mode(controller, sweep):
    c = controller
    c.set_processing(sample_files=files(sweep))
    c.process()
    c.set_processing(reference_mode=ReferenceMode.SELF)  # the sample is the reference
    assert states(c) == (DATA_CURRENT, DATA_CHANGED)
    c.process()
    assert states(c) == (DATA_CURRENT, DATA_CURRENT)
    c.set_processing(smooth=True)  # the reference's own options
    assert states(c) == (DATA_CURRENT, DATA_CHANGED)
    c.process()
    c.set_processing(sample_files=SweepFiles(tuple(sweep["zero"]), tuple(sweep["field"][1:])))
    assert states(c) == (DATA_CHANGED, DATA_CHANGED)  # the sample changes its reference
    c.set_processing(sample_files=files(sweep))

    c.set_processing(reference_mode=ReferenceMode.SEPARATE)
    assert states(c) == (DATA_CURRENT, DATA_EMPTY)  # no reference files
    c.set_processing(reference_files=files(sweep))
    assert states(c) == (DATA_CURRENT, DATA_CHANGED)
    c.process()
    assert states(c) == (DATA_CURRENT, DATA_CURRENT)
    for name, value in (
        ("reference_files", SweepFiles(tuple(sweep["zero"]), tuple(sweep["field"][:3]))),
        ("smooth", False),
        ("sg_window", 21),
        ("custom_field", True),  # the reference's fields too
    ):
        before = getattr(c.processing, name)
        c.set_processing(**{name: value})
        assert c.data_state("reference") == DATA_CHANGED, name
        c.set_processing(**{name: before})
        assert states(c) == (DATA_CURRENT, DATA_CURRENT), name
    c.set_processing(reference_mode=ReferenceMode.NONE)
    assert states(c) == (DATA_CURRENT, DATA_EMPTY)


# ---------------------------------------------------------------------- the rail
def test_reference_rail_label_is_short(window):
    button = window.rail_button("reference")
    assert button.text() == "Ref" and button.accessibleName() == "Reference"
    assert button.toolTip().startswith("Reference")
    assert window.panel_pages["reference"].title.text() == "Reference"
    assert window.rail_button("sample").accessibleName() == "Sample"


def test_rail_buttons_show_the_data_states(window, sweep, errors):
    sample, reference = window.rail_button("sample"), window.rail_button("reference")
    assert sample.data_state == DATA_EMPTY and not sample.badge
    assert sample.toolTip() == "Sample files – no files"
    assert reference.toolTip() == "Reference measurement – none in use"
    assert tile(sample).alpha() == 0  # the outline icon
    load_sweep(window, sweep)
    assert sample.data_state == DATA_CHANGED and sample.badge
    assert sample.toolTip() == "Sample files – not processed yet"
    assert sample.accessibleDescription() == sample.toolTip()
    assert tile(sample).alpha() == 0
    process(window)
    assert sample.data_state == DATA_CURRENT and not sample.badge
    assert sample.toolTip() == "Sample files – processed"
    accent = theme.current_tokens()["accent"]
    assert tile(sample).name() == accent.name()  # the filled icon
    set_unit(window, "meV")
    assert sample.data_state == DATA_CURRENT

    window.panels["reference"].set_reference_mode(ReferenceMode.SELF)
    assert reference.data_state == DATA_CHANGED and reference.badge
    assert reference.toolTip() == (
        "Reference measurement – the sample itself, changed since last Process"
    )
    process(window)
    assert reference.data_state == DATA_CURRENT and tile(reference).name() == accent.name()
    window.panels["reference"].set_reference_mode(ReferenceMode.SEPARATE)
    assert reference.data_state == DATA_EMPTY and reference.toolTip().endswith("– no files")
    window.panels["sample"].measurement.field_list.set_paths(sweep["field"][:2])
    assert sample.data_state == DATA_CHANGED and sample.badge
    assert sample.toolTip() == "Sample files – changed since last Process"

    window.show_panel("library")
    for button, state in ((sample, DATA_CHANGED), (reference, DATA_EMPTY)):
        button.click()  # selection and data state are independent
        assert button.isChecked() and button.data_state == state


def test_filled_rail_icons_follow_the_theme(app, qtbot, sweep):
    t = Theme("light")
    t.apply(app)
    w = MainWindow(theme=t)
    qtbot.addWidget(w)
    try:
        load_sweep(w, sweep)
        process(w)
        sample = w.rail_button("sample")
        assert tile(sample).name() == LIGHT["accent"]
        w.set_appearance("dark")
        assert sample.data_state == DATA_CURRENT and tile(sample).name() == DARK["accent"]
    finally:
        w.close()
        w.deleteLater()


def test_filled_icons_draw_the_glyph_on_a_tile(qapp):
    icons.clear_cache()
    filled = icons.icon("activity", "#ffffff", fill="#7a2a8c")
    assert filled.cacheKey() == icons.icon("activity", "#ffffff", fill="#7a2a8c").cacheKey()
    assert filled.cacheKey() != icons.icon("activity", "#ffffff").cacheKey()
    image = filled.pixmap(20, 20).toImage()
    assert image.pixelColor(10, 0).name() == "#7a2a8c"  # the tile
    colours = {image.pixelColor(x, y).name() for x in range(20) for y in range(20)}
    assert "#ffffff" in colours  # the glyph on it
    for name in ("activity", "layers"):  # the SVGs stay Lucide's outlines
        assert b'fill="none"' in icons.svg_data(name)
