"""The rail's data states (Sample, Reference), its labels, and Process without reference files."""

import dataclasses

import numpy as np
import pytest

import gui_helpers
from gui_helpers import load_sweep, process
from mag_opt_detective.core.pipeline import ReferenceMode
from mag_opt_detective.gui import icons
from mag_opt_detective.gui.controller import (
    NO_REFERENCE_FILES,
    AppController,
    ProcessingState,
    SweepFiles,
)

window, errors = gui_helpers.window, gui_helpers.errors


@pytest.fixture
def controller(qapp):
    return AppController()


def files(sweep) -> SweepFiles:
    return SweepFiles(tuple(sweep["zero"]), tuple(sweep["field"]))


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
