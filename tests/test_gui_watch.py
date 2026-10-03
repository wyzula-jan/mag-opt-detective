"""Watching a measurement folder (gui/watch.py and the Sample panel's Watch folder block)."""

import logging
import os
import time
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QFileDialog

import gui_helpers
from helpers import sweep_name, write_opus, write_text
from mag_opt_detective.core.pipeline import ReferenceMode
from mag_opt_detective.gui import watch
from mag_opt_detective.gui.controller import NO_REFERENCE_FILES, SweepFiles
from mag_opt_detective.gui.main_window import MainWindow
from mag_opt_detective.gui.panels.sample import WatchChip

window, errors = gui_helpers.window, gui_helpers.errors

X = np.linspace(100.0, 1000.0, 91)
BASE = 1.0 + 0.5 * np.sin(X / 50.0)
BEFORE = "Sample_4p2K_Sam1_a00p000T_a00p000T.txt"  # zero field before the sweep
AFTER = "Sample_4p2K_Sam1_a00p000T_a02p000T.txt"  # ... and after it, up to 2 T
SETTLE = watch.SETTLE_S + 0.5


def spectrum(folder: Path, b: float, old: bool = False) -> str:
    """An in-field spectrum (1 + 0.1 B) * BASE; *old*: written long ago (complete)."""
    return _aged(write_text(folder / sweep_name(b), X, (1 + 0.1 * b) * BASE), old)


def zero(folder: Path, name: str = BEFORE, scale: float = 1.0, old: bool = False) -> str:
    return _aged(write_text(folder / name, X, scale * BASE), old)


def _aged(path: Path, old: bool) -> str:
    if old:
        past = time.time() - 3600
        os.utime(path, (past, past))
    return str(path)


class Clock:
    """The watcher's monotonic clock, moved by hand."""

    def __init__(self):
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock(window):
    clock = Clock()
    window.folder_watch.clock = clock
    return clock


def arrive(window, clock) -> None:
    """Let the watcher look at the folder, then again once new files are complete."""
    watcher = window.folder_watch
    watcher.check_now()
    clock.now += SETTLE
    watcher.check_now()


def watch_folder(window, folder: Path, monkeypatch) -> None:
    """Watch *folder* with Watch a folder… (the dialog answers *folder*)."""
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *_args, **_kwargs: str(folder))
    window.panels["sample"].watch.choose_button.click()


def fields(window) -> list[float]:
    return list(window.controller.result.ratio.field)


def state_text(window) -> str:
    return window.panels["sample"].watch.state.text()


def chip(window) -> WatchChip:
    return window.statusBar().findChildren(WatchChip)[0]


def count(signal) -> list:
    calls = []
    signal.connect(lambda *args: calls.append(args))
    return calls


# ---------------------------------------------------------------------- the two ways in
def test_watch_the_folder_of_the_loaded_sweep(window, clock, tmp_path):
    folder = tmp_path / "Run 7"
    folder.mkdir()
    files = SweepFiles(
        (zero(folder, old=True),), tuple(spectrum(folder, b, True) for b in (0.5, 1))
    )
    c = window.controller
    c.set_processing(sample_files=files)
    gui_helpers.process(window)
    box = window.panels["sample"].watch
    assert box.switch.isEnabled() and not box.switch.isChecked()
    assert "Run 7" in box.row.description_label.text().replace("\u200b", "")
    results = count(c.resultChanged)

    box.switch.setChecked(True)
    watcher = window.folder_watch
    assert watcher.watching() and watcher.folder() == str(folder)
    assert results == []  # processed as listed already: nothing to do
    assert c.spectrum_cache() is not None

    spectrum(folder, 1.5)
    watcher.check_now()
    assert len(c.processing.sample_files.field) == 2  # not complete yet
    assert watcher.interval() == watch.RECHECK_MS
    clock.now += SETTLE
    watcher.check_now()
    assert len(results) == 1 and fields(window) == [0.5, 1.0, 1.5]
    assert watcher.interval() == watch.POLL_MS
    assert state_text(window).startswith("Watching · 4 files · last update ")
    assert chip(window).text().startswith("Watching Run 7 · 4 files · ")
    assert not c.changed_since_process()


def test_watch_an_empty_folder_until_a_sweep_arrives(window, clock, sweep, tmp_path, monkeypatch):
    gui_helpers.load_sweep(window, sweep)
    gui_helpers.process(window)
    folder = tmp_path / "running"
    folder.mkdir()
    watch_folder(window, folder, monkeypatch)
    c = window.controller
    assert window.folder_watch.folder() == str(folder)
    assert c.processing.sample_files == SweepFiles()  # a new sweep
    assert state_text(window) == "Waiting for in-field spectra · 0 files"

    spectrum(folder, 0.25)
    arrive(window, clock)
    assert c.processing.sample_files.field and state_text(window).startswith(
        "Waiting for a zero-field spectrum · 1 file"
    )
    assert fields(window) == list(sweep["fields"])  # the old map stays meanwhile

    zero(folder)
    spectrum(folder, 0.5)
    arrive(window, clock)
    assert fields(window) == [0.25, 0.5]
    assert state_text(window).startswith("Watching · 3 files · last update")
    assert {0.25, 0.5} <= set(c.points.field)  # rows for the new fields


def test_watching_a_folder_with_a_sweep_loads_it_at_once(window, clock, tmp_path, monkeypatch):
    zero(tmp_path, old=True)
    for b in (0.5, 1.0):
        spectrum(tmp_path, b, old=True)
    watch_folder(window, tmp_path, monkeypatch)
    assert fields(window) == [0.5, 1.0]


# ---------------------------------------------------------------------- complete files only
def test_a_file_still_written_waits_until_it_stopped_changing(window, clock, tmp_path):
    zero(tmp_path, old=True)
    spectrum(tmp_path, 0.5, old=True)
    watcher = window.folder_watch
    watcher.start(tmp_path)
    c = window.controller
    path = tmp_path / sweep_name(1.0)
    lines = [f"{a:.8f}\t{b:.8f}\r\n" for a, b in zip(X, 1.1 * BASE, strict=True)]
    path.write_text("".join(lines[:40]))
    watcher.check_now()
    clock.now += 1.0
    with path.open("a") as fh:
        fh.write("".join(lines[40:]))
    os.utime(path, ns=(time.time_ns(), time.time_ns() + 1000))
    watcher.check_now()
    clock.now += 1.5  # 2.5 s since it was first seen, but only 1.5 s since it changed
    watcher.check_now()
    assert str(path) not in c.processing.sample_files.field
    clock.now += 1.0
    watcher.check_now()
    assert fields(window) == [0.5, 1.0]
    np.testing.assert_allclose(c.result.data.values[:, -1], 1.1 * BASE, rtol=1e-7)


def test_files_that_are_no_spectra_are_reported_once_and_left_out(window, clock, tmp_path):
    zero(tmp_path, old=True)
    spectrum(tmp_path, 0.5, old=True)
    watcher = window.folder_watch
    watcher.start(tmp_path)
    skipped = count(watcher.skipped)
    (tmp_path / "Sample_4p2K_Sam1_a01p000T.txt").write_text("energy\tintensity\n1\tx\n")
    write_text(tmp_path / "notes.txt", X, BASE)  # a spectrum, but no field in its name
    arrive(window, clock)
    for _ in range(watch.MAX_FAILURES + 2):
        clock.now += 1.0
        watcher.check_now()
    names = sorted(Path(path).name for path, _text in skipped)
    assert names == ["Sample_4p2K_Sam1_a01p000T.txt", "notes.txt"]
    text = dict((Path(p).name, t) for p, t in skipped)
    assert text["notes.txt"].startswith("There is no field in its name")
    assert text["Sample_4p2K_Sam1_a01p000T.txt"].startswith("It does not read as a spectrum")
    assert gui_helpers.infobar_text(window).startswith("Left out ")
    assert c_files(window) == [sweep_name(0.5)]
    assert fields(window) == [0.5]


def c_files(window) -> list[str]:
    return [Path(p).name for p in window.controller.processing.sample_files.field]


# ---------------------------------------------------------------------- updates
def test_files_arriving_together_make_one_update(window, clock, tmp_path):
    zero(tmp_path, old=True)
    spectrum(tmp_path, 0.5, old=True)
    window.folder_watch.start(tmp_path)
    results = count(window.controller.resultChanged)
    for b in (1.0, 1.5, 2.0):
        spectrum(tmp_path, b)
    arrive(window, clock)
    assert len(results) == 1 and fields(window) == [0.5, 1.0, 1.5, 2.0]


def test_files_found_during_an_update_make_one_more_update_after_it(window, clock, tmp_path, qtbot):
    zero(tmp_path, old=True)
    spectrum(tmp_path, 0.5, old=True)
    watcher = window.folder_watch
    watcher.start(tmp_path)
    c = window.controller
    depth, nested, results = [0], [], []

    def during_update() -> None:
        results.append(list(c.result.ratio.field))
        if depth[0]:
            nested.append(True)
        if len(results) == 1:  # files appear while the first update draws its map
            depth[0] += 1
            for b in (1.5, 2.0):
                spectrum(tmp_path, b)
            arrive(window, clock)
            depth[0] -= 1

    c.resultChanged.connect(during_update)
    spectrum(tmp_path, 1.0)
    arrive(window, clock)
    assert results == [[0.5, 1.0]]  # the second update waits for the first
    qtbot.waitUntil(lambda: len(results) == 2, timeout=2000)
    assert results[1] == [0.5, 1.0, 1.5, 2.0] and not nested
    qtbot.wait(20)
    assert len(results) == 2


def test_the_after_zero_field_file_switches_on_drift_correction(window, clock, tmp_path):
    zero(tmp_path, old=True)
    for b in (0.5, 1.0, 2.0):
        spectrum(tmp_path, b, old=True)
    window.folder_watch.start(tmp_path)
    c = window.controller
    note = window.panels["sample"].measurement.zero_note
    assert note.text().startswith("One spectrum for the whole sweep")
    np.testing.assert_allclose(c.result.ratio.values[:, -1], 1.2)  # (1 + 0.1 B) / 1

    zero(tmp_path, AFTER, scale=2.0)  # the zero field after the sweep drifted to 2x
    arrive(window, clock)
    assert [Path(p).name for p in c.processing.sample_files.zero] == [BEFORE, AFTER]
    assert note.text().startswith("Measured before and after the sweep")
    np.testing.assert_allclose(c.result.ratio.values[:, 0], 1.05)
    np.testing.assert_allclose(c.result.ratio.values[:, -1], 0.6)


def test_zero_field_files_that_come_last(window, clock, tmp_path):
    window.folder_watch.start(tmp_path)
    c = window.controller
    for b in (0.5, 1.0):
        spectrum(tmp_path, b)
    arrive(window, clock)
    assert c.result is None
    assert state_text(window) == "Waiting for a zero-field spectrum · 2 files"
    zero(tmp_path)
    arrive(window, clock)
    assert fields(window) == [0.5, 1.0]


def test_points_view_and_levels_stay_across_updates(window, clock, tmp_path):
    zero(tmp_path, old=True)
    for b in (0.5, 1.0):
        spectrum(tmp_path, b, old=True)
    window.folder_watch.start(tmp_path)
    c = window.controller
    c.record_point(1.0, 500.0)
    c.set_ranges(energy_range=(200.0, 800.0))
    c.set_levels("Ratio", 0.95, 1.3)
    spectrum(tmp_path, 1.5)
    arrive(window, clock)
    assert fields(window) == [0.5, 1.0, 1.5]
    b, e = c.points.points("LL 1")
    np.testing.assert_allclose(b, [1.0])
    np.testing.assert_allclose(e, [500.0])
    assert list(c.points.field) == [0.5, 1.0, 1.5]  # a row for the new field
    assert c.view.energy_range == (200.0, 800.0)
    assert c.view.levels["Ratio"] == (0.95, 1.3)


def test_updates_use_the_current_processing_options(window, clock, tmp_path):
    zero(tmp_path, old=True)
    spectrum(tmp_path, 0.5, old=True)
    window.folder_watch.start(tmp_path)
    c = window.controller
    c.set_processing(energy_cut=(200.0, 400.0), baseline=(300.0, 400.0))
    spectrum(tmp_path, 1.0)
    arrive(window, clock)
    assert c.result.ratio.energy[[0, -1]].tolist() == [200.0, 400.0]
    assert c.result.baseline_region == (300.0, 400.0)
    assert not c.changed_since_process()


def test_updates_read_only_the_new_files(window, clock, tmp_path):
    zero(tmp_path, old=True)
    for b in (0.5, 1.0):
        _aged(write_opus(tmp_path / sweep_name(b, ".0"), X, (1 + 0.1 * b) * BASE), True)
    window.folder_watch.start(tmp_path)
    c = window.controller
    cache = c.spectrum_cache()
    reads = cache.reads
    gui_helpers.process(window)  # Process while watching reads through the cache too
    assert cache.reads == reads
    write_opus(tmp_path / sweep_name(1.5, ".0"), X, 1.15 * BASE)
    arrive(window, clock)
    assert fields(window) == [0.5, 1.0, 1.5]
    assert cache.reads == reads + 1


def test_errors_of_updates_are_reported_once(window, clock, tmp_path, errors):
    zero(tmp_path, old=True)
    spectrum(tmp_path, 0.5, old=True)
    c = window.controller
    c.set_processing(energy_cut=(5000.0, 6000.0))  # holds no data
    window.folder_watch.start(tmp_path)
    assert len(errors) == 1 and "contains no data" in errors[0]
    assert state_text(window).startswith("Not processed: energy window")
    assert chip(window).text().endswith(" · 2 files · not processed")  # the reason: tooltip
    for b in (1.0, 1.5):
        spectrum(tmp_path, b)
        arrive(window, clock)
    assert len(errors) == 1  # not at every update
    c.set_processing(energy_cut=None)
    spectrum(tmp_path, 2.0)
    arrive(window, clock)
    assert fields(window) == [0.5, 1.0, 1.5, 2.0]
    assert state_text(window).startswith("Watching · 5 files · last update")


def test_the_missing_reference_note_comes_once(window, clock, tmp_path, caplog):
    zero(tmp_path, old=True)
    spectrum(tmp_path, 0.5, old=True)
    c = window.controller
    c.set_processing(reference_mode=ReferenceMode.SEPARATE)  # without reference files
    with caplog.at_level(logging.WARNING, logger="mag_opt_detective"):
        window.folder_watch.start(tmp_path)
        assert gui_helpers.infobar_text(window).startswith(NO_REFERENCE_FILES)
        for b in (1.0, 1.5):
            spectrum(tmp_path, b)
            arrive(window, clock)
    assert gui_helpers.infobar_text(window) == ""
    assert caplog.text.count(NO_REFERENCE_FILES) == 1
    assert fields(window) == [0.5, 1.0, 1.5]


def test_a_library_map_shown_holds_the_processing(window, clock, tmp_path):
    zero(tmp_path, old=True)
    spectrum(tmp_path, 0.5, old=True)
    window.folder_watch.start(tmp_path)
    c = window.controller
    c.plot_entry(c.save_current_map())
    spectrum(tmp_path, 1.0)
    arrive(window, clock)
    assert c.result_source == "library" and len(c.processing.sample_files.field) == 2
    assert state_text(window).startswith("A library map is shown")
    gui_helpers.process(window)
    assert fields(window) == [0.5, 1.0]


# ---------------------------------------------------------------------- stopping
def test_stop_watching_keeps_the_files_and_the_map(window, clock, tmp_path):
    zero(tmp_path, old=True)
    spectrum(tmp_path, 0.5, old=True)
    watcher = window.folder_watch
    watcher.start(tmp_path)
    c = window.controller
    assert not chip(window).isHidden()
    chip(window).stop_button.click()
    assert not watcher.watching() and chip(window).isHidden()
    assert not window.panels["sample"].watch.switch.isChecked()
    assert c.spectrum_cache() is None
    assert fields(window) == [0.5] and len(c.processing.sample_files.field) == 1
    spectrum(tmp_path, 1.0)
    clock.now += SETTLE
    watcher.check_now()
    assert len(c.processing.sample_files.field) == 1


def test_closing_the_window_stops_watching(window, tmp_path):
    zero(tmp_path, old=True)
    window.folder_watch.start(tmp_path)
    window.close()
    assert not window.folder_watch.watching()


def test_loading_another_sweep_stops_watching(window, clock, sweep, tmp_path):
    folder = tmp_path / "running"
    folder.mkdir()
    zero(folder, old=True)
    window.folder_watch.start(folder)
    gui_helpers.load_sweep(window, sweep)
    assert not window.folder_watch.watching()


def test_watching_does_not_resume_after_a_restart(qtbot, tmp_path):
    ini = str(tmp_path / "settings.ini")
    folder = tmp_path / "running"
    folder.mkdir()
    zero(folder, old=True)
    first = MainWindow(settings=QSettings(ini, QSettings.Format.IniFormat))
    qtbot.addWidget(first)
    first.folder_watch.start(folder)
    assert first.folder_watch.watching()
    first.close()
    second = MainWindow(settings=QSettings(ini, QSettings.Format.IniFormat))
    qtbot.addWidget(second)
    assert not second.folder_watch.watching()
    assert not second.panels["sample"].watch.switch.isChecked()
    assert second.controller.processing.sample_files == SweepFiles()
    second.close()


def test_the_switch_needs_a_folder(window):
    box = window.panels["sample"].watch
    assert not box.switch.isEnabled()
    assert (
        box.row.description_label.text()
        == "Load a sweep folder first, or watch one that may still be empty."
    )
    assert "reference sweep is not watched" in box.row.toolTip()
