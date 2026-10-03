"""Watching a measurement folder (gui/watch.py and the Sample panel's Watch folder block)."""

import logging
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QFileDialog, QLabel

import gui_helpers
from helpers import sweep_name, write_opus, write_text
from mag_opt_detective.core.pipeline import ReferenceMode
from mag_opt_detective.gui import watch
from mag_opt_detective.gui.controller import NO_REFERENCE_FILES, FieldRange, SweepFiles
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
    assert state_text(window) == "Waiting for in-field spectra · no files"
    assert window.panel_pages["sample"].subtitle.text() == "Watching running · waiting for files"

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
    text = dict(skipped)
    assert sorted(text) == ["Left out Sample_4p2K_Sam1_a01p000T.txt", "Left out notes.txt"]
    assert text["Left out notes.txt"].startswith("There is no field in its name")
    assert text["Left out Sample_4p2K_Sam1_a01p000T.txt"].startswith(
        "It does not read as a spectrum"
    )
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
    assert results == [[0.5, 1.0]]  # the second update waits for the first, and the gap
    clock.now += watch.MIN_GAP_S
    qtbot.waitUntil(lambda: len(results) == 2, timeout=3000)  # the gap's timer
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


def test_a_spectrum_written_again_comes_back(window, clock, tmp_path):
    zero(tmp_path, old=True)
    for b in (0.5, 1.0):
        spectrum(tmp_path, b, old=True)
    watcher = window.folder_watch
    watcher.start(tmp_path)
    c = window.controller
    target = tmp_path / sweep_name(1.0)
    os.remove(target)  # measured again: the old file goes first
    clock.now += watch.MIN_GAP_S
    watcher.check_now()
    assert fields(window) == [0.5] and c_files(window) == [sweep_name(0.5)]
    write_text(target, X, 1.3 * BASE)
    arrive(window, clock)
    assert fields(window) == [0.5, 1.0]
    np.testing.assert_allclose(c.result.data.values[:, -1], 1.3 * BASE)

    # a file removed from the list but still in the folder stays out
    c.set_processing(sample_files=SweepFiles(c.processing.sample_files.zero, (str(target),)))
    spectrum(tmp_path, 1.5)
    arrive(window, clock)
    assert c_files(window) == [sweep_name(1.0), sweep_name(1.5)]


def test_a_failed_update_or_check_does_not_end_watching(
    window, clock, tmp_path, monkeypatch, caplog
):
    zero(tmp_path, old=True)
    spectrum(tmp_path, 0.5, old=True)
    watcher = window.folder_watch
    watcher.start(tmp_path)
    c = window.controller

    def boom(*_args, **_kwargs):
        raise RuntimeError("boom")

    with monkeypatch.context() as patch, caplog.at_level(logging.ERROR, "mag_opt_detective"):
        patch.setattr(c, "process_update", boom)
        spectrum(tmp_path, 1.0)
        arrive(window, clock)
    assert state_text(window).startswith("The update failed: RuntimeError: boom; see the log")
    assert chip(window).level() == watch.PROBLEM
    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert [r.exc_info is not None for r in errors] == [True]  # in the Log, with its trace
    assert watcher.watching() and watcher.interval() > 0  # still looking
    spectrum(tmp_path, 1.5)
    arrive(window, clock)
    assert fields(window) == [0.5, 1.0, 1.5]
    assert state_text(window).startswith("Watching · 4 files · last update")

    with monkeypatch.context() as patch:
        patch.setattr(c, "read_spectrum", boom)
        spectrum(tmp_path, 2.0)
        arrive(window, clock)
    assert state_text(window).startswith("The check failed: RuntimeError: boom; see the log")
    assert watcher.interval() > 0
    arrive(window, clock)
    assert fields(window) == [0.5, 1.0, 1.5, 2.0]


def test_a_folder_out_of_reach_is_looked_at_until_it_is_back(window, clock, tmp_path, errors):
    folder = tmp_path / "share"
    folder.mkdir()
    zero(folder, old=True)
    spectrum(folder, 0.5, old=True)
    watcher = window.folder_watch
    watcher.start(folder)
    away = tmp_path / "away"
    folder.rename(away)  # the network drive is gone
    watcher.check_now()
    assert watcher.watching() and watcher.interval() == watch.UNREACHABLE_MS
    assert state_text(window).startswith("Folder not reachable since ")
    assert chip(window).level() == watch.PROBLEM and "not reachable since" in chip(window).text()
    assert fields(window) == [0.5] and len(window.controller.processing.sample_files.field) == 1
    clock.now += 60
    watcher.check_now()
    assert errors == []

    away.rename(folder)  # back, with a spectrum written meanwhile
    spectrum(folder, 1.0)
    watcher.check_now()
    assert state_text(window).startswith("Watching · 2 files")
    assert watcher.interval() == watch.RECHECK_MS
    clock.now += SETTLE
    watcher.check_now()
    assert fields(window) == [0.5, 1.0]


def test_a_file_with_another_energy_axis_is_held_back(window, clock, tmp_path, errors):
    zero(tmp_path, old=True)
    spectrum(tmp_path, 0.5, old=True)
    watcher = window.folder_watch
    watcher.start(tmp_path)
    skipped = count(watcher.skipped)
    path = tmp_path / sweep_name(1.0)
    lines = [f"{a:.8f}\t{b:.8f}\n" for a, b in zip(X, 1.1 * BASE, strict=True)]
    path.write_text("".join(lines[:40]))  # a slow writer paused half-way: it parses
    arrive(window, clock)
    clock.now += 1.0
    watcher.check_now()
    assert c_files(window) == [sweep_name(0.5)] and skipped == [] and errors == []
    clock.now += 1.0
    watcher.check_now()  # the third check with the same axis: reported once
    assert [title for title, _text in skipped] == [f"Left out {sweep_name(1.0)}"]
    assert skipped[0][1].startswith("Its energy axis differs from the sweep's: 40 points")

    with path.open("a") as fh:
        fh.write("".join(lines[40:]))
    os.utime(path, ns=(time.time_ns(), time.time_ns() + 1000))
    arrive(window, clock)
    assert fields(window) == [0.5, 1.0] and errors == []


def test_held_files_are_tried_again_when_the_sweep_is_complete(window, clock, tmp_path, errors):
    def text(y, rows=None) -> str:
        return "".join(f"{a:.8f}\t{b:.8f}\n" for a, b in list(zip(X, y, strict=True))[:rows])

    first = [tmp_path / BEFORE, tmp_path / sweep_name(0.5)]
    for path, y in zip(first, (BASE, 1.05 * BASE), strict=True):
        path.write_text(text(y, 40))  # listed while still half-written
        _aged(path, True)
    watcher = window.folder_watch
    watcher.start(tmp_path)
    assert window.controller.result.ratio.energy.size == 40
    for b in (1.0, 1.25, 1.5):  # complete, so another axis than the (half) sweep's
        spectrum(tmp_path, b)
        arrive(window, clock)
        for _ in range(watch.MAX_FAILURES):
            clock.now += 1.0
            watcher.check_now()
    assert c_files(window) == [sweep_name(0.5)]  # held back, though there are more of them
    for path, y in zip(first, (BASE, 1.05 * BASE), strict=True):
        path.write_text(text(y))  # the first files are finished
    arrive(window, clock)
    for _ in range(2):
        clock.now += watch.MIN_GAP_S
        watcher.check_now()
    assert fields(window) == [0.5, 1.0, 1.25, 1.5]
    assert window.controller.result.ratio.energy.size == X.size
    assert errors == []  # never a sweep with two axes


FINE = np.linspace(100.0, 1000.0, 181)  # another resolution


def fine(folder: Path, name: str, scale: float = 1.0) -> str:
    """A spectrum on the finer axis FINE."""
    return str(write_text(folder / name, FINE, scale * (1.0 + 0.5 * np.sin(FINE / 50.0))))


def test_new_files_held_back_for_their_axis_are_shown(window, clock, tmp_path):
    zero(tmp_path, old=True)
    for b in (0.5, 1.0, 1.5):
        spectrum(tmp_path, b, old=True)
    watcher = window.folder_watch
    watcher.start(tmp_path)
    for b in (2.0, 2.5):  # the resolution was changed during the sweep
        fine(tmp_path, sweep_name(b))
        arrive(window, clock)
    assert fields(window) == [0.5, 1.0, 1.5]
    text = "2 new files have another energy axis: 181 points, 100 – 1000 cm⁻¹"
    assert state_text(window) == f"{text} · 4 files"
    assert chip(window).text().endswith("· 4 files · 2 held back (energy axis)")
    os.remove(tmp_path / sweep_name(2.0))
    os.remove(tmp_path / sweep_name(2.5))
    watcher.check_now()
    assert state_text(window).startswith("Watching · 4 files")


def axes(window) -> str:
    """The axis of each listed in-field spectrum: A (X) or B (FINE)."""
    c = window.controller
    return "".join(
        "A" if c.read_spectrum(p)[0].size == X.size else "B"
        for p in c.processing.sample_files.field
    )


def test_the_listed_spectra_keep_their_axis(window, clock, tmp_path, errors):
    zero(tmp_path, old=True)
    for b in (0.25, 0.5, 0.75):
        spectrum(tmp_path, b, old=True)
    watcher = window.folder_watch
    watcher.start(tmp_path)
    for i in range(5):  # more spectra at the new resolution than at the old one
        fine(tmp_path, sweep_name(1.0 + 0.25 * i))
        arrive(window, clock)
        for _ in range(watch.MAX_FAILURES):
            clock.now += 1.0
            watcher.check_now()
    assert axes(window) == "AAA" and fields(window) == [0.25, 0.5, 0.75] and errors == []
    assert state_text(window).startswith("5 new files have another energy axis: 181 points")


def test_spectra_of_two_axes_never_share_the_lists(window, clock, tmp_path, errors):
    zero(tmp_path, old=True)
    watcher = window.folder_watch
    watcher.start(tmp_path)
    for i, axis in enumerate("ABBAABBBAA"):
        name = sweep_name(0.25 * (i + 1))
        if axis == "A":
            write_text(tmp_path / name, X, BASE)
        else:
            fine(tmp_path, name)
        arrive(window, clock)
        for _ in range(watch.MAX_FAILURES):
            clock.now += 1.0
            watcher.check_now()
        assert set(axes(window)) == {"A"}
    assert axes(window) == "AAAAA" and errors == []


def test_the_axis_most_files_share_wins(window, clock, tmp_path):
    watcher = window.folder_watch
    watcher.start(tmp_path, replace=True)
    c = window.controller
    odd = fine(tmp_path, BEFORE)  # the zero field was measured with other settings
    arrive(window, clock)
    spectrum(tmp_path, 0.5)
    arrive(window, clock)
    for _ in range(watch.MAX_FAILURES):
        clock.now += 1.0
        watcher.check_now()
    assert c_files(window) == []  # one against one: the listed file's axis
    spectrum(tmp_path, 1.0)
    arrive(window, clock)
    clock.now += watch.MIN_GAP_S
    watcher.check_now()  # the file left out before is tried again with the new axis
    assert c_files(window) == [sweep_name(0.5), sweep_name(1.0)]

    os.remove(odd)  # without the odd file the sweep needs a zero field like the others
    zero(tmp_path)
    arrive(window, clock)
    clock.now += watch.MIN_GAP_S
    watcher.check_now()
    assert fields(window) == [0.5, 1.0]
    assert [Path(p).name for p in c.processing.sample_files.zero] == [BEFORE]


def test_removing_the_odd_file_releases_the_held_ones(window, clock, tmp_path):
    watcher = window.folder_watch
    watcher.start(tmp_path, replace=True)
    odd = fine(tmp_path, BEFORE)
    arrive(window, clock)
    spectrum(tmp_path, 0.5)
    arrive(window, clock)
    for _ in range(watch.MAX_FAILURES):
        clock.now += 1.0
        watcher.check_now()
    assert c_files(window) == []  # held back, then left out
    os.remove(odd)
    clock.now += watch.MIN_GAP_S
    watcher.check_now()  # the axis changes: the files held back are checked again
    clock.now += watch.MIN_GAP_S
    watcher.check_now()
    assert c_files(window) == [sweep_name(0.5)]
    assert state_text(window).startswith("Waiting for a zero-field spectrum")


def test_a_custom_field_range_waits_for_its_files(window, clock, tmp_path, errors):
    c = window.controller
    c.set_processing(custom_field=True, sample_field=FieldRange(0.5, 0.5, 2.0))
    zero(tmp_path, old=True)
    for b in (0.5, 1.0):
        spectrum(tmp_path, b, old=True)
    window.folder_watch.start(tmp_path)
    assert state_text(window) == (
        "Waiting for 2 more files to match the custom field range · 3 files"
    )
    spectrum(tmp_path, 1.5)
    arrive(window, clock)
    assert state_text(window).startswith("Waiting for 1 more file to match")
    assert c.result is None and errors == []
    spectrum(tmp_path, 2.0)
    arrive(window, clock)
    assert fields(window) == [0.5, 1.0, 1.5, 2.0] and errors == []


def test_files_left_out_together_are_reported_together(window, clock, tmp_path, caplog):
    zero(tmp_path, old=True)
    spectrum(tmp_path, 0.5, old=True)
    watcher = window.folder_watch
    watcher.start(tmp_path)
    skipped = count(watcher.skipped)
    for i in range(5):
        (tmp_path / f"report_{i}.pdf").write_bytes(bytes(range(256)) * 4)
    with caplog.at_level(logging.WARNING, "mag_opt_detective"):
        arrive(window, clock)
        for _ in range(watch.MAX_FAILURES):
            clock.now += 1.0
            watcher.check_now()
    assert skipped == [
        (
            "Left out 5 files that are not spectra",
            "report_0.pdf, report_1.pdf, report_2.pdf and 2 more. " + watch.TRIED_AGAIN,
        )
    ]
    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1 and "Left out 5 files" in warnings[0]
    assert gui_helpers.infobar_text(window).startswith("Left out 5 files that are not spectra")


def test_updates_keep_a_gap_between_them(window, clock, tmp_path, monkeypatch):
    monkeypatch.setattr(watch, "SETTLE_S", 0.5)
    zero(tmp_path, old=True)
    spectrum(tmp_path, 0.5, old=True)
    watcher = window.folder_watch
    watcher.start(tmp_path)  # an update now
    spectrum(tmp_path, 1.0)
    watcher.check_now()
    clock.now += 0.6
    watcher.check_now()  # complete, but within MIN_GAP_S of the last update
    assert fields(window) == [0.5]
    assert len(window.controller.processing.sample_files.field) == 1
    clock.now += 0.5
    watcher.check_now()
    assert fields(window) == [0.5, 1.0]


def test_the_file_menu_watches_and_stops(window, tmp_path):
    action = window.commands["watch_folder"]
    assert action.isCheckable() and not action.isEnabled()  # no folder yet
    zero(tmp_path, old=True)
    spectrum(tmp_path, 0.5, old=True)
    files = SweepFiles((str(tmp_path / BEFORE),), (str(tmp_path / sweep_name(0.5)),))
    window.controller.set_processing(sample_files=files)
    assert action.isEnabled() and not action.isChecked()
    action.trigger()
    assert window.folder_watch.watching() and action.isChecked()
    assert window.panels["sample"].watch.switch.isChecked()
    assert fields(window) == [0.5]  # listed, not processed yet: processed at once
    action.trigger()
    assert not window.folder_watch.watching() and not action.isChecked()


def test_a_file_gone_and_back_before_the_update_stays_listed(window, clock, tmp_path):
    zero(tmp_path, old=True)
    for b in (0.5, 1.0):
        spectrum(tmp_path, b, old=True)
    watcher = window.folder_watch
    watcher.start(tmp_path)  # an update now: the next one waits for the gap
    target = tmp_path / sweep_name(1.0)
    os.remove(target)
    watcher.check_now()  # gone, but the update waits (the clock stands still)
    write_text(target, X, 1.3 * BASE)
    arrive(window, clock)  # back before the update ran: one update with both
    assert c_files(window) == [sweep_name(0.5), sweep_name(1.0)]
    np.testing.assert_allclose(window.controller.result.data.values[:, -1], 1.3 * BASE)


def test_files_left_out_are_reported_after_a_waiting_update(window, clock, tmp_path, monkeypatch):
    monkeypatch.setattr(watch, "SETTLE_S", 0.5)
    zero(tmp_path, old=True)
    spectrum(tmp_path, 0.5, old=True)
    watcher = window.folder_watch
    watcher.start(tmp_path)  # an update now
    (tmp_path / sweep_name(9.0)).write_text("garbage\n")
    spectrum(tmp_path, 1.0)
    watcher.check_now()
    clock.now += 0.5  # both complete: the junk fails, the update for 1 T waits for the gap
    watcher.check_now()
    for _ in range(watch.MAX_FAILURES - 1):
        clock.now += 0.1
        watcher.check_now()  # the junk is left out while the update still waits
    assert fields(window) == [0.5] and gui_helpers.infobar_text(window) == ""
    clock.now += 0.5
    watcher.check_now()  # the update runs, then the report (its map would close it)
    assert fields(window) == [0.5, 1.0]
    assert gui_helpers.infobar_text(window).startswith(f"Left out {sweep_name(9.0)}")


def test_the_file_menu_follows_a_failed_start(window, tmp_path, errors):
    folder = tmp_path / "share"
    folder.mkdir()
    files = SweepFiles((zero(folder),), (spectrum(folder, 0.5),))
    window.controller.set_processing(sample_files=files)
    shutil.rmtree(folder)  # the sweep's folder is gone
    action = window.commands["watch_folder"]
    action.trigger()
    assert not window.folder_watch.watching() and errors
    assert not action.isChecked()
    assert not window.panels["sample"].watch.switch.isChecked()


def test_the_chip_sits_beside_the_baseline_chip_and_shrinks(window, qtbot, tmp_path):
    zero(tmp_path, old=True)
    for b in (0.5, 1.0):
        spectrum(tmp_path, b, old=True)
    window.controller.set_processing(baseline=(300.0, 400.0))
    window.resize(1100, 800)
    window.show()
    qtbot.waitExposed(window)
    narrowest = window.minimumSizeHint().width()
    window.folder_watch.start(tmp_path)
    watching, baseline = chip(window), window.baseline_chip
    qtbot.waitUntil(lambda: watching.width() >= watching.sizeHint().width())  # all fits
    assert watching.isVisible() and baseline.isVisible()
    labels = window.statusBar().findChildren(QLabel)
    state = next(label for label in labels if label.text() == window.state_text())
    summary = next(label for label in labels if label.text() == window.summary_text())
    # between the state and the summary, beside the baseline chip
    assert state.geometry().right() < watching.geometry().left()
    assert watching.geometry().right() < baseline.geometry().left()
    assert baseline.geometry().right() < summary.geometry().left()
    assert watching.minimumSizeHint().width() == 0
    assert window.minimumSizeHint().width() == narrowest <= 1100  # it adds nothing


def test_a_crowded_status_bar_shortens_the_folder_name_first(window, clock, qtbot, tmp_path):
    folder = tmp_path / "FePS3_BF_T222_eGlob_40kHz-R1_18V_2p2K_Sam1_run"
    folder.mkdir()
    zero(folder, old=True)
    for b in (0.5, 1.0):
        spectrum(folder, b, old=True)
    c = window.controller
    c.set_processing(baseline=(300.0, 400.0))
    window.resize(1100, 800)
    window.show()
    qtbot.waitExposed(window)
    window.folder_watch.start(folder)
    fine(folder, sweep_name(1.5))  # held back for its axis: the longest chip
    arrive(window, clock)
    c.set_processing(energy_cut=(150.0, 950.0))  # "Settings changed · process again"
    window.set_cursor_text("B 7.25 T · E 1234.5 cm⁻¹ · 1.0234")
    watching, baseline = chip(window), window.baseline_chip
    tail = " · 3 files · 1 held back (energy axis)"  # the count stays
    # once the bar is laid out, the baseline chip keeps its value and the count stays
    qtbot.waitUntil(lambda: baseline.form() == 0 and watching.shown_text().endswith(tail))
    shown = watching.shown_text()
    assert shown.startswith("Watching FePS3") and "…" in shown  # the folder name gave way
    assert watching.geometry().right() < baseline.geometry().left()


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
    assert "adds every complete spectrum in the folder" in box.row.toolTip()
    assert "reference sweep is not watched" in box.row.toolTip()
