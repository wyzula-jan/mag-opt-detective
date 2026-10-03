"""Watching a measurement folder: the spectra that appear there are added and processed.

:class:`FolderWatcher` follows one folder while a sweep is measured. A
:class:`QFileSystemWatcher` tells it when the folder changes, and it also looks every
:data:`POLL_MS` (network drives may not send notices). A file counts once it is complete
(:class:`Arrivals`): its size and modification time stayed the same for :data:`SETTLE_S` and
it reads as a spectrum. A file that does not read is tried again; one that keeps failing is
reported once and left out until it changes.

Complete files go to the sample's lists as the files of a dropped folder do (names whose
first field is 0 T to the zero-field list), and the sweep is processed again with the current
options (:meth:`AppController.process_update`). Spectra read before come from the controller's
cache, so an update reads only the new files. At most one update runs at a time; files found
during one make one more update after it. Watching never resumes after a restart, and the
reference sweep is not watched.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QFileSystemWatcher, QObject, QTimer, Signal

from mag_opt_detective.core.readers import parse_field
from mag_opt_detective.gui.controller import (
    DATA_CURRENT,
    EXPECTED_ERRORS,
    AppController,
    SweepFiles,
)
from mag_opt_detective.gui.panels.files import merged, split_zero

logger = logging.getLogger("mag_opt_detective")

SETTLE_S = 2.0  # a file is complete once its size and time stayed the same this long
POLL_MS = 3000  # look at the folder this often, with or without a change notice
RECHECK_MS = 1000  # ... while a file is still written or may be read again
DEBOUNCE_MS = 300  # after a change notice, wait this long for more before looking
MAX_FAILURES = 3  # failed reads of an unchanged file before it is reported and left out

Listing = dict[str, tuple[int, int]]  # path -> (size, modification time in ns)

WAITING, OK, PROBLEM = "info", "ok", "warn"  # levels of WatchStatus (Note levels)


def path_key(path: str | Path) -> str:
    """*path* in one spelling, to compare paths written differently (``/`` or ``\\``)."""
    return os.path.normcase(os.path.normpath(str(path)))


def folder_listing(folder: str | Path) -> Listing:
    """The visible files of *folder* with their size and modification time (ns)."""
    listing: Listing = {}
    with os.scandir(os.path.normpath(str(folder))) as entries:
        for entry in entries:
            if entry.name.startswith("."):
                continue
            try:
                if not entry.is_file():
                    continue
                stat = entry.stat()
            except OSError:  # gone meanwhile
                continue
            listing[entry.path] = (stat.st_size, stat.st_mtime_ns)
    return listing


def sweep_folder(files: SweepFiles) -> str | None:
    """The folder the sample's files come from: all of them, else the in-field ones; None
    without files or when they come from several folders."""
    for paths in ((*files.zero, *files.field), files.field):
        folders = {path_key(os.path.dirname(p)): os.path.dirname(p) for p in paths}
        if len(folders) == 1:
            return folders.popitem()[1]
    return None


@dataclass
class _File:
    signature: tuple[int, int]  # size, modification time
    since: float  # when this signature was first seen
    complete: bool = False
    failures: int = 0  # failed reads with this signature
    skipped: bool = False  # reported and left out until the file changes


class Arrivals:
    """Which files of a folder are complete, from listings taken over time (no Qt).

    A file is complete when it is not empty and was seen with the same size and modification
    time for *settle* seconds; a file that changes starts again.
    """

    def __init__(self, settle: float = SETTLE_S):
        self.settle = settle
        self._files: dict[str, _File] = {}

    def observe(self, listing: Listing, now: float, wall: float | None = None) -> None:
        """Take *listing* (:func:`folder_listing`), seen at *now* (a monotonic clock).

        With *wall* (the time of day, as file times are) a file last changed at least
        *settle* seconds before it is complete at once (the files there when watching starts).
        """
        for path in [p for p in self._files if p not in listing]:
            del self._files[path]
        for path, signature in listing.items():
            known = self._files.get(path)
            if known is None or known.signature != signature:
                old = wall is not None and wall - signature[1] / 1e9 >= self.settle
                self._files[path] = _File(signature, now, complete=old and signature[0] > 0)
            elif not known.complete and signature[0] > 0 and now - known.since >= self.settle:
                known.complete = True

    def complete(self) -> list[str]:
        """The complete files that are not left out."""
        return sorted(p for p, f in self._files.items() if f.complete and not f.skipped)

    def waiting(self) -> bool:
        """A file is still written, or complete but may be read again."""
        return any(
            not f.skipped and (f.failures or (not f.complete and f.signature[0] > 0))
            for f in self._files.values()
        )

    def failed(self, path: str) -> bool:
        """Count a failed read of the complete file *path*; True when it is now left out
        (until it changes), which is the time to report it."""
        file = self._files[path]
        file.failures += 1
        file.skipped = file.failures >= MAX_FAILURES
        return file.skipped

    def read_ok(self, path: str) -> None:
        self._files[path].failures = 0


@dataclass(frozen=True)
class WatchStatus:
    """What :meth:`FolderWatcher.status` reports."""

    folder: str
    files: int  # in the sample's lists
    last: datetime | None  # the last update processed
    note: str  # waiting or why the last update was not processed ("" after a processed one)
    level: str  # OK, WAITING or PROBLEM


class FolderWatcher(QObject):
    """Watches one folder for the sample's spectra (see the module docstring).

    ``changed`` fires when :meth:`status` changes, ``skipped(path, text)`` when a file is
    left out (once until it changes; *text* says why) and ``failed(exception)`` when an
    update cannot be processed (the first time after a processed one). ``clock`` is the
    monotonic clock of the completeness check (tests replace it).
    """

    changed = Signal()
    skipped = Signal(str, str)
    failed = Signal(object)

    def __init__(self, controller: AppController, parent: QObject | None = None):
        super().__init__(parent)
        self.controller = controller
        self.clock: Callable[[], float] = time.monotonic
        self._folder: str | None = None
        self._arrivals = Arrivals()
        self._accepted: Listing = {}  # files read as spectra, with their signature then
        self._handled: set[str] = set()  # keys of files listed once: removed ones stay out
        self._queued: tuple[list[str], list[str], bool] = ([], [], False)  # new, gone, changed
        self._busy = False  # an update runs
        self._runs = 0  # updates run
        self._again = False  # files were found during it
        self._first = True  # no update processed yet
        self._failing = False  # the last update was not processed
        self._last: datetime | None = None
        self._note, self._level = "", OK
        self._notices = QFileSystemWatcher(self)
        self._notices.directoryChanged.connect(self._on_notice)
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(DEBOUNCE_MS)
        self._debounce.timeout.connect(self.check_now)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self.check_now)
        controller.processingChanged.connect(self._follow_files)

    # --- state ------------------------------------------------------------------------------
    def folder(self) -> str | None:
        """The folder watched, or None."""
        return self._folder

    def watching(self) -> bool:
        return self._folder is not None

    def interval(self) -> int:
        """Milliseconds until the folder is looked at again (0 when not watching)."""
        return self._timer.interval() if self._timer.isActive() else 0

    def status(self) -> WatchStatus | None:
        if self._folder is None:
            return None
        files = self.controller.processing.sample_files
        count = len(files.zero) + len(files.field)
        return WatchStatus(self._folder, count, self._last, self._note, self._level)

    # --- start and stop ---------------------------------------------------------------------
    def start(self, folder: str | Path, replace: bool = False) -> None:
        """Watch *folder* (which may be empty). Its complete spectra that are not listed yet
        are added; *replace* empties the sample's lists first (a new sweep). The sweep is
        processed at once if its files have not been processed as they are."""
        folder = os.path.normpath(str(folder))
        if not os.path.isdir(folder):
            raise ValueError(f"{folder}: no such folder")
        self.stop()
        c = self.controller
        if replace:
            c.set_processing(sample_files=SweepFiles())
        self._folder = folder
        self._arrivals = Arrivals(SETTLE_S)
        self._accepted = {}
        self._handled = self._listed()
        self._queued = ([], [], False)
        self._first, self._failing = True, False
        self._last = None
        self._note, self._level = "", OK
        c.set_spectrum_cache(True)
        self._notices.addPath(folder)
        logger.info("Watching %s for new spectra.", folder)
        runs = self._runs
        self._scan(wall=time.time())
        if self._runs == runs and c.data_state("sample") != DATA_CURRENT:
            self._queue([], [], True)  # listed but not processed as they are
        self.changed.emit()

    def stop(self) -> None:
        """Stop watching; the files and the map shown stay."""
        folder = self._folder
        if folder is None:
            return
        self._folder = None
        self._timer.stop()
        self._debounce.stop()
        if self._notices.directories():
            self._notices.removePaths(self._notices.directories())
        self.controller.set_spectrum_cache(False)
        logger.info("Stopped watching %s.", folder)
        self.changed.emit()

    # --- looking at the folder --------------------------------------------------------------
    def check_now(self) -> None:
        """Look at the folder now (the timers do it by themselves)."""
        if self._folder is not None:
            self._debounce.stop()
            self._scan()

    def _on_notice(self, _path: str) -> None:
        self._debounce.start()

    def _scan(self, wall: float | None = None) -> None:
        folder = self._folder
        assert folder is not None
        try:
            listing = folder_listing(folder)
        except OSError as exc:
            self.stop()
            why = exc.strerror or str(exc)
            self.failed.emit(
                OSError(f"the folder {folder} cannot be read ({why}), so watching stopped")
            )
            return
        arrivals = self._arrivals
        arrivals.observe(listing, self.clock(), wall)
        new: list[str] = []
        changed = False
        skips: list[tuple[str, str]] = []
        listed = self._listed()
        for path in arrivals.complete():
            signature = listing[path]
            if self._accepted.get(path) == signature:
                continue
            try:
                self.controller.read_spectrum(path)
            except EXPECTED_ERRORS as exc:
                if arrivals.failed(path):
                    skips.append((path, _reason(path, exc)))
                continue
            arrivals.read_ok(path)
            known = path in self._accepted
            self._accepted[path] = signature
            if not known:
                if path_key(path) not in self._handled:
                    new.append(path)
            elif path_key(path) in listed:
                changed = True  # a listed spectrum changed since it was read
        gone = [p for p in self._accepted if p not in listing]
        for path in gone:
            del self._accepted[path]
        gone = [p for p in gone if path_key(p) in listed]
        if new or gone or changed:
            self._queue(new, gone, changed)
        for path, reason in skips:  # after the update, whose new map closes the info bar
            self._skip(
                path,
                f"It does not read as a spectrum: {reason}",
                "It is read again when it changes.",
            )
        self._schedule()

    def _schedule(self) -> None:
        if self._folder is not None:
            self._timer.start(RECHECK_MS if self._arrivals.waiting() else POLL_MS)

    def _listed(self) -> set[str]:
        """Keys (:func:`path_key`) of the sample's files."""
        files = self.controller.processing.sample_files
        return {path_key(p) for p in (*files.zero, *files.field)}

    def _follow_files(self) -> None:
        """Stop when the sample's files were replaced by files from another folder."""
        if self._folder is None or self._busy:
            return
        folder = path_key(self._folder)
        listed = self._listed()
        if listed and not any(os.path.dirname(p) == folder for p in listed):
            logger.info("The sample's files come from another folder now.")
            self.stop()

    # --- updates ----------------------------------------------------------------------------
    def _queue(self, new: Iterable[str], gone: Iterable[str], changed: bool) -> None:
        """Add files to the next update and run it, or after the one running."""
        queued_new, queued_gone, queued_changed = self._queued
        self._queued = ([*queued_new, *new], [*queued_gone, *gone], queued_changed or changed)
        if self._busy:
            self._again = True
            return
        self._run()

    def _run(self) -> None:
        new, gone, changed = self._queued
        if self._folder is None or self._busy or not (new or gone or changed):
            return
        self._queued = ([], [], False)
        self._busy = True
        self._runs += 1
        try:
            self._update(new, gone, changed)
        finally:
            self._busy = False
        if self._again:  # files found during the update: one more, once the window is drawn
            self._again = False
            QTimer.singleShot(0, self, self._run)

    def _update(self, new: list[str], gone: list[str], changed: bool) -> None:
        c = self.controller
        files = c.processing.sample_files
        zero, field = split_zero(new)
        if not c.processing.custom_field:
            for path in [p for p in field if parse_field(p) is None]:
                field.remove(path)
                self._skip(path, "There is no field in its name (like …_a01p250T)")
        self._handled.update(path_key(p) for p in new)
        if not (zero or field or gone or changed):
            return
        out = {path_key(p) for p in gone}
        c.set_processing(
            sample_files=SweepFiles(
                tuple(p for p in merged(files.zero, zero) if path_key(p) not in out),
                tuple(p for p in merged(files.field, field) if path_key(p) not in out),
            )
        )
        self._process(_change_text([*zero, *field], gone))

    def _process(self, what: str) -> None:
        """Process the sweep if it can be; *what* came with the update (for the log)."""
        c = self.controller
        files = c.processing.sample_files
        head = f"Watched folder: {what}; " if what else "Watched folder: "
        if not files.field or not files.zero:
            missing = "in-field spectra" if not files.field else "a zero-field spectrum"
            self._set_note(f"Waiting for {missing}", WAITING)
            logger.info("%swaiting for %s.", head, missing)
            return
        if c.result_source == "library":
            self._set_note("A library map is shown; Process shows the sweep again", WAITING)
            logger.info("%snot processed while a library map is shown.", head)
            return
        start = time.perf_counter()
        try:
            c.process_update(first=self._first)
        except EXPECTED_ERRORS as exc:
            self._set_note(f"Not processed: {exc}", PROBLEM)
            if not self._failing:  # reported once until an update is processed again
                self._failing = True
                self.failed.emit(exc)
            else:
                logger.warning("%snot processed: %s", head, exc)
            return
        self._first, self._failing = False, False
        self._last = datetime.now()
        self._set_note("", OK)
        ms = 1000 * (time.perf_counter() - start)
        logger.info("%s%d spectra processed in %.0f ms.", head, len(files.field), ms)

    def _skip(self, path: str, why: str, then: str = "") -> None:
        logger.warning("Watched folder: %s left out. %s", Path(path).name, why)
        self.skipped.emit(path, f"{why.rstrip('.')}. {then}".strip())

    def _set_note(self, note: str, level: str) -> None:
        self._note, self._level = note, level
        self.changed.emit()


def _change_text(added: list[str], gone: list[str]) -> str:
    """``1 new file (…_a05p750T.txt)``, ``3 new files``, ``1 file gone`` or ``""``."""
    parts = []
    if added:
        names = f" ({', '.join(Path(p).name for p in added)})" if len(added) <= 2 else ""
        parts.append(f"{len(added)} new file{'s' if len(added) > 1 else ''}{names}")
    if gone:
        parts.append(f"{len(gone)} file{'s' if len(gone) > 1 else ''} gone")
    return ", ".join(parts)


def _reason(path: str, exc: BaseException) -> str:
    """The message of a read error without the file name it starts with."""
    text = str(exc)
    prefix = f"{Path(path).name}: "
    return text[len(prefix) :] if text.startswith(prefix) else text
