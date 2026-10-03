"""Watching a measurement folder: the spectra that appear there are added and processed.

:class:`FolderWatcher` follows one folder while a sweep is measured. A
:class:`QFileSystemWatcher` tells it when the folder changes, and it also looks every
:data:`POLL_MS` (network drives may not send notices). A file counts once it is complete
(:class:`Arrivals`): its size and modification time stayed the same for :data:`SETTLE_S`, it
reads as a spectrum and its energy axis is the sweep's. A file that does not is tried again
(it may still be written); one that keeps failing is reported once and left out until it
changes. A file deleted from the folder leaves the lists, and comes back when it is written
again.

Complete files go to the sample's lists as the files of a dropped folder do (names whose
first field is 0 T to the zero-field list), and the sweep is processed again with the current
options (:meth:`AppController.process_update`). Spectra read before come from the controller's
cache, so an update reads only the new files. At most one update runs at a time, at least
:data:`MIN_GAP_S` after the one before; files found meanwhile make one more update. A folder
that cannot be read (a network drive away) is looked at again every :data:`UNREACHABLE_MS`
until it is back. Watching never resumes after a restart, and the reference sweep is not
watched.
"""

from __future__ import annotations

import logging
import os
import time
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
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
UNREACHABLE_MS = 10_000  # ... while the folder cannot be read
DEBOUNCE_MS = 300  # after a change notice, wait this long for more before looking
MAX_FAILURES = 3  # failed checks of an unchanged file before it is reported and left out
MIN_GAP_S = 1.0  # an update starts at least this long after the end of the one before

Listing = dict[str, tuple[int, int]]  # path -> (size, modification time in ns)

WAITING, OK, PROBLEM = "info", "ok", "warn"  # levels of WatchStatus (Note levels)

# why files are left out: (one file, several files, the title for several of one kind)
NOT_SPECTRUM, NO_FIELD, OTHER_AXIS = "spectrum", "field", "axis"
SKIP_TEXTS = {
    NOT_SPECTRUM: (
        "It does not read as a spectrum: {detail}.",
        "{n} do not read as spectra",
        "that are not spectra",
    ),
    NO_FIELD: (
        "There is no field in its name (like …_a01p250T).",
        "{n} have no field in their names",
        "without a field in their names",
    ),
    OTHER_AXIS: (
        "Its energy axis differs from the sweep's: {detail}.",
        "{n} have another energy axis than the sweep",
        "with another energy axis",
    ),
}
TRIED_AGAIN = "Each is read again when it changes."
LISTED_NAMES = 3  # file names a report of several files gives


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


def same_axis(x: np.ndarray, axis: np.ndarray) -> bool:
    """The energy axes agree as :func:`~mag_opt_detective.core.readers.load_measurement`
    requires."""
    return x.shape == axis.shape and bool(np.allclose(x, axis, rtol=0, atol=1e-6))


def _axis_text(x: np.ndarray) -> str:
    return f"{x.size} points, {x[0]:.6g} – {x[-1]:.6g} cm⁻¹" if x.size else "no points"


@dataclass
class _File:
    signature: tuple[int, int]  # size, modification time
    since: float  # when this signature was first seen
    complete: bool = False
    failures: int = 0  # failed checks with this signature
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
        """Count a failed check of the complete file *path*; True when it is now left out
        (until it changes), which is the time to report it."""
        file = self._files[path]
        file.failures += 1
        file.skipped = file.failures >= MAX_FAILURES
        return file.skipped

    def passed(self, path: str) -> None:
        """*path* passed its check."""
        self._files[path].failures = 0

    def retry(self, path: str) -> None:
        """Check *path* again as if it had not failed (e.g. against another energy axis)."""
        file = self._files.get(path)
        if file is not None:
            file.failures, file.skipped = 0, False


def skip_report(skips: list[tuple[str, str, str]]) -> tuple[str, str]:
    """``(title, text)`` of an info-bar report of the files left out by one check, given as
    ``(path, kind, detail)`` with a kind of :data:`SKIP_TEXTS`."""
    if len(skips) == 1:
        path, kind, detail = skips[0]
        text = SKIP_TEXTS[kind][0].format(detail=detail.rstrip("."))
        if kind != NO_FIELD:
            text += " It is read again when it changes."
        return f"Left out {Path(path).name}", text
    counts = Counter(kind for _path, kind, _detail in skips)
    names = sorted(Path(path).name for path, _kind, _detail in skips)
    more = len(names) - LISTED_NAMES
    shown = ", ".join(names[:LISTED_NAMES]) + (f" and {more} more" if more > 0 else "")
    if len(counts) == 1:
        title = f"Left out {len(skips)} files {SKIP_TEXTS[next(iter(counts))][2]}"
        return title, f"{shown}. {TRIED_AGAIN}"
    parts = [SKIP_TEXTS[kind][1].format(n=n) for kind, n in counts.items()]
    return f"Left out {len(skips)} files", f"{', '.join(parts)}: {shown}. {TRIED_AGAIN}"


@dataclass(frozen=True)
class WatchStatus:
    """What :meth:`FolderWatcher.status` reports."""

    folder: str
    files: int  # in the sample's lists
    last: datetime | None  # the last update processed
    note: str  # waiting or a problem ("" while all is well), a sentence for the panel
    level: str  # OK, WAITING or PROBLEM
    brief: str = ""  # the note in a few words (status bar)


class FolderWatcher(QObject):
    """Watches one folder for the sample's spectra (see the module docstring).

    ``changed`` fires when :meth:`status` changes, ``skipped(title, text)`` once per check
    that left files out (a report for the info bar) and ``failed(exception)`` when an update
    cannot be processed (the first time after a processed one). ``clock`` is the monotonic
    clock of the completeness check and of the gap between updates (tests replace it).
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
        self._handled: set[str] = set()  # keys of files listed once and still in the folder
        self._held: set[str] = set()  # complete files held back for their energy axis
        self._axis_key: tuple | None = None  # (size, first, last) of the sweep's axis
        self._skips: list[tuple[str, str, str]] = []  # left out, not reported yet
        self._queued: tuple[list[str], list[str], bool] = ([], [], False)  # new, gone, changed
        self._busy = False  # an update runs
        self._last_end: float | None = None  # clock time the last update ended
        self._first = True  # no update processed yet
        self._failing = False  # the last update was not processed
        self._last: datetime | None = None
        self._note, self._level, self._brief = "", OK, ""
        self._unreachable: tuple[str, str, str] | None = None  # the note before an outage
        self._notices = QFileSystemWatcher(self)
        self._notices.directoryChanged.connect(self._on_notice)
        self._debounce = _single_shot(self, DEBOUNCE_MS, self.check_now)
        self._timer = _single_shot(self, POLL_MS, self.check_now)
        self._hold = _single_shot(self, 0, self._run_held)  # the gap between updates
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
        return WatchStatus(self._folder, count, self._last, self._note, self._level, self._brief)

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
        self._held, self._axis_key, self._skips = set(), None, []
        self._queued = ([], [], False)
        self._last_end = None
        self._first, self._failing = True, False
        self._last = None
        self._note, self._level, self._brief = "", OK, ""
        self._unreachable = None
        c.set_spectrum_cache(True)
        self._notices.addPath(folder)
        logger.info("Watching %s for new spectra.", folder)
        ended = self._last_end
        self._check(wall=time.time())
        idle = self._folder is not None and self._last_end == ended and not self._has_queued()
        if idle and c.data_state("sample") != DATA_CURRENT:  # listed, not processed as they are
            self._queue([], [], True)
        self.changed.emit()

    def stop(self) -> None:
        """Stop watching; the files and the map shown stay."""
        folder = self._folder
        if folder is None:
            return
        self._folder = None
        for timer in (self._timer, self._debounce, self._hold):
            timer.stop()
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
            self._check()

    def _on_notice(self, _path: str) -> None:
        self._debounce.start()

    def _check(self, wall: float | None = None) -> None:
        """Look at the folder; whatever happens, look again later."""
        try:
            self._scan(wall)
        except Exception as exc:  # never end watching silently
            self._unexpected("check", exc)
        finally:
            self._report_skips()
            self._schedule()

    def _scan(self, wall: float | None = None) -> None:
        folder = self._folder
        assert folder is not None
        try:
            listing = folder_listing(folder)
        except OSError as exc:
            self._lost(folder, exc)
            return
        self._found(folder)
        arrivals = self._arrivals
        arrivals.observe(listing, self.clock(), wall)
        self._handled &= {path_key(p) for p in listing}  # written again: a new file
        listed = self._listed()
        candidates = [p for p in arrivals.complete() if self._accepted.get(p) != listing[p]]
        axis = None
        if candidates or self._held:  # held files are tried again when the axis changes
            axis = self._sweep_axis(listing, listed, set(candidates))
        new: list[str] = []
        changed = False
        for path in candidates:
            try:
                x, _y = self.controller.read_spectrum(path)
            except EXPECTED_ERRORS as exc:
                if arrivals.failed(path):
                    self._skips.append((path, NOT_SPECTRUM, _reason(path, exc)))
                continue
            if axis is not None and not same_axis(x, axis):
                self._held.add(path)  # perhaps still written (a text file parses half-way)
                if arrivals.failed(path):
                    detail = f"{_axis_text(x)}, the sweep has {_axis_text(axis)}"
                    self._skips.append((path, OTHER_AXIS, detail))
                continue
            arrivals.passed(path)
            self._held.discard(path)
            known = path in self._accepted
            self._accepted[path] = listing[path]
            if not known:
                if path_key(path) not in self._handled:
                    new.append(path)
            elif path_key(path) in listed:
                changed = True  # a listed spectrum changed since it was read
        gone = [p for p in self._accepted if p not in listing]
        for path in gone:
            del self._accepted[path]
        self._held &= set(listing)
        gone = [p for p in gone if path_key(p) in listed]
        if new or gone or changed or self._has_queued():
            self._queue(new, gone, changed)

    def _sweep_axis(self, listing: Listing, listed: set[str], candidates: set[str]):
        """The energy axis most of the sweep's spectra from this folder share (read from the
        cache), or None before there are any; files held back for another axis are checked
        again when it changes."""
        axes: dict[tuple, np.ndarray] = {}
        counts: Counter = Counter()
        for path, signature in self._accepted.items():
            if path in candidates or listing.get(path) != signature:
                continue
            if path_key(path) not in listed:
                continue
            try:
                x, _y = self.controller.read_spectrum(path)
            except EXPECTED_ERRORS:
                continue
            key = (x.size, float(x[0]), float(x[-1])) if x.size else (0,)
            axes.setdefault(key, x)
            counts[key] += 1
        if not counts:
            return None
        key = counts.most_common(1)[0][0]
        if key != self._axis_key:
            if self._axis_key is not None:
                for path in self._held:
                    self._arrivals.retry(path)
            self._axis_key = key
        return axes[key]

    def _schedule(self) -> None:
        if self._folder is None:
            return
        if self._unreachable is not None:
            interval = UNREACHABLE_MS
        else:
            interval = RECHECK_MS if self._arrivals.waiting() else POLL_MS
        self._timer.start(interval)

    def _lost(self, folder: str, exc: OSError) -> None:
        """The folder cannot be read: keep the lists and look again until it is back."""
        if self._unreachable is not None:
            return
        self._unreachable = (self._note, self._level, self._brief)
        since = datetime.now()
        why = exc.strerror or str(exc)
        logger.warning(
            "Watched folder %s cannot be read (%s); looking again every %d s.",
            folder,
            why,
            UNREACHABLE_MS // 1000,
        )
        self._set_note(
            f"Folder not reachable since {since:%H:%M:%S}",
            PROBLEM,
            f"not reachable since {since:%H:%M}",
        )

    def _found(self, folder: str) -> None:
        """The folder can be read (again)."""
        if self._unreachable is None:
            return
        logger.info("Watched folder %s can be read again.", folder)
        note = self._unreachable
        self._unreachable = None
        if folder not in self._notices.directories():
            self._notices.addPath(folder)
        self._set_note(*note)

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

    def _report_skips(self) -> None:
        """Report the files left out since the last report, in one message."""
        skips, self._skips = self._skips, []
        if not skips:
            return
        title, text = skip_report(skips)  # the text names the files of a group
        logger.warning("Watched folder: %s. %s", title, text)
        self.skipped.emit(title, text)

    def _unexpected(self, what: str, exc: BaseException) -> None:
        logger.exception("Watched folder: the %s failed", what)
        self._set_note(f"The {what} failed ({exc!r}); see the log", PROBLEM, f"{what} failed")

    # --- updates ----------------------------------------------------------------------------
    def _has_queued(self) -> bool:
        new, gone, changed = self._queued
        return bool(new or gone or changed)

    def _queue(self, new: Iterable[str], gone: Iterable[str], changed: bool) -> None:
        """Add files to the next update and run it when it may run."""
        queued_new, queued_gone, queued_changed = self._queued
        self._queued = ([*queued_new, *new], [*queued_gone, *gone], queued_changed or changed)
        self._try_run()

    def _try_run(self) -> None:
        """Run the queued update now, or once the one running ended and the gap has passed."""
        if self._folder is None or self._busy or not self._has_queued():
            return
        if self._last_end is not None:
            wait = self._last_end + MIN_GAP_S - self.clock()
            if wait > 0:
                self._hold.start(max(1, round(1000 * wait)))
                return
        self._run()

    def _run_held(self) -> None:
        if self._folder is not None:
            self._try_run()
            self._report_skips()

    def _run(self) -> None:
        new, gone, changed = self._queued
        self._queued = ([], [], False)
        self._busy = True
        try:
            self._update(new, gone, changed)
        except Exception as exc:  # the next update may work: keep watching
            self._unexpected("update", exc)
        finally:
            self._busy = False
            self._last_end = self.clock()
        if self._has_queued():  # found during the update: one more, after the gap
            self._try_run()

    def _update(self, new: list[str], gone: list[str], changed: bool) -> None:
        c = self.controller
        files = c.processing.sample_files
        zero, field = split_zero(new)
        if not c.processing.custom_field:
            for path in [p for p in field if parse_field(p) is None]:
                field.remove(path)
                self._skips.append((path, NO_FIELD, ""))
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
        p = c.processing
        files = p.sample_files
        head = f"Watched folder: {what}; " if what else "Watched folder: "
        waiting = self._waiting_for(p)
        if waiting:
            self._set_note(f"Waiting for {waiting}", WAITING, f"waiting for {waiting}")
            logger.info("%swaiting for %s.", head, waiting)
            return
        if c.result_source == "library":
            self._set_note(
                "A library map is shown; Process shows the sweep again",
                WAITING,
                "library map shown",
            )
            logger.info("%snot processed while a library map is shown.", head)
            return
        start = time.perf_counter()
        try:
            c.process_update(first=self._first)
        except EXPECTED_ERRORS as exc:
            self._set_note(f"Not processed: {exc}", PROBLEM, "not processed")
            if not self._failing:  # reported once until an update is processed again
                self._failing = True
                self.failed.emit(exc)
            else:
                logger.warning("%snot processed: %s", head, exc)
            return
        self._first, self._failing = False, False
        self._last = datetime.now()
        self._set_note("", OK, "")
        ms = 1000 * (time.perf_counter() - start)
        logger.info("%s%d spectra processed in %.0f ms.", head, len(files.field), ms)

    @staticmethod
    def _waiting_for(p) -> str:
        """What the sweep lacks before it can be processed ("" if nothing)."""
        files = p.sample_files
        if not files.field:
            return "in-field spectra"
        if not files.zero:
            return "a zero-field spectrum"
        if p.custom_field:
            try:
                missing = p.sample_field.count() - len(files.field)
            except ValueError:  # Process says what is wrong with the range
                return ""
            if missing > 0:
                more = "1 more file" if missing == 1 else f"{missing} more files"
                return f"{more} to match the custom field range"
        return ""

    def _set_note(self, note: str, level: str, brief: str) -> None:
        if self._unreachable is not None and level != PROBLEM:
            self._unreachable = (note, level, brief)  # shown when the folder is back
            return
        self._note, self._level, self._brief = note, level, brief
        self.changed.emit()


def _single_shot(parent: QObject, interval: int, slot: Callable[[], None]) -> QTimer:
    timer = QTimer(parent)
    timer.setSingleShot(True)
    timer.setInterval(interval)
    timer.timeout.connect(slot)
    return timer


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
