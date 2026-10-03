"""Help > Check for updates…: a notice in the info bar when a newer version is released.

The app never installs anything: the bundles are not code-signed, so people download the new
version and replace the app, or update a source checkout with ``git pull && uv sync``. Once a
day, a few seconds after the start, one anonymous request asks GitHub for the newest releases
(:mod:`mag_opt_detective.updates`, on a worker thread); offline or on any error the automatic
check stays silent. The app's start asks for that check (:meth:`UpdateChecker.check_at_startup`);
windows in tests and the smoke test never check by themselves. The notice stays until the user
closes it or uses one of its buttons: when the program closes the bar (Process, a new result)
or shows another message on it, the notice comes back once the bar is free.

Settings (``v2`` prefix): ``updates/check_at_startup`` (Help > Check for updates at startup),
``updates/last_check`` (ISO time of the last check), ``updates/skipped`` (the version skipped:
newer ones show again) and ``updates/channel`` (``stable`` or ``pre-release``; unset, pre-releases
count while the app is 0.x).
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future
from datetime import datetime
from functools import partial

from PySide6.QtCore import QObject, QTimer, Signal
from PySide6.QtGui import QAction

from mag_opt_detective import __version__
from mag_opt_detective import updates as feed
from mag_opt_detective.gui.controller import user_action

logger = logging.getLogger("mag_opt_detective")

STARTUP_DELAY_MS = 4000  # after the start, so the check never slows it down
POLL_MS = 100
DEADLINE_S = 15.0  # a check GitHub has not answered by then fails (a trickling connection)
AGAIN_MS = 250  # a notice the program closed comes back once the bar has been free this long
AT_STARTUP_KEY = "updates/check_at_startup"
LAST_CHECK_KEY = "updates/last_check"
SKIPPED_KEY = "updates/skipped"
CHANNEL_KEY = "updates/channel"
TITLE = "Check for updates"  # how errors name the action ("Can't check for updates")
FROM_BUNDLE = "Download it and replace this app with the new one."
FROM_SOURCE = "Running from source: update with git pull && uv sync."
CHECKING = "Asking GitHub for the newest release."
NEWEST = "There is no newer release on GitHub."


@user_action("Open the release")
def open_release(window, url: str) -> None:
    """Open a release page or file of this repository (the notice's buttons) as the Help
    menu's links open theirs: without a web browser the error bar offers to copy it."""
    window.open_link("Open the release", url)


class ActionSetting:
    """Settings protocol for a checkable action (stored as a bool)."""

    def __init__(self, action: QAction):
        self.action = action

    def settings_value(self) -> bool:
        return self.action.isChecked()

    def set_settings_value(self, value) -> bool:
        if isinstance(value, str):  # ini files store text
            value = {"true": True, "false": False}.get(value.lower(), value)
        if not isinstance(value, bool):
            return False
        self.action.setChecked(value)
        return True


class UpdateChecker(QObject):
    """Checks for a newer version for *window*: once a day after the start
    (:meth:`check_at_startup`) and when asked (:meth:`check_now`), one check at a time.

    ``fetch`` reads the releases on a worker thread (tests give a fake), ``clock`` tells the
    time and ``version`` is the running version. ``finished`` fires on the GUI thread when a
    check has been answered, or has failed, and its result is shown.
    """

    finished = Signal()

    def __init__(self, window) -> None:
        super().__init__(window)
        self.window = window
        self.version = __version__
        self.fetch: Callable[[], list[feed.Release]] = partial(feed.fetch_releases, __version__)
        self.clock: Callable[[], datetime] = datetime.now
        self.check_action: QAction = window.commands["check_updates"]
        self.startup_action: QAction = window.commands["check_updates_at_startup"]
        self._future: Future | None = None
        self._manual = False  # the check running was asked for: say how it ended
        self._started = 0.0  # time.monotonic() at the start of the running check
        self._notice: feed.Release | None = None  # offered until the user is done with it
        self._again = QTimer(self)
        self._again.setSingleShot(True)
        self._again.setInterval(AGAIN_MS)  # a burst of closes (a live baseline) shows it once
        self._again.timeout.connect(self._show_again)
        window.infobar.closed.connect(self._closed)
        window.infobar.closedByUser.connect(self._closed_by_user)
        self._startup = QTimer(self)
        self._startup.setSingleShot(True)
        self._startup.timeout.connect(self._check_automatically)
        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(POLL_MS)
        self._poll_timer.timeout.connect(self._poll)

    # ------------------------------------------------------------------ settings
    def _stored(self, key: str) -> str:
        p = self.window.persistence
        value = p.value(key, "") if p is not None else ""
        return value if isinstance(value, str) else ""

    def last_check(self) -> datetime | None:
        try:
            return datetime.fromisoformat(self._stored(LAST_CHECK_KEY))
        except ValueError:
            return None

    def skipped(self) -> str:
        """The version skipped with *Skip this version* ("" for none)."""
        return self._stored(SKIPPED_KEY)

    def skip(self, version: feed.Version) -> None:
        """No notice of *version* (or older ones) at startup; a newer one shows again."""
        if self.window.persistence is not None:
            self.window.persistence.set_value(SKIPPED_KEY, str(version))
        logger.info("Version %s skipped: the next one will show again.", version)

    def channel(self) -> str:
        """``stable`` or ``pre-release``: as stored, else pre-releases while the app is 0.x."""
        stored = self._stored(CHANNEL_KEY)
        if stored in feed.CHANNELS:
            return stored
        current = self.current()
        return feed.default_channel(current) if current is not None else feed.STABLE

    def current(self) -> feed.Version | None:
        """The running version, or None if it is no version number."""
        try:
            return feed.Version.parse(self.version)
        except ValueError:
            return None

    # ------------------------------------------------------------------ checks
    def check_due(self) -> bool:
        """The daily check is on and has not run today (never without settings)."""
        if self.window.persistence is None or not self.startup_action.isChecked():
            return False
        last = self.last_check()
        return self.current() is not None and (last is None or last.date() != self.clock().date())

    def check_at_startup(self, delay_ms: int = STARTUP_DELAY_MS) -> bool:
        """Check in *delay_ms* if :meth:`check_due`; False if not (off, done today, tests)."""
        if not self.check_due():
            return False
        self._startup.start(delay_ms)
        return True

    def check_now(self) -> None:
        """Help > Check for updates…: check at once and say what came out, also when there
        is nothing new or the check fails."""
        if self.current() is None:
            self.window.report_error(TITLE, f"this version ({self.version}) has no number")
            return
        self.window.infobar.show_message("info", "Checking for updates…", CHECKING)
        self._start(manual=True)

    def pending(self) -> bool:
        """A check is waiting for the start or for GitHub's answer."""
        return self._startup.isActive() or self._future is not None

    def _check_automatically(self) -> None:
        if self.check_due():  # still on, and no check since it was scheduled
            self._start(manual=False)

    def _start(self, manual: bool) -> None:
        if self.window.persistence is not None:
            stamp = self.clock().isoformat(timespec="seconds")
            self.window.persistence.set_value(LAST_CHECK_KEY, stamp)
        self._manual = self._manual or manual
        if self._future is not None:
            return
        future: Future = Future()
        fetch = self.fetch

        def run() -> None:  # worker thread: no Qt objects here
            try:
                future.set_result(fetch())
            except Exception as exc:
                future.set_exception(exc)

        # a daemon thread: a request still waiting for its timeout never holds up the exit
        threading.Thread(target=run, name="update-check", daemon=True).start()
        self._future, self._started = future, time.monotonic()
        self._poll_timer.start()

    def _poll(self) -> None:
        future = self._future
        late = time.monotonic() - self._started > DEADLINE_S
        if future is None or not (future.done() or late):
            return
        self._poll_timer.stop()
        self._future, manual, self._manual = None, self._manual, False
        if not future.done():  # its thread ends by itself; its answer is dropped
            self._failed(f"GitHub did not answer within {DEADLINE_S:g} s", manual)
        elif (exc := future.exception()) is None:
            self._show_result(future.result(), manual)
        elif isinstance(exc, feed.UpdateCheckError):
            self._failed(str(exc), manual)
        else:  # a bug: in the log, never in a dialog
            logger.warning("Checking for updates failed", exc_info=exc)
            self._failed(f"unexpected error: {exc!r}", manual)
        self.finished.emit()

    def _failed(self, message: str, manual: bool) -> None:
        if manual:
            self.window.report_error(TITLE, message)
        else:  # silent: offline, a proxy, GitHub's limit…
            logger.info("Could not check for updates: %s.", message)

    def _show_result(self, releases: list[feed.Release], manual: bool) -> None:
        release = feed.update_for(releases, self.current(), self.channel())
        bar = self.window.infobar
        if release is None:
            self._notice = None
            logger.info("Checked for updates: %s is the newest version.", self.version)
            if manual:
                bar.show_message("info", f"You have the newest version ({self.version})", NEWEST)
            return
        logger.info("Version %s is available (this is %s).", release.version, self.version)
        if not manual and self._is_skipped(release):
            return
        self._notice = release
        if manual or bar.isHidden():
            self.show_notice(release)
        # else never over another message: the notice follows when the bar closes

    def _closed(self) -> None:
        if self._notice is not None:
            self._again.start()

    def _closed_by_user(self) -> None:
        """The user closed the notice, or used one of its buttons: done with it (another
        message the user closed leaves it waiting)."""
        notice = self._notice
        if notice is not None and self.window.infobar.title_label.text() == self._title(notice):
            self._notice = None
            self._again.stop()

    def _show_again(self) -> None:
        if self._notice is not None and self.window.infobar.isHidden():
            self.show_notice(self._notice)  # else after the message on the bar

    def _is_skipped(self, release: feed.Release) -> bool:
        try:
            return release.version <= feed.Version.parse(self.skipped())
        except ValueError:  # none skipped
            return False

    def _title(self, release: feed.Release) -> str:
        return f"Version {release.version} is available (you have {self.version})"

    def notice(self) -> feed.Release | None:
        """The release offered until the user closes its notice or uses a button on it."""
        return self._notice

    def show_notice(self, release: feed.Release) -> None:
        """The notice of *release* in the info bar: Download (the archive for this system in
        an app bundle, else the release page), Release notes and Skip this version. It stays
        offered (:meth:`notice`) until the user is done with it."""
        self._notice = release
        frozen = getattr(sys, "frozen", False)
        download = release.download_url() if frozen else release.page
        self.window.infobar.show_message(
            "info",
            self._title(release),
            FROM_BUNDLE if frozen else FROM_SOURCE,
            actions=[
                ("Download", lambda: open_release(self.window, download)),
                ("Release notes", lambda: open_release(self.window, release.page)),
                ("Skip this version", lambda: self.skip(release.version)),
            ],
        )


def install(window) -> None:
    """Add Help > Check for updates… and Check for updates at startup (before About) and the
    checker (``window.updates``); bind the switch's setting.

    The daily check itself starts only with :meth:`UpdateChecker.check_at_startup`.
    """
    check = QAction("Check for &updates…", window)
    check.setStatusTip("Ask GitHub now whether a newer version has been released")
    at_startup = QAction("Check for updates at s&tartup", window)
    at_startup.setStatusTip("Ask GitHub once a day, a few seconds after the start")
    at_startup.setCheckable(True)
    at_startup.setChecked(True)
    for name, action in (("check_updates", check), ("check_updates_at_startup", at_startup)):
        window.commands[name] = action
        window.help_menu.insertAction(window.commands["about"], action)
    checker = UpdateChecker(window)
    window.updates = checker
    checker.check_action.triggered.connect(lambda _checked=False: checker.check_now())
    if window.persistence is not None:
        window.persistence.bind(AT_STARTUP_KEY, ActionSetting(checker.startup_action))
    else:  # nothing to remember it in (as View > Reset settings)
        checker.startup_action.setEnabled(False)
