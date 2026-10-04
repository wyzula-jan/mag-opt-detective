"""Help > Check for updates…: the daily check, the notice in the info bar and its actions.

GitHub is never asked: each check reads a fake transport (the sample answer in
``tests/data/github_releases.json`` unless a test gives another), browsers are never opened
(``QDesktopServices.openUrl`` is captured), and conftest's guard fails any test that tries to
reach the network.
"""

import io
import json
import sys
import threading
from datetime import datetime, timedelta
from functools import partial
from pathlib import Path
from urllib.error import HTTPError, URLError

import pytest
import shiboken6
from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QDesktopServices, QGuiApplication, QPalette
from PySide6.QtWidgets import QApplication

import gui_helpers
from gui_helpers import infobar_text, load_sweep, process
from mag_opt_detective import app
from mag_opt_detective import updates as feed
from mag_opt_detective.gui import icons, links, plot_panel, theme, updates
from mag_opt_detective.gui.main_window import MainWindow
from mag_opt_detective.gui.settings import PREFIX

window, errors = gui_helpers.window, gui_helpers.errors

SAMPLE = Path(__file__).parent / "data" / "github_releases.json"
NOW = datetime(2026, 10, 3, 9, 30)
PAGE = "https://github.com/wyzula-jan/mag-opt-detective/releases/tag/v0.3.0"
ACTIONS = ["Download", "Release notes", "Skip this version"]


@pytest.fixture
def opened(monkeypatch) -> list[str]:
    """The URLs the app asked the browser to open (none is opened)."""
    urls: list[str] = []
    monkeypatch.setattr(
        QDesktopServices, "openUrl", lambda url: urls.append(url.toString()) or True
    )
    return urls


@pytest.fixture
def ini(tmp_path) -> str:
    return str(tmp_path / "settings.ini")


@pytest.fixture
def stored(qtbot, errors, ini):
    """A window that keeps its settings (in *ini*), checking against the sample answer as
    version 0.2.0 at ``NOW``."""
    w = MainWindow(settings=QSettings(ini, QSettings.Format.IniFormat))
    qtbot.addWidget(w)
    answer(w.updates)
    yield w
    w.close()
    w.deleteLater()


def answer(checker, payload=None, error: Exception | None = None, version="0.2.0", gate=None):
    """Let *checker* read *payload* (default: the sample answer; or fail with *error*) through
    a fake transport, as *version* at ``NOW``; with *gate* the answer waits for it."""

    def transport(request, timeout):
        if gate is not None:
            gate.wait(10)
        if error is not None:
            raise error
        body = SAMPLE.read_bytes() if payload is None else json.dumps(payload).encode()
        return io.BytesIO(body)

    checker.version = version
    checker.clock = lambda: NOW
    checker.fetch = partial(feed.fetch_releases, version, opener=transport)


def release(tag: str, prerelease: bool = False) -> dict:
    return {
        "tag_name": tag,
        "html_url": f"https://github.com/wyzula-jan/mag-opt-detective/releases/tag/{tag}",
        "draft": False,
        "prerelease": prerelease,
        "assets": [],
    }


def check(qtbot, w, manual: bool = True) -> None:
    """Run a check (asked for, or the daily one) and wait for its result."""
    with qtbot.waitSignal(w.updates.finished, timeout=10_000):
        if manual:
            w.updates.check_now()
        else:
            assert w.updates.check_at_startup(delay_ms=0)


def button(w, text: str):
    return next(b for b in w.infobar.row_buttons() if b.text() == text)


def notice_shown(w) -> bool:
    bar = w.infobar
    return not bar.isHidden() and bar.title_label.text().startswith("Version 0.3.0 is available")


def help_menu(w):
    return next(a.menu() for a in w.menuBar().actions() if a.text() == "&Help")


def test_the_help_menu_checks_for_updates(window, stored):
    for w in (window, stored):
        assert help_menu(w) is w.help_menu
        check, at_startup = w.commands["check_updates"], w.commands["check_updates_at_startup"]
        actions = w.help_menu.actions()
        start = actions.index(check)  # in the last group, before About
        assert actions[start - 1].isSeparator()
        assert actions[start : start + 3] == [check, at_startup, w.commands["about"]]
        assert check.text() == "Check for &updates…"
        assert check.statusTip() and at_startup.statusTip()
        assert at_startup.isCheckable() and at_startup.isChecked()
    assert stored.commands["check_updates_at_startup"].isEnabled()
    # nothing would remember it without settings (as View > Reset settings)
    assert not window.commands["check_updates_at_startup"].isEnabled()


def test_windows_never_check_by_themselves(window, stored):
    """Only the app's start asks for the daily check; windows without settings (tests, the
    smoke test) never check at startup."""
    assert not stored.updates.pending() and not window.updates.pending()
    assert not window.updates.check_at_startup()
    assert stored.updates.check_due()  # what the app's start would schedule


def test_a_newer_version_shows_a_notice(qtbot, stored, opened):
    check(qtbot, stored, manual=False)
    bar = stored.infobar
    assert not bar.isHidden() and bar.level() == "info"
    assert bar.title_label.text() == "Version 0.3.0 is available (you have 0.2.0)"
    assert bar.text_label.text() == "Running from source: update with git pull && uv sync."
    assert [b.text() for b in bar.row_buttons()] == ACTIONS
    with qtbot.waitSignal(bar.closed):
        button(stored, "Release notes").click()
    assert opened == [PAGE] and bar.isHidden()
    assert stored.updates.notice() is None  # used: done with it
    stored.updates.show_notice(
        feed.parse_releases(json.loads(SAMPLE.read_text(encoding="utf-8")))[0]
    )
    button(stored, "Download").click()  # run from source: the release page
    assert opened == [PAGE, PAGE] and stored.updates.notice() is None


def test_a_link_no_browser_opens_offers_its_address(qtbot, stored, errors, monkeypatch):
    """As the Help menu's links: the error bar offers to copy the address."""
    monkeypatch.setattr(QDesktopServices, "openUrl", lambda url: False)
    check(qtbot, stored)
    QApplication.clipboard().clear()
    button(stored, "Release notes").click()
    assert errors == [links.NO_BROWSER]
    assert infobar_text(stored) == f"Can't open the release: {links.NO_BROWSER}"
    assert stored.infobar.action_button.text() == "Copy address"
    stored.infobar.action_button.click()
    assert QApplication.clipboard().text() == PAGE


def test_an_app_bundle_downloads_the_archive_for_its_system(qtbot, stored, opened, monkeypatch):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    check(qtbot, stored)
    bar = stored.infobar
    assert bar.title_label.text() == "Version 0.3.0 is available (you have 0.2.0)"
    assert bar.text_label.text() == "Download it and replace this app with the new one."
    assert "git pull" not in infobar_text(stored)
    button(stored, "Download").click()
    archive = feed.ASSETS.get(sys.platform)
    downloads = "https://github.com/wyzula-jan/mag-opt-detective/releases/download"
    assert opened == [f"{downloads}/v0.3.0/{archive}" if archive else PAGE]


def test_the_newest_version_shows_nothing_unless_asked(qtbot, stored, errors):
    # the newest release, and a source build after it, each on a new day
    for day, version in enumerate(("0.3.0", "0.3.1.dev0")):
        answer(stored.updates, version=version)
        stored.updates.clock = lambda day=day: NOW + timedelta(days=day)
        check(qtbot, stored, manual=False)
        assert stored.infobar.isHidden()
        check(qtbot, stored)
        assert stored.infobar.level() == "info"
        assert stored.infobar.title_label.text() == f"You have the newest version ({version})"
        stored.infobar.dismiss()
    answer(stored.updates, version="0.3.0.dev0")  # a source build before 0.3.0
    check(qtbot, stored)
    assert stored.infobar.title_label.text() == "Version 0.3.0 is available (you have 0.3.0.dev0)"
    assert errors == []


def test_pre_releases_count_before_1_0_0_only(qtbot, stored):
    payload = [release("v1.1.0-rc.1", prerelease=True), release("v1.0.1"), release("v1.0.0")]
    answer(stored.updates, payload, version="1.0.0")
    assert stored.updates.channel() == feed.STABLE
    check(qtbot, stored)
    assert stored.infobar.title_label.text() == "Version 1.0.1 is available (you have 1.0.0)"
    stored.persistence.set_value(updates.CHANNEL_KEY, "pre-release")
    check(qtbot, stored)
    assert stored.infobar.title_label.text().startswith("Version 1.1.0-rc.1 is available")
    stored.persistence.set_value(updates.CHANNEL_KEY, "nightly")  # not a channel: the default
    answer(stored.updates, payload, version="1.0.1")
    check(qtbot, stored)
    assert stored.infobar.title_label.text() == "You have the newest version (1.0.1)"
    # pre-releases count while the app is 0.x
    answer(stored.updates, version="0.2.0")
    assert stored.updates.channel() == feed.PRERELEASE


def test_a_skipped_version_shows_again_only_when_a_newer_one_is_out(qtbot, stored, opened, ini):
    c = stored.updates
    check(qtbot, stored, manual=False)
    with qtbot.waitSignal(stored.infobar.closed):
        button(stored, "Skip this version").click()
    assert opened == [] and c.skipped() == "0.3.0"
    assert QSettings(ini, QSettings.Format.IniFormat).value(f"{PREFIX}/updates/skipped") == "0.3.0"

    c.clock = lambda: NOW + timedelta(days=1)
    check(qtbot, stored, manual=False)
    assert stored.infobar.isHidden()  # skipped
    check(qtbot, stored)  # asked for: said all the same
    assert stored.infobar.title_label.text().startswith("Version 0.3.0 is available")
    stored.infobar.dismiss()

    payload = [release("v0.3.1", prerelease=True), *json.loads(SAMPLE.read_text(encoding="utf-8"))]
    answer(c, payload)
    c.clock = lambda: NOW + timedelta(days=2)
    check(qtbot, stored, manual=False)
    assert stored.infobar.title_label.text() == "Version 0.3.1 is available (you have 0.2.0)"


def test_the_daily_check_runs_once_a_day(qtbot, stored):
    c = stored.updates
    assert c.last_check() is None
    check(qtbot, stored, manual=False)
    assert c.last_check() == NOW
    stored.infobar.dismiss()  # closed: not shown again today
    assert not c.check_at_startup()
    c.clock = lambda: NOW + timedelta(hours=14)  # 23:30, the same day
    assert not c.check_at_startup() and not c.pending()
    c.clock = lambda: NOW + timedelta(days=1)
    check(qtbot, stored, manual=False)  # the next day
    assert c.last_check() == NOW + timedelta(days=1)
    c.clock = lambda: NOW + timedelta(days=2)
    check(qtbot, stored)  # asked for: today's check
    assert not c.check_due()


def test_the_daily_check_can_be_switched_off(qtbot, stored, ini):
    c = stored.updates
    stored.commands["check_updates_at_startup"].setChecked(False)
    assert not c.check_due() and not c.check_at_startup()
    stored.commands["check_updates_at_startup"].setChecked(True)
    assert c.check_at_startup(delay_ms=0)
    stored.commands["check_updates_at_startup"].setChecked(False)  # before it started
    qtbot.waitUntil(lambda: not c.pending())
    assert c.last_check() is None and stored.infobar.isHidden()

    stored.close()  # remembered
    again = MainWindow(settings=QSettings(ini, QSettings.Format.IniFormat))
    qtbot.addWidget(again)
    assert not again.commands["check_updates_at_startup"].isChecked()
    assert not again.updates.check_at_startup()
    again.persistence.set_value(updates.SKIPPED_KEY, "0.3.0")
    again.persistence.set_value(updates.LAST_CHECK_KEY, NOW.isoformat())
    again.reset_settings()  # back to on; the last check and the skipped version forgotten
    assert again.commands["check_updates_at_startup"].isChecked()
    assert again.updates.skipped() == "" and again.updates.last_check() is None
    again.close()


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (URLError(OSError(51, "Network is unreachable")), "No connection to GitHub"),
        (URLError(TimeoutError("timed out")), "GitHub did not answer within 5 s"),
        (
            HTTPError(feed.RELEASES_API, 403, "rate limit exceeded", {}, None),
            "GitHub limits how often it answers; try again in an hour",
        ),
    ],
)
def test_failures_are_silent_at_startup_and_said_when_asked(qtbot, stored, errors, error, message):
    answer(stored.updates, error=error)
    check(qtbot, stored, manual=False)
    assert stored.infobar.isHidden() and errors == []
    check(qtbot, stored)
    assert infobar_text(stored) == f"Can't check for updates: {message}"
    assert stored.infobar.level() == "error" and len(errors) == 1


def test_a_failing_check_is_never_a_dialog(qtbot, stored, errors):
    def broken():
        raise RuntimeError("a bug")

    stored.updates.fetch = broken
    check(qtbot, stored, manual=False)
    assert stored.infobar.isHidden() and errors == []
    check(qtbot, stored)
    assert infobar_text(stored).endswith("Unexpected error: RuntimeError('a bug')")
    stored.updates.version = "unknown"
    stored.updates.check_now()
    assert "this version (unknown) has no number" in errors[-1]
    assert not stored.updates.check_due()


def test_asking_says_it_is_checking(qtbot, stored):
    gate = threading.Event()
    answer(stored.updates, gate=gate)
    stored.updates.check_now()
    assert stored.infobar.title_label.text() == "Checking for updates…"
    assert stored.updates.pending()
    with qtbot.waitSignal(stored.updates.finished, timeout=10_000):
        gate.set()
    assert stored.infobar.title_label.text().startswith("Version 0.3.0 is available")


def test_the_notice_waits_for_the_message_on_the_bar(qtbot, stored):
    stored.report_error("Process", "the reference sweep has no files", panel="reference")
    check(qtbot, stored, manual=False)
    assert infobar_text(stored).startswith("Can't process")  # not replaced
    stored.infobar.close_button.click()  # the user closes the error: the notice follows
    qtbot.waitUntil(lambda: notice_shown(stored))
    stored.infobar.close_button.click()  # the user closes the notice: done for today
    stored.report_error("Process", "the reference sweep has no files", panel="reference")
    stored.infobar.close_button.click()
    qtbot.wait(3 * updates.AGAIN_MS)
    assert stored.infobar.isHidden() and stored.updates.notice() is None

    stored.updates.clock = lambda: NOW + timedelta(days=1)
    stored.report_error("Process", "the reference sweep has no files", panel="reference")
    check(qtbot, stored, manual=False)  # waits for the bar
    check(qtbot, stored)  # asked for meanwhile: shown at once, and only then
    assert stored.infobar.title_label.text().startswith("Version 0.3.0 is available")
    stored.infobar.close_button.click()
    assert stored.infobar.isHidden()


def test_only_the_user_closes_the_notice(qtbot, stored, sweep):
    """Process (and any new result) closes the bar: the notice comes back once the bar is
    free, also when it waited behind an error that Process closed."""
    process(stored)  # no files yet: an error on the bar
    assert infobar_text(stored).startswith("Can't process")
    check(qtbot, stored, manual=False)
    assert infobar_text(stored).startswith("Can't process")  # the notice waits
    load_sweep(stored, sweep)
    process(stored)  # closes the error
    assert stored.controller.result is not None
    qtbot.waitUntil(lambda: notice_shown(stored))
    process(stored)  # closes the notice for a moment
    assert stored.infobar.isHidden()
    qtbot.waitUntil(lambda: notice_shown(stored))

    stored.infobar.dismiss()  # the program closes it, then shows another message
    stored.report_error("Process", "the reference sweep has no files", panel="reference")
    qtbot.wait(3 * updates.AGAIN_MS)
    assert infobar_text(stored).startswith("Can't process")  # never over another message
    plot_panel.on_escape(stored)  # Escape in the window: the user closes the error
    qtbot.waitUntil(lambda: notice_shown(stored))

    plot_panel.on_escape(stored)  # and the notice: gone for the day
    process(stored)
    qtbot.wait(3 * updates.AGAIN_MS)
    assert stored.infobar.isHidden() and stored.updates.notice() is None


def test_a_trickling_answer_ends_the_check(qtbot, stored, monkeypatch):
    monkeypatch.setattr(updates, "DEADLINE_S", 0.3)
    gate = threading.Event()
    answer(stored.updates, gate=gate)
    try:
        check(qtbot, stored)
        assert infobar_text(stored) == "Can't check for updates: GitHub did not answer within 0.3 s"
        assert not stored.updates.pending()
        answer(stored.updates)  # the next check starts afresh
        check(qtbot, stored)
        assert notice_shown(stored)
    finally:
        gate.set()


def test_a_real_check_in_a_test_fails_it(qtbot, window, network_guard):
    """Help > Check for updates with the real request: refused on the worker thread, where
    it only becomes a message on the bar, and still recorded, so the test fails."""
    with qtbot.waitSignal(window.updates.finished, timeout=10_000):
        window.commands["check_updates"].trigger()
    assert infobar_text(window).startswith("Can't check for updates: Unexpected error")
    assert network_guard.attempts == ["the update check"]
    with pytest.raises(pytest.fail.Exception, match="tried to reach the network"):
        network_guard.check()
    network_guard.attempts.clear()  # tried on purpose


def test_a_check_still_running_when_the_window_closes_is_dropped(qtbot, ini):
    w = MainWindow(settings=QSettings(ini, QSettings.Format.IniFormat))
    gate = threading.Event()
    answer(w.updates, gate=gate)
    checker = w.updates
    checker.check_now()
    w.close()
    w.deleteLater()
    qtbot.waitUntil(lambda: not shiboken6.isValid(checker))
    gate.set()
    qtbot.wait(300)  # the answer comes back to no one


@pytest.fixture
def app_look(qapp):
    """app.main applies the theme to the shared application: undo it afterwards (as
    test_app's restore_look does)."""
    palette, sheet, style = QPalette(qapp.palette()), qapp.styleSheet(), qapp.style().name()
    yield qapp
    qapp.setStyleSheet(sheet)
    qapp.setPalette(palette)
    if qapp.style().name() != style:
        QApplication.setStyle(style)
    QGuiApplication.styleHints().setColorScheme(Qt.ColorScheme.Unknown)
    theme._active = None
    icons.clear_cache()


def test_the_app_asks_for_the_daily_check_at_its_start(app_look, monkeypatch, ini):
    asked = []
    monkeypatch.setattr(
        updates.UpdateChecker,
        "check_at_startup",
        lambda self, *args: asked.append(self.window.persistence is not None),
    )
    monkeypatch.setattr(app, "default_settings", lambda: QSettings(ini, QSettings.Format.IniFormat))
    monkeypatch.setattr(QApplication, "exec", lambda *_args: 0)  # no event loop
    assert app.main(["mag-opt-detective"]) == 0
    assert asked == [True]  # the window keeps the user's settings
