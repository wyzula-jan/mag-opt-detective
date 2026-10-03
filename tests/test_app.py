import logging
import os
import subprocess
import sys
import textwrap

import pytest
import shiboken6
from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication, QIcon, QPalette
from PySide6.QtWidgets import QApplication, QMenu

from mag_opt_detective import app, export
from mag_opt_detective.gui import icons, licences, teardown, theme
from mag_opt_detective.gui.main_window import MainWindow


@pytest.fixture
def restore_look(qapp):
    """The smoke test applies the theme to the shared application; undo it afterwards."""
    palette, sheet, style = QPalette(qapp.palette()), qapp.styleSheet(), qapp.style().name()
    yield qapp
    qapp.setStyleSheet(sheet)
    qapp.setPalette(palette)
    if qapp.style().name() != style:
        QApplication.setStyle(style)
    QGuiApplication.styleHints().setColorScheme(Qt.ColorScheme.Unknown)
    theme._active = None
    icons.clear_cache()


def test_smoke_test_passes(restore_look):
    assert app.main(["mag-opt-detective", "--smoke-test"]) == 0
    assert theme.current_theme() is not None  # the application look comes from the theme
    assert not QApplication.windowIcon().isNull()


def test_smoke_test_fails_on_a_broken_figure_export(restore_look, monkeypatch):
    """The smoke test saves a journal figure in every format (backends, fonts, Pillow)."""

    def save_nothing(fig, path, *, dpi):
        path.write_bytes(b"")
        return path

    monkeypatch.setattr(export, "save", save_nothing)
    assert app.main(["mag-opt-detective", "--smoke-test"]) == 1


def test_a_bundle_without_licence_notices_fails_the_smoke_test(restore_look, monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(licences, "notices_path", lambda: tmp_path / "THIRD_PARTY_NOTICES.txt")
    assert app.main(["mag-opt-detective", "--smoke-test"]) == 1
    (tmp_path / "THIRD_PARTY_NOTICES.txt").write_text("notices", encoding="utf-8")
    assert app.main(["mag-opt-detective", "--smoke-test"]) == 0


def test_qt_arguments_are_ignored():
    args = app.parse_args(["mag-opt-detective", "-platform", "offscreen", "--smoke-test"])
    assert args.smoke_test


def test_windows_left_open_are_deleted_before_pyside_shuts_down(qapp):
    window, menu = MainWindow(), QMenu()  # a top-level menu that is not ours to delete
    window.show()
    teardown.delete_windows()
    assert not shiboken6.isValid(window)
    assert window.log_handler not in logging.getLogger("mag_opt_detective").handlers
    assert shiboken6.isValid(menu)
    menu.deleteLater()


def test_a_script_that_fails_with_a_window_open_exits_quietly(tmp_path):
    """The window is deleted before PySide shuts down: no cascade of "Internal C++ object
    already deleted" tracebacks from our event filters at the exit."""
    script = tmp_path / "fails.py"
    script.write_text(
        textwrap.dedent(
            """
            import sys
            from PySide6.QtWidgets import QApplication
            from mag_opt_detective.gui.main_window import MainWindow

            app = QApplication([])
            window = MainWindow()
            window.show()
            app.processEvents()
            window.destroyed.connect(lambda *_: print("window deleted", file=sys.stderr))
            raise SystemExit("the script failed")
            """
        )
    )
    run = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
    )
    assert run.returncode == 1, run.stderr
    assert "the script failed\nwindow deleted\n" in run.stderr
    assert "already deleted" not in run.stderr and "Error calling" not in run.stderr


def test_app_icon_is_packaged(qapp):
    icon = app.app_icon()
    assert not icon.isNull()
    assert icon.pixmap(64, 64).width() == 64


def test_main_sets_window_icon(qapp, restore_look):
    qapp.setWindowIcon(QIcon())
    app.main(["mag-opt-detective", "--smoke-test"])
    assert not qapp.windowIcon().isNull()
