import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication, QIcon, QPalette
from PySide6.QtWidgets import QApplication

from mag_opt_detective import app
from mag_opt_detective.gui import icons, theme


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


def test_qt_arguments_are_ignored():
    args = app.parse_args(["mag-opt-detective", "-platform", "offscreen", "--smoke-test"])
    assert args.smoke_test


def test_app_icon_is_packaged(qapp):
    icon = app.app_icon()
    assert not icon.isNull()
    assert icon.pixmap(64, 64).width() == 64


def test_main_sets_window_icon(qapp, restore_look):
    qapp.setWindowIcon(QIcon())
    app.main(["mag-opt-detective", "--smoke-test"])
    assert not qapp.windowIcon().isNull()
