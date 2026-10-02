from PySide6.QtGui import QIcon

from mag_opt_detective import app


def test_smoke_test_passes(qapp):
    assert app.main(["mag-opt-detective", "--smoke-test"]) == 0


def test_qt_arguments_are_ignored():
    args = app.parse_args(["mag-opt-detective", "-platform", "offscreen", "--smoke-test"])
    assert args.smoke_test


def test_app_icon_is_packaged(qapp):
    icon = app.app_icon()
    assert not icon.isNull()
    assert icon.pixmap(64, 64).width() == 64


def test_main_sets_window_icon(qapp):
    qapp.setWindowIcon(QIcon())
    app.main(["mag-opt-detective", "--smoke-test"])
    assert not qapp.windowIcon().isNull()
