"""Application entry point."""

from __future__ import annotations

import argparse
import logging
import sys
from importlib.resources import files

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from mag_opt_detective import __version__
from mag_opt_detective.gui.main_window import MainWindow
from mag_opt_detective.gui.settings import default_settings
from mag_opt_detective.gui.teardown import delete_widget
from mag_opt_detective.gui.theme import Theme


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="mag-opt-detective", description="Magneto-optical FTIR data analysis."
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="process a small synthetic measurement, then exit (0 = success)",
    )
    # Qt options such as -platform are left for QApplication
    args, _qt_args = parser.parse_known_args(argv[1:])
    return args


def app_icon() -> QIcon:
    """Window and Dock icon; on macOS the version with the icon-grid margin and shadow."""
    name = "icon-macos.png" if sys.platform == "darwin" else "icon.png"
    return QIcon(str(files("mag_opt_detective") / "resources" / name))


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv if argv is None else argv
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    app = QApplication.instance() or QApplication(argv)
    app.setApplicationName("Magneto-Optical Detective")
    theme = Theme("system")  # the window restores the user's choice
    theme.apply(app)
    app.setWindowIcon(app_icon())

    if args.smoke_test:
        from mag_opt_detective import smoke

        window = MainWindow(theme=theme)  # no settings: the test must not touch the user's
        try:
            smoke.run(window)
        except Exception:
            logging.getLogger(__name__).exception("Smoke test failed")
            return 1
        finally:
            window.close()
            delete_widget(window)  # now, not at the interpreter's exit (see gui.teardown)
        return 0

    window = MainWindow(settings=default_settings(), theme=theme)
    try:
        window.show()
        if not window.geometry_restored:
            window.center_on_screen()
        return app.exec()
    finally:
        if window.isVisible():  # the event loop ended another way: save as a close does
            window.close()
        delete_widget(window)


if __name__ == "__main__":
    raise SystemExit(main())
