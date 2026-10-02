"""Application entry point."""

from __future__ import annotations

import argparse
import logging
import sys

from PySide6.QtWidgets import QApplication

from mag_opt_detective import __version__
from mag_opt_detective.gui.main_window import MainWindow
from mag_opt_detective.gui.settings import default_settings


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


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv if argv is None else argv
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    app = QApplication.instance() or QApplication(argv)
    app.setApplicationName("Magneto-Optical Detective")

    if args.smoke_test:
        from mag_opt_detective import smoke

        window = MainWindow()  # no settings: the test must not touch the user's
        try:
            smoke.run(window)
        except Exception:
            logging.getLogger(__name__).exception("Smoke test failed")
            return 1
        finally:
            window.close()
        return 0

    window = MainWindow(settings=default_settings())
    window.show()
    if not window.geometry_restored:
        window.center_on_screen()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
