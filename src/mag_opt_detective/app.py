"""Application entry point."""

from __future__ import annotations

import logging
import sys

from PySide6.QtWidgets import QApplication

from mag_opt_detective.gui.main_window import MainWindow
from mag_opt_detective.gui.settings import default_settings


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    app = QApplication.instance() or QApplication(sys.argv if argv is None else argv)
    app.setApplicationName("Magneto-Optical Detective")
    window = MainWindow(settings=default_settings())
    window.show()
    if not window.geometry_restored:
        window.center_on_screen()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
