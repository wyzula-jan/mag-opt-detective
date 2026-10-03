"""Help > About (with the app icon) and its Licences… window: third-party licences.

App bundles carry ``THIRD_PARTY_NOTICES.txt`` (written by ``packaging/third_party_notices.py``
at build time) with every licence text. Run from source, the window lists the installed
packages with their licences instead, and both show the Lucide icon licence.
"""

from __future__ import annotations

import re
from importlib import metadata
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QMessageBox,
    QPlainTextEdit,
    QVBoxLayout,
    QWidget,
)

import mag_opt_detective

NOTICES = "THIRD_PARTY_NOTICES.txt"
LUCIDE = Path(__file__).resolve().parent / "icons" / "LICENSE-lucide.txt"
APP_ICON = Path(__file__).resolve().parent / "app_icon.png"  # packaging/make_icon.py
APP = "mag-opt-detective"


def notices_path() -> Path:
    """Where a bundle keeps its third-party notices (next to the package's modules)."""
    return Path(mag_opt_detective.__file__).resolve().parent / NOTICES


def _installed() -> list[str]:
    """``name version: licence`` of each package the app requires, as installed."""
    try:
        requires = metadata.requires(APP) or []
    except metadata.PackageNotFoundError:
        return []
    rows = []
    for text in requires:
        if "extra ==" in text:
            continue
        name = re.match(r"[A-Za-z0-9._-]+", text).group()
        try:
            meta = metadata.metadata(name)
        except metadata.PackageNotFoundError:
            continue
        licence = meta.get("License-Expression") or (meta.get("License") or "").strip()
        if not licence or "\n" in licence or len(licence) > 80:
            classifiers = [c.split(" :: ")[-1] for c in meta.get_all("Classifier") or []]
            licence = ", ".join(c for c in classifiers if "License" in c) or "see its licence"
        rows.append(f"  {meta['Name']} {meta['Version']}: {licence}")
    return sorted(rows, key=str.lower)


def licences_text() -> str:
    """The bundle's third-party notices, or (run from source) the installed packages."""
    path = notices_path()
    if path.is_file():
        return path.read_text(encoding="utf-8")
    rows = "\n".join(_installed())
    return (
        "Running from source: each package below is installed with its own licence text;\n"
        "the app bundles include all of them in THIRD_PARTY_NOTICES.txt.\n\n"
        f"{rows}\n\n"
        "Qt and PySide6 are used under the GNU Lesser General Public License v3.\n\n"
        f"Lucide icons (gui/icons)\n\n{LUCIDE.read_text(encoding='utf-8')}"
    )


class LicencesDialog(QDialog):
    """The licences of the third-party software, as plain text."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Licences")
        self.resize(760, 600)
        self.text = QPlainTextEdit(licences_text())
        self.text.setReadOnly(True)
        self.text.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addWidget(self.text)
        layout.addWidget(buttons)


def about_box(parent: QWidget, text: str) -> QMessageBox:
    """The About box with *text* (rich text) and a Licences… button that opens the licences.

    The box deletes itself when it closes.
    """
    box = QMessageBox(parent)
    box.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
    box.setWindowTitle("About")
    box.setText(text)
    icon = parent.windowIcon()
    if not icon.isNull():
        box.setIconPixmap(icon.pixmap(64, 64))
    box.addButton(QMessageBox.StandardButton.Ok)
    button = box.addButton("Licences…", QMessageBox.ButtonRole.ActionRole)
    button.clicked.connect(lambda: show_licences(parent))
    box.licences_button = button
    return box


def show_licences(parent: QWidget) -> LicencesDialog:
    """Open the licences window (not modal) and return it."""
    dialog = LicencesDialog(parent)
    dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
    dialog.show()
    return dialog
