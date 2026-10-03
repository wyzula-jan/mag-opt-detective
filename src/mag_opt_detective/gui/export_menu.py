"""Export > Journal figure…: the journal figure window, next to the quick PNG/SVG save.

The window (:mod:`~mag_opt_detective.gui.export_dialog`) is built the first time it opens, so
matplotlib is only imported then. What it remembers (:class:`ExportSettings`) and the user's own
presets (:class:`PresetStore`) are bound to the settings from the start, so they are restored
with everything else.
"""

from __future__ import annotations

import json
import math

from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import QApplication, QMenu

from mag_opt_detective.gui import icons
from mag_opt_detective.gui.controller import user_action

SETTINGS_KEY = "export/figure"
PRESETS_KEY = "export/presets"
SHORTCUT = "Ctrl+Shift+E"
MENU_TEXT = "Journal figure…"  # also the window's title and header

# what the export window remembers; None numbers take the preset's default
DEFAULTS: dict[str, object] = {
    "preset": "nature",
    "width": "",  # column of the preset ("" for its default)
    "custom_width": None,  # mm, Custom preset
    "height": None,  # mm
    "font": None,  # pt
    "line": None,  # pt
    "format": "pdf",
    "dpi": None,
    "colorbar": True,
    "colorbar_label": "",
    "colorbar_location": "right",
    "panel_label": "",
    "colour_range": "window",  # the map's colour range: "window", "auto" or "fixed"
    "fixed_levels": {},  # level key -> [lo, hi]; per-unit energy derivatives per cm^-1
    "tick_direction": "out",
    "tick_mirror": False,
    "tick_length": None,  # pt; None follows the text size
    "tick_width": None,  # pt; None: the line width
    "minor_ticks": False,
    "minor_intervals": 2,
    "minor_length": None,  # pt; None follows the text size
}


def _valid(default, value) -> bool:
    if isinstance(default, bool):
        return isinstance(value, bool)
    if isinstance(default, str):
        return isinstance(value, str)
    if isinstance(default, dict):
        return isinstance(value, dict)
    if isinstance(default, int):
        return isinstance(value, int) and not isinstance(value, bool)
    # numbers (default None): a finite number or None
    if value is None:
        return True
    return not isinstance(value, bool) and isinstance(value, int | float) and math.isfinite(value)


class ExportSettings:
    """Settings protocol for the export window's choices (one JSON object, see ``DEFAULTS``).

    Values of the wrong type are dropped one by one; the window checks the rest against the
    presets when it applies them. *listeners* are called after a restore or reset.
    """

    def __init__(self):
        self.values: dict[str, object] = dict(DEFAULTS)
        self.listeners: list = []

    def update(self, **changes) -> None:
        unknown = set(changes) - set(DEFAULTS)
        if unknown:
            raise KeyError(f"unknown export settings: {', '.join(sorted(unknown))}")
        self.values.update(changes)

    def settings_value(self) -> str:
        return json.dumps(self.values, sort_keys=True)

    def set_settings_value(self, value) -> bool:
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except ValueError:
                return False
        if not isinstance(value, dict):
            return False
        values = dict(DEFAULTS)
        for key, default in DEFAULTS.items():
            if key in value and _valid(default, value[key]):
                values[key] = value[key]
        self.values = values
        for listener in list(self.listeners):
            listener()
        return True


class PresetStore:
    """Settings protocol for the user's figure presets: the JSON text of a preset file ("" for
    none). Only its shape is checked here; the export window reads the presets themselves
    (:func:`export.user_presets.stored_presets`) when it opens, so matplotlib is not imported
    at start. *listeners* are called after a restore or reset."""

    def __init__(self):
        self.text = ""
        self.listeners: list = []

    def settings_value(self) -> str:
        return self.text

    def set_settings_value(self, value) -> bool:
        if not isinstance(value, str):
            return False
        if value:
            try:
                data = json.loads(value)
            except ValueError:
                return False
            if not isinstance(data, dict) or not isinstance(data.get("presets"), list):
                return False
        self.text = value
        for listener in list(self.listeners):
            listener()
        return True


class _CloseWatcher(QObject):
    """Closes the export window together with the main window."""

    def eventFilter(self, watched, event) -> bool:
        if event.type() == QEvent.Type.Close:
            dialog = getattr(watched, "export_dialog", None)
            if dialog is not None:
                dialog.close()
        return False


@user_action("Export figure")
def open_dialog(window) -> None:
    """Show the export window for the plot on screen (built the first time)."""
    if window.controller.result is None:
        raise ValueError("nothing to export – process data first")
    dialog = window.export_dialog
    if dialog is None:
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)  # imports matplotlib
        try:
            from mag_opt_detective.gui.export_dialog import ExportDialog

            dialog = ExportDialog(window, window.export_settings, window.export_presets)
        finally:
            QApplication.restoreOverrideCursor()
        window.export_dialog = dialog
    dialog.open_for_window()


def install(window) -> None:
    """Add Export > Journal figure… (Ctrl+Shift+E) before the quick image save in the menus."""
    window.export_settings = ExportSettings()
    window.export_presets = PresetStore()
    window.export_dialog = None
    action = QAction(MENU_TEXT, window)
    action.setShortcut(QKeySequence(SHORTCUT))
    action.setToolTip("A journal figure (PDF, SVG, EPS, PNG, TIFF) at print size")
    icons.set_icon(action, "file-chart-line")
    action.triggered.connect(lambda _checked=False: open_dialog(window))
    window.addAction(action)
    window.commands["export_figure"] = action
    quick = window.commands["export_image"]
    for menu in quick.associatedObjects():  # the toolbar's Export menu and the File menu
        if isinstance(menu, QMenu):
            menu.insertAction(quick, action)
    window.installEventFilter(_CloseWatcher(window))
    if window.persistence is not None:
        window.persistence.bind(SETTINGS_KEY, window.export_settings)
        window.persistence.bind(PRESETS_KEY, window.export_presets)
