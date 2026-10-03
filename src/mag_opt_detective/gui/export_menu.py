"""Export > Journal figure…: the journal figure window, next to the quick PNG/SVG save.

The window (:mod:`~mag_opt_detective.gui.export_dialog`) is built the first time it opens, so
matplotlib is only imported then. What it remembers (:class:`ExportSettings`) is bound to the
settings from the start, so it is restored with everything else. The user's own presets
(:class:`PresetStore`) are user data: they are kept outside the settings that Reset settings
forgets, and written as soon as they change.
"""

from __future__ import annotations

import json
import logging
import math

from PySide6.QtCore import QEvent, QObject, QSettings, Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import QApplication, QMenu

from mag_opt_detective.gui import icons
from mag_opt_detective.gui.controller import user_action

logger = logging.getLogger("mag_opt_detective")

SETTINGS_KEY = "export/figure"
# outside the v2 prefix: Reset settings (and a prefix bump) keep the presets; the preset
# file's own format version versions them
PRESETS_KEY = "figure-presets"
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


def presets_text(value) -> bool:
    """Whether *value* can be the stored text of the presets: "" (none) or a JSON object (the
    export window reads it: presets of a newer format may look different, and are kept)."""
    if not isinstance(value, str):
        return False
    if not value:
        return True
    try:
        data = json.loads(value)
    except (ValueError, RecursionError):
        return False
    return isinstance(data, dict)


class PresetStore:
    """The user's figure presets as the JSON text of a preset file ("" for none), kept in
    *settings* at ``PRESETS_KEY`` (or only in memory without settings).

    It is read once; :meth:`set_text` writes at once, so a crash loses no preset. The export
    window reads the presets themselves (:func:`export.user_presets.stored_presets`), so
    matplotlib is not imported at start.
    """

    def __init__(self, settings: QSettings | None = None):
        self.settings = settings
        self.text = ""
        if settings is not None:
            value = settings.value(PRESETS_KEY)
            if presets_text(value):
                self.text = value
            elif value is not None:
                logger.warning("Ignoring the stored figure presets: they cannot be read.")

    def set_text(self, text: str) -> None:
        self.text = text
        if self.settings is None:
            return
        if text:
            self.settings.setValue(PRESETS_KEY, text)
        else:
            self.settings.remove(PRESETS_KEY)
        self.settings.sync()


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
    settings = window.persistence.settings if window.persistence is not None else None
    window.export_presets = PresetStore(settings)
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
