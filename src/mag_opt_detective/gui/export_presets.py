"""My presets in the journal figure window: the user's own figure presets.

A menu button next to the journals lists them (the one the window shows is ticked) and saves,
renames, deletes, imports and exports them; names are typed in a row under the journals, never
in a dialog box. A preset is an :class:`~mag_opt_detective.export.UserPreset` (its checks and
JSON format are Qt-free, in ``export.user_presets``), kept by a
:class:`~mag_opt_detective.gui.export_menu.PresetStore`.
"""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path

from PySide6.QtCore import QEvent, QObject, QRect, QSize, Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLineEdit,
    QMenu,
    QStyle,
    QStyleOptionToolButton,
    QStylePainter,
    QToolButton,
    QWidget,
)

from mag_opt_detective.export import PRESETS
from mag_opt_detective.export.user_presets import (
    MAX_NAME,
    PresetError,
    UserPreset,
    clean_name,
    presets_from_json,
    presets_to_json,
    same_name,
    stored_presets,
)
from mag_opt_detective.gui import icons
from mag_opt_detective.gui.export_menu import PresetStore
from mag_opt_detective.gui.export_render import error_text
from mag_opt_detective.gui.export_state import FORMAT_LABELS, PRESET_NAMES, RANGE_NAMES
from mag_opt_detective.gui.kit import SmallButton
from mag_opt_detective.gui.kit._common import TightToolButton
from mag_opt_detective.gui.panels.common import Note, block
from mag_opt_detective.gui.theme import current_tokens
from mag_opt_detective.gui.widgets import open_file, save_file

logger = logging.getLogger("mag_opt_detective")

MY_PRESETS = "My presets"
PRESET_FILTER = "Figure presets (*.json);;All files (*)"
PRESET_KEEPS = "Keeps the journal, size, text, lines, file, colour bar, ticks and colour range."


def sentence(text: str) -> str:
    """*text* as a sentence: a capital first letter and a full stop."""
    text = text.strip()
    if not text:
        return text
    text = text[0].upper() + text[1:]
    return text if text.endswith((".", "!", "?")) else text + "."


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'s' if n != 1 else ''}"


class MenuButton(TightToolButton):
    """A kit button that opens a menu: its text, shortened to fit, and a chevron."""

    GAP = 6
    CHEVRON = 12
    PADDING = 9  # the stylesheet's padding and border, on each side
    MAX_TEXT = 118

    def __init__(self, text: str, menu: QMenu, tooltip: str = "", parent=None):
        super().__init__(parent)
        self.setProperty("kit", "button")
        self.setText(text)
        self.setToolTip(tooltip)
        self.setAccessibleName(text)
        self.setMenu(menu)
        self.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def shown_text(self) -> str:
        """The text as drawn (shortened with an ellipsis when long)."""
        metrics = self.fontMetrics()
        return metrics.elidedText(self.text(), Qt.TextElideMode.ElideRight, self.MAX_TEXT)

    def sizeHint(self) -> QSize:
        text = self.fontMetrics().horizontalAdvance(self.shown_text())
        width = 2 * self.PADDING + text + self.GAP + self.CHEVRON
        return QSize(width, max(super().sizeHint().height(), 26))

    def minimumSizeHint(self) -> QSize:
        return self.sizeHint()

    def paintEvent(self, event) -> None:
        option = QStyleOptionToolButton()
        self.initStyleOption(option)
        option.text = ""
        option.features &= ~QStyleOptionToolButton.ToolButtonFeature.HasMenu
        painter = QStylePainter(self)
        painter.drawComplexControl(QStyle.ComplexControl.CC_ToolButton, option)
        tokens = current_tokens()
        color = tokens["fg"] if self.isEnabled() else tokens["faint"]
        text = self.shown_text()
        text_width = self.fontMetrics().horizontalAdvance(text)
        x = (self.width() - (text_width + self.GAP + self.CHEVRON)) // 2
        painter.setPen(color)
        painter.drawText(
            QRect(x, 0, text_width + 1, self.height()), Qt.AlignmentFlag.AlignVCenter, text
        )
        middle = self.height() // 2
        chevron = QRect(x + text_width + self.GAP, middle - 6, self.CHEVRON, self.CHEVRON)
        icons.icon("chevron-down", color).paint(painter, chevron)
        painter.end()


class MyPresets(QObject):
    """The user's presets of an export window, kept in *store*.

    The window (*dialog*) gives its settings as a preset with ``current_user_preset(name)``
    (ValueError if they cannot be kept), shows a preset's settings with
    ``show_user_preset(preset)`` and messages with ``show_note(level, text)``, and calls
    :meth:`update_button` whenever its settings change. Widgets: :attr:`button` (with
    :attr:`menu`) and :attr:`naming_box` (the name row).
    """

    def __init__(self, dialog, store: PresetStore):
        super().__init__(dialog)
        self.dialog = dialog
        self.store = store
        self._presets: list[UserPreset] = []
        self._active: str | None = None
        self._naming: tuple[str, str] | None = None  # ("save", "") or ("rename", old name)
        self.preset_actions: dict[str, object] = {}
        self.menu_actions: dict[str, object] = {}

        self.menu = QMenu(dialog)
        self.menu.setToolTipsVisible(True)
        self.button = MenuButton(
            MY_PRESETS,
            self.menu,
            "Your presets: choose one, save these settings, rename, delete, import or export",
        )
        self.name_edit = QLineEdit()
        self.name_edit.setMaxLength(MAX_NAME)
        self.name_edit.setPlaceholderText("Preset name")
        self.name_edit.setAccessibleName("Preset name")
        self.name_save = SmallButton("Save", "save")
        self.name_cancel = SmallButton("Cancel")
        self.name_note = Note()
        row = QWidget()
        line = QHBoxLayout(row)
        line.setContentsMargins(0, 0, 0, 0)
        line.setSpacing(6)
        line.addWidget(self.name_edit, 1)
        line.addWidget(self.name_save)
        line.addWidget(self.name_cancel)
        self.naming_box = block(row, self.name_note, spacing=5)
        self.naming_box.hide()

        self.menu.aboutToShow.connect(self.fill_menu)
        self.name_edit.textChanged.connect(self._update_naming)
        self.name_edit.returnPressed.connect(self._finish_naming)
        self.name_edit.installEventFilter(self)  # Escape cancels the name, not the window
        self.name_save.clicked.connect(self._finish_naming)
        self.name_cancel.clicked.connect(self.cancel_naming)

    # ------------------------------------------------------------------ the presets
    def presets(self) -> list[UserPreset]:
        """The user's presets, by name."""
        return list(self._presets)

    def preset(self, name: str | None) -> UserPreset | None:
        """The user's preset called *name* (any case), if there is one."""
        if not name:
            return None
        return next((p for p in self._presets if same_name(p.name, name)), None)

    def active(self) -> str | None:
        """The user's preset whose settings the window shows, if any."""
        return self._active

    def load(self) -> None:
        """Read the stored presets (after a restore or reset too); bad ones are skipped (and
        logged)."""
        text = self.store.text
        presets, problems = stored_presets(text) if text else ([], [])
        for problem in problems:
            logger.warning("Skipped a stored figure preset: %s", problem)
        self._presets = sorted(presets, key=lambda p: p.name.casefold())
        self.update_button()

    def _set(self, presets: list[UserPreset]) -> None:
        self._presets = sorted(presets, key=lambda p: p.name.casefold())
        self.store.text = presets_to_json(self._presets) if self._presets else ""

    def save(self, name: str) -> UserPreset:
        """Keep the window's style settings as the preset *name* (replacing one so called)."""
        preset = self.dialog.current_user_preset(name)
        others = [p for p in self._presets if not same_name(p.name, preset.name)]
        replaced = len(others) < len(self._presets)
        self._set([*others, preset])
        self._active = preset.name
        logger.info("Saved the figure preset %r.", preset.name)
        verb = "Replaced" if replaced else "Saved"
        self.dialog.show_note("ok", f"{verb} the preset “{preset.name}”.")
        return preset

    def apply(self, name: str) -> None:
        """Show the settings of the user's preset *name* in the window."""
        preset = self.preset(name)
        if preset is None:
            raise ValueError(f"there is no preset called {name!r}")
        self._active = preset.name
        self.dialog.show_user_preset(preset)

    def rename(self, old: str, new: str) -> UserPreset:
        """Give the user's preset *old* the name *new*."""
        preset = self.preset(old)
        if preset is None:
            raise ValueError(f"there is no preset called {old!r}")
        name = clean_name(new)
        clash = self.preset(name)
        if clash is not None and clash is not preset:
            raise PresetError(f"a preset called “{clash.name}” exists already")
        renamed = dataclasses.replace(preset, name=name)
        self._set([renamed if p is preset else p for p in self._presets])
        if same_name(self._active or "", preset.name):
            self._active = name
        self.dialog.show_note("ok", f"Renamed “{preset.name}” to “{name}”.")
        return renamed

    def delete(self, name: str) -> None:
        """Forget the user's preset *name*."""
        preset = self.preset(name)
        if preset is None:
            raise ValueError(f"there is no preset called {name!r}")
        self._set([p for p in self._presets if p is not preset])
        if same_name(self._active or "", preset.name):
            self._active = None
        logger.info("Deleted the figure preset %r.", preset.name)
        self.dialog.show_note("ok", f"Deleted the preset “{preset.name}”.")

    def import_file(self, path: str | Path | None = None) -> list[UserPreset]:
        """Add the presets of a preset file (those with a name in use replace the old ones);
        a file that cannot be read changes nothing and says why inline."""
        path = path or open_file(self.dialog, "Import figure presets", PRESET_FILTER)
        if not path:
            return []
        path = Path(path)
        try:
            presets = presets_from_json(path.read_bytes())
        except (OSError, ValueError) as exc:
            reason = error_text(exc) if isinstance(exc, OSError) else str(exc)
            logger.warning("Could not import figure presets from %s: %s", path, reason)
            self.dialog.show_note("err", f"Could not import presets from {path.name}: {reason}.")
            return []
        kept = [p for p in self._presets if not any(same_name(p.name, n.name) for n in presets)]
        replaced = len(self._presets) - len(kept)
        self._set([*kept, *presets])
        logger.info("Imported %d figure presets from %s", len(presets), path)
        text = f"Imported {_plural(len(presets), 'preset')} from {path.name}"
        if replaced:
            text += f" ({replaced} replaced)"
        self.dialog.show_note("ok", text + ".")
        return presets

    def export_file(self, path: str | Path | None = None) -> Path | None:
        """Write the user's presets to a preset file (.json); None if nothing was written."""
        if not self._presets:
            return None
        path = path or save_file(self.dialog, "Export figure presets", PRESET_FILTER)
        if not path:
            return None
        path = Path(path)
        if path.suffix.lower() != ".json":
            path = path.with_name(path.name + ".json")
        try:
            path.write_text(presets_to_json(self._presets), encoding="utf-8")
        except OSError as exc:
            logger.warning("Could not export figure presets: %s", error_text(exc))
            self.dialog.show_note("err", f"Could not export the presets: {error_text(exc)}.")
            return None
        logger.info("Exported %d figure presets to %s", len(self._presets), path)
        self.dialog.show_note(
            "ok", f"Exported {_plural(len(self._presets), 'preset')} to {path.name}."
        )
        return path

    # ------------------------------------------------------------------ button and menu
    def update_button(self) -> None:
        """The button names the user's preset the window shows (it stops when one changes)."""
        try:
            current = self.dialog.current_user_preset("current")
        except ValueError:
            current = None
        active = self.preset(self._active)
        if current is None or (active is not None and not active.same_settings(current)):
            active = None
        if active is None and current is not None:
            active = next((p for p in self._presets if p.same_settings(current)), None)
        self._active = active.name if active is not None else None
        self.button.setText(self._active or MY_PRESETS)
        self.button.setAccessibleName(f"My presets: {self._active or 'none chosen'}")
        self.button.updateGeometry()
        self.button.update()

    def _describe(self, preset: UserPreset) -> str:
        journal = PRESETS[preset.journal]
        width = preset.width_mm if preset.width_mm is not None else journal.widths_mm[preset.width]
        bar = f"colour bar {preset.style.colorbar_location}" if preset.colorbar else "no bar"
        return (
            f"{journal.name} · {width:g} × {preset.height_mm:g} mm · {preset.font_pt:g} pt · "
            f"{FORMAT_LABELS[preset.format]} · {bar} · ticks {preset.style.ticks.direction} · "
            f"colour range {RANGE_NAMES[preset.colour_range]}"
        )

    def fill_menu(self) -> None:
        """The user's presets (the one shown ticked), then save, rename, delete, import and
        export (built each time the menu opens)."""
        menu = self.menu
        menu.clear()
        self.preset_actions = {}
        if not self._presets:
            empty = menu.addAction("No presets saved yet")
            empty.setEnabled(False)
        for preset in self._presets:
            action = menu.addAction(f"{preset.name}\t{PRESET_NAMES[preset.journal]}")
            action.setCheckable(True)
            action.setChecked(same_name(preset.name, self._active or ""))
            action.setToolTip(self._describe(preset))
            action.triggered.connect(lambda _c=False, n=preset.name: self.apply(n))
            self.preset_actions[preset.name] = action
        menu.addSeparator()
        active = self._active
        choose_first = "Choose one of your presets first"
        save = menu.addAction("Save as preset…")
        save.setToolTip("Keep these settings under a name")
        icons.set_icon(save, "save", "muted")
        save.triggered.connect(lambda: self.start_naming("save"))
        rename = menu.addAction(f"Rename “{active}”…" if active else "Rename…")
        rename.setEnabled(active is not None)
        rename.setToolTip("" if active else choose_first)
        rename.triggered.connect(lambda: self.start_naming("rename"))
        delete = menu.addAction(f"Delete “{active}”" if active else "Delete")
        delete.setEnabled(active is not None)
        delete.setToolTip("" if active else choose_first)
        icons.set_icon(delete, "trash-2", "muted")
        delete.triggered.connect(lambda: active and self.delete(active))
        menu.addSeparator()
        load = menu.addAction("Import presets…")
        load.setToolTip("Add the presets of a file (.json)")
        icons.set_icon(load, "upload", "muted")
        load.triggered.connect(lambda: self.import_file())
        write = menu.addAction("Export presets…")
        write.setToolTip("Save your presets to a file (.json)")
        write.setEnabled(bool(self._presets))
        icons.set_icon(write, "download", "muted")
        write.triggered.connect(lambda: self.export_file())
        self.menu_actions = {
            "save": save,
            "rename": rename,
            "delete": delete,
            "import": load,
            "export": write,
        }

    # ------------------------------------------------------------------ naming
    def start_naming(self, mode: str) -> None:
        """Ask for a name inline: to save the settings ("save") or rename ("rename") the
        preset shown."""
        old = self._active or ""
        if mode == "rename" and not old:
            return
        self._naming = (mode, old if mode == "rename" else "")
        self.name_save.setText("Rename" if mode == "rename" else "Save")
        self.name_edit.setText(old)
        self.naming_box.show()
        self.name_edit.setFocus(Qt.FocusReason.OtherFocusReason)
        self.name_edit.selectAll()
        self._update_naming()

    def naming(self) -> str | None:
        """ "save" or "rename" while a name is asked for, else None."""
        return None if self._naming is None else self._naming[0]

    def cancel_naming(self) -> None:
        self._naming = None
        self.naming_box.hide()

    def _update_naming(self) -> None:
        if self._naming is None:
            return
        mode, old = self._naming
        try:
            name = clean_name(self.name_edit.text())
        except PresetError:
            name = ""
        clash = self.preset(name)
        level, text = "muted", PRESET_KEEPS if mode == "save" else f"A new name for “{old}”."
        if clash is not None and mode == "save":
            level, text = "warn", f"Replaces your preset “{clash.name}”."
        elif clash is not None and not same_name(clash.name, old):
            level, text = "err", f"A preset “{clash.name}” exists already."
        self.name_note.set_text(text, level)
        self.name_save.setEnabled(bool(name) and level != "err")

    def _finish_naming(self) -> None:
        if self._naming is None or not self.name_save.isEnabled():
            return
        mode, old = self._naming
        try:
            if mode == "rename":
                self.rename(old, self.name_edit.text())
            else:
                self.save(self.name_edit.text())
        except ValueError as exc:
            self.name_note.set_text(sentence(str(exc)), "err")
            return
        self.cancel_naming()

    def eventFilter(self, watched, event) -> bool:
        if (
            watched is self.name_edit
            and event.type() == QEvent.Type.KeyPress
            and event.key() == Qt.Key.Key_Escape
        ):
            self.cancel_naming()
            return True
        return super().eventFilter(watched, event)
