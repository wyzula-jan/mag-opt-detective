"""Sample panel: where the field values come from and the sample's measurement files.

The panel follows ``controller.processing`` both ways: edits are pushed to the controller,
and values set elsewhere (e.g. files opened with the toolbar's Open sweep) are shown here.
The rail button shows the state of the sample's files (:func:`show_data_state`).

Its Watch folder block watches the folder of the sweep, or one that may still be empty, for
new spectra (:mod:`gui.watch`); the status bar shows the watching state with a stop button.
"""

from __future__ import annotations

import html
from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import QEvent, QObject, QSize, Qt, QTimer
from PySide6.QtGui import QAction, QKeySequence, QPainter
from PySide6.QtWidgets import QFileDialog, QHBoxLayout, QLabel, QToolButton, QVBoxLayout, QWidget

from mag_opt_detective.core.opus import is_opus_file
from mag_opt_detective.core.readers import parse_field
from mag_opt_detective.gui import icons
from mag_opt_detective.gui.controller import (
    DATA_CHANGED,
    DATA_CURRENT,
    DATA_EMPTY,
    AppController,
    SweepFiles,
    panel_error,
    user_action,
)
from mag_opt_detective.gui.kit import SegmentedControl
from mag_opt_detective.gui.panels.common import (
    LinkButton,
    Note,
    SwitchRow,
    WidthWatcher,
    block,
    content_width,
    fit_segments,
    hint,
    panel_layout,
    scaled_font,
    section_label,
)
from mag_opt_detective.gui.panels.files import (
    FieldRangeInputs,
    SweepFilesBox,
    breakable,
    count_text,
    fmt_field,
)
from mag_opt_detective.gui.theme import current_tokens
from mag_opt_detective.gui.watch import OK, PROBLEM, FolderWatcher, WatchStatus, sweep_folder
from mag_opt_detective.gui.widgets import last_dir, set_last_dir

QWIDGETSIZE_MAX = (1 << 24) - 1  # Qt's largest widget size

NAMES, CUSTOM = "names", "custom"
NAMES_HINT = "Read from names like …_a01p250T.txt (1.25 T)."


LABELS = {NAMES: ("From file names", "From names"), CUSTOM: ("Custom range", "Custom")}
# the state of a sweep's files on its rail button (AppController.data_state)
STATE_TEXTS = {
    DATA_EMPTY: "no files",
    DATA_CHANGED: "changed since last Process",
    DATA_CURRENT: "processed",
}


def field_source_control() -> SegmentedControl:
    control = SegmentedControl(size="sm", expand=True)
    control.add_option(NAMES, "From file names", "Read the field from names like …_a01p250T.txt")
    control.add_option(CUSTOM, "Custom range", "Give the fields as start, step and end")
    control.setAccessibleName("Field values")
    return control


# ---------------------------------------------------------------------- watching
NO_WATCH_FOLDER = "Load a sweep folder first, or watch one that may still be empty."
WATCH_TIP = (
    "Turning it on adds every complete spectrum in the folder that is not listed yet, then "
    "each new one as it appears. The reference sweep is not watched."
)


def watch_text(status: WatchStatus) -> tuple[str, str]:
    """``(text, Note level)`` of the watching state, e.g.
    ``Watching · 23 files · last update 12:03:10``."""
    files = count_text(status.files)
    if status.note:
        return f"{status.note} · {files}", status.level
    text = f"Watching · {files}"
    if status.last is not None:
        text += f" · last update {status.last:%H:%M:%S}"
    return text, OK


class WatchBox(QWidget):
    """Watch the sweep's folder for new spectra: a switch for the folder the sample's files
    come from, *Watch a folder…* for one that may still be empty, and the state."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.choose_button = LinkButton(
            "Watch a folder…",
            "Watch a folder that may still be empty: the spectra that appear in it make the "
            "sample's sweep",
        )
        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.setSpacing(6)
        head.addWidget(section_label("Watch folder"))
        head.addStretch(1)
        head.addWidget(self.choose_button)
        self.row = SwitchRow("Watch for new files", NO_WATCH_FOLDER)
        self.switch = self.row.switch
        self.state = Note()
        self.state.setVisible(False)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(7)
        layout.addLayout(head)
        layout.addWidget(self.row)
        layout.addWidget(self.state)

    def show_state(self, folder: str | None, status: WatchStatus | None) -> None:
        """Show the *folder* that is (or would be) watched and the watching *status*
        (None: not watching)."""
        if folder is None:
            text, tip = NO_WATCH_FOLDER, WATCH_TIP
        else:
            name = html.escape(breakable(Path(folder).name or folder))
            text = f"Adds and processes the spectra that appear in <b>{name}</b>."
            tip = f"{folder}\n{WATCH_TIP}"
        self.row.description_label.setText(text)
        self.row.setToolTip(tip)
        self.switch.setEnabled(folder is not None)
        self.switch.setChecked(status is not None)
        self.state.setVisible(status is not None)
        if status is not None:
            self.state.set_text(*watch_text(status))


class _ChipText(QLabel):
    """The chip's text, a head ("Watching <folder>") and the rest (" · 23 files · …"),
    painted in the theme token *token* (read at paint time). Given less width than it needs,
    the head is elided first, then left out, then the rest is elided."""

    token = "muted"
    SEPARATOR = " · "

    def __init__(self, parent=None):
        super().__init__(parent)
        self._head, self._rest = "", ""

    def set_parts(self, head: str, rest: str) -> None:
        self._head, self._rest = head, rest
        self.setText(f"{head}{self.SEPARATOR}{rest}" if rest else head)
        self.update()

    def minimumSizeHint(self) -> QSize:
        return QSize(0, super().minimumSizeHint().height())

    def shown(self) -> str:
        """The text as it fits the label's width."""
        metrics, width = self.fontMetrics(), self.contentsRect().width()
        if metrics.horizontalAdvance(self.text()) <= width or not self._rest:
            return metrics.elidedText(self.text(), Qt.TextElideMode.ElideRight, width)
        rest = self.SEPARATOR + self._rest
        room = width - metrics.horizontalAdvance(rest)
        head = metrics.elidedText(self._head, Qt.TextElideMode.ElideRight, room)
        if room > 0 and head not in ("", "…") and len(head) > len("Watching …"):
            return head + rest
        return metrics.elidedText(self._rest, Qt.TextElideMode.ElideRight, width)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setPen(current_tokens()[self.token])
        painter.setFont(self.font())
        flags = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        painter.drawText(self.contentsRect(), int(flags), self.shown())
        painter.end()


class _YieldRoom(QObject):
    """Lets *chip* have only the room of its status-bar row that the other widgets leave at
    their preferred widths, so it shrinks before them (e.g. before the baseline chip)."""

    def __init__(self, bar: QWidget, chip: QWidget):
        super().__init__(chip)
        self._bar, self._chip = bar, chip
        self._pending = False
        bar.installEventFilter(self)

    def eventFilter(self, watched, event) -> bool:
        if event.type() in (QEvent.Type.Resize, QEvent.Type.LayoutRequest) and not self._pending:
            self._pending = True  # once the bar is laid out
            QTimer.singleShot(0, self, self.fit)
        return False

    def _row(self):
        layouts = [self._bar.layout()]
        while layouts:
            layout = layouts.pop()
            if layout.indexOf(self._chip) >= 0:
                return layout
            layouts.extend(
                item.layout() for i in range(layout.count()) if (item := layout.itemAt(i)).layout()
            )
        return None

    def fit(self) -> None:
        self._pending = False
        row = self._row()
        if row is None:
            return
        items = [row.itemAt(i) for i in range(row.count())]
        others = sum(item.sizeHint().width() for item in items if item.widget() is not self._chip)
        shown = [item for item in items if not item.isEmpty()]  # (spacing between these)
        width = row.geometry().width()
        room = width - others - row.spacing() * (len(shown) - 1) if width > 0 else QWIDGETSIZE_MAX
        room = max(0, min(room, QWIDGETSIZE_MAX))  # (no limit before the first layout)
        if room != self._chip.maximumWidth():
            self._chip.setMaximumWidth(room)


class WatchChip(QWidget):
    """The watching state in the status bar (hidden while not watching), with a button that
    stops it: ``Watching Demo_sweep · 23 files · 12:03:10``; a problem shows in the warning
    colour. Its width may be anything down to zero: the text is elided first."""

    MAX_NAME = 28  # characters of the folder name shown

    def __init__(self, parent=None):
        super().__init__(parent)
        self.icon_label = QLabel()
        self.icon_label.setFixedSize(16, 16)
        self.icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.text_label = _ChipText()
        self.text_label.setFont(scaled_font(self.text_label, 0.94))
        self.stop_button = QToolButton()
        self.stop_button.setProperty("kit", "tool")
        self.stop_button.setFixedSize(20, 20)
        self.stop_button.setIconSize(QSize(12, 12))
        icons.set_icon(self.stop_button, "x", "muted")
        self.stop_button.setToolTip("Stop watching (the files and the map stay)")
        self.stop_button.setAccessibleName("Stop watching")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 6, 0)
        layout.setSpacing(5)
        layout.setSizeConstraint(QHBoxLayout.SizeConstraint.SetNoConstraint)  # (it shrinks)
        layout.addWidget(self.icon_label)
        layout.addWidget(self.text_label, 1)
        layout.addWidget(self.stop_button)
        self._level = OK
        self.setVisible(False)

    def minimumSizeHint(self) -> QSize:
        return QSize(0, super().minimumSizeHint().height())

    def text(self) -> str:
        return self.text_label.text()

    def show_status(self, status: WatchStatus | None) -> None:
        self.setVisible(status is not None)
        if status is None:
            return
        name = Path(status.folder).name or status.folder
        if len(name) > self.MAX_NAME:
            name = name[: self.MAX_NAME - 1] + "…"
        parts = [count_text(status.files)]
        if status.brief:  # the whole note is in the tooltip and the panel
            parts.append(status.brief)
        elif status.last is not None:
            parts.append(f"{status.last:%H:%M:%S}")
        self.text_label.set_parts(f"Watching {name}", " · ".join(parts))
        self.setToolTip(f"{status.folder}\n{watch_text(status)[0]}")
        self._level = status.level
        self.text_label.token = "warn" if status.level == PROBLEM else "muted"
        self.text_label.update()
        self._update_icon()

    def level(self) -> str:
        return self._level

    def shown_text(self) -> str:
        """The text as it fits the chip's width (the folder name is elided first)."""
        return self.text_label.shown()

    def _update_icon(self) -> None:
        color = "warn" if self._level == PROBLEM else "muted"
        ratio = self.devicePixelRatioF()
        self.icon_label.setPixmap(icons.pixmap("eye", 14, color, scale=ratio))

    def changeEvent(self, event) -> None:
        super().changeEvent(event)
        if event.type() == QEvent.Type.PaletteChange:
            self._update_icon()


class _StopOnClose(QObject):
    """Stops *watcher* when *window* closes."""

    def __init__(self, window: QWidget, watcher: FolderWatcher):
        super().__init__(window)
        self._watcher = watcher
        window.installEventFilter(self)

    def eventFilter(self, watched, event) -> bool:
        if event.type() == QEvent.Type.Close:
            self._watcher.stop()
        return False


class SamplePanel(QWidget):
    """Field values (from the names or a custom range), the sample's files and watching
    their folder."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.field_source = field_source_control()
        self.names_hint = hint(NAMES_HINT)
        self.field_range = FieldRangeInputs()
        self.measurement = SweepFilesBox("sample", drop_zone=True)
        self.watch = WatchBox()
        layout = panel_layout(self)
        source = block(
            section_label("Field values"), self.field_source, self.names_hint, self.field_range
        )
        layout.addWidget(source)
        layout.addWidget(self.measurement)
        layout.addWidget(self.watch)
        layout.addStretch(1)
        self.field_source.valueChanged.connect(self._sync)
        self.measurement.changed.connect(self._sync)
        self._sync()

    def custom_field(self) -> bool:
        return self.field_source.value() == CUSTOM

    def set_custom_field(self, custom: bool) -> None:
        self.field_source.set_value(CUSTOM if custom else NAMES)

    def _sync(self) -> None:
        custom = self.custom_field()
        self.names_hint.setVisible(not custom)
        self.field_range.setVisible(custom)
        self.field_range.set_file_count(self.measurement.field_list.count())
        self.measurement.set_gap_check(not custom)


# ---------------------------------------------------------------------- wiring
def connect_files(window, box: SweepFilesBox, field_range: FieldRangeInputs, which: str) -> None:
    """Keep the files and the custom field range of *box* / *field_range* in the controller.

    *which* is "sample" or "reference". Both follow the controller back, so values set
    elsewhere show here; an edit pushes only its own value.
    """
    c = window.controller
    files_key, field_key = f"{which}_files", f"{which}_field"
    syncing = False

    def push_files() -> None:
        if not syncing:
            files = SweepFiles(tuple(box.zero_paths()), tuple(box.field_paths()))
            c.set_processing(**{files_key: files})

    def push_field() -> None:
        if not syncing:
            c.set_processing(**{field_key: field_range.field_range()})

    def pull() -> None:
        nonlocal syncing
        files: SweepFiles = getattr(c.processing, files_key)
        syncing = True
        try:
            box.set_files(files.zero, files.field)
            field_range.show_range(getattr(c.processing, field_key))
        finally:
            syncing = False

    box.changed.connect(push_files)
    field_range.changed.connect(push_field)
    c.processingChanged.connect(pull)
    push_field()
    pull()


def state_text(controller: AppController, state: str) -> str:
    """How a rail button's tooltip words a data *state*."""
    if state == DATA_CHANGED and controller.processed_at is None:
        return "not processed yet"
    return STATE_TEXTS[state]


def show_data_state(window, part: str, text: Callable[[str], str]) -> Callable[[], None]:
    """Show the controller's data state of *part* ("sample" or "reference") on its rail
    button, worded by *text(state)*, whenever it changes; returns the function that shows it
    (for other changes of the wording)."""
    c = window.controller
    button = window.rail_button(part)

    def show(*_args) -> None:
        state = c.data_state(part)
        button.set_data_state(state, text(state))

    c.dataStateChanged.connect(show)
    show()
    return show


_opus_cache: dict[str, bool] = {}


def _file_kind(path: str) -> str:
    if path not in _opus_cache:
        _opus_cache[path] = is_opus_file(path)
    return "OPUS files" if _opus_cache[path] else "text export"


def summary(files: SweepFiles, custom: bool, field_values=None, watched: str | None = None) -> str:
    """Header line of the panel: ``64 spectra · 0.25 – 16 T · text export``; with the folder
    *watched* and no files yet ``Watching <folder> · waiting for files``."""
    if not files.field:
        if files.zero:
            return f"{len(files.zero)} zero-field file(s), no in-field files yet"
        if watched is not None:
            return f"Watching {Path(watched).name or watched} · waiting for files"
        return "No files yet. Drop a sweep folder below or use Open sweep."
    parts = [f"{len(files.field)} spectr{'um' if len(files.field) == 1 else 'a'}"]
    fields = list(field_values) if custom and field_values is not None else None
    if fields is None:
        parsed = [parse_field(p) for p in files.field]
        fields = [b for b in parsed if b is not None] if None not in parsed else None
    if fields:
        lo, hi = min(fields), max(fields)
        parts.append(f"{fmt_field(lo)} T" if lo == hi else f"{fmt_field(lo)} – {fmt_field(hi)} T")
    else:
        parts.append("no field in the names")
    if Path(files.field[0]).is_file():
        parts.append(_file_kind(files.field[0]))
    return " · ".join(parts)


@user_action("Watch the folder")
def watch_folder(window, folder: str | None = None, replace: bool = False) -> None:
    """Watch *folder*, by default the one the sample's files come from (see
    :meth:`FolderWatcher.start` for *replace*)."""
    if folder is None:
        folder = sweep_folder(window.controller.processing.sample_files)
        if folder is None:
            raise panel_error(NO_WATCH_FOLDER[:1].lower() + NO_WATCH_FOLDER[1:-1], "sample")
    window.folder_watch.start(folder, replace=replace)


def choose_watch_folder(window) -> None:
    """Ask for a folder (it may be empty) and watch it: its spectra replace the sample's."""
    window.show_panel("sample")
    folder = QFileDialog.getExistingDirectory(
        window, "Watch a folder for the spectra of a sweep", last_dir()
    )
    if folder:
        set_last_dir(folder)
        watch_folder(window, folder, replace=True)


def open_sweep(window) -> None:
    """Show the Sample panel and ask for the files of a sweep (zero-field names go to the
    zero-field list)."""
    window.show_panel("sample")
    window.panels["sample"].measurement.open_sweep_dialog()


def install(window) -> None:
    c = window.controller
    panel = SamplePanel()
    page = window.add_panel("sample", "Sample", "activity", "Sample files", panel)
    box = panel.measurement
    connect_files(window, box, panel.field_range, "sample")
    show_data_state(window, "sample", lambda state: state_text(c, state))
    WidthWatcher(page, lambda _w: fit_segments(panel.field_source, LABELS, content_width(page)))

    watcher = install_watch(window, panel)

    def push_source() -> None:
        c.set_processing(custom_field=panel.custom_field())

    def pull() -> None:
        p = c.processing
        if panel.custom_field() != p.custom_field:
            panel.set_custom_field(p.custom_field)
        try:
            values = p.sample_field.values() if p.custom_field else None
        except ValueError:
            values = None
        page.set_subtitle(summary(p.sample_files, p.custom_field, values, watcher.folder()))

    panel.field_source.valueChanged.connect(push_source)
    push_source()
    c.processingChanged.connect(pull)
    watcher.changed.connect(pull)
    pull()

    window.toolbar.open_button.clicked.connect(lambda: open_sweep(window))
    for text, slot, shortcut in (
        ("Open sample sweep…", box.open_sweep_dialog, "Ctrl+L"),
        ("Load sample zero field…", box.load_zero_dialog, "Ctrl+Shift+L"),
    ):
        action = QAction(text, window)
        action.setShortcut(QKeySequence(shortcut))

        def run(_checked=False, s=slot) -> None:
            window.show_panel("sample")
            s()

        action.triggered.connect(run)
        window.add_file_action(action)
    add_watch_actions(window, panel.watch, watcher)

    # watching is never remembered: it does not resume after a restart
    p = window.persistence
    if p is not None:
        p.bind("sample/field_source", panel.field_source)
        for part, edit in zip(("start", "step", "end"), panel.field_range.edits(), strict=True):
            p.bind(f"sample/field_{part}", edit)


def install_watch(window, panel: SamplePanel) -> FolderWatcher:
    """The folder watcher (``window.folder_watch``, stopped when the window closes), its block
    in the panel and its chip in the status bar; files left out and updates that cannot be
    processed go to the info bar."""
    c = window.controller
    watcher = FolderWatcher(c, window)
    window.folder_watch = watcher
    _StopOnClose(window, watcher)
    box = panel.watch
    chip = WatchChip()
    window.add_status_chip(chip)  # beside the state and the baseline chip
    _YieldRoom(window.statusBar(), chip)
    syncing = False

    def show() -> None:
        nonlocal syncing
        status = watcher.status()
        folder = status.folder if status is not None else sweep_folder(c.processing.sample_files)
        syncing = True
        try:
            box.show_state(folder, status)
        finally:
            syncing = False
        chip.show_status(status)

    def on_switch(on: bool) -> None:
        if syncing:
            return
        if on:
            watch_folder(window)
        else:
            watcher.stop()
        show()  # also when watching could not start

    def on_skipped(title: str, text: str) -> None:
        window.infobar.show_message("warning", title, text)

    def on_failed(exc: BaseException) -> None:
        window.report_error("Process new files", str(exc), panel=getattr(exc, "panel", None))

    box.switch.toggled.connect(on_switch)
    box.choose_button.clicked.connect(lambda: choose_watch_folder(window))
    chip.stop_button.clicked.connect(watcher.stop)
    watcher.changed.connect(show)
    watcher.skipped.connect(on_skipped)
    watcher.failed.connect(on_failed)
    c.processingChanged.connect(show)
    show()
    return watcher


def add_watch_actions(window, box: WatchBox, watcher: FolderWatcher) -> None:
    """File menu: *Watch a folder…*, and *Watch for new files* (``commands["watch_folder"]``),
    checkable like the panel's switch."""
    choose = QAction("Watch a folder…", window)
    choose.triggered.connect(lambda _checked=False: choose_watch_folder(window))
    window.add_file_action(choose)
    toggle = QAction("Watch for new files", window)
    toggle.setCheckable(True)
    toggle.triggered.connect(lambda checked: box.switch.setChecked(checked))
    window.add_file_action(toggle)
    window.commands["watch_folder"] = toggle

    def show(*_args) -> None:
        toggle.setChecked(watcher.watching())
        toggle.setEnabled(box.switch.isEnabled())

    for signal in (watcher.changed, window.controller.processingChanged, box.switch.toggled):
        signal.connect(show)  # the switch: also when watching could not start
    show()
