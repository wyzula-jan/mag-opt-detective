"""Sample panel: where the field values come from and the sample's measurement files.

The panel follows ``controller.processing`` both ways: edits are pushed to the controller,
and values set elsewhere (e.g. files opened with the toolbar's Open sweep) are shown here.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import QWidget

from mag_opt_detective.core.opus import is_opus_file
from mag_opt_detective.core.readers import parse_field
from mag_opt_detective.gui.controller import SweepFiles
from mag_opt_detective.gui.kit import SegmentedControl
from mag_opt_detective.gui.panels.common import (
    WidthWatcher,
    block,
    content_width,
    fit_segments,
    hint,
    panel_layout,
    section_label,
)
from mag_opt_detective.gui.panels.files import FieldRangeInputs, SweepFilesBox, fmt_field

NAMES, CUSTOM = "names", "custom"
NAMES_HINT = "Read from names like …_a01p250T.txt (1.25 T)."


LABELS = {NAMES: ("From file names", "From names"), CUSTOM: ("Custom range", "Custom")}


def field_source_control() -> SegmentedControl:
    control = SegmentedControl(size="sm", expand=True)
    control.add_option(NAMES, "From file names", "Read the field from names like …_a01p250T.txt")
    control.add_option(CUSTOM, "Custom range", "Give the fields as start, step and end")
    control.setAccessibleName("Field values")
    return control


class SamplePanel(QWidget):
    """Field values (from the names or a custom range) and the sample's files."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.field_source = field_source_control()
        self.names_hint = hint(NAMES_HINT)
        self.field_range = FieldRangeInputs()
        self.measurement = SweepFilesBox("sample", drop_zone=True)
        layout = panel_layout(self)
        source = block(
            section_label("Field values"), self.field_source, self.names_hint, self.field_range
        )
        layout.addWidget(source)
        layout.addWidget(self.measurement)
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


_opus_cache: dict[str, bool] = {}


def _file_kind(path: str) -> str:
    if path not in _opus_cache:
        _opus_cache[path] = is_opus_file(path)
    return "OPUS files" if _opus_cache[path] else "text export"


def summary(files: SweepFiles, custom: bool, field_values=None) -> str:
    """Header line of the panel: ``64 spectra · 0.25 – 16 T · text export``."""
    if not files.field:
        if files.zero:
            return f"{len(files.zero)} zero-field file(s), no in-field files yet"
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
    WidthWatcher(page, lambda _w: fit_segments(panel.field_source, LABELS, content_width(page)))

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
        page.set_subtitle(summary(p.sample_files, p.custom_field, values))

    panel.field_source.valueChanged.connect(push_source)
    push_source()
    c.processingChanged.connect(pull)
    pull()

    window.toolbar.open_button.clicked.connect(lambda: open_sweep(window))
    for text, slot, shortcut in (
        ("Open Sweep (Sample)…", window.panels["sample"].measurement.open_sweep_dialog, "Ctrl+L"),
        ("Load Sample Zero Field…", box.load_zero_dialog, "Ctrl+Shift+L"),
    ):
        action = QAction(text, window)
        action.setShortcut(QKeySequence(shortcut))

        def run(_checked=False, s=slot) -> None:
            window.show_panel("sample")
            s()

        action.triggered.connect(run)
        window.add_file_action(action)

    p = window.persistence
    if p is not None:
        p.bind("sample/field_source", panel.field_source)
        for part, edit in zip(("start", "step", "end"), panel.field_range.edits(), strict=True):
            p.bind(f"sample/field_{part}", edit)
