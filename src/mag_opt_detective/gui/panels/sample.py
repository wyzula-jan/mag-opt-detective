"""Sample panel: the sample's file lists and where the field values come from."""

from __future__ import annotations

from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import QComboBox, QGridLayout, QLabel, QVBoxLayout, QWidget

from mag_opt_detective.gui.controller import FieldRange, SweepFiles
from mag_opt_detective.gui.measurement_tab import MeasurementTab
from mag_opt_detective.gui.widgets import parse_float

FIELD_FROM_NAMES = "Field from file names"
FIELD_CUSTOM = "Custom field range"


class SamplePanel(QWidget):
    """Field source and the sample's zero-field and in-field files."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.field_source = QComboBox()
        self.field_source.addItems([FIELD_FROM_NAMES, FIELD_CUSTOM])
        self.field_source.setToolTip(
            "File type (OPUS or text) is detected automatically.\n"
            "Field from names expects e.g. ..._a01p250T.txt (= 1.25 T)."
        )
        top = QGridLayout()
        top.setContentsMargins(9, 9, 9, 0)
        top.addWidget(QLabel("Field:"), 0, 0)
        top.addWidget(self.field_source, 0, 1)
        top.setColumnStretch(1, 1)
        self.measurement = MeasurementTab()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(top)
        layout.addWidget(self.measurement, stretch=1)

    def custom_field(self) -> bool:
        return self.field_source.currentText() == FIELD_CUSTOM


def field_range_of(tab: MeasurementTab) -> FieldRange:
    box = tab.field_range
    return FieldRange(*(parse_float(edit.text()) for edit in (box.start, box.step, box.end)))


def connect_measurement(window, tab: MeasurementTab, which: str) -> None:
    """Keep the file lists and the custom field range of *tab* in the controller.

    *which* is "sample" or "reference". File lists follow the controller both ways, so
    files set elsewhere (e.g. the Open sweep button) show here too.
    """
    c = window.controller
    files_key, field_key = f"{which}_files", f"{which}_field"
    syncing = False

    def push_files() -> None:
        if not syncing:
            files = SweepFiles(tuple(tab.zero_paths()), tuple(tab.field_paths()))
            c.set_processing(**{files_key: files})

    def pull() -> None:
        nonlocal syncing
        files: SweepFiles = getattr(c.processing, files_key)
        syncing = True
        try:
            if tuple(tab.zero_paths()) != files.zero:
                tab.zero_list.set_paths(files.zero)
            if tuple(tab.field_paths()) != files.field:
                tab.field_list.set_paths(files.field)
        finally:
            syncing = False
        tab.set_custom_field_enabled(c.processing.custom_field)

    def push_field() -> None:
        c.set_processing(**{field_key: field_range_of(tab)})

    tab.zero_list.pathsChanged.connect(push_files)
    tab.field_list.pathsChanged.connect(push_files)
    for edit in (tab.field_range.start, tab.field_range.step, tab.field_range.end):
        edit.textChanged.connect(push_field)
    c.processingChanged.connect(pull)
    push_field()
    pull()


def _summary(files: SweepFiles) -> str:
    if not files.field and not files.zero:
        return "No files yet: use Open sweep or drop files on the lists."
    return f"{len(files.field)} in-field and {len(files.zero)} zero-field file(s)"


def open_sweep(window) -> None:
    """Show the Sample panel and ask for the in-field files."""
    window.show_panel("sample")
    window.panels["sample"].measurement.load_field_dialog()


def install(window) -> None:
    c = window.controller
    panel = SamplePanel()
    page = window.add_panel("sample", "Sample", "activity", "Sample files", panel)
    tab = panel.measurement
    connect_measurement(window, tab, "sample")

    def on_field_source() -> None:
        c.set_processing(custom_field=panel.custom_field())

    panel.field_source.currentIndexChanged.connect(on_field_source)
    on_field_source()
    c.processingChanged.connect(lambda: page.set_subtitle(_summary(c.processing.sample_files)))
    page.set_subtitle(_summary(c.processing.sample_files))

    window.toolbar.open_button.clicked.connect(lambda: open_sweep(window))
    for text, slot, shortcut in (
        ("Open Sweep (Sample Field)…", lambda: open_sweep(window), "Ctrl+L"),
        ("Load Sample Zero Field…", tab.load_zero_dialog, "Ctrl+Shift+L"),
    ):
        action = QAction(text, window)
        action.setShortcut(QKeySequence(shortcut))
        action.triggered.connect(lambda _checked=False, s=slot: s())
        window.add_file_action(action)

    p = window.persistence
    if p is not None:
        p.bind("sample/field_source", panel.field_source)
        for part in ("start", "step", "end"):
            p.bind(f"sample/field_{part}", getattr(tab.field_range, part))
