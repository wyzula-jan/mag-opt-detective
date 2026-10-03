"""Reference panel: the reference mode, the reference sweep's files and the smoothing.

The sweep's files show only for a separate sweep; the custom field range only when the
Sample panel uses one too. Smoothing is disabled without a reference.
"""

from __future__ import annotations

from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import QGridLayout, QWidget

from mag_opt_detective.core.pipeline import ReferenceMode
from mag_opt_detective.gui.kit import SegmentedControl
from mag_opt_detective.gui.panels.common import (
    Note,
    SpinBox,
    SwitchRow,
    WidthWatcher,
    block,
    content_width,
    fit_segments,
    hint,
    labelled,
    panel_layout,
    section_label,
)
from mag_opt_detective.gui.panels.files import FieldRangeInputs, SweepFilesBox
from mag_opt_detective.gui.panels.sample import connect_files

MODES = (
    (ReferenceMode.NONE, "None", "No reference"),
    (ReferenceMode.SEPARATE, "Separate sweep", "Divide by a second, separately measured sweep"),
    (ReferenceMode.SELF, "The sample", "Divide the sweep by itself, smoothed"),
)
SHORT_LABELS = {ReferenceMode.SEPARATE: "Separate", ReferenceMode.SELF: "Self"}
HINTS = {
    ReferenceMode.NONE: "The sample ratio is used as measured.",
    ReferenceMode.SEPARATE: (
        "Divides by a second sweep measured the same way, for example the bare substrate."
    ),
    ReferenceMode.SELF: "Divides the sweep by itself, smoothed, to remove slow spectral features.",
}
SUBTITLES = {
    ReferenceMode.NONE: "Corrects the sample ratio with a second sweep.",
    ReferenceMode.SEPARATE: "Corrects the sample ratio with a second sweep.",
    ReferenceMode.SELF: "Corrects the sample with its own smoothed sweep.",
}


class ReferencePanel(QWidget):
    """Reference mode, the separate sweep's files and the Savitzky–Golay smoothing."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.mode = SegmentedControl(size="sm", expand=True)
        for mode, text, tooltip in MODES:
            self.mode.add_option(mode.value, text, tooltip)
        self.mode.setAccessibleName("Reference")
        self.mode_hint = hint()

        self.field_range = FieldRangeInputs()
        self.field_range_hint = hint("Custom field range, as for the sample.")
        self.field_block = block(self.field_range_hint, self.field_range)
        self.measurement = SweepFilesBox("reference", drop_zone=False)
        self.zero_list = self.measurement.zero_list
        self.field_list = self.measurement.field_list
        self.files_block = block(self.field_block, self.measurement, spacing=16)

        self.smooth_row = SwitchRow("Smooth the reference", "Savitzky–Golay, before dividing")
        self.smooth = self.smooth_row.switch
        self.sg_window = SpinBox(3, 9999, 11, step=2)
        self.sg_window.setAccessibleName("Smoothing window (points)")
        self.sg_poly = SpinBox(0, 10, 2)
        self.sg_poly.setAccessibleName("Smoothing polynomial order")
        spins = QWidget()
        grid = QGridLayout(spins)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(6)
        grid.addWidget(labelled("Window (points)", self.sg_window), 0, 0)
        grid.addWidget(labelled("Polynomial order", self.sg_poly), 0, 1)
        self.sg_note = Note()
        self.smooth_block = block(self.smooth_row, spins, self.sg_note)

        layout = panel_layout(self)
        layout.addWidget(block(section_label("Reference"), self.mode, self.mode_hint))
        layout.addWidget(self.files_block)
        layout.addWidget(self.smooth_block)
        layout.addStretch(1)

        self._custom_field = False
        self.mode.valueChanged.connect(self._sync)
        self.smooth.toggled.connect(self._sync)
        self.sg_window.valueChanged.connect(self._sync)
        self.sg_poly.valueChanged.connect(self._sync)
        self.measurement.changed.connect(self._sync)
        self._sync()

    def fit_width(self, width: int) -> None:
        """Shorten the mode labels when the panel is narrow."""
        labels = {m.value: (text, SHORT_LABELS.get(m, text)) for m, text, _tip in MODES}
        fit_segments(self.mode, labels, width)

    def reference_mode(self) -> ReferenceMode:
        return ReferenceMode(self.mode.value())

    def set_reference_mode(self, mode: ReferenceMode) -> None:
        self.mode.set_value(ReferenceMode(mode).value)

    def set_custom_field(self, custom: bool) -> None:
        """Show the custom field range (the Sample panel uses one)."""
        self._custom_field = custom
        self._sync()

    def _sync(self) -> None:
        mode = self.reference_mode()
        self.mode_hint.setText(HINTS[mode])
        self.files_block.setVisible(mode is ReferenceMode.SEPARATE)
        self.field_block.setVisible(self._custom_field)
        self.field_range.set_file_count(self.field_list.count())
        self.measurement.set_gap_check(not self._custom_field)
        used = mode is not ReferenceMode.NONE
        self.smooth_block.setEnabled(used)
        self.smooth_row.setToolTip("" if used else "Choose a reference to smooth it")
        for spin in (self.sg_window, self.sg_poly):
            spin.setEnabled(used and self.smooth.isChecked())
        window, poly = self.sg_window.value(), self.sg_poly.value()
        bad = self.smooth.isChecked() and (window % 2 == 0 or window <= poly)
        self.sg_note.set_text(
            "The window must be odd and larger than the polynomial order." if bad else "", "warn"
        )
        self.sg_note.setVisible(bad)


def install(window) -> None:
    c = window.controller
    panel = ReferencePanel()
    page = window.add_panel(
        "reference",
        "Reference",
        "layers",
        "Reference measurement",
        panel,
        SUBTITLES[c.processing.reference_mode],
    )
    connect_files(window, panel.measurement, panel.field_range, "reference")
    WidthWatcher(page, lambda _width: panel.fit_width(content_width(page)))

    pulling = False

    def push() -> None:
        if pulling:
            return
        c.set_processing(
            reference_mode=panel.reference_mode(),
            smooth=panel.smooth.isChecked(),
            sg_window=panel.sg_window.value(),
            sg_poly=panel.sg_poly.value(),
        )

    def pull() -> None:
        """Show options set through the controller (e.g. by another area)."""
        nonlocal pulling
        p = c.processing
        pulling = True
        try:
            if panel.reference_mode() is not p.reference_mode:
                panel.set_reference_mode(p.reference_mode)
            panel.smooth.setChecked(p.smooth)
            panel.sg_window.setValue(p.sg_window)
            panel.sg_poly.setValue(p.sg_poly)
            panel.set_custom_field(p.custom_field)
        finally:
            pulling = False
        page.set_subtitle(SUBTITLES[p.reference_mode])

    panel.mode.valueChanged.connect(push)
    panel.smooth.toggled.connect(push)
    panel.sg_window.valueChanged.connect(push)
    panel.sg_poly.valueChanged.connect(push)
    push()
    c.processingChanged.connect(pull)
    pull()

    box = panel.measurement
    for text, slot, shortcut in (
        ("Open reference sweep…", box.open_sweep_dialog, "Ctrl+R"),
        ("Load reference zero field…", box.load_zero_dialog, "Ctrl+Shift+R"),
    ):
        action = QAction(text, window)
        action.setShortcut(QKeySequence(shortcut))

        def run(_checked=False, s=slot) -> None:
            window.show_panel("reference")
            s()

        action.triggered.connect(run)
        window.add_file_action(action)

    p = window.persistence
    if p is not None:
        p.bind("reference/mode", panel.mode)
        p.bind("reference/smooth", panel.smooth)
        p.bind("reference/sg_window", panel.sg_window)
        p.bind("reference/sg_poly", panel.sg_poly)
        for part, edit in zip(("start", "step", "end"), panel.field_range.edits(), strict=True):
            p.bind(f"reference/field_{part}", edit)
