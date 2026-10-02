"""Processing panel: the energy window and the baseline region, and the Process action.

Both ranges are typed in the display unit and kept in cm^-1 (:class:`EnergyEdit`).
"""

from __future__ import annotations

from PySide6.QtWidgets import QGridLayout, QLabel, QVBoxLayout, QWidget

from mag_opt_detective.core.units import Range
from mag_opt_detective.gui.controller import user_action
from mag_opt_detective.gui.kit import Switch
from mag_opt_detective.gui.widgets import EnergyEdit

APPLIED = "Applied the next time you process."
CHANGED = "Changed since the last run. Process again to apply."


def _hint(text: str) -> QLabel:
    label = QLabel(text)
    label.setProperty("kit", "muted")
    label.setWordWrap(True)
    return label


class ProcessingPanel(QWidget):
    """Energy window (drops data outside it) and baseline region (normalised to 1)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.cut_on = Switch("Energy window")
        self.cut_lo = EnergyEdit(name="Energy window from")
        self.cut_hi = EnergyEdit(name="Energy window to")
        self.baseline_on = Switch("Baseline")
        self.baseline_lo = EnergyEdit(name="Baseline from")
        self.baseline_hi = EnergyEdit(name="Baseline to")
        self.unit_labels: list[QLabel] = []
        for edit in (self.cut_lo, self.baseline_lo):
            edit.setPlaceholderText("from")
        for edit in (self.cut_hi, self.baseline_hi):
            edit.setPlaceholderText("to")
        self.status = _hint(APPLIED)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 16)
        layout.setSpacing(14)
        for switch, lo, hi, hint in (
            (
                self.cut_on,
                self.cut_lo,
                self.cut_hi,
                "Drops data outside it when processing, e.g. noisy detector edges. "
                "To zoom only, use the energy range of the View section.",
            ),
            (
                self.baseline_on,
                self.baseline_lo,
                self.baseline_hi,
                "Shifts each spectrum so that this region averages 1.",
            ),
        ):
            block = QGridLayout()
            block.setHorizontalSpacing(6)
            block.setVerticalSpacing(6)
            block.addWidget(switch, 0, 0, 1, 4)
            block.addWidget(_hint(hint), 1, 0, 1, 4)
            block.addWidget(lo, 2, 0)
            block.addWidget(QLabel("–"), 2, 1)
            block.addWidget(hi, 2, 2)
            unit = QLabel()
            unit.setProperty("kit", "muted")
            self.unit_labels.append(unit)
            block.addWidget(unit, 2, 3)
            block.setColumnStretch(0, 1)
            block.setColumnStretch(2, 1)
            layout.addLayout(block)
        layout.addWidget(self.status)
        layout.addStretch(1)

    def edits(self) -> tuple[EnergyEdit, ...]:
        return (self.cut_lo, self.cut_hi, self.baseline_lo, self.baseline_hi)

    def set_unit(self, unit) -> None:
        for edit in self.edits():
            edit.set_unit(unit)
        for label in self.unit_labels:
            label.setText(str(unit))

    def energy_cut(self) -> Range | None:
        """The window in cm^-1, or None when it is off or both ends are empty."""
        rng = (self.cut_lo.cm1(), self.cut_hi.cm1())
        return rng if self.cut_on.isChecked() and rng != (None, None) else None

    def baseline(self) -> Range | None:
        """The baseline region in cm^-1 (an end may be None), or None when it is off."""
        if not self.baseline_on.isChecked():
            return None
        return self.baseline_lo.cm1(), self.baseline_hi.cm1()

    def show_ranges(self, energy_cut: Range | None, baseline: Range | None) -> None:
        """Show the two ranges (cm^-1); None switches one off and keeps its typed ends."""
        for switch, lo, hi, rng in (
            (self.cut_on, self.cut_lo, self.cut_hi, energy_cut),
            (self.baseline_on, self.baseline_lo, self.baseline_hi, baseline),
        ):
            if rng is None:
                switch.setChecked(False)
                continue
            for edit, value in zip((lo, hi), rng, strict=True):
                if edit.cm1() != value:
                    edit.set_cm1(value)
            switch.setChecked(True)


@user_action("Process")
def process(window) -> None:
    window.infobar.dismiss()
    window.controller.process()


def install(window) -> None:
    c = window.controller
    panel = ProcessingPanel()
    window.add_panel(
        "processing", "Processing", "sliders-horizontal", "Processing options", panel,
        "Energy window and baseline, in the energy unit of the toolbar.", rail_text="Process",
    )  # fmt: skip

    pulling = False

    def push() -> None:
        if not pulling:
            c.set_processing(energy_cut=panel.energy_cut(), baseline=panel.baseline())

    def pull() -> None:
        """Show ranges set through the controller (e.g. by another area)."""
        nonlocal pulling
        state = c.processing
        if (panel.energy_cut(), panel.baseline()) == (state.energy_cut, state.baseline):
            return
        pulling = True
        try:
            panel.show_ranges(state.energy_cut, state.baseline)
        finally:
            pulling = False

    for switch in (panel.cut_on, panel.baseline_on):
        switch.toggled.connect(push)
    for edit in panel.edits():
        edit.valueChanged.connect(push)

    def sync_enabled() -> None:
        for on, edits in (
            (panel.cut_on, (panel.cut_lo, panel.cut_hi)),
            (panel.baseline_on, (panel.baseline_lo, panel.baseline_hi)),
        ):
            for edit in edits:
                edit.setEnabled(on.isChecked())

    for switch in (panel.cut_on, panel.baseline_on):
        switch.toggled.connect(sync_enabled)
    sync_enabled()

    panel.set_unit(c.unit)
    c.unitChanged.connect(lambda _old, new: panel.set_unit(new))
    c.changedSinceProcess.connect(
        lambda changed: panel.status.setText(CHANGED if changed else APPLIED)
    )
    push()
    c.processingChanged.connect(pull)

    window.commands["process"].triggered.connect(lambda: process(window))

    p = window.persistence
    if p is not None:
        p.bind("processing/cut_on", panel.cut_on)
        p.bind("processing/cut_lo_cm1", panel.cut_lo)
        p.bind("processing/cut_hi_cm1", panel.cut_hi)
        p.bind("processing/baseline_on", panel.baseline_on)
        p.bind("processing/baseline_lo_cm1", panel.baseline_lo)
        p.bind("processing/baseline_hi_cm1", panel.baseline_hi)
