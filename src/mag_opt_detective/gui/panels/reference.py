"""Reference panel: the reference mode, its smoothing and the reference sweep's files."""

from __future__ import annotations

from PySide6.QtGui import QAction, QKeySequence

from mag_opt_detective.core.pipeline import ReferenceMode
from mag_opt_detective.gui.measurement_tab import MeasurementTab
from mag_opt_detective.gui.panels.sample import connect_measurement

SUBTITLES = {
    ReferenceMode.NONE: "No reference: the sample ratio is shown as it is.",
    ReferenceMode.SEPARATE: "Corrects the sample with a separately measured sweep.",
    ReferenceMode.SELF: "Uses the (smoothed) sample itself as the reference.",
}


def install(window) -> None:
    c = window.controller
    tab = MeasurementTab(reference=True)
    page = window.add_panel(
        "reference",
        "Reference",
        "layers",
        "Reference measurement",
        tab,
        SUBTITLES[c.processing.reference_mode],
    )
    connect_measurement(window, tab, "reference")

    def push() -> None:
        c.set_processing(
            reference_mode=tab.reference_mode(),
            smooth=tab.smooth.isChecked(),
            sg_window=tab.sg_window.value(),
            sg_poly=tab.sg_poly.value(),
        )
        page.set_subtitle(SUBTITLES[c.processing.reference_mode])

    tab.ref_group.buttonToggled.connect(lambda _b, checked: checked and push())
    tab.smooth.toggled.connect(push)
    tab.sg_window.valueChanged.connect(push)
    tab.sg_poly.valueChanged.connect(push)
    push()

    for text, slot, shortcut in (
        ("Load Reference Field…", tab.load_field_dialog, "Ctrl+R"),
        ("Load Reference Zero Field…", tab.load_zero_dialog, "Ctrl+Shift+R"),
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
        for part in ("start", "step", "end"):
            p.bind(f"reference/field_{part}", getattr(tab.field_range, part))
        for attr in ("ref_none", "ref_separate", "ref_self", "smooth", "sg_window", "sg_poly"):
            p.bind(f"reference/{attr}", getattr(tab, attr))
