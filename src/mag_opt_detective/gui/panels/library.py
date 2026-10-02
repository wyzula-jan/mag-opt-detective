"""Library panel: slots of processed maps to plot again, merge or average.

The slots live in ``controller.slots`` (cm^-1); the table keeps each slot's name, whether it
is used, and its energy (shown in the display unit) and field limits.
"""

from __future__ import annotations

from pathlib import Path

from mag_opt_detective.gui.controller import panel_error, user_action
from mag_opt_detective.gui.processed_tab import ProcessedTab
from mag_opt_detective.gui.widgets import open_file


def _tab(window) -> ProcessedTab:
    return window.panels["library"]


def _used_slots(window) -> list[int]:
    used = _tab(window).used_slots(window.controller.slots)
    if not used:
        raise panel_error(
            "no slot selected - load slots and tick them in the Use column", "library"
        )
    return used


@user_action("Load slot")
def load_slot(window, slot: int) -> None:
    tab = _tab(window)
    path = open_file(window, f"Load processed table into slot {slot}")
    if not path:
        return
    c = window.controller
    field_values = None if tab.auto_field.isChecked() else c.processing.sample_field.values()
    c.load_slot(slot, path, field_values)
    tab.set_slot_name(slot, Path(path).name)


@user_action("Save slot")
def save_slot(window, slot: int) -> None:
    window.controller.save_slot(slot)
    _tab(window).set_slot_name(slot, f"Saved R(B)/R(0) [{slot}]")


@user_action("Plot slot")
def plot_slot(window, slot: int) -> None:
    tab = _tab(window)
    rng = None if tab.full_energy.isChecked() else tab.energy_range(slot)
    window.controller.plot_slot(slot, rng)


@user_action("Merge by energy")
def merge_by_energy(window) -> None:
    tab = _tab(window)
    used = _used_slots(window)
    window.controller.merge_by_energy([(i, tab.energy_range(i)) for i in used])


@user_action("Merge by field")
def merge_by_field(window) -> None:
    tab = _tab(window)
    used = _used_slots(window)
    window.controller.merge_by_field([(i, tab.field_range(i)) for i in used])


@user_action("Average")
def average(window) -> None:
    tab = _tab(window)
    used = _used_slots(window)
    window.controller.average([(i, tab.energy_range(i), tab.field_range(i)) for i in used])


def install(window) -> None:
    c = window.controller
    tab = ProcessedTab()
    window.add_panel(
        "library", "Library", "library-big", "Processed maps", tab,
        "Processed maps to plot again, merge or average.",
    )  # fmt: skip
    tab.loadRequested.connect(lambda slot: load_slot(window, slot))
    tab.saveRequested.connect(lambda slot: save_slot(window, slot))
    tab.plotRequested.connect(lambda slot: plot_slot(window, slot))
    tab.mergeEnergyRequested.connect(lambda: merge_by_energy(window))
    tab.mergeFieldRequested.connect(lambda: merge_by_field(window))
    tab.averageRequested.connect(lambda: average(window))
    tab.set_unit(c.unit)
    c.unitChanged.connect(lambda _old, new: tab.set_unit(new))

    p = window.persistence
    if p is not None:
        p.bind("library/full_energy", tab.full_energy)
        p.bind("library/auto_field", tab.auto_field)
