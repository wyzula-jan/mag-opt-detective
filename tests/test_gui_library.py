import numpy as np

import gui_helpers
from gui_helpers import energy_label, load_sweep, open_from, process, save_to, set_unit, shown_image
from mag_opt_detective.core.pipeline import PlotKind
from mag_opt_detective.core.spectra import load_tsv
from mag_opt_detective.core.units import Unit, to_cm1
from mag_opt_detective.gui.panels import library

window, errors = gui_helpers.window, gui_helpers.errors  # shared fixtures


def test_export_slots_and_merge(window, sweep, tmp_path, monkeypatch, errors):
    load_sweep(window, sweep)
    set_unit(window, "meV")
    process(window)
    c = window.controller
    assert c.result.ratio.unit is Unit.CM1  # processed in cm-1, shown in meV
    assert energy_label(window) == "Energy (meV)"
    save_to(monkeypatch, tmp_path / "S1.csv")
    window.commands["export_table"].trigger()
    exported = tmp_path / "S1_Ratio.csv"
    assert exported.read_text().startswith("Energy (meV)\t0.50T")  # legacy header
    fmap = load_tsv(exported)
    assert fmap.unit is Unit.MEV
    np.testing.assert_allclose(fmap.energy, sweep["x"] / 8.0656)
    np.testing.assert_allclose(fmap.values, c.result.ratio.values, rtol=1e-11)

    open_from(monkeypatch, exported)
    library.load_slot(window, 0)
    library.save_slot(window, 1)
    assert c.slots[0].unit is c.slots[1].unit is Unit.CM1
    np.testing.assert_allclose(c.slots[0].energy, sweep["x"], rtol=1e-11)
    tab = window.panels["library"]
    assert tab.slot_name(0) == "S1_Ratio.csv"
    tab.set_energy_range(0, None, 60)  # meV, the display unit
    tab.set_energy_range(1, 60, None)
    tab.merge_energy_button.click()
    assert not errors
    merged = c.result.ratio
    np.testing.assert_allclose(merged.energy[[0, -1]], sweep["x"][[0, -1]], rtol=1e-11)
    np.testing.assert_allclose(np.diff(merged.energy), np.diff(merged.energy)[0])
    assert energy_label(window) == "Energy (meV)"

    tab.full_energy.setChecked(False)
    tab.slot_spin.setValue(0)
    tab.plot_button.click()
    assert c.result.ratio.energy.max() <= to_cm1(60, Unit.MEV)
    assert c.result.ratio.to_unit(Unit.MEV).energy.max() <= 60
    library.plot_slot(window, 1)
    assert c.result.ratio.to_unit(Unit.MEV).energy.min() >= 60
    assert not errors


def test_slots_from_different_units_merge(window, sweep, tmp_path, monkeypatch, errors):
    """Slots keep cm-1, so slots saved while another unit was shown still combine."""
    load_sweep(window, sweep)
    c, tab = window.controller, window.panels["library"]
    set_unit(window, "meV")
    process(window)
    library.save_slot(window, 0)
    save_to(monkeypatch, tmp_path / "S1.csv")
    window.commands["export_table"].trigger()  # an meV table

    set_unit(window, "THz")
    open_from(monkeypatch, tmp_path / "S1_Ratio.csv")
    library.load_slot(window, 1)
    process(window)
    library.save_slot(window, 2)
    for slot in (1, 2):
        np.testing.assert_allclose(c.slots[slot].energy, c.slots[0].energy, rtol=1e-11)
    tab.set_used(2, False)
    tab.set_energy_range(0, None, 15)  # THz, about 500 cm-1
    tab.set_energy_range(1, 15, None)
    library.merge_by_energy(window)
    assert not errors
    assert c.unit is Unit.THZ
    assert energy_label(window) == "Energy (THz)"
    shown = c.result.get(PlotKind.RATIO, unit=Unit.THZ)
    np.testing.assert_allclose(shown_image(window), shown.values)
    np.testing.assert_allclose(c.result.ratio.energy[[0, -1]], [100, 1000], rtol=1e-11)

    tab.set_used(2, True)
    for slot in (0, 1):
        tab.set_energy_range(slot, None, None)
    library.average(window)
    library.merge_by_field(window)
    assert not errors
    np.testing.assert_allclose(c.result.ratio.values, c.slots[2].values, rtol=1e-9)


def test_slot_energy_limits_follow_the_unit(window, sweep, errors):
    load_sweep(window, sweep)
    process(window)
    tab = window.panels["library"]
    library.save_slot(window, 0)
    set_unit(window, "meV")
    tab.set_energy_range(0, 30, 60)
    lo, hi = tab.energy_range(0)
    assert (lo, hi) == (30 * 8.0656, 60 * 8.0656)
    set_unit(window, "cm-1")
    assert tab.shown_energy_range(0) == (241.968, 483.936)
    assert tab.energy_range(0) == (lo, hi)  # kept in cm-1: no rounding
    set_unit(window, "meV")
    assert tab.shown_energy_range(0) == (30, 60)

    tab.set_energy_range(0, 200, 300)  # meV: beyond the data (12.4 - 124 meV)
    tab.full_energy.setChecked(False)
    library.plot_slot(window, 0)
    assert "slot 0: E range 200 – 300 meV contains no data" in errors[-1]
    assert window.infobar.action_button.text() == "Open Library"


def test_merge_by_field_uses_ticked_slots(window, sweep, errors):
    load_sweep(window, sweep)
    process(window)
    tab = window.panels["library"]
    for slot in (0, 1, 2):
        library.save_slot(window, slot)
    c = window.controller
    assert tab.used_slots(c.slots) == [0, 1, 2]
    tab.set_field_range(0, None, 1.0)
    tab.set_field_range(1, 1.5, None)
    tab.set_used(2, False)
    tab.merge_field_button.click()
    assert not errors
    np.testing.assert_allclose(c.result.ratio.field, sweep["fields"])

    for slot in (0, 1):
        tab.set_used(slot, False)
    library.merge_by_field(window)
    assert errors and "Use column" in errors[-1]


def test_average_slots(window, sweep, errors):
    load_sweep(window, sweep)
    process(window)
    library.save_slot(window, 0)
    library.save_slot(window, 1)
    window.panels["library"].average_button.click()
    assert not errors
    np.testing.assert_allclose(
        window.controller.result.ratio.values, window.controller.slots[0].values
    )
