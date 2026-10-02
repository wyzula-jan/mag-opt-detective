import numpy as np

import gui_helpers
from gui_helpers import (
    energy_label,
    infobar_text,
    load_sweep,
    process,
    save_to,
    set_unit,
    shown_image,
)
from mag_opt_detective.core.pipeline import PlotKind
from mag_opt_detective.core.spectra import load_tsv
from mag_opt_detective.core.units import Unit, to_cm1
from mag_opt_detective.gui.panels import library

window, errors = gui_helpers.window, gui_helpers.errors  # shared fixtures


def rows(window):
    """The Library panel's rows, in library order."""
    panel = window.panels["library"]
    return [panel.rows[entry.key] for entry in window.controller.library]


def test_export_load_save_and_merge(window, sweep, tmp_path, monkeypatch, errors):
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

    library.load_tables(window, [str(exported)])
    library.save_current(window)
    first, second = c.library
    assert first.fmap.unit is second.fmap.unit is Unit.CM1
    np.testing.assert_allclose(first.fmap.energy, sweep["x"], rtol=1e-11)
    assert first.name == "S1_Ratio.csv" and second.name == "Sample_4p2K_Sam1"
    row0, row1 = rows(window)
    row0.e_max.setText("60")  # meV, the display unit
    row1.e_min.setText("60")
    window.panels["library"].merge_energy_button.click()
    assert not errors
    merged = c.result.ratio
    np.testing.assert_allclose(merged.energy[[0, -1]], sweep["x"][[0, -1]], rtol=1e-11)
    np.testing.assert_allclose(np.diff(merged.energy), np.diff(merged.energy)[0])
    assert energy_label(window) == "Energy (meV)"

    window.panels["library"].full_energy.setChecked(False)
    row0.plot_button.click()
    assert c.result.ratio.energy.max() <= to_cm1(60, Unit.MEV)
    assert c.result.ratio.to_unit(Unit.MEV).energy.max() <= 60
    library.plot_entry(window, second.key)
    assert c.result.ratio.to_unit(Unit.MEV).energy.min() >= 60
    assert not errors


def test_maps_from_different_units_merge(window, sweep, tmp_path, monkeypatch, errors):
    """Library maps keep cm-1, so maps saved while another unit was shown still combine."""
    load_sweep(window, sweep)
    c = window.controller
    set_unit(window, "meV")
    process(window)
    library.save_current(window)
    save_to(monkeypatch, tmp_path / "S1.csv")
    window.commands["export_table"].trigger()  # an meV table

    set_unit(window, "THz")
    library.load_tables(window, [str(tmp_path / "S1_Ratio.csv")])
    process(window)
    library.save_current(window)
    for entry in c.library[1:]:
        np.testing.assert_allclose(entry.fmap.energy, c.library[0].fmap.energy, rtol=1e-11)
    row0, row1, row2 = rows(window)
    row2.use.setChecked(False)
    row0.e_max.setText("15")  # THz, about 500 cm-1
    row1.e_min.setText("15")
    library.merge_by_energy(window)
    assert not errors
    assert c.unit is Unit.THZ
    assert energy_label(window) == "Energy (THz)"
    shown = c.result.get(PlotKind.RATIO, unit=Unit.THZ)
    np.testing.assert_allclose(shown_image(window), shown.values)
    np.testing.assert_allclose(c.result.ratio.energy[[0, -1]], [100, 1000], rtol=1e-11)

    row2.use.setChecked(True)
    for row in (row0, row1):
        row.e_min.setText("")
        row.e_max.setText("")
    library.average(window)
    library.merge_by_field(window)
    assert not errors
    np.testing.assert_allclose(c.result.ratio.values, c.library[2].fmap.values, rtol=1e-9)


def test_energy_limits_follow_the_unit(window, sweep, errors):
    load_sweep(window, sweep)
    process(window)
    library.save_current(window)
    entry = window.controller.library[0]
    (row,) = rows(window)
    set_unit(window, "meV")
    row.e_min.setText("30")
    row.e_max.setText("60")
    lo, hi = entry.energy_cut
    assert (lo, hi) == (30 * 8.0656, 60 * 8.0656)
    set_unit(window, "cm-1")
    assert (row.e_min.text(), row.e_max.text()) == ("241.968", "483.936")
    assert entry.energy_cut == (lo, hi)  # kept in cm-1: no rounding
    set_unit(window, "meV")
    assert (row.e_min.text(), row.e_max.text()) == ("30", "60")

    row.e_min.setText("200")  # meV: beyond the data (12.4 - 124 meV)
    row.e_max.setText("300")
    window.panels["library"].full_energy.setChecked(False)
    library.plot_entry(window, entry.key)
    assert "Sample_4p2K_Sam1: E range 200 – 300 meV contains no data" in errors[-1]
    assert window.infobar.action_button.text() == "Open Library"


def test_merge_by_field_uses_ticked_maps(window, sweep, errors):
    load_sweep(window, sweep)
    process(window)
    for _ in range(3):
        library.save_current(window)
    c = window.controller
    assert [e.name for e in c.library] == [
        "Sample_4p2K_Sam1",
        "Sample_4p2K_Sam1 (2)",
        "Sample_4p2K_Sam1 (3)",
    ]
    assert c.ticked() == c.library
    row0, row1, row2 = rows(window)
    row0.b_max.setText("1.0")
    row1.b_min.setText("1.5")
    row2.use.setChecked(False)
    window.panels["library"].merge_field_button.click()
    assert not errors
    np.testing.assert_allclose(c.result.ratio.field, sweep["fields"])

    row0.use.setChecked(False)
    assert not window.panels["library"].merge_field_button.isEnabled()
    library.merge_by_field(window)
    assert errors and "tick at least two maps" in errors[-1]


def test_average_maps(window, sweep, errors):
    load_sweep(window, sweep)
    process(window)
    library.save_current(window)
    library.save_current(window)
    window.panels["library"].average_button.click()
    assert not errors
    np.testing.assert_allclose(
        window.controller.result.ratio.values, window.controller.library[0].fmap.values
    )


def test_a_new_result_closes_the_error_bar(window, sweep, errors):
    load_sweep(window, sweep)
    process(window)
    library.save_current(window)
    library.save_current(window)
    row0, _row1 = rows(window)
    row0.use.setChecked(False)
    library.average(window)
    assert "Tick at least two maps" in infobar_text(window)
    row0.use.setChecked(True)
    library.average(window)
    assert window.infobar.isHidden()
    assert len(errors) == 1
