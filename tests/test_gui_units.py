"""The live energy-unit switch: everything is shown in the new unit, nothing is reprocessed."""

import numpy as np
import pytest

import gui_helpers
from gui_helpers import (
    click_map,
    current_marker_energies,
    energy_label,
    load_sweep,
    process,
    save_to,
    select,
    set_unit,
    shown_image,
)
from mag_opt_detective.core.models import dirac_interband
from mag_opt_detective.core.pipeline import PlotKind, ReferenceMode
from mag_opt_detective.core.processing import Axis
from mag_opt_detective.core.spectra import load_tsv
from mag_opt_detective.core.units import Unit, convert
from mag_opt_detective.gui import display

window, errors = gui_helpers.window, gui_helpers.errors  # shared fixtures

MEV = 8.0656


def view_page(window):
    return window.inspector["view"].body_layout().itemAt(0).widget()


def dirac_card(window):
    """The card of the massive Dirac model (the first in the Models section)."""
    models = window.inspector["models"].body_layout().itemAt(0).widget().models
    return models.cards[models.entries[0]]


def test_live_unit_switch(window, sweep, errors):
    load_sweep(window, sweep)
    window.panels["reference"].set_reference_mode(ReferenceMode.SELF)
    process(window)
    c = window.controller
    result = c.result
    page = view_page(window)
    page.energy.lo_spin.setValue(200.0)
    page.energy.hi_spin.setValue(800.0)
    assert c.view.energy_range == (200.0, 800.0)
    window.tools.set_active("pick")
    click_map(window, 1.0, 300.0)
    card = dirac_card(window)
    card.editor.n_lines.field.edit.setText("2")
    card.editor.delta.field.edit.setText(f"{10.0 * MEV}")  # 10 meV, typed in cm-1
    card.visible_switch.setChecked(True)
    select(window, kind=PlotKind.DATA, order=1, per_unit=True)
    window.plots.map.hist.region.setRegion((-0.002, 0.003))  # fixed levels per cm-1
    assert c.view.levels["Data_der1_E_unit"] == pytest.approx((-0.002, 0.003))
    ratio_levels = c.view.levels["Ratio"]

    set_unit(window, "meV")
    assert c.result is result  # not processed again
    assert not errors
    expected = result.get(PlotKind.DATA, 1, Axis.ENERGY, physical=True, unit=Unit.MEV)
    np.testing.assert_allclose(shown_image(window), expected.values)
    assert energy_label(window) == "Energy (meV)"
    assert window.plots.stacked.plot.getAxis("bottom").labelText == "Energy (meV)"
    assert window.plots.reference.plot.getAxis("left").labelText == "Energy (meV)"
    assert c.view.energy_range == pytest.approx((200 / MEV, 800 / MEV))
    assert window.plots.map.plot.vb.viewRange()[1] == pytest.approx([200 / MEV, 800 / MEV])
    assert page.energy.range() == pytest.approx((200 / MEV, 800 / MEV))
    assert page.energy.unit() == "meV" and not page.energy.is_auto()
    assert c.view.levels["Data_der1_E_unit"] == pytest.approx((-0.002 * MEV, 0.003 * MEV))
    assert window.plots.map.levels() == pytest.approx((-0.002 * MEV, 0.003 * MEV))
    assert c.view.levels["Ratio"] == ratio_levels  # other levels do not depend on the unit
    np.testing.assert_allclose(current_marker_energies(window), [300 / MEV])
    model = window.panels["points"].model
    assert model.data(model.index(0, 1)) == f"{300 / MEV:.5g}"  # the point's E
    field, energy = window.plots.map.model_curve_data()[1]
    np.testing.assert_allclose(energy, dirac_interband(field, 5.0, 10.0, 2)[1])  # meV
    assert window.summary_text().endswith("E 12.4 – 124 meV")
    assert window.plot_area.description.text().endswith("per meV")

    set_unit(window, "THz")
    field, energy = window.plots.map.model_curve_data()[1]
    np.testing.assert_allclose(
        energy, convert(dirac_interband(field, 5.0, 10.0, 2)[1], "meV", "THz")
    )
    set_unit(window, "cm-1")
    assert c.result is result
    assert c.view.energy_range == pytest.approx((200.0, 800.0))
    assert c.view.levels["Data_der1_E_unit"] == pytest.approx((-0.002, 0.003))
    assert page.energy.range() == pytest.approx((200.0, 800.0))
    np.testing.assert_allclose(c.points.points("LL 1")[1], [300.0])  # kept in cm-1
    assert not errors


def test_derivatives_along_field_keep_their_levels(window, sweep):
    load_sweep(window, sweep)
    process(window)
    c = window.controller
    select(window, order=1, axis="B", per_unit=True)
    window.plots.map.hist.region.setRegion((-0.1, 0.1))
    set_unit(window, "THz")
    assert c.view.levels["Ratio_der1_B_unit"] == pytest.approx((-0.1, 0.1))
    select(window, axis="E", per_unit=False)
    window.plots.map.hist.region.setRegion((-0.02, 0.02))
    set_unit(window, "meV")
    assert c.view.levels["Ratio_der1_E"] == pytest.approx((-0.02, 0.02))  # per point: no scaling


def test_energy_fields_are_kept_in_cm1(window, sweep):
    panel = window.panels["processing"]
    set_unit(window, "meV")
    panel.baseline_on.setChecked(True)
    panel.baseline_lo.setText("60")
    panel.baseline_hi.setText("70.5")
    region = window.controller.processing.baseline
    assert region == pytest.approx((60 * MEV, 70.5 * MEV))
    set_unit(window, "cm-1")
    assert (panel.baseline_lo.text(), panel.baseline_hi.text()) == ("483.936", "568.625")
    assert panel.unit_labels[0].text() == "cm⁻¹"
    for unit in ("THz", "meV"):
        set_unit(window, unit)
    assert (panel.baseline_lo.text(), panel.baseline_hi.text()) == ("60", "70.5")
    assert window.controller.processing.baseline == region  # exactly: no round trips
    assert not window.controller.changed_since_process()


def test_exports_use_the_current_unit(window, sweep, tmp_path, monkeypatch, errors):
    load_sweep(window, sweep)
    process(window)
    set_unit(window, "THz")
    save_to(monkeypatch, tmp_path / "T.csv")
    window.commands["export_table"].trigger()
    fmap = load_tsv(tmp_path / "T_Ratio.csv")
    assert fmap.unit is Unit.THZ
    np.testing.assert_allclose(fmap.energy, sweep["x"] / 33.35641)
    state = window.controller.figure_state()
    assert state.unit is Unit.THZ and state.fmap.unit is Unit.THZ
    assert not errors


def test_figure_state(window, sweep):
    load_sweep(window, sweep)
    process(window)
    c = window.controller
    window.tools.set_active("pick")
    click_map(window, 1.0, 300.0)
    dirac_card(window).visible_switch.setChecked(True)
    set_unit(window, "meV")
    state = c.figure_state()
    assert state.description == "R(B)/R(0)"
    assert state.colormap == "magma" and state.fixed_levels
    assert state.levels == (0.9, 1.1)
    np.testing.assert_allclose(state.points["LL 1"][1], [300 / MEV])
    assert len(state.overlays["dirac"]) == 5
    np.testing.assert_allclose(state.fmap.values, c.result.ratio.values)


def test_units_and_numbers_are_written_alike_everywhere(window, sweep):
    """cm⁻¹ wherever people read the unit (files keep "cm-1"), and four significant digits."""
    assert [display.unit_text(u) for u in Unit] == ["cm⁻¹", "meV", "THz"]
    assert display.energy_label("cm-1") == "Energy (cm⁻¹)"
    numbers = (16.0, 0.25, 396.75, 43.394, 3200.0, 0.0, -0.5, 1.5e-6, 2.5e7, None, float("nan"))
    assert [display.format_number(v) for v in numbers] == [
        "16", "0.25", "396.8", "43.39", "3200", "0", "-0.5", "1.5e-6", "2.5e7", "–", "–",
    ]  # fmt: skip
    load_sweep(window, sweep)
    process(window)
    select(window, order=1, per_unit=True)
    assert window.plot_area.description.text() == "R(B)/R(0) · 1st derivative d/dE per cm⁻¹"
    assert window.plots.map.plot.getAxis("bottom").labelText == "Magnetic field B (T)"
    stacked = window.plots.stacked.plot
    assert stacked.getAxis("bottom").labelText == "Energy (cm⁻¹)"
    assert stacked.getAxis("left").labelText == "1st derivative + offset"
    select(window, order=0)
    assert stacked.getAxis("left").labelText == "R(B)/R(0) + offset"
