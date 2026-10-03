"""The massive Dirac model, the first card of the Models section (the former Overlays)."""

import numpy as np
import pytest

import gui_helpers
from gui_helpers import load_sweep, process, set_unit
from mag_opt_detective.core.models import dirac_interband
from mag_opt_detective.core.units import Unit, convert

window, errors = gui_helpers.window, gui_helpers.errors  # shared fixtures

THZ = 33.35641 / 8.0656  # meV per THz


def dirac_card(window):
    models = window.inspector["models"].body_layout().itemAt(0).widget().models
    return models.cards[models.entries[0]]


def test_dirac_overlay(window, sweep, errors):
    load_sweep(window, sweep)
    set_unit(window, "THz")
    process(window)
    plot = window.plots.map
    assert plot.model_curve_data() == []
    editor = dirac_card(window).editor
    editor.n_lines.field.edit.setText("3")
    editor.delta.field.edit.setText(f"{10.0 / THZ}")  # the half-gap is typed in THz
    editor.velocity.field.edit.setText("5")
    dirac_card(window).visible_switch.setChecked(True)
    curves = plot.model_curve_data()  # the first model draws in the "models" layer
    assert len(curves) == 3
    field, energy = curves[1]
    expected = convert(dirac_interband(field, 5.0, 10.0, 3)[1], Unit.MEV, Unit.THZ)
    np.testing.assert_allclose(energy, expected)
    editor.n_lines.field.edit.setText("2")
    assert len(plot.model_curve_data()) == 2
    editor.n_lines.field.edit.setText("400")  # beyond the 40 of the former overlay
    assert editor.n_lines.field.is_invalid() and len(plot.model_curve_data()) == 2
    assert dirac_card(window).entry.model.n_lines == 2
    dirac_card(window).visible_switch.setChecked(False)
    assert plot.model_curve_data() == []
    assert not errors


def test_overlay_without_a_result(window, errors):
    dirac_card(window).visible_switch.setChecked(True)
    assert window.plots.map.model_curve_data() == []
    with pytest.raises(ValueError, match="process data first"):
        window.controller.figure_state()
    assert not errors
