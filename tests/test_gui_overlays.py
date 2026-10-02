import numpy as np
import pytest

import gui_helpers
from gui_helpers import load_sweep, process, set_unit
from mag_opt_detective.core.models import dirac_interband
from mag_opt_detective.core.units import Unit, convert

window, errors = gui_helpers.window, gui_helpers.errors  # shared fixtures


def test_dirac_overlay(window, sweep, errors):
    load_sweep(window, sweep)
    set_unit(window, "THz")
    process(window)
    plot = window.plots.map
    assert plot.model_curve_data() == []
    page = window.inspector["overlays"].body_layout().itemAt(0).widget()
    page.n_lines.setValue(3)
    page.delta.setValue(10.0)
    page.velocity.setValue(5.0)
    page.show_dirac.setChecked(True)
    curves = plot.model_curve_data()
    assert len(curves) == 3
    field, energy = curves[1]
    expected = convert(dirac_interband(field, 5.0, 10.0, 3)[1], Unit.MEV, Unit.THZ)
    np.testing.assert_allclose(energy, expected)
    page.n_lines.setValue(2)
    assert len(plot.model_curve_data()) == 2
    page.show_dirac.setChecked(False)
    assert plot.model_curve_data() == []
    assert not errors


def test_overlay_without_a_result(window, errors):
    page = window.inspector["overlays"].body_layout().itemAt(0).widget()
    page.show_dirac.setChecked(True)
    assert window.plots.map.model_curve_data() == []
    with pytest.raises(ValueError, match="process data first"):
        window.controller.figure_state()
    assert not errors
