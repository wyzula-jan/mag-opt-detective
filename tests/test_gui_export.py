"""Journal figures of the window's plot: the figure state, the print size and file names."""

from pathlib import Path

import numpy as np
import pytest

from mag_opt_detective.core.spectra import FieldMap
from mag_opt_detective.core.units import Unit, from_cm1
from mag_opt_detective.export import APS, CUSTOM, NATURE, StackedOptions
from mag_opt_detective.gui.controller import AppController
from mag_opt_detective.gui.export_state import (
    POINTS_CURRENT,
    POINTS_NONE,
    FigureContent,
    auto_labels,
    figure_state,
    file_name,
    print_size,
    safe_name,
    with_format,
)

MEV = 8.0656


# ---------------------------------------------------------------------- figure state (no window)
@pytest.fixture
def controller(qapp):
    """A controller showing a small cm-1 map, with two curves of points and one model line."""
    c = AppController()
    energy = np.linspace(100.0, 1000.0, 46)
    field = np.array([0.5, 1.0, 1.5, 2.0])
    values = 1 + 0.01 * np.add.outer(np.sin(energy / 50), field)
    c.from_map(FieldMap(energy, field, values, unit="cm-1"))
    c.record_points([1.0, 1.5], [300.0, 320.0], unit="cm-1")
    c.add_curve("LL 2")
    c.record_points([2.0], [500.0], unit="cm-1")
    c.set_overlay("model", lambda unit: [(field, from_cm1(np.full(4, 400.0), unit))])
    return c


def test_figure_state_of_the_controller(controller):
    c = controller
    c.set_view(colormap="viridis")
    c.set_ranges(field_range=(0.6, 1.9), energy_range=(200.0, 900.0))
    c.set_unit("meV")
    state = figure_state(c, FigureContent())
    assert state.kind == "map" and state.fmap.unit is Unit.MEV and state.cmap == "viridis"
    assert state.levels == pytest.approx((0.9, 1.1))  # the fixed R(B)/R(0) default
    assert state.x_range == (0.6, 1.9)
    assert state.y_range == pytest.approx((200 / MEV, 900 / MEV))
    (curve,) = state.curves
    assert curve.style == "model" and curve.label == "model"
    np.testing.assert_allclose(curve.y, 400 / MEV)
    assert [(p.label, p.current) for p in state.points] == [("LL 1", False), ("LL 2", True)]
    np.testing.assert_allclose(state.points[0].y, [300 / MEV, 320 / MEV])

    content = FigureContent(models=False, points=POINTS_CURRENT, x_label="B", y_label=" ")
    state = figure_state(c, content)
    assert state.curves == [] and [p.label for p in state.points] == ["LL 2"]
    assert (state.x_label, state.y_label) == ("B", "")
    assert figure_state(c, FigureContent(points=POINTS_NONE)).points == []

    c.set_view(stacked_offset=0.02, stacked_every=2, stacked_by_field=False)
    state = figure_state(c, FigureContent(kind="stacked"))
    assert state.kind == "stacked" and state.stacked == StackedOptions(0.02, 2, False)
    assert state.x_range == pytest.approx((200 / MEV, 900 / MEV)) and state.y_range is None
    assert state.cmap == "viridis"  # the colour map of the traces
    assert not state.colorbar  # traces are not coloured by field
    assert auto_labels("map", Unit.CM1) == ("Magnetic field (T)", "Energy (cm⁻¹)")
    assert auto_labels("stacked", Unit.THZ) == ("Energy (THz)", "Intensity (a.u.)")
    with pytest.raises(ValueError):
        figure_state(AppController(), FigureContent())  # nothing processed


def test_file_name_from_the_data(controller):
    name = file_name(controller, "map", NATURE, "single", "pdf")
    assert name == "Library_map_Ratio_nature-single.pdf"
    assert file_name(controller, "stacked", NATURE, "1.5 wide", "tif").endswith(
        "_Ratio_stacked_nature-1.5-wide.tif"
    )
    assert file_name(controller, "map", CUSTOM, "", "png").endswith("_Ratio_custom.png")


def test_print_size_rules():
    size = print_size(APS, 86, 65, 8, 0.25, 600, "eps")
    assert size.ok
    assert size.warnings == [f"{APS.name}: lines should be at least 0.5 pt, not 0.25 pt."]
    size = print_size(NATURE, 89, None, 7, 0.5, 450)
    assert not size.ok and size.invalid == {"height"}
    size = print_size(CUSTOM, 120, 90, 30, 1, 600, "svg")
    assert size.font_pt == 24 and size.warnings == ["Custom: text is 4–24 pt; using 24 pt."]


def test_file_names():
    assert with_format("/a/fig", "pdf") == Path("/a/fig.pdf")
    assert with_format("/a/fig.PDF", "pdf") == Path("/a/fig.PDF")
    assert with_format("/a/fig.png", "eps") == Path("/a/fig.eps")
    assert with_format("/a/fig.tiff", "tif") == Path("/a/fig.tiff")
    assert with_format("/a/B_1.5T", "svg") == Path("/a/B_1.5T.svg")
    assert safe_name("Merged by energy: a + b") == "Merged_by_energy_a_+_b"
