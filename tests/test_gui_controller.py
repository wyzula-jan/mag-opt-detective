"""The application controller on its own (no window)."""

import dataclasses

import numpy as np
import pytest

from mag_opt_detective.core.pipeline import PlotKind, ReferenceMode
from mag_opt_detective.core.processing import Axis
from mag_opt_detective.core.units import Unit
from mag_opt_detective.gui.controller import (
    AppController,
    FieldRange,
    PlotSelection,
    ProcessingState,
    SweepFiles,
    ViewState,
    level_key,
    parse_level_key,
    user_action,
)

MEV = 8.0656


@pytest.fixture
def controller(qapp):
    return AppController()


def test_level_keys():
    assert level_key(PlotKind.STEP) == "Ratio_Step"
    assert level_key(PlotKind.DATA, 1, Axis.FIELD) == "Data_der1_B"  # per kind, order, axis
    assert level_key(PlotKind.DATA, 2, Axis.ENERGY, physical=True) == "Data_der2_E_unit"
    assert level_key(PlotKind.STEP, 1, Axis.FIELD, physical=True) == "Ratio_Step_der1_B_unit"
    for kind in PlotKind:
        for order in (0, 1, 2):
            for axis in Axis:
                for physical in (False, True):
                    key = level_key(kind, order, axis, physical)
                    expected = (kind, order, axis, physical) if order else (kind, 0)
                    assert parse_level_key(key)[: len(expected)] == expected
    k = parse_level_key("Ratio_AVR_der2_E_unit")
    assert (k.kind, k.order, k.axis_is_energy, k.physical) == (PlotKind.AVERAGE, 2, True, True)
    assert not parse_level_key("Ratio_der1_B").axis_is_energy
    with pytest.raises(ValueError):
        parse_level_key("der1")  # the keys of phase 2 are not read any more


def test_view_state_converts_only_unit_dependent_values():
    view = ViewState(
        field_range=(1.0, 2.0),
        energy_range=(100.0, None),
        levels={"Ratio": (0.9, 1.1), "Ratio_der1_E": (-1.0, 1.0),
                "Data_der2_E_unit": (-1.0, 2.0), "Ratio_der1_B_unit": (-3.0, 3.0)},
        stacked_range=(0.5, 2.0),
    )  # fmt: skip
    mev = view.converted(Unit.CM1, Unit.MEV)
    assert mev.field_range == (1.0, 2.0) and mev.stacked_range == (0.5, 2.0)
    assert mev.energy_range == pytest.approx((100 / MEV, None))
    assert mev.levels["Ratio"] == (0.9, 1.1) and mev.levels["Ratio_der1_E"] == (-1.0, 1.0)
    assert mev.levels["Ratio_der1_B_unit"] == (-3.0, 3.0)
    assert mev.levels["Data_der2_E_unit"] == pytest.approx((-(MEV**2), 2 * MEV**2))
    back = mev.converted(Unit.MEV, Unit.CM1)
    assert back.energy_range == pytest.approx(view.energy_range)
    assert back.levels["Data_der2_E_unit"] == pytest.approx((-1.0, 2.0))
    # the stacked intensities follow the selection shown: only per-unit energy derivatives
    per_unit = PlotSelection(order=2, axis=Axis.ENERGY, physical=True)
    shown = view.converted(Unit.CM1, Unit.MEV, per_unit)
    assert shown.stacked_range == pytest.approx((0.5 * MEV**2, 2 * MEV**2))
    assert shown.stacked_offset == pytest.approx(0.01 * MEV**2)
    for selection in (PlotSelection(order=2), PlotSelection(order=1, axis=Axis.FIELD)):
        same = view.converted(Unit.CM1, Unit.THZ, selection)
        assert same.stacked_range == (0.5, 2.0) and same.stacked_offset == 0.01


def test_view_state_levels_and_colours():
    view = ViewState(levels={"Ratio": (0.9, 1.1), "Data": (0.0, 2.0)})
    assert view.levels_for("Ratio") == (0.9, 1.1)  # ratios start at fixed levels
    assert view.levels_for("Data") is None  # raw data starts on Auto
    assert view.levels_for("Ratio_der1_E") is None  # nothing kept yet
    auto = ViewState(levels={"Ratio": (0.9, 1.1)}, level_modes={"Ratio": "auto"})
    assert auto.levels_for("Ratio") is None
    assert ViewState().levels["Ratio_Step"] == (0.98, 1.02)
    assert view.colormap_for(0) == "magma" and view.colormap_for(1) == "grey"
    assert ViewState(colormap="viridis").colormap_for(2) == "viridis"
    assert view.trace_colormap() == "viridis"  # Auto colours the stacked fields with viridis
    assert ViewState(colormap="grey").trace_colormap() == "grey"


def test_selection_names():
    s = PlotSelection(kind=PlotKind.DATA, order=1, axis=Axis.ENERGY, physical=True)
    assert s.export_name() == "Data_1stDer_perUnit"
    assert s.description(Unit.THZ) == "Data · 1st derivative d/dE per THz"
    assert PlotSelection(physical=True).export_name() == "Ratio"  # per unit needs an order
    field = PlotSelection(order=2, axis=Axis.FIELD)
    assert field.description(Unit.MEV) == "R(B)/R(0) · 2nd derivative d/dB per point"


def test_field_range():
    np.testing.assert_allclose(FieldRange(1.0, 0.5, 2.0).values(), [1.0, 1.5, 2.0])
    with pytest.raises(ValueError, match="enter a number for the step"):
        FieldRange(1.0, None, 2.0).values()
    with pytest.raises(ValueError, match="positive"):
        FieldRange(1.0, 0.0, 2.0).values()


def test_unit_switch_converts_the_view_except_while_restoring(controller):
    changes = []
    controller.unitChanged.connect(lambda old, new: changes.append((old, new)))
    controller.set_view(energy_range=(100.0, 200.0), levels={"Ratio_der1_E_unit": (-1.0, 1.0)})
    controller.set_unit("meV")
    assert changes == [(Unit.CM1, Unit.MEV)]
    assert controller.view.energy_range == pytest.approx((100 / MEV, 200 / MEV))
    assert controller.view.levels["Ratio_der1_E_unit"] == pytest.approx((-MEV, MEV))
    restored = []
    controller.restored.connect(lambda: restored.append(True))
    with controller.restoring():
        assert controller.is_restoring()
        controller.set_view(energy_range=(10.0, 20.0))  # restored in the restored unit
        controller.set_unit("THz")
        assert not restored
    assert restored == [True]
    assert controller.unit is Unit.THZ
    assert controller.view.energy_range == (10.0, 20.0)  # not converted


def test_changed_since_process_compares_with_the_options_used(controller, sweep):
    flags = []
    controller.changedSinceProcess.connect(flags.append)
    files = SweepFiles(tuple(sweep["zero"]), tuple(reversed(sweep["field"])))
    controller.set_processing(sample_files=files)
    assert controller.processing.sample_files.field == tuple(map(str, sweep["field"]))  # sorted
    assert not controller.changed_since_process()  # nothing processed yet
    controller.process()
    controller.set_processing(smooth=True)  # no reference: smoothing changes nothing
    assert not controller.changed_since_process()
    controller.set_processing(reference_mode=ReferenceMode.SELF)
    controller.set_processing(reference_mode=ReferenceMode.NONE)
    assert flags == [True, False]


def test_effective_processing_state_drops_unused_options():
    plain = ProcessingState()
    unused = dict(
        smooth=True,
        sg_window=21,
        sample_field=FieldRange(1.0, 1.0, 2.0),
        reference_field=FieldRange(1.0, 1.0, 2.0),
        reference_files=SweepFiles(("zero",), ("field",)),
    )
    assert dataclasses.replace(plain, **unused).effective() == plain.effective()
    separate = ProcessingState(reference_mode=ReferenceMode.SEPARATE, custom_field=True)
    for name, value in unused.items():
        if name != "sg_window":  # used only with smoothing
            changed = dataclasses.replace(separate, **{name: value})
            assert changed.effective() != separate.effective(), name
    smoothed = ProcessingState(reference_mode=ReferenceMode.SELF, smooth=True)
    assert dataclasses.replace(smoothed, sg_window=21).effective() != smoothed.effective()
    files = dataclasses.replace(smoothed, reference_files=SweepFiles(("zero",), ("field",)))
    assert files.effective() == smoothed.effective()


def test_library_maps_do_not_count_as_processed(controller, sweep):
    controller.set_processing(sample_files=SweepFiles(tuple(sweep["zero"]), tuple(sweep["field"])))
    result = controller.process()
    processed_at = controller.processed_at
    controller.set_processing(energy_cut=(200.0, 800.0))
    assert controller.changed_since_process()
    controller.from_map(result.ratio)
    assert controller.result_source == "library" and controller.processed_at == processed_at
    assert controller.changed_since_process()  # the sweep still needs processing
    controller.process()
    assert controller.result_source == "process" and not controller.changed_since_process()


def test_points_are_kept_in_cm1(controller, sweep):
    controller.set_processing(sample_files=SweepFiles(tuple(sweep["zero"]), tuple(sweep["field"])))
    controller.process()
    controller.set_unit("meV")
    controller.record_point(1.0, 40.0)
    np.testing.assert_allclose(controller.points.points("LL 1")[1], [40.0 * MEV])
    controller.set_curve(" ")
    with pytest.raises(ValueError, match="curve name") as info:
        controller.record_point(1.0, 40.0)
    assert info.value.panel == "points"


def test_user_action_reports_errors():
    reports = []

    class Window:
        def report_error(self, title, message, panel=None, expected=True):
            reports.append((title, message, panel, expected))

    @user_action("Thing")
    def fails(_window, exc):
        raise exc

    error = ValueError("bad value")
    error.panel = "library"
    assert fails(Window(), error) is None
    fails(Window(), RuntimeError("bug"))
    assert reports[0] == ("Thing", "bad value", "library", True)
    assert reports[1][0] == "Thing" and "RuntimeError('bug')" in reports[1][1]
    assert reports[1][3] is False
