import dataclasses
import itertools

import numpy as np
import pytest

import golden
from mag_opt_detective.core import processing as proc
from mag_opt_detective.core.pipeline import (
    PlotKind,
    ProcessOptions,
    ProcessResult,
    ReferenceMode,
    process,
)
from mag_opt_detective.core.points import PointTable
from mag_opt_detective.core.processing import Axis
from mag_opt_detective.core.readers import Measurement, load_measurement, sort_paths
from mag_opt_detective.core.spectra import FieldMap
from mag_opt_detective.core.units import Unit


def make_map(energy=None, field=None, values=None, unit="cm-1"):
    energy = np.linspace(0, 10, 11) if energy is None else energy
    field = np.array([1.0, 2.0, 3.0]) if field is None else field
    if values is None:
        values = np.add.outer(energy, field)
    return FieldMap(energy=energy, field=field, values=values, unit=unit)


def test_fieldmap_validates_shape():
    with pytest.raises(ValueError, match="does not match"):
        FieldMap(energy=np.arange(3), field=np.arange(2), values=np.zeros((2, 3)))


def test_zero_reference_single_and_drift():
    zero = np.array([[1.0], [2.0]])
    field = np.array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(proc.zero_reference(zero, field), [[1, 1, 1], [2, 2, 2]])
    zero2 = np.array([[1.0, 3.0]])
    np.testing.assert_allclose(proc.zero_reference(zero2, field), [[1.0, 2.0, 3.0]])


def test_ratio_to_zero_removes_drift(sweep):
    m = load_measurement(sweep["zero"], sweep["field"])
    ratio = proc.ratio_to_zero(m)
    # sample = (1 + 0.1 B) base, zero drifts from base (B=0.5) to 2 base (B=2.0)
    t = (sweep["fields"] - 0.5) / 1.5
    expected = (1 + 0.1 * sweep["fields"]) / (1 + t)
    np.testing.assert_allclose(ratio.values, np.broadcast_to(expected, ratio.values.shape), 1e-6)


def test_interpolate_field_extrapolates_linearly():
    fmap = make_map(field=np.array([1.0, 2.0]))
    out = proc.interpolate_field(fmap, np.array([0.0, 1.5, 3.0]))
    np.testing.assert_allclose(out.values, np.add.outer(fmap.energy, [0.0, 1.5, 3.0]))


def test_divide_aligns_energy_and_field():
    num = make_map(values=np.full((11, 3), 6.0))
    den = make_map(energy=np.linspace(0, 10, 21), field=np.array([1.0, 3.0]))
    den = den.with_values(np.full(den.values.shape, 2.0))
    np.testing.assert_allclose(proc.divide(num, den).values, 3.0)


def test_ratio_to_average():
    fmap = make_map(values=np.array([[1.0, 3.0]] * 11), field=np.array([1.0, 2.0]))
    np.testing.assert_allclose(proc.ratio_to_average(fmap).values, [[0.5, 1.5]] * 11)


def test_baseline_normalize_inclusive_region():
    fmap = make_map()
    out = proc.baseline_normalize(fmap, (2, 4))
    np.testing.assert_allclose(out.values[2:5].mean(axis=0), 1.0)
    with pytest.raises(ValueError, match="no data"):
        proc.baseline_normalize(fmap, (20, 30))


def test_derivative_axes():
    fmap = make_map(values=np.outer(np.arange(11.0) ** 2, [1.0, 1.0, 1.0]))
    d_energy = proc.derivative(fmap, Axis.ENERGY).values
    np.testing.assert_allclose(d_energy[1:-1, 0], 2 * np.arange(1, 10))
    np.testing.assert_allclose(proc.derivative(fmap, Axis.FIELD).values, 0)


def test_savgol_validation_and_identity_on_polynomial():
    fmap = make_map()
    np.testing.assert_allclose(proc.savgol(fmap, 5, 2).values, fmap.values, atol=1e-12)
    with pytest.raises(ValueError, match="odd"):
        proc.savgol(fmap, 4, 2)
    with pytest.raises(ValueError, match="longer"):
        proc.savgol(fmap, 31, 2)


def test_crop_and_merge_energy():
    low = make_map(energy=np.linspace(0, 10, 11))
    high = make_map(energy=np.linspace(8, 20, 7))
    merged = proc.merge_energy([(low, 0, 10), (high, 10, 20)])
    # every measured energy is kept (nothing re-gridded): steps of 1, then of 2
    np.testing.assert_allclose(merged.energy, [*range(11), 12, 14, 16, 18, 20])
    np.testing.assert_allclose(merged.values, np.add.outer(merged.energy, low.field))
    other_field = make_map(field=np.array([5.0, 6.0, 7.0]))
    with pytest.raises(ValueError, match="same field"):
        proc.merge_energy([(low, None, None), (other_field, None, None)])


def test_process_reference_modes(sweep):
    m = load_measurement(sweep["zero"], sweep["field"])
    plain = process(m)
    np.testing.assert_allclose(plain.data.values, m.spectra.values)
    assert plain.reference_data is None

    separate = process(m, m, ProcessOptions(reference_mode=ReferenceMode.SEPARATE))
    np.testing.assert_allclose(separate.data.values, 1.0)
    np.testing.assert_allclose(separate.ratio.values, 1.0)

    self_ref = process(m, options=ProcessOptions(reference_mode=ReferenceMode.SELF))
    np.testing.assert_allclose(self_ref.ratio.values, 1.0)

    with pytest.raises(ValueError, match="no reference"):
        process(m, None, ProcessOptions(reference_mode=ReferenceMode.SEPARATE))


def test_process_baseline_and_derivatives(sweep):
    m = load_measurement(sweep["zero"], sweep["field"])
    res = process(m, options=ProcessOptions(baseline_region=(100, 1000)))
    np.testing.assert_allclose(res.ratio.values.mean(axis=0), 1.0)
    np.testing.assert_allclose(res.average.values.mean(axis=0), 1.0)
    second = res.get(PlotKind.RATIO, order=2, axis=Axis.FIELD)
    expected = proc.derivative(proc.derivative(res.ratio, Axis.FIELD), Axis.FIELD)
    np.testing.assert_allclose(second.values, expected.values)


def test_result_from_map():
    fmap = make_map()
    res = ProcessResult.from_map(fmap, baseline_region=(0, 2))
    assert res.data is fmap
    np.testing.assert_allclose(res.ratio.values[:3].mean(axis=0), 1.0)


def test_real_reference_correction(data_dir):
    """Sample and reference measured on different field grids (0.25 T vs 0.5 T steps)."""

    def split(folder):
        files = sort_paths(folder.glob("*.txt"))
        zero = [f for f in files if "_a00p000T_a" in f]
        return zero, [f for f in files if f not in zero]

    sam = load_measurement(*split(data_dir / "TR" / "Sam1" / "txt_files"))
    ref = load_measurement(*split(data_dir / "TR" / "Ref1" / "txt_files"))
    assert sam.spectra.field.size == 2 * ref.spectra.field.size
    opts = ProcessOptions(reference_mode=ReferenceMode.SEPARATE, smooth_reference=True)
    res = process(sam, ref, opts)
    np.testing.assert_allclose(res.reference_ratio.field, sam.spectra.field)
    assert res.ratio.values.shape == sam.spectra.values.shape
    assert res.ratio.unit is res.reference_data.unit is Unit.CM1
    shown = res.get(PlotKind.RATIO, unit=Unit.MEV)
    np.testing.assert_allclose(shown.energy, sam.spectra.energy / 8.0656)
    assert shown.values is res.ratio.values


def test_physical_derivative_on_non_uniform_grids():
    energy = np.linspace(1.0, 3.0, 400) ** 2  # non-uniform spacing
    field = np.array([0.5, 1.0, 2.0, 4.0])
    values = np.sin(energy)[:, None] * field[None, :]
    fmap = make_map(energy=energy, field=field, values=values)

    d_energy = proc.derivative(fmap, Axis.ENERGY, physical=True).values
    expected = np.cos(energy)[:, None] * field[None, :]
    np.testing.assert_allclose(d_energy[1:-1], expected[1:-1], atol=2e-3)

    d_field = proc.derivative(fmap, Axis.FIELD, physical=True).values
    np.testing.assert_allclose(d_field, np.broadcast_to(np.sin(energy)[:, None], values.shape))

    per_point = proc.derivative(fmap, Axis.FIELD).values
    np.testing.assert_allclose(per_point[:, 0], d_field[:, 0] * 0.5)


def test_physical_derivative_rejects_repeated_axis():
    fmap = make_map(field=np.array([1.0, 1.0, 2.0]))
    with pytest.raises(ValueError, match="repeated"):
        proc.derivative(fmap, Axis.FIELD, physical=True)


def test_result_get_physical(sweep):
    m = load_measurement(sweep["zero"], sweep["field"])
    res = process(m)
    got = res.get(PlotKind.DATA, order=1, axis=Axis.FIELD, physical=True)
    expected = proc.derivative(res.data, Axis.FIELD, physical=True)
    np.testing.assert_allclose(got.values, expected.values)


def assert_scaled(got: FieldMap, cm1: FieldMap, factor: float, base: FieldMap) -> None:
    """*got* equals *cm1* times *factor*, up to rounding of the *base* values."""
    atol = 1e-11 * np.abs(base.values).max() * factor
    np.testing.assert_allclose(got.values, cm1.values * factor, rtol=1e-9, atol=atol)


@pytest.mark.parametrize("unit", [Unit.MEV, Unit.THZ])
@pytest.mark.parametrize("kind", [PlotKind.RATIO, PlotKind.DATA, PlotKind.STEP])
def test_result_get_in_unit(sweep, unit, kind):
    """d/dE per unit scales with CM1_PER_UNIT ** order; d/dB and per-point do not."""
    res = process(load_measurement(sweep["zero"], sweep["field"]))
    data = res.get(PlotKind.DATA, 1, Axis.ENERGY, physical=True, unit=unit).values
    assert np.abs(data).max() > 1e-3  # a real slope, so the scaling below is visible
    k = {Unit.MEV: 8.0656, Unit.THZ: 33.35641}[unit]
    plain = res.get(kind, unit=unit)
    assert plain.unit is unit
    np.testing.assert_allclose(plain.energy, res.base(kind).energy / k)
    np.testing.assert_array_equal(plain.values, res.base(kind).values)
    for order in (1, 2):
        per_unit = res.get(kind, order, Axis.ENERGY, physical=True, unit=unit)
        assert per_unit.unit is unit
        cm1 = res.get(kind, order, Axis.ENERGY, physical=True)
        assert_scaled(per_unit, cm1, k**order, plain)
        for axis, physical in ((Axis.ENERGY, False), (Axis.FIELD, True), (Axis.FIELD, False)):
            got = res.get(kind, order, axis, physical=physical, unit=unit)
            expected = res.get(kind, order, axis, physical=physical)
            np.testing.assert_array_equal(got.values, expected.values)  # not scaled at all


def test_process_and_from_map_keep_cm1(sweep):
    m = load_measurement(sweep["zero"], sweep["field"])
    mev = Measurement(spectra=m.spectra.to_unit(Unit.MEV), zero=m.zero)
    region = (300.0, 600.0)  # cm-1
    expected = process(m, m, ProcessOptions(ReferenceMode.SEPARATE, baseline_region=region))
    res = process(mev, mev, ProcessOptions(ReferenceMode.SEPARATE, baseline_region=region))
    for name in ("data", "ratio", "average", "step", "reference_data", "reference_ratio"):
        got, ref = getattr(res, name), getattr(expected, name)
        assert got.unit is Unit.CM1, name
        np.testing.assert_allclose(got.energy, ref.energy, rtol=1e-12)
        np.testing.assert_allclose(got.values, ref.values, rtol=1e-12)

    wrapped = ProcessResult.from_map(expected.data.to_unit(Unit.THZ), baseline_region=region)
    direct = ProcessResult.from_map(expected.data, baseline_region=region)
    assert wrapped.ratio.unit is Unit.CM1
    np.testing.assert_allclose(wrapped.ratio.values, direct.ratio.values, rtol=1e-12)
    np.testing.assert_allclose(wrapped.step.values, direct.step.values, rtol=1e-12)


def test_crop_field():
    fmap = make_map(field=np.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(proc.crop_field(fmap, 2, None).field, [2, 3])
    with pytest.raises(ValueError, match="no data"):
        proc.crop_field(fmap, 5, 6)


def test_merge_field_joins_sorts_and_averages():
    energy = np.linspace(0, 10, 11)
    low = make_map(energy=energy, field=np.array([1.0, 2.0, 3.0]))
    high = make_map(energy=energy, field=np.array([3.0, 4.0, 5.0]))
    high = high.with_values(high.values + 1.0)  # makes the 3 T overlap visible
    merged = proc.merge_field([(high, None, None), (low, None, None)])
    np.testing.assert_allclose(merged.field, [1, 2, 3, 4, 5])
    np.testing.assert_allclose(merged.values[:, 2], energy + 3 + 0.5)  # mean of both 3 T
    np.testing.assert_allclose(merged.values[:, 4], energy + 5 + 1.0)

    cut = proc.merge_field([(low, None, 2.5), (high, 3.5, None)])
    np.testing.assert_allclose(cut.field, [1, 2, 4, 5])


def test_merge_field_interpolates_to_common_energy():
    a = make_map(energy=np.linspace(0, 10, 11), field=np.array([1.0]))
    b = make_map(energy=np.linspace(2, 12, 21), field=np.array([2.0]))
    merged = proc.merge_field([(a, None, None), (b, None, None)])
    np.testing.assert_allclose(merged.energy, np.linspace(2, 10, 9))
    np.testing.assert_allclose(merged.values, np.add.outer(merged.energy, [1.0, 2.0]))
    with pytest.raises(ValueError, match="common energy"):
        proc.merge_field([(a, None, None), (make_map(energy=np.linspace(20, 30, 5)), None, None)])
    with pytest.raises(ValueError, match="cannot merge"):
        proc.merge_field([(a, None, None), (make_map(unit="meV"), None, None)])


def test_average_maps():
    a = make_map(values=np.full((11, 3), 1.0))
    b = make_map(energy=np.linspace(0, 10, 21), values=np.full((21, 3), 3.0))
    avg = proc.average_maps([a, b])
    np.testing.assert_allclose(avg.energy, a.energy)
    np.testing.assert_allclose(avg.values, 2.0)
    assert proc.average_maps([a]) is a
    with pytest.raises(ValueError, match="same field"):
        proc.average_maps([a, make_map(field=np.array([1.0, 2.0, 4.0]))])
    with pytest.raises(ValueError, match="cannot average"):
        proc.average_maps([a, make_map(unit="THz")])


def test_step_ratio():
    fmap = make_map(field=np.array([1.0, 2.0, 4.0]), values=np.array([[1.0, 2.0, 8.0]] * 11))
    step = proc.step_ratio(fmap)
    np.testing.assert_allclose(step.field, [2.0, 4.0])
    np.testing.assert_allclose(step.values, [[2.0, 4.0]] * 11)
    with pytest.raises(ValueError, match="two field"):
        proc.step_ratio(make_map(field=np.array([1.0]), values=np.ones((11, 1))))


def test_process_step_ratio(sweep):
    m = load_measurement(sweep["zero"], sweep["field"])
    res = process(m, options=ProcessOptions(baseline_region=(100, 1000)))
    step = res.get(PlotKind.STEP)
    np.testing.assert_allclose(step.field, sweep["fields"][1:])
    np.testing.assert_allclose(step.values.mean(axis=0), 1.0)  # baseline applied
    single = make_map(field=np.array([1.0]), values=np.ones((11, 1)))
    with pytest.raises(ValueError, match="two field"):
        ProcessResult.from_map(single).get(PlotKind.STEP)
    np.testing.assert_allclose(ProcessResult.from_map(make_map()).step.field, [2.0, 3.0])


@pytest.mark.parametrize("mode", [ReferenceMode.NONE, ReferenceMode.SELF])
def test_with_baseline_equals_processing_again(tmp_path, mode):
    """Another region applied to a result gives exactly the maps of processing with it."""
    sample = load_measurement(*golden.write_sweep(tmp_path))
    options = ProcessOptions(reference_mode=mode, smooth_reference=True, sg_window=5)
    first = process(sample, options=dataclasses.replace(options, baseline_region=(450, 550)))
    region = (900.0, 1100.0)
    again = first.with_baseline(region)
    fresh = process(sample, options=dataclasses.replace(options, baseline_region=region))
    assert again.baseline_region == region and again.before_baseline is first.before_baseline
    for kind, order in itertools.product(PlotKind, (0, 1)):
        np.testing.assert_array_equal(again.get(kind, order).values, fresh.get(kind, order).values)
    assert again.data is first.data  # never normalised, as the reference maps
    assert again.reference_ratio is first.reference_ratio
    mask = (again.ratio.energy >= 900) & (again.ratio.energy <= 1100)
    np.testing.assert_allclose(again.ratio.values[mask].mean(axis=0), 1.0)

    plain = process(sample, options=options)
    off = again.with_baseline(None)
    assert off is first.before_baseline and off.baseline_region is None
    np.testing.assert_array_equal(off.ratio.values, plain.ratio.values)
    np.testing.assert_array_equal(off.step.values, plain.step.values)
    assert plain.with_baseline(None) is plain
    np.testing.assert_array_equal(plain.with_baseline(region).average.values, fresh.average.values)
    with pytest.raises(ValueError, match="contains no data"):
        first.with_baseline((2000, 3000))


def test_with_baseline_needs_the_maps_before_it():
    fmap = make_map()
    res = ProcessResult.from_map(fmap, baseline_region=(0, 2))
    assert res.baseline_region == (0.0, 2.0) and res.before_baseline.ratio is fmap
    assert res.ratio is res.average  # normalised once for both
    np.testing.assert_allclose(res.with_baseline((8, 10)).ratio.values[-3:].mean(axis=0), 1.0)
    bare = ProcessResult(data=fmap, ratio=res.ratio, average=res.ratio, baseline_region=(0, 2))
    with pytest.raises(ValueError, match="before the baseline"):
        bare.with_baseline((8, 10))


@pytest.fixture(scope="module")
def golden_maps(tmp_path_factory) -> dict[str, FieldMap]:
    return golden.exports(tmp_path_factory.mktemp("golden"))


@pytest.mark.parametrize("name", golden.MAPS)
def test_golden_exports(golden_maps, name):
    golden.assert_matches(golden_maps[name], name)


def test_golden_points(golden_maps, tmp_path):
    """The golden points file has no unit in its header (it is in meV)."""
    path = golden.GOLDEN_DIR / golden.POINTS
    expected = PointTable.load_tsv(path, default_unit=Unit.MEV)
    table = golden.points(golden_maps["Ratio"])  # cm-1
    assert table.names == expected.names == ["peak", "single"]
    np.testing.assert_allclose(table.field, expected.field)
    for name in table.names:
        np.testing.assert_allclose(table.column(name), expected.column(name), rtol=1e-9)

    out = tmp_path / "points.tsv"
    table.save_tsv(out, unit=Unit.MEV)
    assert out.read_text().splitlines()[0] == "Energy (meV)\tpeak\tsingle"
    assert path.read_text().splitlines()[0] == "\tpeak\tsingle"

    def cells(p):
        return np.genfromtxt(p, delimiter="\t", skip_header=1)

    np.testing.assert_allclose(cells(out), cells(path), rtol=1e-9)


def test_common_energy_keeps_samples_of_rounded_tables():
    a = make_map(energy=np.linspace(100, 200, 11))
    b = a.replace(energy=a.energy * (1 + 1e-12))  # read back from a 12-digit table
    assert proc.average_maps([a, b]).energy.size == 11
    shifted = b.replace(field=b.field + 3)
    assert proc.merge_field([(a, None, None), (shifted, None, None)]).energy.size == 11


def test_merge_energy_takes_each_energy_from_one_map():
    """An overlap is split at its middle: no samples of the two maps interleaved."""
    low = make_map(energy=np.arange(0.0, 10.01, 0.5), values=np.ones((21, 3)))
    high = make_map(energy=np.arange(6.0, 20.01, 2.0), values=np.full((8, 3), 2.0))
    merged = proc.merge_energy([(high, None, None), (low, None, None)])
    # the measured energies of each side of the middle of the overlap 6 - 10
    np.testing.assert_allclose(merged.energy, [*np.arange(0.0, 8.01, 0.5), 10, 12, 14, 16, 18, 20])
    below = merged.energy <= 8.0
    np.testing.assert_allclose(merged.values[below], 1.0)
    above = merged.energy >= 10.0
    np.testing.assert_allclose(merged.values[above], 2.0)
    (seam,) = proc.energy_seams([(high, None, None), (low, None, None)])
    assert (seam.lower, seam.upper, seam.kind, seam.middle) == (1, 0, "overlap", 8.0)


def test_merge_energy_leaves_a_gap_empty():
    low = make_map(energy=np.linspace(0, 10, 11))
    high = make_map(energy=np.linspace(20, 30, 11))
    merged = proc.merge_energy([(low, None, None), (high, None, None)])
    gap = (merged.energy > 10) & (merged.energy < 20)
    # one empty sample a step beyond each side marks the gap
    np.testing.assert_allclose(merged.energy[gap], [11, 19])
    assert np.isnan(merged.values[gap]).all()
    np.testing.assert_allclose(merged.values[~gap], np.add.outer(merged.energy[~gap], low.field))
    coarse = make_map(energy=np.linspace(20, 30, 3))  # steps of 5: the gap is two steps
    merged = proc.merge_energy(
        [(make_map(energy=np.linspace(0, 10, 3)), None, None), (coarse, None, None)]
    )
    np.testing.assert_allclose(merged.energy[(merged.energy > 10) & (merged.energy < 20)], [15])
    seams = proc.energy_seams([(low, None, None), (high, None, None)])
    assert [(s.kind, s.lo, s.hi) for s in seams] == [("gap", 10.0, 20.0)]
    touching = make_map(energy=np.linspace(11, 21, 11))
    assert proc.energy_seams([(low, None, None), (touching, None, None)])[0].kind == "touch"


def test_merge_energy_refuses_a_map_inside_another():
    whole = make_map(energy=np.linspace(0, 10, 11))
    part = make_map(energy=np.linspace(2, 8, 7))
    for parts in ([(whole, None, None), (part, None, None)], [(whole, None, None)] * 2):
        assert proc.energy_seams(parts)[0].kind == "inside"
        with pytest.raises(ValueError, match="inside"):
            proc.merge_energy(parts)
    # cut to its limits, it is a neighbour
    merged = proc.merge_energy([(whole, None, 5), (part, 5, None)])
    assert merged.energy[-1] == 8


def test_baseline_normalize_skips_missing_values():
    values = np.add.outer(np.linspace(0, 10, 11), [1.0, 2.0, 3.0])
    values[3] = np.nan  # e.g. the gap of a map merged by energy
    out = proc.baseline_normalize(make_map(values=values), (2, 4))
    np.testing.assert_allclose(np.nanmean(out.values[2:5], axis=0), 1.0)
    assert np.isnan(out.values[3]).all() and np.isfinite(out.values[[2, 4]]).all()


def test_baseline_region_without_values_contains_no_data():
    values = np.add.outer(np.linspace(0, 10, 11), [1.0, 2.0, 3.0])
    values[2:5] = np.nan  # the region lies in a gap
    with pytest.raises(ValueError, match="contains no data"):
        proc.baseline_normalize(make_map(values=values), (2, 4))


def test_interpolation_keeps_missing_values_local():
    values = np.add.outer(np.linspace(0, 10, 11), [1.0, 2.0])
    values[4] = np.nan
    out = proc.interp_linear(
        np.array([1.5, 3.5, 4.5, 6.5, 12.0]), np.linspace(0, 10, 11), values, 0
    )
    assert np.isnan(out[1:3]).all()
    np.testing.assert_allclose(out[[0, 3, 4]], np.add.outer([1.5, 6.5, 12.0], [1.0, 2.0]))


def test_maps_with_a_gap_merge_and_average_again():
    """A merged map with an empty gap (or a table with empty cells) can be combined again; its
    missing values stay missing, and only those."""
    low = make_map(energy=np.linspace(0, 10, 11))
    high = make_map(energy=np.linspace(20, 30, 11))
    gapped = proc.merge_energy([(low, None, None), (high, None, None)])
    gap = np.isnan(gapped.values[:, 0])
    np.testing.assert_allclose(gapped.energy[gap], [11, 19])
    other = make_map(energy=np.linspace(0, 30, 61), field=np.array([4.0, 5.0, 6.0]))
    # on the gapped map's energies: the empty samples stay empty, the other map is whole
    merged = proc.merge_field([(gapped, None, None), (other, None, None)])
    empty = np.isnan(merged.values)
    np.testing.assert_allclose(merged.energy[empty.any(axis=1)], [11, 19])
    assert empty[:, :3].sum() == 6 and not empty[:, 3:].any()
    # the gapped map put on another map's energies: missing only inside the gap
    merged = proc.merge_field([(other, None, None), (gapped, None, None)])
    missing = merged.energy[np.isnan(merged.values[:, 0])]
    assert missing.min() > 10 and missing.max() < 20 and missing.size == 19  # 10.5 ... 19.5
    kept = (merged.energy <= 10) | (merged.energy >= 20)
    np.testing.assert_allclose(
        merged.values[kept, :3], np.add.outer(merged.energy[kept], [1.0, 2.0, 3.0])
    )
    averaged = proc.average_maps([gapped, make_map(energy=np.linspace(0, 30, 31))])
    np.testing.assert_allclose(averaged.energy[np.isnan(averaged.values[:, 0])], [11, 19])
    assert np.isfinite(averaged.values[~np.isin(averaged.energy, [11, 19])]).all()


def test_per_point_derivative_on_an_even_axis_counts_samples():
    """Unchanged on an even axis, also one that is even only within rounding."""
    for energy in (np.linspace(0, 10, 11), np.linspace(0, 10, 11) + 1e-9 * np.arange(11) ** 2):
        fmap = make_map(energy=energy, values=np.sin(np.add.outer(energy, [1.0, 2.0, 3.0])))
        out = proc.derivative(fmap, Axis.ENERGY).values
        assert np.array_equal(out, np.gradient(fmap.values, axis=0))
    assert proc.typical_step(np.linspace(0, 10, 11)) is None
    assert proc.typical_step(np.array([0.0, 1.0, 2.0, 2.5, 3.0, 3.5])) == 0.5
    assert proc.typical_step(np.array([3.0, 2.0, 1.0, 0.0, -2.0])) == -1.0  # as it runs
    assert proc.typical_step(np.array([0.0, 1.0, 1.0, 2.0])) is None  # repeated


def test_per_point_derivative_keeps_merged_parts_comparable():
    """A line sampled finely in one part and coarsely in another has (about) the same per-point
    derivatives in both, of first and second order."""
    field = np.array([1.0, 2.0])

    def line(energy, centre):
        dip = 0.05 * np.exp(-0.5 * ((energy - centre) / 4.0) ** 2)
        return make_map(energy=energy, field=field, values=np.ones((energy.size, 2)) - dip[:, None])

    fine, coarse = np.arange(30.0, 500.001, 0.25), np.arange(400.0, 3000.001, 1.0)
    merged = proc.merge_energy(
        [(line(fine, 200.0), None, None), (line(coarse, 1000.0), None, None)]
    )
    result = ProcessResult.from_map(merged)
    near = (
        (merged.energy > 150) & (merged.energy < 250),
        (merged.energy > 950) & (merged.energy < 1050),
    )
    for order in (1, 2):
        d = result.get(PlotKind.RATIO, order).values[:, 0]
        ratio = np.abs(d[near[0]]).max() / np.abs(d[near[1]]).max()
        assert ratio == pytest.approx(1.0, rel=0.1), order  # (was 0.25 and 0.07)
        per_unit = result.get(PlotKind.RATIO, order, physical=True).values[:, 0]
        np.testing.assert_allclose(d, per_unit * 1.0**order)  # the typical step: 1 cm^-1
