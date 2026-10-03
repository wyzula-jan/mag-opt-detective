import numpy as np
import pytest

from mag_opt_detective.core.pipeline import ProcessResult
from mag_opt_detective.core.processing import step_ratio
from mag_opt_detective.core.spectra import (
    FieldMap,
    cell_edges,
    energy_mask,
    field_label,
    load_tsv,
    sample_at,
    save_tsv,
)
from mag_opt_detective.core.units import (
    CM1_PER_UNIT,
    Unit,
    axis_label,
    convert,
    convert_levels,
    convert_range,
    derivative_scale,
    from_cm1,
    parse_axis_label,
    to_cm1,
)


def test_units():
    np.testing.assert_allclose(convert([8.0656], Unit.CM1, Unit.MEV), [1.0])
    np.testing.assert_allclose(convert([1.0], Unit.THZ, Unit.CM1), [33.35641])
    assert parse_axis_label(axis_label(Unit.MEV)) is Unit.MEV
    assert parse_axis_label("") is None
    assert parse_axis_label("Energy") is None
    assert parse_axis_label(" Energy (cm^-1) ") is Unit.CM1
    assert parse_axis_label("E (thz)") is Unit.THZ
    with pytest.raises(ValueError, match="unknown energy unit 'eV'"):
        parse_axis_label("Energy (eV)")


def test_to_and_from_cm1():
    x = np.array([100.0, 806.56, 3335.641])
    np.testing.assert_allclose(from_cm1(x, Unit.MEV), [100 / 8.0656, 100.0, 3335.641 / 8.0656])
    np.testing.assert_allclose(from_cm1(x, "THz"), x / 33.35641)
    np.testing.assert_allclose(to_cm1([1.0, 2.0], Unit.MEV), [8.0656, 16.1312])
    np.testing.assert_allclose(to_cm1(1.0, Unit.THZ), 33.35641)
    for unit in Unit:
        np.testing.assert_allclose(to_cm1(from_cm1(x, unit), unit), x, rtol=1e-15)
    # the same unit keeps values bit for bit and returns a copy
    same = convert(x, Unit.MEV, Unit.MEV)
    assert same is not x
    np.testing.assert_array_equal(same, x)


def test_convert_range():
    assert convert_range(None, Unit.MEV, Unit.CM1) is None
    lo, hi = convert_range((10.0, 20.0), Unit.MEV, Unit.CM1)
    assert (lo, hi) == pytest.approx((80.656, 161.312))
    assert isinstance(lo, float)
    assert convert_range((None, 20.0), Unit.MEV, Unit.CM1) == pytest.approx((None, 161.312))
    assert convert_range((None, None), Unit.THZ, Unit.MEV) == (None, None)
    assert convert_range((3.0, 1.0), Unit.CM1, Unit.THZ) == pytest.approx(
        (3 / 33.35641, 1 / 33.35641)
    )  # the order is kept, not sorted
    assert convert_range((0.5, 7.25), Unit.THZ, Unit.THZ) == (0.5, 7.25)
    back = convert_range(convert_range((61.25, 99.0), Unit.MEV, Unit.THZ), Unit.THZ, Unit.MEV)
    assert back == pytest.approx((61.25, 99.0), rel=1e-14)


@pytest.mark.parametrize("unit", list(Unit))
def test_derivative_scale(unit):
    k = CM1_PER_UNIT[unit]
    assert derivative_scale(unit, 1, True, True) == pytest.approx(k)
    assert derivative_scale(unit, 2, True, True) == pytest.approx(k**2)
    assert derivative_scale(unit, 0, True, True) == 1.0
    assert derivative_scale(unit, 2, True, False) == 1.0  # per data point
    assert derivative_scale(unit, 2, False, True) == 1.0  # along B


def test_convert_levels():
    k = CM1_PER_UNIT[Unit.THZ] / CM1_PER_UNIT[Unit.MEV]  # meV per THz
    assert convert_levels(None, Unit.MEV, Unit.THZ, 1, True, True) is None
    assert convert_levels((-1.0, 2.0), Unit.MEV, Unit.THZ, 1, True, True) == pytest.approx(
        (-k, 2 * k)
    )
    assert convert_levels((-1.0, 2.0), Unit.MEV, Unit.THZ, 2, True, True) == pytest.approx(
        (-(k**2), 2 * k**2)
    )
    assert convert_levels((-1.0, 2.0), Unit.CM1, Unit.MEV, 1, True, True) == pytest.approx(
        (-8.0656, 16.1312)
    )
    for args in ((0, True, True), (1, True, False), (2, False, True)):
        assert convert_levels((0.9, 1.1), Unit.MEV, Unit.THZ, *args) == (0.9, 1.1)
    assert convert_levels((0.9, 1.1), Unit.THZ, Unit.THZ, 2, True, True) == (0.9, 1.1)


def test_tsv_round_trip(tmp_path):
    fmap = FieldMap(
        energy=np.linspace(1, 2, 5),
        field=np.array([0.25, 0.5, 16.0]),
        values=np.arange(15.0).reshape(5, 3) / 7,
        unit=Unit.MEV,
    )
    path = tmp_path / "out.csv"
    save_tsv(fmap, path)
    header = path.read_text().splitlines()[0]
    assert header == "Energy (meV)\t0.25T\t0.50T\t16.00T"
    back = load_tsv(path)
    assert back.unit is Unit.MEV
    np.testing.assert_allclose(back.field, fmap.field)
    np.testing.assert_allclose(back.energy, fmap.energy)
    np.testing.assert_allclose(back.values, fmap.values, rtol=1e-11)


def test_load_legacy_merged_export(tmp_path):
    """A table with an empty first header cell and CRLF line ends loads."""
    path = tmp_path / "merged.csv"
    path.write_bytes(b"\t0.25T\t0.50T\r\n0.0\t1.0\t2.0\r\n0.1\t1.5\t2.5\r\n")
    fmap = load_tsv(path, default_unit=Unit.MEV)
    assert fmap.unit is Unit.MEV
    np.testing.assert_allclose(fmap.values, [[1.0, 2.0], [1.5, 2.5]])


def test_load_tsv_errors(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("Energy (meV)\tfoo\n1\t2\n")
    with pytest.raises(ValueError, match="field"):
        load_tsv(path)
    assert field_label(1.0) == "1.00T"


@pytest.mark.parametrize(
    ("text", "message"),
    [
        (b"Energy (eV)\t0.25T\n1\t2\n", r"bad\.csv: unknown energy unit 'eV'"),
        (b"Energy (meV)\t0.25T\n1\t2\n1\t2\t3\n", r"(?s)bad\.csv: .*got 3 columns"),
        (b"Energy (meV)\t0.25T\n1\t\x80\n", r"bad\.csv: not a text file in UTF-8"),
    ],
)
def test_load_tsv_errors_name_the_file(tmp_path, text, message):
    path = tmp_path / "bad.csv"
    path.write_bytes(text)
    with pytest.raises(ValueError, match=message):
        load_tsv(path, default_unit=Unit.MEV)


def test_load_tsv_sorts_the_field_columns(tmp_path):
    """A table with falling field columns gives the same map, step ratio and d/dB."""
    energy = np.linspace(100.0, 200.0, 11)
    field = np.array([1.0, 2.0, 3.0, 4.0])
    rising = FieldMap(energy, field, 1 + 0.005 * field[None, :] + 0 * energy[:, None])
    falling = FieldMap(energy, field[::-1], rising.values[:, ::-1])
    save_tsv(rising, tmp_path / "up.csv")
    save_tsv(falling, tmp_path / "down.csv")
    up, down = load_tsv(tmp_path / "up.csv"), load_tsv(tmp_path / "down.csv")
    np.testing.assert_array_equal(down.field, field)
    np.testing.assert_allclose(down.values, up.values)
    np.testing.assert_allclose(step_ratio(down).values, step_ratio(up).values)
    # a map added in falling order (not from a file) is sorted when it is shown
    shown = ProcessResult.from_map(falling)
    np.testing.assert_array_equal(shown.ratio.field, field)
    assert (shown.step.values > 1).all()  # R(B)/R(B - dB), not R(B)/R(B + dB)


def test_cells_of_a_sample_reach_the_midpoints():
    axis = np.array([0.5, 1.0, 1.25, 1.5])
    np.testing.assert_allclose(cell_edges(axis), [0.25, 0.75, 1.125, 1.375, 1.625])
    np.testing.assert_allclose(cell_edges(np.array([2.0])), [1.5, 2.5])
    assert [sample_at(axis, x) for x in (0.25, 0.74, 0.75, 1.2, 1.625)] == [0, 0, 1, 2, 3]
    assert sample_at(axis, 0.2) is None and sample_at(axis, 1.7) is None
    assert sample_at(axis[::-1], 0.6) == 3  # any order: the index in *axis*
    assert sample_at(np.array([1.0, 2.0, 1.0]), 1.1) == 0  # a repeated value: its first
    assert sample_at(axis, float("nan")) is None


def make_cm1_map() -> FieldMap:
    rng = np.random.default_rng(1)
    return FieldMap(
        energy=np.linspace(400.0, 4000.0, 3601),
        field=np.array([0.0, 0.5, 1.0]),
        values=rng.normal(size=(3601, 3)),
    )


def test_to_unit_round_trip():
    fmap = make_cm1_map()
    assert fmap.to_unit(Unit.CM1) is fmap
    mev = fmap.to_unit("meV")
    assert mev.unit is Unit.MEV
    assert mev.values is fmap.values  # values do not depend on the unit
    np.testing.assert_array_equal(mev.field, fmap.field)
    np.testing.assert_allclose(mev.energy, fmap.energy / 8.0656, rtol=1e-15)
    assert mev.to_unit(Unit.MEV) is mev
    for unit in (Unit.MEV, Unit.THZ):
        back = fmap.to_unit(unit).to_unit(Unit.CM1)
        assert back.unit is Unit.CM1
        np.testing.assert_allclose(back.energy, fmap.energy, rtol=1e-12, atol=0)
    np.testing.assert_allclose(
        fmap.to_unit(Unit.THZ).to_unit(Unit.MEV).energy, mev.energy, rtol=1e-12, atol=0
    )


def test_energy_mask_keeps_converted_limits():
    energy = np.linspace(400.0, 4000.0, 3601)
    np.testing.assert_array_equal(energy_mask(energy, None, None), True)
    assert energy_mask(energy, 500.0, 600.0).sum() == 101  # inclusive
    for unit in (Unit.MEV, Unit.THZ):
        # every sample, used as both limits after a round trip through *unit*, is kept
        limits = convert(convert(energy, Unit.CM1, unit), unit, Unit.CM1)
        kept = [energy_mask(energy, lim, lim).sum() for lim in limits]
        assert kept == [1] * energy.size
