import numpy as np
import pytest

from mag_opt_detective.core.spectra import FieldMap, field_label, load_tsv, save_tsv
from mag_opt_detective.core.units import Unit, axis_label, convert, parse_axis_label


def test_units():
    np.testing.assert_allclose(convert([8.0656], Unit.CM1, Unit.MEV), [1.0])
    np.testing.assert_allclose(convert([1.0], Unit.THZ, Unit.CM1), [33.35641])
    assert parse_axis_label(axis_label(Unit.MEV)) is Unit.MEV
    assert parse_axis_label("") is None
    assert parse_axis_label("Energy (eV)") is None


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
    """Merged legacy exports have an empty first header cell and CRLF line ends."""
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
