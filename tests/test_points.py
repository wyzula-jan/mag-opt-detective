import numpy as np
import pytest

from mag_opt_detective.core.points import PointTable
from mag_opt_detective.core.units import Unit


def test_set_and_clear_nearest():
    table = PointTable(np.array([0.25, 0.5, 0.75]))
    assert table.set_nearest("LL 1", 0.55, 12.5) == 1
    table.set_nearest("LL 1", 0.7, 13.0)
    b, e = table.points("LL 1")
    np.testing.assert_allclose(b, [0.5, 0.75])
    np.testing.assert_allclose(e, [12.5, 13.0])
    table.clear_nearest("LL 1", 0.5)
    np.testing.assert_allclose(table.points("LL 1")[0], [0.75])
    table.drop("LL 1")
    assert table.names == []


def test_a_value_never_lands_on_another_field():
    """Beyond half a field step past the first or last row, set_nearest refuses."""
    table = PointTable(np.array([0.5, 1.0, 4.0]))
    assert table.row_at(2.4) == 1 and table.row_at(2.6) == 2  # the gap splits at 2.5
    assert table.set_nearest("LL 1", 4.25, 700.0) == 2  # the last cell reaches 5.5
    assert table.row_at(0.25) == 0
    for b in (0.2, 5.6):
        with pytest.raises(ValueError, match=f"B = {b:g} T is outside the field rows"):
            table.set_nearest("LL 1", b, 720.0)
    np.testing.assert_allclose(table.points("LL 1")[1], [700.0])


def test_with_fields_keeps_every_point_on_its_field():
    table = PointTable(np.array([0.5, 1.0, 1.5]), {"LL 1": [10.0, np.nan, 30.0]})
    assert table.with_fields(np.array([1.0 + 1e-9, 0.5])) is table  # nothing new
    union = table.with_fields(np.array([2.0, 0.25, 1.0, 2.0]))
    np.testing.assert_allclose(union.field, [0.25, 0.5, 1.0, 1.5, 2.0])
    b, e = union.points("LL 1")
    np.testing.assert_allclose(b, [0.5, 1.5])
    np.testing.assert_allclose(e, [10.0, 30.0])
    assert union.set_nearest("LL 1", 2.0, 40.0) == 4
    np.testing.assert_allclose(table.field, [0.5, 1.0, 1.5])  # the old table is unchanged


def test_round_trip(tmp_path):
    table = PointTable(np.array([0.25, 0.5]))
    table.add_column("LL 1")
    table.set_nearest("LL 2", 0.5, 3.25)
    path = tmp_path / "points.csv"
    table.save_tsv(path)
    lines = ["Energy (cm-1)\tLL 1\tLL 2", "0.25\t\t", "0.5\t\t3.25"]
    assert path.read_text().splitlines() == lines
    back = PointTable.load_tsv(path, default_unit=Unit.MEV)  # the header wins
    assert back.names == ["LL 1", "LL 2"]
    np.testing.assert_allclose(back.field, table.field)
    np.testing.assert_allclose(back.column("LL 2"), [np.nan, 3.25])


@pytest.mark.parametrize("unit", [Unit.MEV, Unit.THZ])
def test_round_trip_in_unit(tmp_path, unit):
    table = PointTable(np.array([0.25, 0.5, 0.75]))  # energies in cm-1
    table.set_nearest("LL 1", 0.25, 806.56)
    table.set_nearest("LL 1", 0.75, 333.5641)
    path = tmp_path / "points.csv"
    table.save_tsv(path, unit=unit)
    header, first, *_ = path.read_text().splitlines()
    assert header == f"Energy ({unit})\tLL 1"
    assert first == ("0.25\t100" if unit is Unit.MEV else "0.25\t24.18006014")
    back = PointTable.load_tsv(path)
    np.testing.assert_allclose(back.column("LL 1"), table.column("LL 1"), rtol=1e-9)


def test_load_points_with_empty_column_name(tmp_path):
    path = tmp_path / "unitless.csv"
    path.write_text("\t\tLL 1\n0.25\t\t40.1\n0.5\t\t\n")
    table = PointTable.load_tsv(path)
    assert table.names == ["unnamed_1", "LL 1"]
    np.testing.assert_allclose(table.column("LL 1"), [40.1, np.nan])


@pytest.mark.parametrize(("unit", "factor"), [(Unit.MEV, 8.0656), (Unit.THZ, 33.35641)])
def test_points_without_unit_are_read_in_the_default_unit(tmp_path, unit, factor):
    """Files without a unit in the header (first cell empty) use *default_unit*."""
    path = tmp_path / "unitless.csv"
    path.write_bytes(b"\tLL 1\tLL 2\r\n0.25\t40.1\t\r\n0.5\t\t2\r\n")
    table = PointTable.load_tsv(path, default_unit=unit)
    np.testing.assert_allclose(table.column("LL 1"), [40.1 * factor, np.nan])
    np.testing.assert_allclose(table.column("LL 2"), [np.nan, 2 * factor])


@pytest.mark.parametrize(
    ("text", "message"),
    [
        (b"Energy (meV)\tLL 1\tLL 1\n0.5\t1\t2\n", "more than one curve is named 'LL 1'"),
        (b"Energy (eV)\tLL 1\n0.5\t1\n", "unknown energy unit 'eV'"),
        (b"Energy (meV)\tLL 1\n0.5\t1\n\n1.0\tx\n", "line 4: .*'x'"),
        (b"Energy (meV)\tLL 1\n0.5\t\xff\n", "not a text file in UTF-8"),
    ],
)
def test_point_import_errors_name_the_file(tmp_path, text, message):
    path = tmp_path / "picked.csv"
    path.write_bytes(text)
    with pytest.raises(ValueError, match=f"picked.csv: {message}"):
        PointTable.load_tsv(path)
