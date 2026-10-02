import numpy as np

from mag_opt_detective.core.points import PointTable


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


def test_round_trip(tmp_path):
    table = PointTable(np.array([0.25, 0.5]))
    table.add_column("LL 1")
    table.set_nearest("LL 2", 0.5, 3.25)
    path = tmp_path / "points.csv"
    table.save_tsv(path)
    assert path.read_text().splitlines() == ["\tLL 1\tLL 2", "0.25\t\t", "0.5\t\t3.25"]
    back = PointTable.load_tsv(path)
    assert back.names == ["LL 1", "LL 2"]
    np.testing.assert_allclose(back.field, table.field)
    np.testing.assert_allclose(back.column("LL 2"), [np.nan, 3.25])


def test_load_legacy_points_with_empty_column_name(tmp_path):
    path = tmp_path / "Points_V1.csv"
    path.write_text("\t\tLL 1\n0.25\t\t40.1\n0.5\t\t\n")
    table = PointTable.load_tsv(path)
    assert table.names == ["unnamed_1", "LL 1"]
    np.testing.assert_allclose(table.column("LL 1"), [40.1, np.nan])
