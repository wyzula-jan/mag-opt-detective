import subprocess
import sys

import numpy as np
import pytest

from mag_opt_detective.core import colormaps


def test_names():
    assert colormaps.names() == (
        "magma",
        "inferno",
        "viridis",
        "plasma",
        "turbo",
        "grey",
        "bipolar",
    )


@pytest.mark.parametrize("name", colormaps.names())
def test_stops_are_monotonic_from_0_to_1(name):
    table = colormaps.stops(name)
    pos = np.array([p for p, _ in table])
    assert pos[0] == 0.0
    assert pos[-1] == 1.0
    assert np.all(np.diff(pos) > 0)
    for _, colour in table:
        assert len(colour) == 3
        assert all(0 <= c <= 255 for c in colour)


def test_known_endpoints():
    assert colormaps.stops("grey") == [(0.0, (0, 0, 0)), (1.0, (255, 255, 255))]
    bipolar = colormaps.stops("bipolar")
    assert [c for _, c in bipolar] == [
        (0, 255, 255),
        (0, 0, 255),
        (0, 0, 0),
        (255, 0, 0),
        (255, 255, 0),
    ]
    assert [p for p, _ in bipolar] == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert colormaps.stops("magma")[0][1] == (0, 0, 4)
    assert colormaps.stops("magma")[-1][1] == (252, 253, 191)
    assert colormaps.stops("viridis")[0][1] == (68, 1, 84)
    assert colormaps.stops("viridis")[-1][1] == (253, 231, 37)


@pytest.mark.parametrize("name", colormaps.names())
@pytest.mark.parametrize("n", [2, 256, 512])
def test_lut_shape_and_ends(name, n):
    table = colormaps.lut(name, n)
    assert table.shape == (n, 3)
    assert table.dtype == np.uint8
    stops = colormaps.stops(name)
    np.testing.assert_array_equal(table[0], stops[0][1])
    np.testing.assert_array_equal(table[-1], stops[-1][1])


def test_grey_lut_is_linear():
    table = colormaps.lut("grey")
    np.testing.assert_array_equal(table[:, 0], np.arange(256))
    np.testing.assert_array_equal(table[:, 0], table[:, 2])


@pytest.mark.parametrize("name", ["magma", "inferno", "viridis", "plasma", "turbo"])
def test_sampled_maps_match_the_full_tables(name):
    pg = pytest.importorskip("pyqtgraph")
    reference = pg.colormap.get(name).getLookupTable(nPts=256, alpha=False)
    diff = np.abs(colormaps.lut(name).astype(int) - reference.astype(int))
    assert diff.max() <= 2


def test_unknown_name():
    with pytest.raises(ValueError, match="unknown colour map"):
        colormaps.stops("jet")
    with pytest.raises(ValueError):
        colormaps.lut("grey", 1)


def test_stops_are_copies():
    colormaps.stops("grey").clear()
    assert len(colormaps.stops("grey")) == 2


def test_import_does_not_load_qt():
    code = (
        "import sys\n"
        "import mag_opt_detective.core.colormaps as c\n"
        "c.lut('bipolar')\n"
        "bad = sorted(m for m in sys.modules if m.split('.')[0] in ('PySide6', 'pyqtgraph'))\n"
        "print(','.join(bad))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout
    assert out.strip() == ""
