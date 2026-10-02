import numpy as np
import pytest

from helpers import write_opus, write_text
from mag_opt_detective.core.readers import (
    load_measurement,
    parse_field,
    read_spectrum,
    sort_paths,
)
from mag_opt_detective.core.units import Unit, convert


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("NbAs_Sam2_a00p250T.txt", 0.25),
        ("NbAs_Sam2_a12p750T.0", 12.75),
        ("/some/dir/FePS3_Sam1_a18p000T.12", 18.0),
        ("NbAs_Sam2_a00p000T_a16p000T.txt", 16.0),
        ("NbAs_Sam2_a00p000T_a00p000T.txt", 0.0),
        ("no_field_here.txt", None),
    ],
)
def test_parse_field(name, expected):
    assert parse_field(name) == expected


def test_sort_paths_by_field_then_name():
    paths = ["s_a10p000T.0", "s_a02p500T.0", "b.txt", "s_a00p250T.0", "a10.txt", "a9.txt"]
    assert sort_paths(paths) == [
        "s_a00p250T.0",
        "s_a02p500T.0",
        "s_a10p000T.0",
        "a9.txt",
        "a10.txt",
        "b.txt",
    ]


def test_read_spectrum_text_and_opus(tmp_path):
    x = np.linspace(10, 20, 11)
    y = x**2
    tx, ty = read_spectrum(write_text(tmp_path / "a.txt", x[::-1], y[::-1], sep=" ", eol="\n"))
    np.testing.assert_allclose(tx, x)
    np.testing.assert_allclose(ty, y)
    ox, oy = read_spectrum(write_opus(tmp_path / "a.0", x, y))
    np.testing.assert_allclose(ox, x)
    np.testing.assert_allclose(oy, y, rtol=1e-6)


def test_load_measurement_field_from_names(sweep):
    m = load_measurement(sweep["zero"], sweep["field"], unit=Unit.MEV)
    np.testing.assert_allclose(m.spectra.field, sweep["fields"])
    np.testing.assert_allclose(m.spectra.energy, convert(sweep["x"], Unit.CM1, Unit.MEV))
    assert m.spectra.values.shape == (sweep["x"].size, 4)
    assert m.zero.shape == (sweep["x"].size, 2)


def test_load_measurement_custom_field_and_cut(sweep):
    m = load_measurement(
        sweep["zero"][:1], sweep["field"], field=np.arange(1, 5), energy_limits=(200, 500)
    )
    np.testing.assert_allclose(m.spectra.field, [1, 2, 3, 4])
    assert m.spectra.energy.min() >= 200
    assert m.spectra.energy.max() <= 500
    assert m.zero.shape[0] == m.spectra.energy.size


def test_load_measurement_errors(sweep, tmp_path):
    with pytest.raises(ValueError, match="no field files"):
        load_measurement(sweep["zero"], [])
    with pytest.raises(ValueError, match="no zero-field"):
        load_measurement([], sweep["field"])
    with pytest.raises(ValueError, match="one or two"):
        load_measurement(sweep["zero"] * 2, sweep["field"])
    with pytest.raises(ValueError, match="4 files"):
        load_measurement(sweep["zero"], sweep["field"], field=np.arange(3))
    unnamed = write_text(tmp_path / "plain.txt", sweep["x"], sweep["base"])
    with pytest.raises(ValueError, match="custom field range"):
        load_measurement(sweep["zero"], [*sweep["field"], unnamed])
    shifted = write_text(tmp_path / "s_a05p000T.txt", sweep["x"] + 1, sweep["base"])
    with pytest.raises(ValueError, match="energy axis differs"):
        load_measurement(sweep["zero"], [*sweep["field"], shifted])
