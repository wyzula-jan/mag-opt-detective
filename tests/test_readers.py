import os

import numpy as np
import pytest

from helpers import sweep_name, write_opus, write_text
from mag_opt_detective.core.readers import (
    SpectrumCache,
    load_measurement,
    parse_field,
    read_spectrum,
    sort_paths,
)
from mag_opt_detective.core.units import Unit, convert_range, from_cm1


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
    m = load_measurement(sweep["zero"], sweep["field"])
    np.testing.assert_allclose(m.spectra.field, sweep["fields"])
    assert m.spectra.unit is Unit.CM1  # always the unit of the files
    np.testing.assert_allclose(m.spectra.energy, sweep["x"])
    assert m.spectra.values.shape == (sweep["x"].size, 4)
    assert m.zero.shape == (sweep["x"].size, 2)


def test_load_measurement_custom_field_and_cut(sweep):
    m = load_measurement(
        sweep["zero"][:1], sweep["field"], field=np.arange(1, 5), energy_limits=(200, 500)
    )
    np.testing.assert_allclose(m.spectra.field, [1, 2, 3, 4])
    np.testing.assert_allclose(m.spectra.energy[[0, -1]], [200, 500])  # cut in cm-1, inclusive
    assert m.zero.shape[0] == m.spectra.energy.size
    upper = load_measurement(sweep["zero"], sweep["field"], energy_limits=(None, 150))
    np.testing.assert_allclose(upper.spectra.energy, [100, 110, 120, 130, 140, 150])
    with pytest.raises(ValueError, match="cut 1200 to open cm-1 leaves no data"):
        load_measurement(sweep["zero"], sweep["field"], energy_limits=(1200, None))


@pytest.mark.parametrize("unit", [Unit.MEV, Unit.THZ])
def test_cut_typed_in_another_unit_keeps_boundary_samples(sweep, unit):
    """A cut typed in *unit* exactly at a sample keeps that sample after conversion."""
    x = sweep["x"]
    shown = from_cm1(x, unit)  # what the user sees (and may type) for each sample
    for k in range(x.size):
        cut = convert_range((shown[k], shown[k]), unit, Unit.CM1)
        m = load_measurement(sweep["zero"][:1], sweep["field"][:1], energy_limits=cut)
        np.testing.assert_array_equal(m.spectra.energy, x[k : k + 1])


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


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (b"100.0\t1.0\n101.0\tx\n", r"broken_a01p000T\.txt: .*'x'"),
        (b"100.0\t1.0\n" + bytes(range(128, 160)), r"broken_a01p000T\.txt: not a text file"),
    ],
)
def test_a_broken_spectrum_names_its_file(sweep, tmp_path, data, message):
    broken = tmp_path / "broken_a01p000T.txt"
    broken.write_bytes(data)
    with pytest.raises(ValueError, match=message):
        load_measurement(sweep["zero"], [*sweep["field"], broken])


def _touch(path, seconds: int) -> None:
    """Give *path* a modification time of its own (a rewrite within one clock tick may not)."""
    ns = (1_700_000_000 + seconds) * 1_000_000_000
    os.utime(path, ns=(ns, ns))


def test_spectrum_cache_reads_each_file_once_until_it_changes(tmp_path):
    x = np.linspace(100.0, 200.0, 11)
    text = write_text(tmp_path / "a.txt", x, x)
    opus = write_opus(tmp_path / "b.0", x, 2 * x)
    cache = SpectrumCache()
    first = cache.read(text)
    np.testing.assert_allclose(cache.read(opus)[1], 2 * x, rtol=1e-6)
    assert cache.read(str(text)) is first  # the same arrays, not read again
    assert cache.reads == 2 and len(cache) == 2 and text in cache
    with pytest.raises(ValueError, match="read-only"):
        first[1][0] = 0.0  # shared arrays cannot be changed by accident

    write_text(text, x, 3 * x)
    _touch(text, 1)
    np.testing.assert_allclose(cache.read(text)[1], 3 * x)  # changed: read again
    assert cache.reads == 3

    cache.retain([opus])
    assert text not in cache and opus in cache


def test_spectrum_cache_keeps_neither_broken_nor_missing_files(tmp_path):
    cache = SpectrumCache()
    broken = tmp_path / "broken.txt"
    broken.write_bytes(b"100.0\t1.0\n101.0\tx\n")
    with pytest.raises(ValueError, match=r"broken\.txt"):
        cache.read(broken)
    with pytest.raises(OSError):
        cache.read(tmp_path / "missing.txt")
    assert len(cache) == 0


def test_load_measurement_through_the_cache_reads_only_new_files(sweep, tmp_path):
    cache = SpectrumCache()
    plain = load_measurement(sweep["zero"], sweep["field"])
    cached = load_measurement(sweep["zero"], sweep["field"], read=cache.read)
    np.testing.assert_array_equal(cached.spectra.values, plain.spectra.values)
    np.testing.assert_array_equal(cached.zero, plain.zero)
    assert cache.reads == 6

    new = write_text(tmp_path / sweep_name(2.5), sweep["x"], 1.25 * sweep["base"])
    grown = load_measurement(sweep["zero"], [*sweep["field"], new], read=cache.read)
    assert cache.reads == 7  # only the new file
    np.testing.assert_allclose(grown.spectra.field, [*sweep["fields"], 2.5])
    np.testing.assert_allclose(grown.spectra.values[:, -1], 1.25 * sweep["base"])
