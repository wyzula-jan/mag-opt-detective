import numpy as np
import pytest

from helpers import write_opus
from mag_opt_detective.core.opus import OpusError, is_opus_file, read_opus


def test_read_synthetic_opus(tmp_path):
    x = np.linspace(400.0, 4000.0, 50)
    y = np.linspace(0.1, 2.0, 50)
    path = write_opus(tmp_path / "spec.0", x, y)
    assert is_opus_file(path)
    rx, ry = read_opus(path)
    np.testing.assert_allclose(rx, x)
    np.testing.assert_allclose(ry, y.astype(np.float32))


def test_descending_axis_is_flipped_and_scaled(tmp_path):
    x = np.linspace(4000.0, 400.0, 20)
    y = np.arange(20, dtype=float)
    rx, ry = read_opus(write_opus(tmp_path / "spec.1", x, y, csf=2.0))
    np.testing.assert_allclose(rx, x[::-1])
    np.testing.assert_allclose(ry, 2.0 * y[::-1])


def test_not_opus(tmp_path):
    path = tmp_path / "x.txt"
    path.write_text("1 2\n3 4\n", encoding="utf-8")
    assert not is_opus_file(path)
    with pytest.raises(OpusError):
        read_opus(path)


def test_matches_macro_text_exports(data_dir):
    """The OPUS macro .txt files contain exactly the ScSm block (rounded to 8 decimals)."""
    opus_files = sorted((data_dir / "Data_Opus").glob("*.0"))
    assert opus_files
    for path in opus_files:
        x, y = read_opus(path)
        text = np.loadtxt(data_dir / "Data_Macro" / (path.stem + ".txt"))
        np.testing.assert_allclose(x, text[:, 0], atol=1e-6, rtol=0)
        np.testing.assert_allclose(y, text[:, 1], atol=1e-8, rtol=0)
