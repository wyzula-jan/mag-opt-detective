import os
from pathlib import Path

import numpy as np
import pytest

from helpers import sweep_name, write_text

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

DATA_DIR = Path(__file__).resolve().parents[1] / "Data_to_test"


@pytest.fixture
def data_dir() -> Path:
    """Real measurement data; not part of the repository, so tests skip without it."""
    if not DATA_DIR.is_dir():
        pytest.skip("Data_to_test/ not available")
    return DATA_DIR


@pytest.fixture
def sweep(tmp_path: Path):
    """Synthetic sweep: zero files before/after and field files 0.5..2.0 T.

    Intensity is ``(1 + 0.1 B) * spectrum`` and the zero reference drifts from 1x to 2x.
    """
    x = np.linspace(100.0, 1000.0, 91)
    base = 1.0 + 0.5 * np.sin(x / 50.0)
    fields = np.array([0.5, 1.0, 1.5, 2.0])
    zero = [
        write_text(tmp_path / "Sample_4p2K_Sam1_a00p000T_a00p000T.txt", x, base),
        write_text(tmp_path / "Sample_4p2K_Sam1_a00p000T_a02p000T.txt", x, 2 * base),
    ]
    field = [write_text(tmp_path / sweep_name(b), x, (1 + 0.1 * b) * base) for b in fields]
    return {"x": x, "base": base, "fields": fields, "zero": zero, "field": field}
