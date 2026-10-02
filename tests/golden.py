"""Golden exports of a small synthetic sweep, the reference for the energy-unit refactor.

The files in ``tests/data/`` were written in meV by the code that converted the energy
axis when loading. Regenerate them with ``python tests/golden.py [out_dir]``.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

from helpers import sweep_name, write_text
from mag_opt_detective.core.pipeline import PlotKind, ProcessOptions, process
from mag_opt_detective.core.points import PointTable
from mag_opt_detective.core.processing import Axis
from mag_opt_detective.core.readers import load_measurement
from mag_opt_detective.core.spectra import FieldMap, save_tsv
from mag_opt_detective.core.units import Unit

GOLDEN_DIR = Path(__file__).resolve().parent / "data"
UNIT = Unit.MEV
FIELDS = (0.5, 1.0, 1.5, 2.0)
ENERGY_CUT = (55.0, 140.0)  # meV; no sample sits close to these limits
BASELINE = (60.0, 70.0)  # meV
MAPS = ("Ratio", "Data", "Ratio_1stDer", "Ratio_1stDer_perUnit")
POINTS = "golden_points.tsv"


def map_path(name: str, folder: Path = GOLDEN_DIR) -> Path:
    return folder / f"golden_{name}.tsv"


def write_sweep(folder: Path) -> tuple[list[Path], list[Path]]:
    """Zero-field spectra before/after the sweep and one spectrum per field (cm-1 axis).

    A line at ``600 + 150 B`` cm-1 moves with field on a sloped background; the zero
    reference drifts by 10 % over the sweep.
    """
    x = np.linspace(400.0, 1200.0, 81)
    base = 1.0 + 0.3 * np.sin(x / 60.0)
    zero = [
        write_text(folder / "Golden_a00p000T_a00p000T.txt", x, base),
        write_text(folder / "Golden_a00p000T_a02p000T.txt", x, 1.1 * base),
    ]
    field = []
    for b in FIELDS:
        slope = 1.0 + 0.05 * b * x / 800.0
        line = 1.0 + 0.2 * np.exp(-(((x - 600.0 - 150.0 * b) / 40.0) ** 2))
        field.append(write_text(folder / sweep_name(b), x, base * slope * line))
    return zero, field


def exports(folder: Path) -> dict[str, FieldMap]:
    """The exported maps (in meV), keyed by their export name."""
    zero, field = write_sweep(folder)
    sample = load_measurement(zero, field, unit=UNIT, energy_limits=ENERGY_CUT)
    result = process(sample, options=ProcessOptions(baseline_region=BASELINE))
    return {
        "Ratio": result.get(PlotKind.RATIO),
        "Data": result.get(PlotKind.DATA),
        "Ratio_1stDer": result.get(PlotKind.RATIO, 1, Axis.ENERGY),
        "Ratio_1stDer_perUnit": result.get(PlotKind.RATIO, 1, Axis.ENERGY, physical=True),
    }


def points(ratio: FieldMap) -> PointTable:
    """The maximum of every spectrum of *ratio*, and a curve with one point."""
    table = PointTable(ratio.field)
    for j, b in enumerate(ratio.field):
        table.set_nearest("peak", b, ratio.energy[ratio.values[:, j].argmax()])
    table.set_nearest("single", ratio.field[1], ratio.energy[10])
    return table


def write(folder: Path = GOLDEN_DIR) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        maps = exports(Path(tmp))
    for name in MAPS:
        save_tsv(maps[name], map_path(name, folder))
    points(maps["Ratio"]).save_tsv(folder / POINTS)


if __name__ == "__main__":
    write(Path(sys.argv[1]) if len(sys.argv) > 1 else GOLDEN_DIR)
