"""Energy units used on the spectral axis.

Data are kept in cm^-1, the unit of the measured files; the other units are applied
only when showing or exporting. The helpers below convert values, ranges and colour
levels between units.
"""

from __future__ import annotations

import re
from enum import StrEnum

import numpy as np


class Unit(StrEnum):
    CM1 = "cm-1"
    MEV = "meV"
    THZ = "THz"


# How many cm^-1 one unit corresponds to (values kept from the legacy code).
CM1_PER_UNIT: dict[Unit, float] = {
    Unit.CM1: 1.0,
    Unit.MEV: 8.0656,
    Unit.THZ: 33.35641,
}

Bound = float | None
Range = tuple[Bound, Bound]

_LABEL_RE = re.compile(r"^\s*Energy\s*\((?P<unit>[^)]+)\)\s*$")


def from_cm1(x: np.ndarray, unit: Unit | str) -> np.ndarray:
    """Convert wavenumbers in cm^-1 to *unit*."""
    return convert(x, Unit.CM1, unit)


def to_cm1(x: np.ndarray, unit: Unit | str) -> np.ndarray:
    """Convert *x* in *unit* to wavenumbers in cm^-1."""
    return convert(x, unit, Unit.CM1)


def convert(x: np.ndarray, src: Unit | str, dst: Unit | str) -> np.ndarray:
    """Convert *x* from unit *src* to unit *dst* (values are kept exactly if they agree)."""
    src, dst = Unit(src), Unit(dst)
    x = np.asarray(x, dtype=float)
    if src is dst:
        return x * 1.0  # a new array (or NumPy scalar) with the same values
    return x * CM1_PER_UNIT[src] / CM1_PER_UNIT[dst]


def convert_range(rng: Range | None, src: Unit | str, dst: Unit | str) -> Range | None:
    """Convert a ``(lo, hi)`` range; None, or a None end (no limit), passes through.

    The ends are converted one by one and keep their order.
    """
    if rng is None:
        return None
    lo, hi = (None if v is None else float(convert(v, src, dst)) for v in rng)
    return lo, hi


def derivative_scale(unit: Unit | str, order: int, axis_is_energy: bool, physical: bool) -> float:
    """Factor from a cm^-1 based map derivative to the same derivative shown in *unit*.

    Only derivatives per energy unit (*physical* along energy) change, by
    ``CM1_PER_UNIT[unit] ** order``; per-point derivatives and d/dB do not.
    """
    if not (physical and axis_is_energy and order):
        return 1.0
    return CM1_PER_UNIT[Unit(unit)] ** order


def convert_levels(
    levels: tuple[float, float] | None,
    src: Unit | str,
    dst: Unit | str,
    order: int,
    axis_is_energy: bool,
    physical: bool,
) -> tuple[float, float] | None:
    """Rescale fixed colour levels of a derivative map from unit *src* to unit *dst*.

    Levels of per-unit energy derivatives scale with :func:`derivative_scale`; every
    other map has the same values in any unit, so its levels are returned unchanged.
    None (automatic levels) passes through.
    """
    if levels is None:
        return None
    factor = derivative_scale(dst, order, axis_is_energy, physical) / derivative_scale(
        src, order, axis_is_energy, physical
    )
    lo, hi = levels
    return float(lo) * factor, float(hi) * factor


def axis_label(unit: Unit | str) -> str:
    """Header/axis label, e.g. ``Energy (meV)``."""
    return f"Energy ({Unit(unit)})"


def parse_axis_label(label: str) -> Unit | None:
    """Inverse of :func:`axis_label`; returns None for unknown labels."""
    match = _LABEL_RE.match(label)
    if match is None:
        return None
    try:
        return Unit(match["unit"].strip())
    except ValueError:
        return None
