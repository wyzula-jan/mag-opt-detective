"""Energy units used on the spectral axis."""

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

_LABEL_RE = re.compile(r"^\s*Energy\s*\((?P<unit>[^)]+)\)\s*$")


def from_cm1(x: np.ndarray, unit: Unit | str) -> np.ndarray:
    """Convert wavenumbers in cm^-1 to *unit*."""
    return np.asarray(x, dtype=float) / CM1_PER_UNIT[Unit(unit)]


def convert(x: np.ndarray, src: Unit | str, dst: Unit | str) -> np.ndarray:
    """Convert *x* from unit *src* to unit *dst*."""
    return np.asarray(x, dtype=float) * CM1_PER_UNIT[Unit(src)] / CM1_PER_UNIT[Unit(dst)]


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
