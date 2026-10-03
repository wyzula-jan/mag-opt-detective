"""How units, axis titles, numbers and shortcuts are written on screen (no Qt).

Files keep the ASCII unit ("Energy (cm-1)", see :func:`core.units.axis_label`); everything
people read in the window uses these helpers, so the unit is always written ``cm⁻¹`` and a
range shows the same digits wherever it appears.
"""

from __future__ import annotations

import math
import sys

import numpy as np

from mag_opt_detective.core.units import Unit

UNIT_TEXT = {Unit.CM1: "cm⁻¹", Unit.MEV: "meV", Unit.THZ: "THz"}
FIELD_LABEL = "Magnetic field B (T)"
NO_NUMBER = "–"


def unit_text(unit: Unit | str) -> str:
    """The unit as people read it: ``cm⁻¹``, ``meV`` or ``THz``."""
    return UNIT_TEXT[Unit(unit)]


def energy_label(unit: Unit | str) -> str:
    """Title of an energy axis, e.g. ``Energy (cm⁻¹)``."""
    return f"Energy ({unit_text(unit)})"


def format_number(value: float | None, digits: int = 4) -> str:
    """A number with *digits* significant digits and no trailing zeros (``16``, ``0.25``,
    ``396.8``); very large or small ones in exponent form, and "–" for None or NaN."""
    if value is None or not math.isfinite(value):
        return NO_NUMBER
    if value == 0:
        return "0"
    if 1e-4 <= abs(value) < 1e6:
        return positional(value, digits)
    mantissa, exponent = f"{value:.{max(0, digits - 1)}e}".split("e")
    if "." in mantissa:
        mantissa = mantissa.rstrip("0").rstrip(".")
    return f"{mantissa}e{int(exponent)}"


def positional(value: float, digits: int = 4) -> str:
    """*value* with *digits* significant digits, never in exponent form (for number fields)."""
    if not math.isfinite(value):
        return NO_NUMBER
    text = np.format_float_positional(
        float(value), precision=digits, unique=False, fractional=False, trim="-"
    )
    return "0" if text in ("-0", "0") else text


def format_range(lo: float, hi: float, unit: str = "") -> str:
    """``350 – 3200 cm⁻¹``: two ends written alike, with an optional unit."""
    text = f"{format_number(lo)} – {format_number(hi)}"
    return f"{text} {unit}" if unit else text


def process_key() -> str:
    """The Process shortcut as written on this system: ⌘↵ on macOS, else Ctrl+↵."""
    return "⌘↵" if sys.platform == "darwin" else "Ctrl+↵"
