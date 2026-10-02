"""Model transition energies to compare with the measured maps."""

from __future__ import annotations

import numpy as np
from scipy.constants import e, hbar

VELOCITY_UNIT = 1e5  # Fermi velocities are given in 10^5 m/s


def dirac_level(n: int | np.ndarray, field: np.ndarray, velocity: float, delta: float):
    """Energy (meV) of Landau level *n* >= 0 of a massive Dirac band.

    ``E_n = sqrt(2 e hbar v^2 B n + delta^2)``, with *velocity* in 10^5 m/s, *field* in T
    and *delta* the half-gap (mass term) in meV.
    """
    v = velocity * VELOCITY_UNIT
    # 2 e hbar v^2 B n is in J^2; divide by e^2 for eV^2 and scale to meV^2
    kinetic = 2.0 * hbar * v**2 * np.asarray(field, dtype=float) * n / e * 1e6
    return np.sqrt(kinetic + delta**2)


def dirac_interband(field: np.ndarray, velocity: float, delta: float, n_lines: int) -> np.ndarray:
    """Interband transitions L(-n) -> L(n+1) for n = 0 .. n_lines-1, in meV.

    Returns an array of shape ``(n_lines, field.size)``:
    ``E = E_n + E_(n+1)`` (the formula of the legacy draft).
    """
    if velocity < 0 or delta < 0:
        raise ValueError("velocity and gap must not be negative")
    field = np.asarray(field, dtype=float)
    n = np.arange(n_lines)[:, None]
    return dirac_level(n, field, velocity, delta) + dirac_level(n + 1, field, velocity, delta)
