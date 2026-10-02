"""Automatic picking of transitions in spectra and field maps (pure numpy/scipy).

Nothing here converts units: energies, windows and slopes are in the unit of the energy
axis (or :class:`FieldMap`) passed in, fields in tesla.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from scipy.signal import find_peaks, savgol_filter

Smoothing = tuple[int, int]
"""Savitzky-Golay ``(window, polyorder)``; the window is in samples and odd."""

Range = tuple[float | None, float | None]
"""Inclusive ``(lo, hi)``; None leaves that side open."""


class Feature(StrEnum):
    """What marks a transition in a spectrum."""

    MAX = "max"
    MIN = "min"
    RISING = "rising"
    """Inflection point where the slope has a positive maximum."""
    FALLING = "falling"
    """Inflection point where the slope has a negative minimum."""


@dataclass(frozen=True)
class Peak:
    """One feature of a spectrum.

    *energy* is refined below one sample and *index* is the nearest valid sample of the
    input arrays. *strength* is the prominence for MAX/MIN and ``|dV/dE|`` at the
    inflection point for RISING/FALLING.
    """

    energy: float
    strength: float
    index: int


def find_features(
    energy: np.ndarray,
    values: np.ndarray,
    feature: Feature | str,
    *,
    smooth: Smoothing | None = None,
    prominence: float | None = None,
    e_range: Range | None = None,
) -> list[Peak]:
    """Features of one spectrum, sorted by energy.

    NaN samples are ignored and the energy axis may be non-uniform or unsorted. *smooth*
    applies a Savitzky-Golay filter to the whole spectrum first; its window is clipped to
    the data length. Only samples inside *e_range* then take part in the detection.

    MAX/MIN are local extrema with at least *prominence* (in value units), refined by the
    vertex of the parabola through the three samples around them. RISING/FALLING are
    extrema of the slope ``dV/dE`` (``np.gradient`` on the energy axis) of the right sign
    with at least *prominence* (in value per energy unit), placed at the zero crossing of
    ``d2V/dE2`` (three-point stencil on the energy axis) next to them by linear
    interpolation.
    """
    feature = Feature(feature)
    energy = np.asarray(energy, dtype=float)
    values = np.asarray(values, dtype=float)
    if energy.ndim != 1 or values.shape != energy.shape:
        raise ValueError("energy and values must be 1D arrays of equal length")
    _check_smoothing(smooth)
    keep = np.flatnonzero(np.isfinite(energy) & np.isfinite(values))
    keep = keep[np.argsort(energy[keep], kind="stable")]
    x = energy[keep]
    if np.any(np.diff(x) == 0):
        raise ValueError("energy axis has repeated values")
    y = _smooth(values[keep], smooth)
    if e_range is not None:
        inside = _inside(x, e_range)
        keep, x, y = keep[inside], x[inside], y[inside]
    if x.size < 3:
        return []
    min_prominence = 0.0 if prominence is None else prominence
    if feature in (Feature.MAX, Feature.MIN):
        found = _extrema(x, y if feature is Feature.MAX else -y, min_prominence)
    else:
        found = _inflections(x, y, feature is Feature.RISING, min_prominence)
    positions, strengths = found
    nearest = keep[_nearest_index(x, positions)]
    peaks = [
        Peak(float(e), float(s), int(i))
        for e, s, i in zip(positions, strengths, nearest, strict=True)
    ]
    return sorted(peaks, key=lambda p: p.energy)


def _check_smoothing(smooth: Smoothing | None) -> None:
    if smooth is None:
        return
    window, poly = smooth
    if window < 1 or window % 2 == 0 or poly < 0 or window <= poly:
        raise ValueError(
            "SG window must be odd and larger than the polynomial order "
            f"(got window {window}, order {poly})"
        )


def _smooth(y: np.ndarray, smooth: Smoothing | None) -> np.ndarray:
    if smooth is None:
        return y
    window, poly = smooth
    window = min(window, y.size if y.size % 2 else y.size - 1)
    if window <= poly:  # too few samples for this polynomial order
        return y
    return savgol_filter(y, window, poly)


def _inside(x: np.ndarray, bounds: Range | None) -> np.ndarray:
    """Mask of *x* within inclusive *bounds* (given in either order)."""
    mask = np.ones(x.shape, dtype=bool)
    if bounds is None:
        return mask
    lo, hi = bounds
    if lo is not None and hi is not None and lo > hi:
        lo, hi = hi, lo
    if lo is not None:
        mask &= x >= lo
    if hi is not None:
        mask &= x <= hi
    return mask


def _vertex(x: np.ndarray, y: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Vertex of the parabola through samples ``k-1, k, k+1`` (any spacing)."""
    x0, x1, x2 = x[k - 1], x[k], x[k + 1]
    slope01 = (y[k] - y[k - 1]) / (x1 - x0)
    slope12 = (y[k + 1] - y[k]) / (x2 - x1)
    curvature = (slope12 - slope01) / (x2 - x0)
    with np.errstate(divide="ignore", invalid="ignore"):
        vertex = 0.5 * (x0 + x1) - slope01 / (2.0 * curvature)
    return np.clip(np.where(curvature != 0, vertex, x1), x0, x2)


def _extrema(x: np.ndarray, y: np.ndarray, prominence: float) -> tuple[np.ndarray, np.ndarray]:
    """Refined positions and prominences of the maxima of *y*."""
    k, props = find_peaks(y, prominence=prominence)
    return _vertex(x, y, k), props["prominences"]


def _inflections(
    x: np.ndarray, y: np.ndarray, rising: bool, prominence: float
) -> tuple[np.ndarray, np.ndarray]:
    """Refined positions and ``|slope|`` of the rising or falling inflection points."""
    signed = y if rising else -y
    slope = np.gradient(signed, x)
    bend = _curvature(x, signed)  # goes from + to - where the signed slope peaks
    k, _ = find_peaks(slope, prominence=prominence)
    right = (bend[k] > 0) & (bend[k + 1] <= 0)
    left = (bend[k] <= 0) & (bend[k - 1] > 0)
    i = np.where(right, k, k - 1)
    b0, b1 = bend[i], bend[i + 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        crossing = x[i] + b0 / (b0 - b1) * (x[i + 1] - x[i])
    # no sign change next to a flat-topped slope peak: fall back to its parabola vertex
    position = np.where(right | left, crossing, _vertex(x, slope, k))
    strength = np.interp(position, x, slope)
    good = strength > 0
    return position[good], strength[good]


def _curvature(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Second derivative from three neighbouring samples (exact for parabolas).

    The compact stencil has a four times smaller bias than ``np.gradient`` applied twice,
    which shifts inflection points of narrow lines by a sizeable part of a sample.
    """
    d2 = np.empty_like(y)
    d2[1:-1] = 2.0 * np.diff(np.diff(y) / np.diff(x)) / (x[2:] - x[:-2])
    d2[0], d2[-1] = d2[1], d2[-2]
    return d2


def _nearest_index(x: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Index of the sample of sorted *x* nearest to each position."""
    i = np.clip(np.searchsorted(x, positions), 1, x.size - 1)
    return np.where(positions - x[i - 1] <= x[i] - positions, i - 1, i)
