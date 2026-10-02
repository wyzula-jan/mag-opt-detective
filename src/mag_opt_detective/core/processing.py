"""Numerical treatment of field maps (all functions are pure)."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from enum import IntEnum

import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.signal import savgol_filter

from mag_opt_detective.core.readers import Measurement
from mag_opt_detective.core.spectra import FieldMap

logger = logging.getLogger(__name__)


class Axis(IntEnum):
    ENERGY = 0
    FIELD = 1


def interp_linear(x_new: np.ndarray, x: np.ndarray, y: np.ndarray, axis: int) -> np.ndarray:
    """Piecewise-linear interpolation of *y* along *axis* with linear extrapolation."""
    x = np.asarray(x, dtype=float)
    x_new = np.asarray(x_new, dtype=float)
    if x.size == 1:
        return np.repeat(np.take(y, [0], axis=axis), x_new.size, axis=axis)
    order = np.argsort(x, kind="stable")
    spline = make_interp_spline(x[order], np.take(y, order, axis=axis), k=1, axis=axis)
    return spline(x_new, extrapolate=True)


def zero_reference(zero: np.ndarray, field: np.ndarray) -> np.ndarray:
    """Zero-field reference for every field point, shape ``(n_energy, n_field)``.

    One zero spectrum is used as-is. Two spectra (before/after the sweep) are
    interpolated linearly between the first and the last field point to correct drift.
    """
    if zero.shape[1] == 1:
        return np.repeat(zero, field.size, axis=1)
    span = field[-1] - field[0]
    t = np.zeros_like(field) if span == 0 else (field - field[0]) / span
    return zero[:, [0]] * (1.0 - t) + zero[:, [1]] * t


def ratio_to_zero(measurement: Measurement) -> FieldMap:
    """R(B)/R(0)."""
    spectra = measurement.spectra
    return spectra.with_values(spectra.values / zero_reference(measurement.zero, spectra.field))


def interpolate_field(fmap: FieldMap, field: np.ndarray) -> FieldMap:
    values = interp_linear(field, fmap.field, fmap.values, axis=Axis.FIELD)
    return fmap.replace(field=np.asarray(field, dtype=float), values=values)


def interpolate_energy(fmap: FieldMap, energy: np.ndarray) -> FieldMap:
    values = interp_linear(energy, fmap.energy, fmap.values, axis=Axis.ENERGY)
    return fmap.replace(energy=np.asarray(energy, dtype=float), values=values)


def align_to(fmap: FieldMap, target: FieldMap) -> FieldMap:
    """Bring *fmap* onto the energy and field grid of *target*."""
    if fmap.unit != target.unit:
        raise ValueError(f"unit mismatch: {fmap.unit} vs {target.unit}")
    same_energy = fmap.energy.shape == target.energy.shape and np.allclose(
        fmap.energy, target.energy, rtol=1e-9, atol=0
    )
    if not same_energy:
        logger.warning("Reference energy axis differs from sample; interpolating reference.")
        fmap = interpolate_energy(fmap, target.energy)
    same_field = fmap.field.shape == target.field.shape and np.allclose(fmap.field, target.field)
    if not same_field:
        fmap = interpolate_field(fmap, target.field)
    return fmap


def divide(numerator: FieldMap, denominator: FieldMap) -> FieldMap:
    """Element-wise ratio; *denominator* is first aligned to the numerator's grid."""
    denominator = align_to(denominator, numerator)
    return numerator.with_values(numerator.values / denominator.values)


def ratio_to_average(fmap: FieldMap) -> FieldMap:
    """R(B)/R(B-average): each spectrum divided by the mean over all fields."""
    return fmap.with_values(fmap.values / fmap.values.mean(axis=1, keepdims=True))


def energy_mask(energy: np.ndarray, lo: float | None, hi: float | None) -> np.ndarray:
    mask = np.ones(energy.size, dtype=bool)
    if lo is not None:
        mask &= energy >= lo
    if hi is not None:
        mask &= energy <= hi
    return mask


def baseline_normalize(fmap: FieldMap, region: tuple[float, float]) -> FieldMap:
    """Shift every spectrum so that its mean over *region* (inclusive) equals one."""
    mask = energy_mask(fmap.energy, *region)
    if not mask.any():
        raise ValueError(f"baseline region {region[0]} to {region[1]} {fmap.unit} contains no data")
    offset = fmap.values[mask].mean(axis=0) - 1.0
    return fmap.with_values(fmap.values - offset)


def derivative(fmap: FieldMap, axis: Axis = Axis.ENERGY, physical: bool = False) -> FieldMap:
    """Gradient along *axis*.

    By default the spacing is one sample (per data point), as in the legacy tool. With
    *physical* the real axis values are used, giving d/dE per energy unit or d/dB per
    tesla; non-uniform grids are handled to second order.
    """
    if fmap.values.shape[axis] < 2:
        return fmap.with_values(np.zeros_like(fmap.values))
    if not physical:
        return fmap.with_values(np.gradient(fmap.values, axis=int(axis)))
    coords = fmap.energy if axis == Axis.ENERGY else fmap.field
    if np.any(np.diff(coords) == 0):
        raise ValueError(f"{axis.name.lower()} axis has repeated values; cannot differentiate")
    return fmap.with_values(np.gradient(fmap.values, coords, axis=int(axis)))


def savgol(fmap: FieldMap, window: int, poly: int) -> FieldMap:
    """Savitzky-Golay smoothing of every spectrum along energy."""
    if window % 2 == 0 or window <= poly:
        raise ValueError("SG window must be odd and larger than the polynomial order")
    if window > fmap.energy.size:
        raise ValueError(f"SG window {window} is longer than the spectrum ({fmap.energy.size})")
    return fmap.with_values(savgol_filter(fmap.values, window, poly, axis=Axis.ENERGY))


def crop_energy(fmap: FieldMap, lo: float | None, hi: float | None) -> FieldMap:
    mask = energy_mask(fmap.energy, lo, hi)
    if not mask.any():
        raise ValueError(f"energy range {lo} to {hi} {fmap.unit} contains no data")
    return fmap.replace(energy=fmap.energy[mask], values=fmap.values[mask])


def merge_energy(parts: Sequence[tuple[FieldMap, float | None, float | None]]) -> FieldMap:
    """Join maps measured in different spectral ranges into one map.

    Every part is cropped to its ``(lo, hi)`` range, the parts are stacked along
    energy and re-gridded onto a uniform step (the mean step of the stacked data).
    """
    if not parts:
        raise ValueError("nothing to merge")
    first = parts[0][0]
    cropped = []
    for fmap, lo, hi in parts:
        if fmap.unit != first.unit:
            raise ValueError(f"cannot merge {fmap.unit} with {first.unit} data")
        if fmap.field.shape != first.field.shape or not np.allclose(fmap.field, first.field):
            raise ValueError("all merged datasets must share the same field values")
        cropped.append(crop_energy(fmap, lo, hi))
    energy = np.concatenate([c.energy for c in cropped])
    values = np.concatenate([c.values for c in cropped])
    # overlapping range limits produce duplicate energies -> average them
    energy, inverse, counts = np.unique(energy, return_inverse=True, return_counts=True)
    summed = np.zeros((energy.size, values.shape[1]))
    np.add.at(summed, inverse, values)
    values = summed / counts[:, None]
    if energy.size < 2:
        return first.replace(energy=energy, values=values)
    grid = np.linspace(energy[0], energy[-1], energy.size)
    values = interp_linear(grid, energy, values, axis=Axis.ENERGY)
    return first.replace(energy=grid, values=values)


def crop_field(fmap: FieldMap, lo: float | None, hi: float | None) -> FieldMap:
    """Keep the field columns inside ``[lo, hi]`` (inclusive)."""
    mask = np.ones(fmap.field.size, dtype=bool)
    if lo is not None:
        mask &= fmap.field >= lo
    if hi is not None:
        mask &= fmap.field <= hi
    if not mask.any():
        raise ValueError(f"field range {lo} to {hi} T contains no data")
    return fmap.replace(field=fmap.field[mask], values=fmap.values[:, mask])


def _common_energy(maps: Sequence[FieldMap]) -> np.ndarray:
    """Energy axis of the first map restricted to the range all maps cover."""
    lo = max(m.energy[0] for m in maps)
    hi = min(m.energy[-1] for m in maps)
    energy = maps[0].energy
    energy = energy[(energy >= lo) & (energy <= hi)]
    if energy.size < 2:
        raise ValueError("the datasets have no common energy range")
    return energy


def _on_energy(fmap: FieldMap, energy: np.ndarray) -> FieldMap:
    if fmap.energy.shape == energy.shape and np.allclose(fmap.energy, energy, rtol=1e-9, atol=0):
        return fmap
    return interpolate_energy(fmap, energy)


def merge_field(parts: Sequence[tuple[FieldMap, float | None, float | None]]) -> FieldMap:
    """Join maps measured over different field ranges into one map.

    Every part is cut to its ``(lo, hi)`` field range and interpolated onto the energy
    axis of the first part, restricted to the energy range all parts cover. Fields that
    occur in more than one part are averaged; the result is sorted by field.
    """
    if not parts:
        raise ValueError("nothing to merge")
    first = parts[0][0]
    for fmap, _, _ in parts:
        if fmap.unit != first.unit:
            raise ValueError(f"cannot merge {fmap.unit} with {first.unit} data")
    cropped = [crop_field(fmap, lo, hi) for fmap, lo, hi in parts]
    energy = _common_energy(cropped)
    aligned = [_on_energy(c, energy) for c in cropped]
    field = np.concatenate([a.field for a in aligned])
    values = np.concatenate([a.values for a in aligned], axis=1)
    field, inverse, counts = np.unique(field, return_inverse=True, return_counts=True)
    summed = np.zeros((energy.size, field.size))
    np.add.at(summed.T, inverse, values.T)
    return first.replace(energy=energy, field=field, values=summed / counts)
