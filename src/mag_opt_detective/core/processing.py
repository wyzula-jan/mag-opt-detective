"""Numerical treatment of field maps (all functions are pure).

Energy limits are in the unit of the maps they apply to: cm^-1 for processed data.
Maps that are combined must share one unit.
"""

from __future__ import annotations

import itertools
import logging
import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from enum import IntEnum

import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.signal import savgol_filter

from mag_opt_detective.core.readers import Measurement
from mag_opt_detective.core.spectra import FieldMap, energy_mask

logger = logging.getLogger(__name__)


class Axis(IntEnum):
    ENERGY = 0
    FIELD = 1


def interp_linear(x_new: np.ndarray, x: np.ndarray, y: np.ndarray, axis: int) -> np.ndarray:
    """Piecewise-linear interpolation of *y* along *axis* with linear extrapolation.

    A missing value (NaN) in *y* makes only the points next to it missing.
    """
    x = np.asarray(x, dtype=float)
    x_new = np.asarray(x_new, dtype=float)
    if x.size == 1:
        return np.repeat(np.take(y, [0], axis=axis), x_new.size, axis=axis)
    order = np.argsort(x, kind="stable")
    x, y = x[order], np.take(y, order, axis=axis)
    # linear B-splines: the coefficients are the values, so a NaN stays where it is
    spline = make_interp_spline(x, y, k=1, axis=axis, check_finite=False)
    out = spline(x_new, extrapolate=True)
    if not np.isfinite(y).all():  # a sample next to a missing one keeps its value (not 0 * NaN)
        k = np.minimum(np.searchsorted(x, x_new), x.size - 1)
        on = np.flatnonzero(x[k] == x_new)
        out = np.moveaxis(out, axis, 0)
        out[on] = np.moveaxis(y, axis, 0)[k[on]]
        out = np.moveaxis(out, 0, axis)
    return out


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


def step_ratio(fmap: FieldMap) -> FieldMap:
    """R(B)/R(B - dB): every spectrum divided by the one at the previous field step.

    The result is assigned to the higher field, so it has one field point less.
    """
    if fmap.field.size < 2:
        raise ValueError("the field-step ratio needs at least two field values")
    return fmap.replace(field=fmap.field[1:], values=fmap.values[:, 1:] / fmap.values[:, :-1])


def baseline_normalize(fmap: FieldMap, region: tuple[float, float]) -> FieldMap:
    """Shift every spectrum so that its mean over the energy *region* (inclusive) equals one.

    Missing values (NaN, e.g. the gap of a map merged by energy) are left out of the mean; a
    region with no value at all (only missing ones) contains no data.
    """
    mask = energy_mask(fmap.energy, *region)
    if not mask.any() or not np.isfinite(fmap.values[mask]).any():
        raise ValueError(f"baseline region {region[0]} to {region[1]} {fmap.unit} contains no data")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # a spectrum without data there: NaN
        offset = np.nanmean(fmap.values[mask], axis=0) - 1.0
    return fmap.with_values(fmap.values - offset)


def derivative(fmap: FieldMap, axis: Axis = Axis.ENERGY, physical: bool = False) -> FieldMap:
    """Gradient along *axis*.

    By default the spacing is one sample (per data point). On an uneven energy axis (a map
    merged by energy, whose step changes where its parts meet; see :func:`typical_step`) a
    point is the axis's typical step instead, so parts sampled finer or coarser give
    comparable values (a second derivative, taken twice, then is per typical step squared).
    With *physical* the real axis values are used, giving d/dE per energy unit or d/dB per
    tesla; non-uniform grids are handled to second order.
    """
    if fmap.values.shape[axis] < 2:
        return fmap.with_values(np.zeros_like(fmap.values))
    if not physical:
        step = typical_step(fmap.energy) if axis == Axis.ENERGY else None
        if step is None:  # an even axis: one sample
            return fmap.with_values(np.gradient(fmap.values, axis=int(axis)))
        return fmap.with_values(np.gradient(fmap.values, fmap.energy, axis=int(axis)) * step)
    coords = fmap.energy if axis == Axis.ENERGY else fmap.field
    if np.any(np.diff(coords) == 0):
        raise ValueError(f"{axis.name.lower()} axis has repeated values; cannot differentiate")
    return fmap.with_values(np.gradient(fmap.values, coords, axis=int(axis)))


UNEVEN_STEPS = 0.01  # an axis with a step this much (relative) off the typical one is uneven


def typical_step(axis: np.ndarray) -> float | None:
    """The typical (median) step of an uneven *axis*, signed as the axis runs; None for an
    even one (every step within :data:`UNEVEN_STEPS` of it) or one with repeated values."""
    steps = np.diff(np.asarray(axis, dtype=float))
    if steps.size < 2 or np.any(steps == 0):
        return None
    step = float(np.median(steps))
    if np.all(np.abs(steps - step) <= UNEVEN_STEPS * abs(step)):
        return None
    return step


def savgol(fmap: FieldMap, window: int, poly: int) -> FieldMap:
    """Savitzky-Golay smoothing of every spectrum along energy."""
    if window % 2 == 0 or window <= poly:
        raise ValueError("SG window must be odd and larger than the polynomial order")
    if window > fmap.energy.size:
        raise ValueError(f"SG window {window} is longer than the spectrum ({fmap.energy.size})")
    return fmap.with_values(savgol_filter(fmap.values, window, poly, axis=Axis.ENERGY))


def crop_energy(fmap: FieldMap, lo: float | None, hi: float | None) -> FieldMap:
    """Keep the energies inside ``[lo, hi]`` (inclusive; None = no limit)."""
    mask = energy_mask(fmap.energy, lo, hi)
    if not mask.any():
        raise ValueError(f"energy range {lo} to {hi} {fmap.unit} contains no data")
    return fmap.replace(energy=fmap.energy[mask], values=fmap.values[mask])


GAP_STEPS = 1.5  # parts further apart than this many (coarser) steps leave a gap


@dataclass(frozen=True)
class Seam:
    """Where two neighbouring parts of :func:`merge_energy` meet (indices into its parts).

    *kind* is ``"overlap"`` (each part keeps its side of the middle of ``lo`` - ``hi``),
    ``"touch"`` (the ranges follow each other), ``"gap"`` (``lo`` - ``hi`` has no data and
    stays empty) or ``"inside"`` (the upper part lies inside the lower one: no merge).
    """

    lower: int
    upper: int
    kind: str
    lo: float
    hi: float

    @property
    def middle(self) -> float:
        return (self.lo + self.hi) / 2


def _median_step(energy: np.ndarray) -> float:
    steps = np.diff(np.sort(energy))
    steps = steps[steps > 0]
    return float(np.median(steps)) if steps.size else math.inf


def energy_seams(parts: Sequence[tuple[FieldMap, float | None, float | None]]) -> list[Seam]:
    """The seams between the parts of :func:`merge_energy` (each cut to its ``(lo, hi)``
    range), lowest first. Two parts overlap when the upper one starts below the end of the
    lower one; a part starting on the last energy of the other one overlaps it in that sample.
    """
    spans = []
    for fmap, lo, hi in parts:
        energy = crop_energy(fmap, lo, hi).energy
        spans.append((float(energy.min()), float(energy.max()), _median_step(energy)))
    order = sorted(range(len(spans)), key=lambda i: spans[i][:2])
    seams = []
    for a, b in itertools.pairwise(order):
        (a_lo, a_hi, a_step), (b_lo, b_hi, b_step) = spans[a], spans[b]
        steps = [s for s in (a_step, b_step) if math.isfinite(s)]
        coarse = max(steps) if steps else 0.0
        if b_lo == a_lo:  # same start: the shorter one lies inside the other
            seams.append(Seam(b, a, "inside", a_lo, a_hi))
        elif b_hi <= a_hi:
            seams.append(Seam(a, b, "inside", b_lo, b_hi))
        elif b_lo <= a_hi:
            seams.append(Seam(a, b, "overlap", b_lo, a_hi))
        elif b_lo - a_hi > GAP_STEPS * coarse:
            seams.append(Seam(a, b, "gap", a_hi, b_lo))
        else:
            seams.append(Seam(a, b, "touch", a_hi, b_lo))
    return seams


def _gap_energies(seam: Seam, low_step: float, high_step: float) -> list[float]:
    """Energies of the empty samples that mark the gap of *seam*: one step beyond each side
    (so the samples next to the gap keep their own width when drawn), or its middle."""
    inner = [seam.lo + low_step, seam.hi - high_step]
    if all(math.isfinite(e) for e in inner) and seam.lo < inner[0] < inner[1] < seam.hi:
        return inner
    return [seam.middle]


def merge_energy(parts: Sequence[tuple[FieldMap, float | None, float | None]]) -> FieldMap:
    """Join maps measured in different spectral ranges into one map.

    Every part is cropped to its ``(lo, hi)`` range. Where two parts overlap, each energy
    comes from one of them: the lower part up to the middle of the overlap, the upper one
    above it (:func:`energy_seams`). Nothing is re-gridded: the result keeps the measured
    energies of every part, so its step changes where the parts meet. A gap between two
    parts stays empty: it is marked by missing values (NaN), one step beyond each side. A
    part that lies inside another cannot be merged.
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
    seams = energy_seams(parts)
    keep = [np.ones(c.energy.size, dtype=bool) for c in cropped]
    for seam in seams:
        if seam.kind == "inside":
            raise ValueError(
                f"part {seam.upper + 1} lies inside the energy range of part {seam.lower + 1}"
            )
        if seam.kind == "overlap":
            keep[seam.lower] &= cropped[seam.lower].energy <= seam.middle
            keep[seam.upper] &= cropped[seam.upper].energy > seam.middle
    energies = [c.energy[k] for c, k in zip(cropped, keep, strict=True)]
    values = [c.values[k] for c, k in zip(cropped, keep, strict=True)]
    for seam in seams:
        if seam.kind == "gap":
            steps = (
                _median_step(cropped[seam.lower].energy),
                _median_step(cropped[seam.upper].energy),
            )
            gap = np.array(_gap_energies(seam, *steps))
            energies.append(gap)
            values.append(np.full((gap.size, first.field.size), np.nan))
    energy, values = np.concatenate(energies), np.concatenate(values)
    # a part listing an energy twice: average the duplicates
    energy, inverse, counts = np.unique(energy, return_inverse=True, return_counts=True)
    summed = np.zeros((energy.size, values.shape[1]))
    np.add.at(summed, inverse, values)
    return first.replace(energy=energy, values=summed / counts[:, None])


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
    energy = energy[energy_mask(energy, lo, hi)]  # inclusive, robust to rounded tables
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


def average_maps(maps: Sequence[FieldMap]) -> FieldMap:
    """Mean of repeated measurements of the same field points.

    All maps must have the same unit and field values. They are interpolated onto the
    energy axis of the first map, restricted to the range all maps cover.
    """
    if not maps:
        raise ValueError("nothing to average")
    first = maps[0]
    for fmap in maps[1:]:
        if fmap.unit != first.unit:
            raise ValueError(f"cannot average {fmap.unit} with {first.unit} data")
        if fmap.field.shape != first.field.shape or not np.allclose(fmap.field, first.field):
            raise ValueError("averaged datasets must have the same field values")
    if len(maps) == 1:
        return first
    energy = _common_energy(maps)
    stack = np.stack([_on_energy(m, energy).values for m in maps])
    return first.replace(energy=energy, values=stack.mean(axis=0))
