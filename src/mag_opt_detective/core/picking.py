"""Automatic picking of transitions in spectra and field maps (pure numpy/scipy).

Nothing here converts units: energies, windows and slopes are in the unit of the energy
axis (or :class:`FieldMap`) passed in, fields in tesla.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import find_peaks, savgol_filter

from mag_opt_detective.core.spectra import FieldMap

Smoothing = tuple[int, int]
"""Savitzky-Golay ``(window, polyorder)``; the window is in samples and odd."""

Range = tuple[float | None, float | None]
"""Inclusive ``(lo, hi)``; None leaves that side open."""

Direction = Literal["both", "up", "down"]

#: default linking distance of :func:`detect`, in median energy steps
JUMP_SAMPLES = 3.0

#: :func:`auto_prominence`: the threshold in noise spreads (noise rarely reaches it) ...
AUTO_NOISE = 6.0
#: ... of the signal minus its running median over so many samples ...
AUTO_TREND = 51
#: ... in this many columns of the box at most
AUTO_COLUMNS = 32


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


@dataclass(frozen=True, eq=False)
class Track:
    """One feature followed across field columns, sorted by field."""

    field: np.ndarray
    energy: np.ndarray
    strength: np.ndarray

    def __post_init__(self) -> None:
        field = np.asarray(self.field, dtype=float)
        energy = np.asarray(self.energy, dtype=float)
        strength = np.asarray(self.strength, dtype=float)
        if field.ndim != 1 or not field.shape == energy.shape == strength.shape:
            raise ValueError("field, energy and strength must be 1D arrays of equal length")
        order = np.argsort(field, kind="stable")
        object.__setattr__(self, "field", field[order])
        object.__setattr__(self, "energy", energy[order])
        object.__setattr__(self, "strength", strength[order])

    def __len__(self) -> int:
        return self.field.size


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


def track(
    fmap: FieldMap,
    seed: tuple[float, float],
    feature: Feature | str,
    *,
    window: float,
    smooth: Smoothing | None = None,
    prominence: float | None = None,
    max_misses: int = 2,
    history: int = 3,
    direction: Direction = "both",
) -> Track:
    """Follow one feature through *fmap*, starting at *seed* = ``(field, energy)``.

    The feature nearest the seed energy within ±*window* in the field column nearest the
    seed starts the track. From there the walk goes column by column up and/or down in
    field (*direction*): the next energy is predicted by a straight line through the last
    *history* accepted points and the feature nearest the prediction within ±*window* is
    accepted. A direction stops after more than *max_misses* consecutive columns without
    a feature. Columns without any finite value are skipped and do not count as misses.
    Features are searched in the full column with :func:`find_features`.
    """
    feature = Feature(feature)
    if not window > 0:
        raise ValueError("the search window must be positive")
    if max_misses < 0 or history < 1:
        raise ValueError("max_misses must be >= 0 and history >= 1")
    if direction not in ("both", "up", "down"):
        raise ValueError(f"unknown direction {direction!r}")
    if fmap.field.size == 0 or fmap.energy.size == 0:
        raise ValueError("the map is empty")
    b, e = (float(v) for v in seed)
    in_field = np.nanmin(fmap.field) <= b <= np.nanmax(fmap.field)
    in_energy = np.nanmin(fmap.energy) <= e <= np.nanmax(fmap.energy)
    if not (in_field and in_energy):
        raise ValueError(f"seed ({b:g} T, {e:g} {fmap.unit}) is outside the map")

    def features(j: int) -> list[Peak]:
        return find_features(
            fmap.energy, fmap.values[:, j], feature, smooth=smooth, prominence=prominence
        )

    start = int(np.abs(fmap.field - b).argmin())
    first = _nearest_peak(features(start), e, window)
    if first is None:
        raise ValueError(
            f"no {feature} feature within ±{window:g} {fmap.unit} of the seed "
            f"at {fmap.field[start]:g} T"
        )
    order = np.argsort(fmap.field, kind="stable")
    pos = int(np.flatnonzero(order == start)[0])
    points = [(float(fmap.field[start]), first)]
    walks = {"up": [order[pos + 1 :]], "down": [order[:pos][::-1]]}
    walks["both"] = walks["up"] + walks["down"]
    for walk in walks[direction]:
        accepted = [points[0]]
        misses = 0
        for j in walk:
            if not np.isfinite(fmap.values[:, j]).any():
                continue
            b_j = float(fmap.field[j])
            recent = accepted[-history:]
            guess = _predict([p[0] for p in recent], [p[1].energy for p in recent], b_j)
            peak = _nearest_peak(features(j), guess, window)
            if peak is None:
                misses += 1
                if misses > max_misses:
                    break
                continue
            misses = 0
            accepted.append((b_j, peak))
        points += accepted[1:]
    return _to_track(points)


def detect(
    fmap: FieldMap,
    *,
    feature: Feature | str,
    b_range: Range | None = None,
    e_range: Range | None = None,
    smooth: Smoothing | None = None,
    prominence: float | None = None,
    max_jump: float | None = None,
    min_length: int = 3,
    max_misses: int = 0,
    history: int = 1,
    mask: np.ndarray | None = None,
) -> list[Track]:
    """All tracks of *feature* inside the box *b_range* x *e_range*, sorted by mean energy.

    Features are found in every column of the box with :func:`find_features` and linked
    between neighbouring columns (in field order) by nearest neighbour: the closest pairs
    of a track end and a feature of the next column within *max_jump* are joined first,
    each track and each feature at most once; features left over start new tracks.
    *max_jump* defaults to :data:`JUMP_SAMPLES` median energy steps. Columns without any
    finite value inside the box are skipped: tracks link across them, with the jump limit
    multiplied by the number of field steps bridged. Tracks with fewer than *min_length*
    points are dropped.

    *max_misses*: a track that finds no feature in a column with data stays open for up to
    this many such columns in a row (the jump limit grows with the field steps bridged, as
    across empty columns); 0 ends it at once. *history*: a track's end is predicted by a
    straight line through its last *history* points (as in :func:`track`); 1 compares with
    the last point.

    *mask* (bool, the shape of ``fmap.values``, e.g. from :func:`polygon_mask`) restricts
    the search to a region inside the box: each column is searched between the lowest and
    highest energy of its samples in the mask, and a feature counts only if its nearest
    sample is in the mask. Columns without any finite value in the mask are skipped as
    above. The box still limits the search, so pass the region's bounds for speed.
    """
    feature = Feature(feature)
    if min_length < 1:
        raise ValueError("min_length must be >= 1")
    if max_misses < 0 or history < 1:
        raise ValueError("max_misses must be >= 0 and history >= 1")
    if max_jump is None:
        steps = np.diff(np.sort(fmap.energy[np.isfinite(fmap.energy)]))
        max_jump = JUMP_SAMPLES * float(np.median(steps)) if steps.size else 0.0
    elif not max_jump > 0:
        raise ValueError("max_jump must be positive")
    if mask is not None:
        mask = _check_mask(fmap, mask)
    columns = np.flatnonzero(_inside(fmap.field, b_range))
    columns = columns[np.argsort(fmap.field[columns], kind="stable")]
    rows = _inside(fmap.energy, e_range)

    done: list[_Open] = []
    active: list[_Open] = []
    for n, j in enumerate(columns):
        keep = rows if mask is None else rows & mask[:, j]
        if not np.isfinite(fmap.values[keep, j]).any():
            continue
        b_j = float(fmap.field[j])
        peaks = find_features(
            fmap.energy,
            fmap.values[:, j],
            feature,
            smooth=smooth,
            prominence=prominence,
            e_range=e_range if mask is None else _span(fmap.energy[keep]),
        )
        if mask is not None:
            peaks = [p for p in peaks if keep[p.index]]
        guess = np.array([t.predict(b_j, history) for t in active])
        limit = np.array([max_jump * (n - t.column) for t in active])
        found = np.array([p.energy for p in peaks])
        dist = np.abs(guess[:, None] - found[None, :])
        t_idx, p_idx = np.nonzero(dist <= limit[:, None])
        by_distance = np.argsort(dist[t_idx, p_idx], kind="stable")
        used_t: set[int] = set()
        used_p: set[int] = set()
        for t, p in zip(t_idx[by_distance], p_idx[by_distance], strict=True):
            if t in used_t or p in used_p:
                continue
            used_t.add(int(t))
            used_p.add(int(p))
            active[t].add(b_j, peaks[p], n)
        still: list[_Open] = []
        for i, tr in enumerate(active):
            if i not in used_t:
                tr.misses += 1
            (still if tr.misses <= max_misses else done).append(tr)
        active = still + [_Open([(b_j, pk)], n) for i, pk in enumerate(peaks) if i not in used_p]
    done += active
    tracks = [_to_track(tr.points) for tr in done if len(tr.points) >= min_length]
    return sorted(tracks, key=lambda tr: float(tr.energy.mean()))


def auto_prominence(
    fmap: FieldMap,
    feature: Feature | str,
    *,
    smooth: Smoothing | None = None,
    b_range: Range | None = None,
    e_range: Range | None = None,
    columns: int = AUTO_COLUMNS,
    mask: np.ndarray | None = None,
) -> float:
    """A prominence threshold for *feature* above the noise of *fmap* in the box.

    The searched signal (the values, or the slope ``dV/dE`` for RISING and FALLING, smoothed
    as :func:`find_features` smooths it) of up to *columns* columns of the box *b_range* x
    *e_range*, sampled evenly, minus its running median over :data:`AUTO_TREND` samples, is
    mostly noise; its robust spread (1.4826 x the median absolute deviation) times
    :data:`AUTO_NOISE` is the threshold. Smoothing lowers it, lines and slow backgrounds hardly
    change it, and a map without noise gets a threshold near 0.0 (0.0 for an empty box).
    A *mask* (as in :func:`detect`) limits each column to the energies of its samples in it.
    """
    feature = Feature(feature)
    _check_smoothing(smooth)
    inside = _inside(fmap.field, b_range)
    if mask is not None:
        mask = _check_mask(fmap, mask) & _inside(fmap.energy, e_range)[:, None]
        inside &= mask.any(axis=0)
    inside = np.flatnonzero(inside)
    if inside.size > columns:
        inside = inside[np.linspace(0, inside.size - 1, columns).round().astype(int)]
    residuals = []
    for j in inside:
        keep = np.flatnonzero(np.isfinite(fmap.energy) & np.isfinite(fmap.values[:, j]))
        keep = keep[np.argsort(fmap.energy[keep], kind="stable")]
        x, y = fmap.energy[keep], fmap.values[keep, j]
        if x.size < 3 or np.any(np.diff(x) == 0):
            continue
        y = _smooth(y, smooth)
        span = e_range if mask is None else _span(fmap.energy[mask[:, j]])
        if span is not None:
            cut = _inside(x, span)
            x, y = x[cut], y[cut]
        if x.size < 3:
            continue
        signal = y if feature in (Feature.MAX, Feature.MIN) else np.gradient(y, x)
        trend = median_filter(signal, size=min(AUTO_TREND, signal.size), mode="nearest")
        residuals.append(signal - trend)
    if not residuals:
        return 0.0
    residual = np.concatenate(residuals)
    spread = 1.4826 * float(np.median(np.abs(residual - np.median(residual))))
    return AUTO_NOISE * spread


def polygon_mask(fmap: FieldMap, polygon: np.ndarray) -> np.ndarray:
    """The samples of *fmap* inside *polygon*, a mask for :func:`detect` (bool, the shape of
    ``fmap.values``).

    *polygon* is an ``(n, 2)`` array of ``(field, energy)`` vertices in the map's units, closed
    from the last vertex back to the first; it may be concave or cross itself (even-odd rule).
    Each field column is cut by the polygon's edges, so any field and energy grid works
    (uneven, unsorted, with NaN samples left outside). Samples exactly on an edge may fall on
    either side.
    """
    vertices = np.asarray(polygon, dtype=float).reshape(-1, 2)
    field, energy = fmap.field, fmap.energy
    mask = np.zeros(fmap.values.shape, dtype=bool)
    vertices = vertices[np.isfinite(vertices).all(axis=1)]
    if len(vertices) < 3:
        return mask
    x0, y0 = vertices[:, 0], vertices[:, 1]
    x1, y1 = np.roll(x0, -1), np.roll(y0, -1)
    b = field[:, None]
    cuts = (x0 > b) != (x1 > b)  # edges that cross each column (half-open: vertices once)
    with np.errstate(divide="ignore", invalid="ignore"):
        at = y0 + (b - x0) / (x1 - x0) * (y1 - y0)
    crossings = np.sort(np.where(cuts, at, np.inf), axis=1)
    counts = cuts.sum(axis=1)
    for j in np.flatnonzero(counts):
        below = np.searchsorted(crossings[j, : counts[j]], energy, side="right")
        mask[:, j] = (below % 2 == 1) & np.isfinite(energy)
    return mask


class _Open:
    """A track while :func:`detect` builds it: its points, last column and misses since."""

    def __init__(self, points: list[tuple[float, Peak]], column: int):
        self.points = points
        self.column = column
        self.misses = 0

    def add(self, field: float, peak: Peak, column: int) -> None:
        self.points.append((field, peak))
        self.column = column
        self.misses = 0

    def predict(self, field: float, history: int) -> float:
        recent = self.points[-history:]
        return _predict([b for b, _ in recent], [p.energy for _, p in recent], field)


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


def _check_mask(fmap: FieldMap, mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != fmap.values.shape:
        raise ValueError(
            f"mask shape {mask.shape} does not match the map values {fmap.values.shape}"
        )
    return mask


def _span(x: np.ndarray) -> Range:
    """``(min, max)`` of the finite *x*; a range that holds nothing if there are none."""
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return math.nan, math.nan
    return float(finite.min()), float(finite.max())


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


def _nearest_peak(peaks: list[Peak], target: float, window: float) -> Peak | None:
    near = [p for p in peaks if abs(p.energy - target) <= window]
    return min(near, key=lambda p: abs(p.energy - target), default=None)


def _predict(field: list[float], energy: list[float], b: float) -> float:
    """Energy at *b* from a straight line through the given points (or the last one).

    The least-squares line of a few points, in plain Python: called for every open track in
    every column, where ``np.polyfit`` would take most of the time.
    """
    n = len(field)
    mean_b, mean_e = sum(field) / n, sum(energy) / n
    sxx = sum((x - mean_b) ** 2 for x in field)
    if n < 2 or sxx == 0:
        return float(energy[-1])
    sxy = sum((x - mean_b) * (y - mean_e) for x, y in zip(field, energy, strict=True))
    return float(mean_e + sxy / sxx * (b - mean_b))


def _to_track(points: list[tuple[float, Peak]]) -> Track:
    return Track(
        field=np.array([b for b, _ in points]),
        energy=np.array([p.energy for _, p in points]),
        strength=np.array([p.strength for _, p in points]),
    )
