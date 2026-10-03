"""Field-dependent spectra container and its tab-separated file format."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mag_opt_detective.core.units import Unit, axis_label, convert, parse_axis_label

# Relative slack on energy limits: a limit converted from another unit is a few ulp off
# and must still include the sample it was converted from.
LIMIT_RTOL = 1e-9


@dataclass(frozen=True, eq=False)
class FieldMap:
    """Spectra as a function of energy and magnetic field.

    ``values[i, j]`` is the intensity at ``energy[i]`` and ``field[j]``.
    """

    energy: np.ndarray
    field: np.ndarray
    values: np.ndarray
    unit: Unit = Unit.CM1

    def __post_init__(self) -> None:
        energy = np.asarray(self.energy, dtype=float)
        field = np.asarray(self.field, dtype=float)
        values = np.asarray(self.values, dtype=float)
        if energy.ndim != 1 or field.ndim != 1:
            raise ValueError("energy and field must be 1D arrays")
        if values.shape != (energy.size, field.size):
            raise ValueError(
                f"values shape {values.shape} does not match "
                f"(energy, field) = ({energy.size}, {field.size})"
            )
        object.__setattr__(self, "energy", energy)
        object.__setattr__(self, "field", field)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "unit", Unit(self.unit))

    def replace(self, **changes) -> FieldMap:
        return dataclasses.replace(self, **changes)

    def with_values(self, values: np.ndarray) -> FieldMap:
        return dataclasses.replace(self, values=values)

    def to_unit(self, unit: Unit | str) -> FieldMap:
        """This map with the energy axis in *unit*; the values (shared) do not change.

        Returns the map itself if it is already in *unit*.
        """
        unit = Unit(unit)
        if unit is self.unit:
            return self
        return dataclasses.replace(self, energy=convert(self.energy, self.unit, unit), unit=unit)

    @property
    def field_labels(self) -> list[str]:
        return [field_label(b) for b in self.field]


def energy_mask(energy: np.ndarray, lo: float | None, hi: float | None) -> np.ndarray:
    """Samples of *energy* inside ``[lo, hi]`` (inclusive, see LIMIT_RTOL; None = open)."""
    energy = np.asarray(energy, dtype=float)
    mask = np.ones(energy.size, dtype=bool)
    if lo is not None:
        mask &= energy >= lo - abs(lo) * LIMIT_RTOL
    if hi is not None:
        mask &= energy <= hi + abs(hi) * LIMIT_RTOL
    return mask


def cell_edges(axis: np.ndarray) -> np.ndarray:
    """Edges of the cells around the samples of a rising *axis*: the midpoints between
    neighbours and half a step beyond the ends (a single sample is 1 wide)."""
    axis = np.asarray(axis, dtype=float)
    if axis.size == 1:
        return np.array([axis[0] - 0.5, axis[0] + 0.5])
    mid = (axis[:-1] + axis[1:]) / 2
    return np.concatenate([[2 * axis[0] - mid[0]], mid, [2 * axis[-1] - mid[-1]]])


def sample_at(axis: np.ndarray, x: float) -> int | None:
    """Index of the sample of *axis* (any order) whose cell (:func:`cell_edges`) holds *x*,
    which is the nearest sample; None beyond the outer cells. A value that occurs twice is
    its first sample."""
    axis = np.asarray(axis, dtype=float)
    if axis.size == 0 or not np.isfinite(x):
        return None
    values, first = np.unique(axis, return_index=True)
    edges = cell_edges(values)
    k = int(np.searchsorted(edges, x, side="right")) - 1
    if k == values.size and x == edges[-1]:
        k -= 1  # the outer edge belongs to the last cell
    if not 0 <= k < values.size:
        return None
    return int(first[k])


def field_label(b: float) -> str:
    """Column label used in exported files, e.g. ``0.25T``."""
    return f"{b:1.2f}T"


def parse_field_label(label: str) -> float:
    label = label.strip()
    if label[-1:] in ("T", "t"):
        label = label[:-1]
    return float(label)


def save_tsv(fmap: FieldMap, path: str | Path) -> None:
    """Write *fmap* as a tab-separated table (energy rows, field columns).

    Format kept compatible with the legacy exports::

        Energy (meV)<TAB>0.25T<TAB>0.50T ...
        12.5<TAB>1.001<TAB>0.998 ...
    """
    header = "\t".join([axis_label(fmap.unit), *fmap.field_labels])
    table = np.column_stack([fmap.energy, fmap.values])
    np.savetxt(path, table, delimiter="\t", header=header, comments="", fmt="%.12g")


@contextmanager
def file_errors(path: str | Path) -> Iterator[None]:
    """Name the file *path* in the ValueErrors raised inside the block (parse and decoding
    errors), unless the message names it already."""
    name = Path(path).name
    try:
        yield
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"{name}: not a text file in UTF-8 ({exc.reason} at byte {exc.start})"
        ) from exc
    except ValueError as exc:
        if str(exc).startswith(f"{name}:"):
            raise
        raise ValueError(f"{name}: {exc}") from exc


def load_tsv(path: str | Path, default_unit: Unit | str = Unit.CM1) -> FieldMap:
    """Read a table written by :func:`save_tsv` or by the legacy pandas exports.

    The first header cell may be empty (merged legacy exports); then *default_unit*
    is assumed. Field values are parsed from column labels such as ``0.25T``; the columns
    are sorted by field and the rows by energy.
    """
    name = Path(path).name
    with file_errors(path):
        with open(path, encoding="utf-8", newline=None) as fh:
            header = fh.readline().rstrip("\r\n").split("\t")
        if len(header) < 2:
            raise ValueError(f"{name}: expected a tab-separated table with a header")
        unit = parse_axis_label(header[0]) or Unit(default_unit)
        try:
            field = np.array([parse_field_label(h) for h in header[1:]])
        except ValueError as exc:
            raise ValueError(f"{name}: cannot parse field from header ({exc})") from exc
        try:
            table = np.loadtxt(path, delimiter="\t", skiprows=1, ndmin=2)
        except ValueError:
            # empty cells (NaN) are not supported by loadtxt
            table = np.genfromtxt(path, delimiter="\t", skip_header=1)
            table = np.atleast_2d(table)
    if table.shape[1] != field.size + 1:
        raise ValueError(
            f"{name}: {table.shape[1] - 1} data columns but {field.size} header labels"
        )
    rows = np.argsort(table[:, 0], kind="stable")
    columns = np.argsort(field, kind="stable")
    values = table[np.ix_(rows, columns + 1)]
    return FieldMap(energy=table[rows, 0], field=field[columns], values=values, unit=unit)
