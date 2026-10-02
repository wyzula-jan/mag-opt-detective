"""Field-dependent spectra container and its tab-separated file format."""

from __future__ import annotations

import dataclasses
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


def load_tsv(path: str | Path, default_unit: Unit | str = Unit.CM1) -> FieldMap:
    """Read a table written by :func:`save_tsv` or by the legacy pandas exports.

    The first header cell may be empty (merged legacy exports); then *default_unit*
    is assumed. Field values are parsed from column labels such as ``0.25T``.
    """
    with open(path, encoding="utf-8", newline=None) as fh:
        header = fh.readline().rstrip("\r\n").split("\t")
    if len(header) < 2:
        raise ValueError(f"{Path(path).name}: expected a tab-separated table with a header")
    unit = parse_axis_label(header[0]) or Unit(default_unit)
    try:
        field = np.array([parse_field_label(h) for h in header[1:]])
    except ValueError as exc:
        raise ValueError(f"{Path(path).name}: cannot parse field from header ({exc})") from exc

    try:
        table = np.loadtxt(path, delimiter="\t", skiprows=1, ndmin=2)
    except ValueError:
        # empty cells (NaN) are not supported by loadtxt
        table = np.genfromtxt(path, delimiter="\t", skip_header=1)
        table = np.atleast_2d(table)
    if table.shape[1] != field.size + 1:
        raise ValueError(
            f"{Path(path).name}: {table.shape[1] - 1} data columns but {field.size} header labels"
        )
    order = np.argsort(table[:, 0], kind="stable")
    table = table[order]
    return FieldMap(energy=table[:, 0], field=field, values=table[:, 1:], unit=unit)
