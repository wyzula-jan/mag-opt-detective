"""Points picked on the colour map: one energy per field value and named curve."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from mag_opt_detective.core.spectra import file_errors
from mag_opt_detective.core.units import Unit, axis_label, from_cm1, parse_axis_label, to_cm1


class PointTable:
    """Table indexed by field, one column of energies per named curve (NaN = empty).

    Energies are kept in cm^-1; the files can use any unit (see :meth:`save_tsv`).
    """

    def __init__(self, field: np.ndarray, columns: dict[str, np.ndarray] | None = None):
        self.field = np.asarray(field, dtype=float)
        self._columns: dict[str, np.ndarray] = {}
        for name, values in (columns or {}).items():
            values = np.asarray(values, dtype=float)
            if values.shape != self.field.shape:
                raise ValueError(
                    f"column {name!r} has {values.size} rows, expected {self.field.size}"
                )
            self._columns[name] = values.copy()

    @property
    def names(self) -> list[str]:
        return list(self._columns)

    def column(self, name: str) -> np.ndarray:
        return self._columns[name]

    def add_column(self, name: str) -> None:
        if name not in self._columns:
            self._columns[name] = np.full(self.field.shape, np.nan)

    def drop(self, name: str) -> None:
        self._columns.pop(name, None)

    def nearest_row(self, b: float) -> int:
        return int(np.abs(self.field - b).argmin())

    def set_nearest(self, name: str, b: float, energy: float) -> int:
        """Store *energy* at the field row closest to *b*; creates the column if needed."""
        self.add_column(name)
        row = self.nearest_row(b)
        self._columns[name][row] = energy
        return row

    def clear_nearest(self, name: str, b: float) -> int:
        self.add_column(name)
        row = self.nearest_row(b)
        self._columns[name][row] = np.nan
        return row

    def points(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """(field, energy) pairs of the non-empty cells of column *name*."""
        values = self._columns[name]
        mask = ~np.isnan(values)
        return self.field[mask], values[mask]

    def save_tsv(self, path: str | Path, unit: Unit | str = Unit.CM1) -> None:
        """Tab-separated table with the energies in *unit* and empty cells for NaN.

        The first header cell names the unit: ``Energy (meV)<TAB>LL 1<TAB>LL 2``.
        """
        columns = [from_cm1(self._columns[name], unit) for name in self.names]
        lines = ["\t".join([axis_label(unit), *self.names])]
        for row, b in enumerate(self.field):
            cells = [repr(float(b))]
            for values in columns:
                v = values[row]
                cells.append("" if math.isnan(v) else f"{v:.10g}")
            lines.append("\t".join(cells))
        Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")

    @classmethod
    def load_tsv(cls, path: str | Path, default_unit: Unit | str = Unit.CM1) -> PointTable:
        """Read a table written by :meth:`save_tsv`, converting the energies to cm^-1.

        Legacy tables have an empty first header cell; their energies are taken to be
        in *default_unit*. Two curves with one name are an error.
        """
        name = Path(path).name
        with file_errors(path):
            text = Path(path).read_text(encoding="utf-8").splitlines()
            rows = [
                (n, line.rstrip("\r").split("\t"))
                for n, line in enumerate(text, start=1)
                if line.strip()
            ]
            if not rows:
                raise ValueError(f"{name}: empty file")
            header = rows[0][1]
            unit = parse_axis_label(header[0]) or Unit(default_unit)
            names = [n.strip() or f"unnamed_{i}" for i, n in enumerate(header[1:], start=1)]
            twice = sorted({n for n in names if names.count(n) > 1})
            if twice:
                raise ValueError(f"{name}: more than one curve is named {twice[0]!r}")
            field = []
            data = []
            for line, cells in rows[1:]:
                cells = cells + [""] * (len(names) + 1 - len(cells))
                try:
                    field.append(float(cells[0]))
                    data.append(
                        [float(c) if c.strip() else np.nan for c in cells[1 : len(names) + 1]]
                    )
                except ValueError as exc:
                    raise ValueError(f"{name}: line {line}: {exc}") from exc
        values = to_cm1(np.array(data, dtype=float).reshape(len(field), len(names)), unit)
        return cls(np.array(field), {n: values[:, i] for i, n in enumerate(names)})
