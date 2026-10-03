"""Loading of measured spectra (OPUS binary or OPUS-macro text files)."""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mag_opt_detective.core.opus import is_opus_file, read_opus
from mag_opt_detective.core.spectra import FieldMap, energy_mask, file_errors
from mag_opt_detective.core.units import Unit

# Field encoded in file names, e.g. ``..._Sam2_a01p250T.txt`` -> 1.25 T.
# Zero-field references look like ``..._a00p000T_a16p000T`` -> the last match wins (16 T).
FIELD_PATTERN = re.compile(r"_a(\d+)p(\d+)T")


def parse_field(path: str | Path) -> float | None:
    """Magnetic field (T) encoded in the file name, or None if there is none."""
    matches = FIELD_PATTERN.findall(Path(path).name)
    if not matches:
        return None
    whole, frac = matches[-1]
    return float(f"{int(whole)}.{frac}")


def _natural_key(name: str) -> list[int | str]:
    return [int(tok) if tok.isdigit() else tok.lower() for tok in re.split(r"(\d+)", name)]


def sort_paths(paths: Iterable[str | Path]) -> list[str]:
    """Sort by the field in the file name, falling back to natural name order."""

    def key(p: str) -> tuple[bool, float, list[int | str]]:
        b = parse_field(p)
        return (b is None, b if b is not None else 0.0, _natural_key(Path(p).name))

    return sorted((str(p) for p in paths), key=key)


def read_text(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Two-column text file (x, y), any whitespace delimiter, CRLF safe."""
    with file_errors(path):
        data = np.loadtxt(path, ndmin=2)
    if data.shape[1] < 2:
        raise ValueError(f"{Path(path).name}: expected two columns (energy, intensity)")
    x, y = data[:, 0], data[:, 1]
    if x.size > 1 and x[0] > x[-1]:
        x, y = x[::-1], y[::-1]
    return x, y


def read_spectrum(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read one spectrum, auto-detecting OPUS binary vs. text. x is in cm^-1."""
    if is_opus_file(path):
        return read_opus(path)
    return read_text(path)


Reader = Callable[[str], tuple[np.ndarray, np.ndarray]]


def _signature(path: str) -> tuple[int, int]:
    stat = os.stat(path)
    return stat.st_size, stat.st_mtime_ns


class SpectrumCache:
    """Spectra read before, kept with the size and modification time of their file: a file is
    read again only when one of them changed (e.g. the growing folder of a running sweep).

    The arrays handed out are shared and read-only. A file that changes while it is read is
    not kept. :attr:`reads` counts the files actually read.
    """

    def __init__(self, read: Reader = read_spectrum):
        self._read = read
        self._spectra: dict[str, tuple[tuple[int, int], tuple[np.ndarray, np.ndarray]]] = {}
        self.reads = 0

    def __len__(self) -> int:
        return len(self._spectra)

    def __contains__(self, path: object) -> bool:
        return str(path) in self._spectra

    def read(self, path: str | Path) -> tuple[np.ndarray, np.ndarray]:
        """The spectrum of *path* (as :func:`read_spectrum`), from the cache if unchanged."""
        path = str(path)
        try:
            before = _signature(path)
        except OSError:
            self._spectra.pop(path, None)
            return self._read(path)  # the reader's own error for a missing file
        kept = self._spectra.pop(path, None)
        if kept is not None and kept[0] == before:
            self._spectra[path] = kept
            return kept[1]
        spectrum = self._read(path)
        self.reads += 1
        for array in spectrum:
            array.flags.writeable = False
        try:
            unchanged = _signature(path) == before
        except OSError:
            unchanged = False
        if unchanged:
            self._spectra[path] = (before, spectrum)
        return spectrum

    def retain(self, paths: Iterable[str | Path]) -> None:
        """Forget the spectra of every file but *paths*."""
        keep = {str(p) for p in paths}
        for path in [p for p in self._spectra if p not in keep]:
            del self._spectra[path]


@dataclass(frozen=True, eq=False)
class Measurement:
    """Field sweep of one sample: spectra in field plus one or two zero-field spectra.

    ``zero`` has shape ``(n_energy, n_zero)``. With two zero-field spectra (measured
    before and after the sweep) the zero reference drifts linearly over the sweep.
    :func:`load_measurement` gives the energy axis in cm^-1.
    """

    spectra: FieldMap
    zero: np.ndarray


def _read_stack(paths: Sequence[str], read: Reader) -> tuple[np.ndarray, np.ndarray]:
    x0: np.ndarray | None = None
    columns = []
    for p in paths:
        x, y = read(p)
        if x0 is None:
            x0 = x
        elif x.shape != x0.shape or not np.allclose(x, x0, rtol=0, atol=1e-6):
            raise ValueError(
                f"{Path(p).name}: energy axis differs from {Path(paths[0]).name}; "
                "all files of one measurement must share the same axis"
            )
        columns.append(y)
    assert x0 is not None
    return x0, np.column_stack(columns)


def load_measurement(
    zero_paths: Sequence[str | Path],
    field_paths: Sequence[str | Path],
    field: np.ndarray | None = None,
    energy_limits: tuple[float | None, float | None] = (None, None),
    read: Reader = read_spectrum,
) -> Measurement:
    """Load a field sweep; the energy axis stays in cm^-1, the unit of the files.

    Args:
        zero_paths: one or two zero-field spectra (before / after the sweep).
        field_paths: spectra measured in field, sorted with :func:`sort_paths`.
        field: field values; if None they are parsed from the file names.
        energy_limits: optional inclusive energy cut in cm^-1 (None = no limit).
        read: reads one file (e.g. :meth:`SpectrumCache.read`); :func:`read_spectrum`.
    """
    if not field_paths:
        raise ValueError("no field files loaded")
    if not zero_paths:
        raise ValueError("no zero-field files loaded")
    if len(zero_paths) > 2:
        raise ValueError("load one or two zero-field files (before and after the sweep)")

    field_paths = [str(p) for p in field_paths]
    if field is None:
        parsed = [parse_field(p) for p in field_paths]
        missing = [Path(p).name for p, b in zip(field_paths, parsed, strict=True) if b is None]
        if missing:
            raise ValueError(
                f"cannot read the field from file name(s) {missing[:3]}; "
                "use a custom field range instead"
            )
        field = np.array(parsed, dtype=float)
    else:
        field = np.asarray(field, dtype=float)
        if field.size != len(field_paths):
            raise ValueError(
                f"custom field range has {field.size} values "
                f"but {len(field_paths)} files are loaded"
            )

    x, values = _read_stack(field_paths, read)
    x_zero, zero = _read_stack([str(p) for p in zero_paths], read)
    if x_zero.shape != x.shape or not np.allclose(x_zero, x, rtol=0, atol=1e-6):
        raise ValueError("zero-field and field spectra have different energy axes")

    mask = energy_mask(x, *energy_limits)
    if not mask.any():
        lo, hi = ("open" if v is None else f"{v:.6g}" for v in energy_limits)
        raise ValueError(f"energy cut {lo} to {hi} cm-1 leaves no data")

    spectra = FieldMap(energy=x[mask], field=field, values=values[mask], unit=Unit.CM1)
    return Measurement(spectra=spectra, zero=zero[mask])
