"""What a journal figure shows: a snapshot of the plot, filled in by the GUI (no Qt)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from mag_opt_detective.core.spectra import FieldMap

Kind = Literal["map", "stacked"]
Range = tuple[float | None, float | None]

KINDS: tuple[str, ...] = ("map", "stacked")
CURVE_STYLES: tuple[str, ...] = ("model", "plain")


def _floats(values) -> np.ndarray:
    return np.atleast_1d(np.asarray(values, dtype=float))


@dataclass
class Curve:
    """A line over the map: energies *y* (display unit) at fields *x* (T).

    ``style="model"`` is drawn dashed, ``"plain"`` solid.
    """

    x: np.ndarray
    y: np.ndarray
    style: str = "model"
    label: str = ""

    def __post_init__(self) -> None:
        self.x, self.y = _floats(self.x), _floats(self.y)
        if self.x.shape != self.y.shape:
            raise ValueError(f"curve x {self.x.shape} and y {self.y.shape} differ in shape")
        if self.style not in CURVE_STYLES:
            raise ValueError(f"unknown curve style {self.style!r}; choose from {CURVE_STYLES}")


@dataclass
class PointSet:
    """Picked points of one curve: energies *y* (display unit) at fields *x* (T).

    The *current* curve is drawn with filled markers, the others open.
    """

    x: np.ndarray
    y: np.ndarray
    label: str = ""
    current: bool = False

    def __post_init__(self) -> None:
        self.x, self.y = _floats(self.x), _floats(self.y)
        if self.x.shape != self.y.shape:
            raise ValueError(f"points x {self.x.shape} and y {self.y.shape} differ in shape")


@dataclass
class StackedOptions:
    """Stacked spectra: every *every*-th field, shifted by *offset* per shown trace."""

    offset: float = 0.0
    every: int = 1
    color_by_field: bool = True

    def __post_init__(self) -> None:
        if int(self.every) < 1:
            raise ValueError("every must be at least 1")
        self.every = int(self.every)
        self.offset = float(self.offset)


@dataclass
class FigureState:
    """Everything a journal figure shows, in the display unit of *fmap*.

    *kind* "map" plots field (x) against energy (y) in colour; "stacked" plots the
    spectra (x = energy) on top of each other. Curves and points are always given as
    (field, energy); a stacked figure places the points on their traces and does not
    draw curves. Ranges are ``(lo, hi)`` with None (or a None end) for the data extent;
    *levels* None means the 1st/99th percentile of the map. Labels None are filled in
    automatically, "" leaves the axis without a label.
    """

    kind: Kind
    fmap: FieldMap
    x_range: Range | None = None
    y_range: Range | None = None
    levels: tuple[float, float] | None = None
    cmap: str = "magma"
    colorbar: bool = True
    colorbar_label: str = ""
    x_label: str | None = None
    y_label: str | None = None
    curves: list[Curve] = field(default_factory=list)
    points: list[PointSet] = field(default_factory=list)
    stacked: StackedOptions = field(default_factory=StackedOptions)
    title: str = ""

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"unknown figure kind {self.kind!r}; choose from {KINDS}")
