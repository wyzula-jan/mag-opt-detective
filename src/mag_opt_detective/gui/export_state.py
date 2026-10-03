"""What the export window draws: the window's plot as an export :class:`FigureState`, the
print size checked against a journal preset, and the default file name.

:func:`figure_state` adapts :meth:`AppController.figure_state` (the map as shown: display unit,
plot kind and derivative, ranges, colour map and the levels in effect, model overlays and the
picked points) and the stacked options of the view. :func:`print_size` turns the typed sizes
into the values drawn with, plus the errors that block saving and the warnings shown inline.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from mag_opt_detective.core.units import Unit
from mag_opt_detective.export import (
    FORMATS,
    Curve,
    FigureState,
    JournalPreset,
    PointSet,
    StackedOptions,
)
from mag_opt_detective.export.figure import FIELD_LABEL, INTENSITY_LABEL, MM_PER_INCH
from mag_opt_detective.gui.display import energy_label

KINDS = ("map", "stacked")
POINTS_ALL, POINTS_CURRENT, POINTS_NONE = "all", "current", "none"
FORMAT_LABELS = {"pdf": "PDF", "svg": "SVG", "eps": "EPS", "png": "PNG", "tif": "TIFF"}
SUFFIXES = {"pdf": ".pdf", "svg": ".svg", "eps": ".eps", "png": ".png", "tif": ".tif"}
RASTER_FORMATS = ("png", "tif")

# what any figure may be (beyond this the values are refused, not just warned about)
SIZE_LIMITS_MM = (5.0, 1000.0)
FONT_LIMITS_PT = (1.0, 72.0)
LINE_LIMITS_PT = (0.05, 10.0)
DPI_LIMITS = (50.0, 2400.0)
MAX_PIXELS = 120e6  # a larger image would need gigabytes of memory to draw


# ---------------------------------------------------------------------- content
@dataclass(frozen=True)
class FigureContent:
    """What the figure shows besides the map: view, colour bar, overlays and labels.

    *points*: ``"all"`` curves, the ``"current"`` one or ``"none"``. An empty axis label is
    filled in automatically; one of only spaces leaves the axis without a label.
    """

    kind: str = "map"
    colorbar: bool = True
    colorbar_label: str = ""
    models: bool = True
    points: str = POINTS_ALL
    x_label: str = ""
    y_label: str = ""


def auto_labels(kind: str, unit: Unit) -> tuple[str, str]:
    """The automatic (x, y) axis labels as shown to people (cm⁻¹ in Unicode)."""
    energy = energy_label(unit)
    return (FIELD_LABEL, energy) if kind == "map" else (energy, INTENSITY_LABEL)


def _label(text: str) -> str | None:
    """A typed label: None (automatic) when empty, "" (no label) when only spaces."""
    return None if not text else text.strip()


def model_curves(snapshot) -> list[Curve]:
    """The model overlays of the window as dashed figure curves."""
    curves = []
    for name, lines in snapshot.overlays.items():
        for b, e in lines:
            b, e = np.asarray(b, float), np.asarray(e, float)
            if b.size and np.isfinite(e).any():
                curves.append(Curve(b, e, style="model", label=name))
    return curves


def point_sets(snapshot, which: str = POINTS_ALL) -> list[PointSet]:
    """The picked points of *which* curves (the current one drawn filled); empty ones skipped."""
    if which == POINTS_NONE:
        return []
    sets = []
    for name, (b, e) in snapshot.points.items():
        if which == POINTS_CURRENT and name != snapshot.curve:
            continue
        keep = np.isfinite(b) & np.isfinite(e)
        if keep.any():
            sets.append(PointSet(b[keep], e[keep], label=name, current=name == snapshot.curve))
    return sets


def figure_state(controller, content: FigureContent, snapshot=None) -> FigureState:
    """The plot on screen as an export figure, in the display unit.

    *snapshot*: ``controller.figure_state()`` if taken already. ValueError without data.
    """
    if content.kind not in KINDS:
        raise ValueError(f"unknown figure view {content.kind!r}; choose from {KINDS}")
    snapshot = controller.figure_state() if snapshot is None else snapshot
    view = controller.view
    common = {
        "fmap": snapshot.fmap,
        "colorbar_label": content.colorbar_label.strip(),
        "x_label": _label(content.x_label),
        "y_label": _label(content.y_label),
        "points": point_sets(snapshot, content.points),
    }
    if content.kind == "map":
        return FigureState(
            kind="map",
            x_range=snapshot.field_range,
            y_range=snapshot.energy_range,
            levels=tuple(snapshot.levels),
            cmap=snapshot.colormap,
            colorbar=content.colorbar,
            curves=model_curves(snapshot) if content.models else [],
            **common,
        )
    return FigureState(
        kind="stacked",
        x_range=view.energy_range,
        y_range=view.stacked_range,
        cmap=view.trace_colormap(),
        colorbar=content.colorbar and view.stacked_by_field,
        stacked=StackedOptions(view.stacked_offset, view.stacked_every, view.stacked_by_field),
        **common,
    )


# ---------------------------------------------------------------------- print size
@dataclass(frozen=True)
class PrintSize:
    """Typed sizes made usable: the values drawn with, errors (no figure) and warnings."""

    width_mm: float
    height_mm: float
    font_pt: float
    line_pt: float
    dpi: float
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    invalid: frozenset[str] = frozenset()  # names of the refused fields

    @property
    def ok(self) -> bool:
        return not self.errors

    def pixels(self) -> tuple[int, int]:
        """Pixel size of a raster file (or of the map image inside a vector file)."""
        return (
            round(self.width_mm / MM_PER_INCH * self.dpi),
            round(self.height_mm / MM_PER_INCH * self.dpi),
        )


def _number(
    name: str,
    title: str,
    value: float | None,
    limits: tuple[float, float],
    unit: str,
    errors: list[str],
    invalid: set[str],
) -> float:
    if value is None or not math.isfinite(value):
        errors.append(f"Enter the {title} in {unit}.")
    elif not limits[0] <= value <= limits[1]:
        errors.append(f"The {title} must be {limits[0]:g}–{limits[1]:g} {unit}.")
    else:
        return float(value)
    invalid.add(name)
    return math.nan


def format_list(formats: tuple[str, ...]) -> str:
    names = [FORMAT_LABELS.get("tif" if f == "tiff" else f, f.upper()) for f in formats]
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " or " + names[-1]


def print_size(
    preset: JournalPreset,
    width_mm: float | None,
    height_mm: float | None,
    font_pt: float | None,
    line_pt: float | None,
    dpi: float | None,
    fmt: str = "pdf",
) -> PrintSize:
    """Check typed sizes against *preset*.

    The height is capped at the preset's maximum, the text size kept within its range and the
    resolution raised to its minimum, each with a warning; :meth:`JournalPreset.check` adds the
    rest (lines, resolution above the range). Values beyond what any figure can be are errors.
    """
    errors: list[str] = []
    invalid: set[str] = set()
    warnings: list[str] = []
    width = _number("width", "width", width_mm, SIZE_LIMITS_MM, "mm", errors, invalid)
    height = _number("height", "height", height_mm, SIZE_LIMITS_MM, "mm", errors, invalid)
    font = _number("font", "text size", font_pt, FONT_LIMITS_PT, "pt", errors, invalid)
    line = _number("line", "line width", line_pt, LINE_LIMITS_PT, "pt", errors, invalid)
    res = _number("dpi", "resolution", dpi, DPI_LIMITS, "dpi", errors, invalid)
    name = preset.name
    if preset.max_height_mm is not None and height > preset.max_height_mm:  # NaN: False
        height = preset.max_height_mm
        warnings.append(f"{name}: height is at most {height:g} mm; using {height:g} mm.")
    if math.isfinite(font):
        lo, hi = preset.font_size_range_pt
        if not lo <= font <= hi:
            font = min(max(font, lo), hi)
            warnings.append(f"{name}: text is {lo:g}–{hi:g} pt; using {font:g} pt.")
    if math.isfinite(res) and res < preset.raster_dpi_range[0]:
        res = float(preset.raster_dpi_range[0])
        warnings.append(f"{name}: images need at least {res:g} dpi; using {res:g} dpi.")
    if fmt not in preset.formats and not (fmt == "tif" and "tiff" in preset.formats):
        warnings.append(f"{name} asks for {format_list(preset.formats)} files.")
    if not errors:
        warnings += preset.check(width, height, line_width_pt=line, dpi=res)
        pixels = (width / MM_PER_INCH * res) * (height / MM_PER_INCH * res)
        if pixels > MAX_PIXELS:
            errors.append(
                f"At {res:g} dpi the image would have {pixels / 1e6:.0f} million pixels; "
                "use a lower resolution or a smaller size."
            )
            invalid.add("dpi")
    return PrintSize(width, height, font, line, res, errors, warnings, frozenset(invalid))


# ---------------------------------------------------------------------- file name
_UNSAFE = re.compile(r"[^A-Za-z0-9._+-]+")


def safe_name(text: str) -> str:
    """*text* usable in a file name on every system (other characters become ``_``)."""
    return _UNSAFE.sub("_", text).strip("._") or "figure"


def file_name(controller, kind: str, preset: JournalPreset, width: str, fmt: str) -> str:
    """A file name from the data, e.g. ``Sample_4p2K_Ratio_nature-single.pdf``."""
    parts = [controller.result_name(), controller.selection.export_name()]
    if kind == "stacked":
        parts.append("stacked")
    size = preset.key if preset.free_size or not width else f"{preset.key}-{width}"
    parts.append(size.replace(" ", "-"))
    return safe_name("_".join(parts)) + SUFFIXES[fmt]


def with_format(path: str | Path, fmt: str) -> Path:
    """*path* with the suffix of *fmt* (another figure suffix is replaced, anything else kept)."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == SUFFIXES[fmt] or (fmt == "tif" and suffix == ".tiff"):
        return path
    if suffix in FORMATS:
        return path.with_suffix(SUFFIXES[fmt])
    return path.with_name(path.name + SUFFIXES[fmt])
