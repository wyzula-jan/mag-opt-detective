"""The user's own figure presets: style settings saved under a name, read from and written to
JSON (the settings store and preset files share the format).

A :class:`UserPreset` keeps the figure's *style*: the journal it is based on (whose rules and
warnings still apply), the size, text and line width, file format and resolution, the colour
bar (on or off, and where) with the ticks, and the colour range mode. What one figure shows is
not part of it: the view, axis and colour bar labels, the panel letter, model curves, points
and the values of a fixed colour range.

A preset file is a JSON object ``{"content": ..., "version": 1, "presets": [...]}``. Files of a
newer format version, and anything else that is not a valid preset (including values beyond
what any figure may be, ``SIZE_LIMITS_MM`` and the like), raise :class:`PresetError` with a
message for people.
"""

from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import dataclass, field

from mag_opt_detective.export.presets import (
    DPI_LIMITS,
    FONT_LIMITS_PT,
    LINE_LIMITS_PT,
    MAX_PIXELS,
    PRESETS,
    SIZE_LIMITS_MM,
)
from mag_opt_detective.export.style import FigureStyle

FORMAT_VERSION = 1
CONTENT = "Magneto-Optical Detective journal figure presets"
# what a figure can be saved as (file suffixes without the dot)
FIGURE_FORMATS: tuple[str, ...] = ("pdf", "svg", "eps", "png", "tif")
# colour range of a map: the main window's levels, 1st-99th percentile, or typed values
COLOUR_RANGES: tuple[str, ...] = ("window", "auto", "fixed")
MAX_NAME = 60


class PresetError(ValueError):
    """A preset (or a preset file) that cannot be read; the message says why."""


def clean_name(name) -> str:
    """*name* without surrounding spaces; PresetError if it is empty or too long."""
    if not isinstance(name, str) or not name.strip():
        raise PresetError("a preset needs a name")
    name = " ".join(name.split())
    if len(name) > MAX_NAME:
        raise PresetError(f"a preset name has at most {MAX_NAME} characters")
    return name


def _size(name: str, value, limits: tuple[float, float], unit: str) -> float:
    """A number within *limits* (PresetError if it is not one)."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise PresetError(f"{name} must be a number")
    try:
        value = float(value)
    except OverflowError:
        raise PresetError(f"{name} is too large") from None
    lo, hi = limits
    if not (math.isfinite(value) and lo <= value <= hi):
        raise PresetError(f"{name} must be {lo:g}–{hi:g} {unit}, not {value:g}")
    return value


@dataclass(frozen=True)
class UserPreset:
    """Style settings under a name; sizes in mm, text and lines in pt.

    *journal* is the key of the journal preset it is based on; *width* one of that journal's
    columns, or "" with *width_mm* for a free width (Custom). *colour_range* is one of
    ``COLOUR_RANGES``.
    """

    name: str
    journal: str
    width: str
    width_mm: float | None
    height_mm: float
    font_pt: float
    line_pt: float
    format: str
    dpi: float
    colorbar: bool = True
    style: FigureStyle = field(default_factory=FigureStyle)
    colour_range: str = "window"

    def __post_init__(self) -> None:
        set_ = object.__setattr__
        set_(self, "name", clean_name(self.name))
        journal = PRESETS.get(self.journal) if isinstance(self.journal, str) else None
        if journal is None:
            raise PresetError(f"unknown journal {self.journal!r}; use one of {', '.join(PRESETS)}")
        if journal.free_size:
            if self.width:
                raise PresetError(f"{journal.name} has no column {self.width!r}")
            set_(self, "width_mm", _size("the width", self.width_mm, SIZE_LIMITS_MM, "mm"))
        else:
            if self.width not in journal.widths_mm:
                columns = ", ".join(journal.widths_mm)
                raise PresetError(f"{journal.name} has no column {self.width!r} (use {columns})")
            set_(self, "width_mm", None)
        for name, label, limits, unit in (
            ("height_mm", "the height", SIZE_LIMITS_MM, "mm"),
            ("font_pt", "the text size", FONT_LIMITS_PT, "pt"),
            ("line_pt", "the line width", LINE_LIMITS_PT, "pt"),
            ("dpi", "the resolution", DPI_LIMITS, "dpi"),
        ):
            set_(self, name, _size(label, getattr(self, name), limits, unit))
        width = self.width_mm if journal.free_size else journal.widths_mm[self.width]
        pixels = width * self.height_mm * (self.dpi / 25.4) ** 2
        if pixels > MAX_PIXELS:
            raise PresetError(
                f"at {self.dpi:g} dpi the image would have {pixels / 1e6:.0f} million pixels"
            )
        if self.format not in FIGURE_FORMATS:
            raise PresetError(f"unknown file format {self.format!r}")
        if not isinstance(self.colorbar, bool):
            raise PresetError("colour bar must be true or false")
        if not isinstance(self.style, FigureStyle):
            raise PresetError("the style must be a FigureStyle")
        if self.colour_range not in COLOUR_RANGES:
            raise PresetError(f"unknown colour range {self.colour_range!r}")

    def same_settings(self, other: UserPreset) -> bool:
        """Whether *other* holds the same settings (whatever its name)."""
        return dataclasses.replace(other, name=self.name) == self

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "journal": self.journal,
            "width": self.width,
            "width_mm": self.width_mm,
            "height_mm": self.height_mm,
            "font_pt": self.font_pt,
            "line_pt": self.line_pt,
            "format": self.format,
            "dpi": self.dpi,
            "colorbar": self.colorbar,
            **self.style.to_dict(),
            "colour_range": self.colour_range,
        }

    @classmethod
    def from_dict(cls, data) -> UserPreset:
        """A preset from :meth:`to_dict`; missing settings take the journal's defaults and
        the default style. PresetError if it cannot be read."""
        if not isinstance(data, dict):
            raise PresetError("a preset must be an object")
        name = clean_name(data.get("name"))
        key = data.get("journal")
        journal = PRESETS.get(key) if isinstance(key, str) else None
        if journal is None:
            raise PresetError(f"unknown journal {key!r}; use one of {', '.join(PRESETS)}")
        try:
            style = FigureStyle.from_dict(data)
        except ValueError as exc:
            raise PresetError(str(exc)) from None
        width = data.get("width", journal.default_width or "")
        if not isinstance(width, str):
            raise PresetError("the column must be text")
        return cls(
            name=name,
            journal=journal.key,
            width=width,
            width_mm=data.get("width_mm", journal.default_width_mm if journal.free_size else None),
            height_mm=data.get("height_mm", journal.default_height_mm),
            font_pt=data.get("font_pt", journal.font_size_pt),
            line_pt=data.get("line_pt", journal.line_width_pt),
            format=data.get("format", "pdf"),
            dpi=data.get("dpi", journal.raster_dpi),
            colorbar=data.get("colorbar", True),
            style=style,
            colour_range=data.get("colour_range", "window"),
        )


def same_name(a: str, b: str) -> bool:
    """Preset names are compared without case."""
    return a.casefold() == b.casefold()


def presets_to_json(presets: list[UserPreset]) -> str:
    """A preset file (or stored settings) holding *presets*."""
    document = {
        "content": CONTENT,
        "version": FORMAT_VERSION,
        "presets": [preset.to_dict() for preset in presets],
    }
    return json.dumps(document, indent=2, ensure_ascii=False) + "\n"


def _document(text) -> list:
    """The list of presets of a preset file's JSON text (PresetError if it is not one)."""
    try:
        if isinstance(text, bytes):
            text = text.decode("utf-8")
        data = json.loads(text)
    except (ValueError, TypeError, RecursionError) as exc:  # bad UTF-8 or JSON, deep nesting
        where = f" (line {exc.lineno})" if isinstance(exc, json.JSONDecodeError) else ""
        raise PresetError(f"not a presets file: the JSON cannot be read{where}") from None
    if not isinstance(data, dict) or not isinstance(data.get("presets"), list):
        raise PresetError("not a presets file: it has no list of presets")
    version = data.get("version")
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise PresetError("not a presets file: it has no format version")
    if version > FORMAT_VERSION:
        raise PresetError(
            f"the presets were saved by a newer version of the app (format {version}); "
            f"this version reads format {FORMAT_VERSION}"
        )
    return data["presets"]


def presets_from_json(text: str | bytes) -> list[UserPreset]:
    """Every preset of a preset file; PresetError naming the first problem (nothing is read
    from a file with a bad preset, or with two presets of the same name)."""
    presets: list[UserPreset] = []
    for number, data in enumerate(_document(text), start=1):
        name = data.get("name") if isinstance(data, dict) else None
        what = f"preset {number}" + (f" ({name!r})" if isinstance(name, str) else "")
        try:
            preset = UserPreset.from_dict(data)
        except PresetError as exc:
            raise PresetError(f"{what}: {exc}") from None
        if any(same_name(preset.name, other.name) for other in presets):
            raise PresetError(f"the name {preset.name!r} is used twice")
        presets.append(preset)
    return presets


def stored_presets(text: str | bytes) -> tuple[list[UserPreset], list[str]]:
    """The readable presets of stored settings and the problems of the others (skipped)."""
    try:
        items = _document(text)
    except PresetError as exc:
        return [], [str(exc)]
    presets: list[UserPreset] = []
    problems: list[str] = []
    for data in items:
        try:
            preset = UserPreset.from_dict(data)
        except PresetError as exc:
            problems.append(str(exc))
            continue
        if not any(same_name(preset.name, other.name) for other in presets):
            presets.append(preset)
    return presets, problems
