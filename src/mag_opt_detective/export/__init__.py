"""Journal-quality figure export with matplotlib (no Qt, no pyplot)."""

from mag_opt_detective.export.figure import (
    FORMATS,
    colormap,
    energy_label,
    rasterize,
    render,
    resolve_font,
    save,
)
from mag_opt_detective.export.presets import (
    APS,
    CUSTOM,
    NATURE,
    PRESETS,
    JournalPreset,
    get_preset,
)
from mag_opt_detective.export.state import Curve, FigureState, PointSet, StackedOptions

__all__ = [
    "APS",
    "CUSTOM",
    "FORMATS",
    "NATURE",
    "PRESETS",
    "Curve",
    "FigureState",
    "JournalPreset",
    "PointSet",
    "StackedOptions",
    "colormap",
    "energy_label",
    "get_preset",
    "rasterize",
    "render",
    "resolve_font",
    "save",
]
