"""Journal-quality figure export with matplotlib (no Qt, no pyplot)."""

from mag_opt_detective.export.figure import (
    FORMATS,
    colormap,
    energy_label,
    layout_problem,
    rasterize,
    render,
    resolve_font,
    robust_levels,
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
from mag_opt_detective.export.style import FigureStyle, TickStyle
from mag_opt_detective.export.user_presets import (
    PresetError,
    UserPreset,
    presets_from_json,
    presets_to_json,
)

__all__ = [
    "APS",
    "CUSTOM",
    "FORMATS",
    "NATURE",
    "PRESETS",
    "Curve",
    "FigureState",
    "FigureStyle",
    "JournalPreset",
    "PointSet",
    "PresetError",
    "StackedOptions",
    "TickStyle",
    "UserPreset",
    "colormap",
    "energy_label",
    "get_preset",
    "layout_problem",
    "presets_from_json",
    "presets_to_json",
    "rasterize",
    "render",
    "resolve_font",
    "robust_levels",
    "save",
]
