"""How a figure is drawn beyond its journal preset: where the colour bar goes and the ticks.

:class:`FigureStyle` collects these choices, so a new one is one more field (with a default
that keeps today's look) and one more key in :meth:`FigureStyle.to_dict`. The defaults draw
exactly what the presets drew before they existed: outward ticks on the bottom and left axes,
sized from the text size and line width, no minor ticks, the colour bar on the right.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

TICK_DIRECTIONS: tuple[str, ...] = ("in", "out")
COLORBAR_LOCATIONS: tuple[str, ...] = ("right", "top")
MINOR_INTERVALS: tuple[int, int] = (2, 10)  # intervals between labelled ticks (AutoMinorLocator)

# tick sizes that follow the text size and the line width (the presets' own look)
MAJOR_LENGTH = 0.45  # × text size
MINOR_LENGTH = 0.25  # × text size
MINOR_WIDTH = 0.75  # × major tick width


def _length(name: str, value) -> float | None:
    """A length in pt (None: automatic); ValueError if it is not a number of 0 or more."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be a number of pt, not {value!r}")
    value = float(value)
    if not (math.isfinite(value) and value >= 0):
        raise ValueError(f"{name} must be 0 pt or more, not {value:g}")
    return value


@dataclass(frozen=True)
class TickStyle:
    """Ticks of the axes; the colour bar follows their direction and size.

    *direction* is "in" or "out"; *mirror* repeats the ticks on the top and right axes
    (without labels). Lengths and the width are in pt; None follows the text size (major
    0.45×, minor 0.25×) and the line width. *minor* adds unlabelled ticks that split each
    step between labelled ticks into *minor_intervals* parts; minor ticks are 0.75× as wide.
    The colour bar gets no minor ticks and no mirrored ticks (its outline frames it).
    """

    direction: str = "out"
    mirror: bool = False
    length_pt: float | None = None
    width_pt: float | None = None
    minor: bool = False
    minor_intervals: int = 2
    minor_length_pt: float | None = None

    def __post_init__(self) -> None:
        if self.direction not in TICK_DIRECTIONS:
            raise ValueError(f"tick direction must be one of {TICK_DIRECTIONS}")
        for name in ("mirror", "minor"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be true or false")
        object.__setattr__(self, "length_pt", _length("tick length", self.length_pt))
        object.__setattr__(
            self, "minor_length_pt", _length("minor tick length", self.minor_length_pt)
        )
        width = _length("tick width", self.width_pt)
        if width == 0:
            raise ValueError("tick width must be more than 0 pt")
        object.__setattr__(self, "width_pt", width)
        lo, hi = MINOR_INTERVALS
        n = self.minor_intervals
        if isinstance(n, bool) or not isinstance(n, int) or not lo <= n <= hi:
            raise ValueError(f"minor intervals must be a whole number from {lo} to {hi}")

    def major_length(self, font_pt: float) -> float:
        return MAJOR_LENGTH * font_pt if self.length_pt is None else self.length_pt

    def major_width(self, line_pt: float) -> float:
        return line_pt if self.width_pt is None else self.width_pt

    def minor_length(self, font_pt: float) -> float:
        return MINOR_LENGTH * font_pt if self.minor_length_pt is None else self.minor_length_pt

    def to_dict(self) -> dict:
        return {
            "direction": self.direction,
            "mirror": self.mirror,
            "length_pt": self.length_pt,
            "width_pt": self.width_pt,
            "minor": self.minor,
            "minor_intervals": self.minor_intervals,
            "minor_length_pt": self.minor_length_pt,
        }

    @classmethod
    def from_dict(cls, data) -> TickStyle:
        """The ticks of :meth:`to_dict`; missing keys take the defaults (ValueError if bad)."""
        if not isinstance(data, dict):
            raise ValueError("ticks must be an object")
        known = {key: data[key] for key in cls().to_dict() if key in data}
        try:
            return cls(**known)
        except TypeError as exc:
            raise ValueError(str(exc)) from None


@dataclass(frozen=True)
class FigureStyle:
    """Style choices of a figure that no journal prescribes.

    *colorbar_location*: "right" (a vertical bar) or "top" (a horizontal bar above the plot,
    its ticks and label on top). The figure keeps its size: the axes make room.
    """

    colorbar_location: str = "right"
    ticks: TickStyle = field(default_factory=TickStyle)

    def __post_init__(self) -> None:
        if self.colorbar_location not in COLORBAR_LOCATIONS:
            raise ValueError(f"colour bar location must be one of {COLORBAR_LOCATIONS}")
        if not isinstance(self.ticks, TickStyle):
            raise ValueError("ticks must be a TickStyle")

    def to_dict(self) -> dict:
        return {"colorbar_location": self.colorbar_location, "ticks": self.ticks.to_dict()}

    @classmethod
    def from_dict(cls, data) -> FigureStyle:
        """The style of :meth:`to_dict`; missing keys take the defaults (ValueError if bad)."""
        if not isinstance(data, dict):
            raise ValueError("the style must be an object")
        location = data.get("colorbar_location", "right")
        if not isinstance(location, str):
            raise ValueError("the colour bar location must be text")
        ticks = TickStyle.from_dict(data["ticks"]) if "ticks" in data else TickStyle()
        return cls(colorbar_location=location, ticks=ticks)
