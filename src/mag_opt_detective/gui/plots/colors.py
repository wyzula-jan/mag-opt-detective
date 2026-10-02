"""Plot colours that a theme can change."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields

from PySide6.QtGui import QColor


def qcolor(value: str) -> QColor:
    """QColor from ``#rgb``, ``#rrggbb`` or ``#rrggbbaa`` (CSS order) or a colour name."""
    text = value.strip()
    if text.startswith("#") and len(text) == 9:
        rgba = [int(text[i : i + 2], 16) for i in (1, 3, 5, 7)]
        return QColor(*rgba)
    color = QColor(text)
    if not color.isValid():
        raise ValueError(f"invalid colour {value!r}")
    return color


@dataclass(frozen=True)
class PlotColors:
    """Colours of the plot views as hex strings; the defaults are pyqtgraph's dark look."""

    background: str = "#000000"
    foreground: str = "#969696"
    grid: str = "#96969640"  # for grid lines; the views draw none yet
    crosshair: str = "#c8c8c896"
    accent: str = "#fe9f6d"
    well: str = "#000000"  # behind the image, shows where there is no data (NaN)

    def __post_init__(self) -> None:
        for field in fields(self):
            qcolor(getattr(self, field.name))  # validate early

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, str]) -> PlotColors:
        """Build from a theme's ``plot_colors()``; missing keys keep their defaults."""
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in mapping.items() if k in known})

    def q(self, name: str) -> QColor:
        """The colour *name* as a QColor."""
        return qcolor(getattr(self, name))
