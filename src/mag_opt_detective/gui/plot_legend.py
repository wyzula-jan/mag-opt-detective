"""The legend inside the map and the stacked plot (:class:`~.plots.legend.PlotLegend`).

The legend button of the plot toolbar shows or hides the legend of the plot on screen; each
plot keeps its own choice and the place the legend was dragged to (``plot/legend_map``,
``plot/legend_stacked``). It is off by default. The rows follow what the plot draws: first the
picked curves with their markers (the rows of the plot's "points" layer, which
:func:`~.points_view.draw_markers` sets), then, on the map, every model drawn with its line,
its branches in one row, and a fit waiting to be applied. The stacked plot draws no models.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QSignalBlocker
from PySide6.QtGui import QColor

from mag_opt_detective.gui.inspector import model_state as ms
from mag_opt_detective.gui.inspector.models import SHADOW_PEN, curve_pen, preview_pen
from mag_opt_detective.gui.plots.legend import LegendColors, LegendEntry, PlotLegend

VIEWS = ("map", "stacked")
PANEL_OPACITY = 0.88  # of the panel over the colour map


def window_models(window):
    """The model list of the Models section (``inspector.models.Models``), None without it."""
    section = window.inspector.get("models")
    item = section.body_layout().itemAt(0) if section is not None else None
    return getattr(item.widget(), "models", None) if item is not None else None


def model_names(window) -> dict[str, str]:
    """The name of each model by its key (the overlay names of ``figure_state()``)."""
    models = window_models(window)
    return {} if models is None else {entry.key: entry.name for entry in models.entries}


def _branches(entry: ms.ModelEntry, n: int) -> str:
    """How many lines a model draws, in its own words ("" for one)."""
    if n < 2:
        return ""
    if entry.kind == ms.DIRAC:
        return f"{n} transitions"
    if entry.kind == ms.ZEEMAN and entry.model.coupled:
        return f"{n} modes"
    return f"{n} branches"


def model_entries(window) -> list[LegendEntry]:
    """Legend rows of the models drawn on the map: one per model, and one per fit preview."""
    models = window_models(window)
    if models is None:
        return []
    drawn = models.curve_data()
    rows = []
    for entry in models.entries:
        n = sum(bool(np.isfinite(e).any()) for _b, e in drawn.get(entry.key, []))
        if n:
            pens = (SHADOW_PEN, curve_pen(entry.color))
            rows.append(LegendEntry(entry.name, pens, detail=_branches(entry, n), group="models"))
        if models.preview_data(entry):
            pens = (SHADOW_PEN, preview_pen(entry.color))
            rows.append(LegendEntry(entry.name, pens, detail="fit", group="models"))
    return rows


def refresh(window, view: str) -> None:
    """List what plot *view* draws now."""
    rows = window.plots[view].layer("points").legend()
    if view == "map":
        rows += model_entries(window)
    window.legends[view].set_entries(rows)


def legend_colors(window) -> LegendColors:
    """The window theme's surface (translucent), hairline, text and muted text."""
    tokens = window.theme.tokens()
    panel = QColor(tokens["surface"])
    panel.setAlphaF(PANEL_OPACITY)
    return LegendColors(panel, tokens["line-strong"], tokens["fg"], tokens["muted"])


def sync_button(window) -> None:
    """The toolbar button shows whether the plot on screen has its legend on."""
    button = window.plot_area.legend_button
    legend = window.legends.get(window.plot_area.current_view())
    button.setEnabled(legend is not None)
    with QSignalBlocker(button):
        button.setChecked(legend is not None and legend.is_shown())


def install(window) -> None:
    """A legend on the map and the stacked plot, switched by ``plot_area.legend_button``."""
    area = window.plot_area
    window.legends = {view: PlotLegend(window.plots[view].plot) for view in VIEWS}
    for view, legend in window.legends.items():
        legend.set_colors(legend_colors(window))
        legend.shownChanged.connect(lambda _shown: sync_button(window))
        window.plots[view].layer("points").on_legend(lambda v=view: refresh(window, v))
    window.controller.overlaysChanged.connect(lambda: refresh(window, "map"))

    def on_button(checked: bool) -> None:
        legend = window.legends.get(area.current_view())
        if legend is not None:
            legend.set_shown(checked)

    def on_theme() -> None:
        for legend in window.legends.values():
            legend.set_colors(legend_colors(window))

    area.legend_button.clicked.connect(on_button)
    area.tabs.currentChanged.connect(lambda _index: sync_button(window))
    window.themeChanged.connect(on_theme)
    sync_button(window)
    if window.persistence is not None:
        for view, legend in window.legends.items():
            window.persistence.bind(f"plot/legend_{view}", legend)
