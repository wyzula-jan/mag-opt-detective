"""pyqtgraph views: field/energy colour map and stacked spectra, with their colour scales."""

from mag_opt_detective.gui.plots.base import (
    IMAGE_SUFFIXES,
    AxisCells,
    PlotView,
    Range,
    axis_cells,
    map_cells,
    pixel_rect,
    robust_levels,
    set_range,
)
from mag_opt_detective.gui.plots.colormaps import colormap, lookup_table
from mag_opt_detective.gui.plots.colors import PlotColors
from mag_opt_detective.gui.plots.colorscale import (
    SCALE_STYLES,
    BarScale,
    ColorBar,
    ColorScale,
    HistogramScale,
    make_scale,
)
from mag_opt_detective.gui.plots.map_view import ColorMapPlot
from mag_opt_detective.gui.plots.overlays import OverlayLayer
from mag_opt_detective.gui.plots.stacked_view import StackedPlot

__all__ = [
    "IMAGE_SUFFIXES",
    "SCALE_STYLES",
    "AxisCells",
    "BarScale",
    "ColorBar",
    "ColorMapPlot",
    "ColorScale",
    "HistogramScale",
    "OverlayLayer",
    "PlotColors",
    "PlotView",
    "Range",
    "StackedPlot",
    "axis_cells",
    "colormap",
    "lookup_table",
    "make_scale",
    "map_cells",
    "pixel_rect",
    "robust_levels",
    "set_range",
]
