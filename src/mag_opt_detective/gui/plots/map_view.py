"""Colour map of intensity over field and energy, with its colour scale."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QPainter, QPalette
from PySide6.QtWidgets import QVBoxLayout, QWidget

from mag_opt_detective.core.spectra import FieldMap
from mag_opt_detective.gui.display import FIELD_LABEL, energy_label, unit_text
from mag_opt_detective.gui.plots.base import (
    AxisCells,
    PartPainter,
    PlotView,
    Range,
    map_cells,
    robust_levels,
    set_range,
)
from mag_opt_detective.gui.plots.colors import PlotColors
from mag_opt_detective.gui.plots.colorscale import (
    ColorScale,
    HistogramScale,
    make_scale,
    same_values,
)


def _drawn(fmap: FieldMap) -> tuple[np.ndarray, tuple[AxisCells, AxisCells]]:
    """The values of *fmap* as an image and its cells (an uneven axis is drawn on a finer,
    even one, see :func:`~mag_opt_detective.gui.plots.base.axis_cells`)."""
    x, y = cells = map_cells(fmap.field, fmap.energy)
    values = fmap.values
    if y.index is not None:
        values = values[y.index]
    if x.index is not None:
        values = values[:, x.index]
    return values, cells


class _LeadImage(pg.ImageItem):
    """The map's image. The images drawn with it (:attr:`followers`, other maps below it)
    take its levels and lookup table, so the colour scale drives them all."""

    def __init__(self, *args, **kwargs):
        self.followers: list[pg.ImageItem] = []
        super().__init__(*args, **kwargs)

    def setLevels(self, levels, update: bool = True):
        super().setLevels(levels, update)
        for image in self.followers:
            image.setLevels(levels, update)

    def setLookupTable(self, lut, update: bool = True):
        super().setLookupTable(lut, update)
        for image in self.followers:
            image.setLookupTable(lut, update)


class ColorMapPlot(PlotView):
    """Intensity as a function of field (x) and energy (y) with a colour scale beside it.

    The scale sits in :attr:`scale_container`, a separate widget right of the plot.
    Other maps can be drawn below the map with its colours (:meth:`set_overlays`); where they
    overlap, the upper ones are translucent.
    Emits :attr:`pointClicked` ``(field, energy)`` on a left click inside the plot,
    :attr:`levelsEdited` when the user changes the levels on the scale and
    :attr:`cursorMoved` ``(field, energy, value or None)``.
    """

    pointClicked = Signal(float, float)
    levelsEdited = Signal(float, float)  # the user changed the levels on the scale

    def __init__(self, parent=None, scale_style: str = "histogram"):
        super().__init__(parent)
        self._unit = ""
        self._cmap = ""
        self._fmap: FieldMap | None = None
        self._cells: tuple[AxisCells, AxisCells] | None = None  # (field, energy) as drawn
        self._auto_levels = True
        self._follow_levels = False  # the scale's value axis follows the levels (auto-scale)
        self._margins = (0, 0)
        self._overlays: list[tuple[FieldMap, tuple[AxisCells, AxisCells]]] = []  # bottom first
        self._base: list[pg.ImageItem] = []  # the opaque pass of the overlays (map first)
        self._blend: list[pg.ImageItem] = []  # the translucent pass (without the map)
        self._opacity = 1.0
        self._auto = (True, True)  # the field and energy ranges fit the data (set_map)
        self.plot.setLabel("bottom", FIELD_LABEL)
        self.plot.setLabel("left", "Energy")
        self.image = _LeadImage(axisOrder="row-major")
        self.plot.addItem(self.image)
        self.layer("points")
        self.layer("models")

        self.scale_container = QWidget()
        self.scale_container.setAutoFillBackground(True)  # the plot background
        self._scale_layout = QVBoxLayout(self.scale_container)
        self._scale_layout.setContentsMargins(0, 0, 0, 0)
        self._row.addWidget(self.scale_container)
        self._scale: ColorScale = make_scale(scale_style)
        self._install_scale(self._scale)

        self.apply_theme(self._colors)
        self.plot.vb.sigResized.connect(self._align_scale)
        self.plot.scene().sigMouseClicked.connect(self._on_click)

    # ------------------------------------------------------------------ colour scale
    @property
    def scale(self) -> ColorScale:
        """The colour scale in use."""
        return self._scale

    @property
    def hist(self) -> pg.HistogramLUTItem | None:
        """The HistogramLUTItem while the histogram style is active, else None."""
        return self._scale.hist if isinstance(self._scale, HistogramScale) else None

    def scale_style(self) -> str:
        return self._scale.style

    def set_scale_style(self, style: str) -> None:
        """Swap the colour scale ("histogram" or "bar"), keeping levels and colour map."""
        if style == self._scale.style:
            return
        new = make_scale(style)
        old = self._scale
        lo, hi = old.levels()
        old.levelsEdited.disconnect(self._on_scale_edited)
        old.detach()
        self._scale_layout.removeWidget(old.widget)
        old.widget.hide()
        old.widget.deleteLater()
        new.set_colormap(old.colormap())
        new.set_follow_levels(self._follow_levels)
        new.set_levels(lo, hi, auto_range=self._auto_levels)
        self._scale = new
        self._install_scale(new)

    def _install_scale(self, scale: ColorScale) -> None:
        scale.apply_theme(self._colors)
        scale.attach(self.image)
        scale.levelsEdited.connect(self._on_scale_edited)
        self._scale_layout.addWidget(scale.widget)
        self._align_scale()

    def scale_follows_levels(self) -> bool:
        return self._follow_levels

    def set_scale_follows_levels(self, follow: bool) -> None:
        """Auto-scale the colour scale's value axis to the levels whenever they change
        (*follow*), or keep it still: then it is fitted only to new data (see
        :class:`HistogramScale`)."""
        self._follow_levels = bool(follow)
        self._scale.set_follow_levels(follow)

    def set_scale_visible(self, visible: bool) -> None:
        """Show or hide the colour scale (exports follow)."""
        self.scale_container.setVisible(visible)

    def scale_visible(self) -> bool:
        return not self.scale_container.isHidden()

    def _align_scale(self, *_args) -> None:
        plot = self.plot.sceneBoundingRect()
        data = self.plot.vb.sceneBoundingRect()
        self._margins = self._scale.margins(plot, data, self.view.height())
        self._scale_layout.setContentsMargins(0, self._margins[0], 0, self._margins[1])

    def _on_scale_edited(self, lo: float, hi: float) -> None:
        if self._fmap is None:
            return
        self._auto_levels = False
        self.levelsEdited.emit(lo, hi)

    def levels(self) -> tuple[float, float]:
        return self._scale.levels()

    def set_levels(self, lo: float, hi: float) -> None:
        """Set the colour levels without emitting :attr:`levelsEdited`."""
        self._auto_levels = False
        self._scale.set_levels(lo, hi)
        self.image.setLevels((lo, hi))

    def colormap(self) -> str:
        return self._scale.colormap()

    def set_colormap(self, name: str) -> None:
        self._scale.set_colormap(name)
        self._cmap = name

    def apply_theme(self, colors: PlotColors) -> None:
        """Plot, scale and the well behind missing data (NaN) from *colors*."""
        super().apply_theme(colors)
        self.plot.vb.setBackgroundColor(colors.q("well"))
        palette = self.scale_container.palette()
        palette.setColor(QPalette.ColorRole.Window, colors.q("background"))
        self.scale_container.setPalette(palette)
        self._scale.apply_theme(colors)

    # ------------------------------------------------------------------ overlays
    def set_model_curves(self, field: np.ndarray, lines: np.ndarray | None) -> None:
        """Draw model energies ``lines[i, j]`` at ``field[j]``; None removes them."""
        layer = self.layer("models")
        if lines is None or len(lines) == 0:
            layer.clear()
            return
        pen = pg.mkPen((255, 255, 255, 210), width=1.5, style=Qt.PenStyle.DashLine)
        layer.set_curves([(field, line) for line in lines], pen)

    def model_curve_data(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """(field, energy) of the visible model curves."""
        return self.layer("models").curve_data()

    def set_points(
        self,
        current: tuple[np.ndarray, np.ndarray] | None,
        others: list[tuple[np.ndarray, np.ndarray, object]] | None = None,
    ) -> None:
        """Mark picked points: *current* curve as crosses, *others* as coloured circles."""
        layer = self.layer("points")
        layer.clear()
        if others:
            x = np.concatenate([np.asarray(b, dtype=float) for b, _, _ in others])
            y = np.concatenate([np.asarray(e, dtype=float) for _, e, _ in others])
            pens = [pg.mkPen(color, width=2) for b, _, color in others for _ in range(len(b))]
            layer.add_points(x, y, symbol="o", size=8, pen=pens, brush=None)
        if current is not None:
            pen = pg.mkPen("r", width=2)
            layer.add_points(current[0], current[1], symbol="x", size=11, pen=pen, brush=None)

    # ------------------------------------------------------------------ data
    def value_at(self, b: float, energy: float) -> float | None:
        """Map value drawn at (b, energy), or None outside the image."""
        fmap, cells = self._fmap, self._cells
        if fmap is None or cells is None:
            return None
        col, row = cells[0].sample_at(b), cells[1].sample_at(energy)
        if col is None or row is None:
            return None
        return float(fmap.values[row, col])

    def drawn_cells(self) -> tuple[AxisCells, AxisCells] | None:
        """How the map is drawn: the (field, energy) cells, or None without a map."""
        return self._cells

    def _cursor_value(self, x: float, y: float) -> float | None:
        return self.drawn_value_at(x, y)

    def drawn_value_at(self, b: float, energy: float) -> float | None:
        """Value of the top map drawn at (b, energy): the map's, else the overlays' (top
        first), None where nothing is drawn."""
        value = self.value_at(b, energy)
        if value is not None:
            return value
        for fmap, cells in reversed(self._overlays):
            col, row = cells[0].sample_at(b), cells[1].sample_at(energy)
            if col is not None and row is not None:
                return float(fmap.values[row, col])
        return None

    def _cursor_text(self, x: float, y: float) -> str:
        text = f"B = {x:.3f} T    E = {y:.3f} {self._unit}"
        value = self.drawn_value_at(x, y)
        return text if value is None else f"{text}    value = {value:.5g}"

    def set_map(
        self,
        fmap: FieldMap,
        levels: Range = None,
        cmap: str = "magma",
        x_range: Range = None,
        y_range: Range = None,
    ) -> None:
        self._show(fmap, levels, cmap)
        self._auto = (x_range is None, y_range is None)
        if x_range is None:  # (y_range None auto-ranges over the images, overlays too)
            x_range = self.data_extent()[0]
        set_range(self.plot, x_range, y_range)

    def _show(self, fmap: FieldMap, levels: Range, cmap: str) -> None:
        old, self._fmap = self._fmap, fmap
        new_data = old is None or not same_values(old.values, fmap.values)
        self._unit = unit_text(fmap.unit)
        self.plot.setLabel("left", energy_label(fmap.unit))
        scale = self._scale
        if cmap != self._cmap:
            scale.set_colormap(cmap)
            self._cmap = cmap
        self._auto_levels = levels is None
        lo, hi = robust_levels(fmap.values) if levels is None else levels
        x, y = self._cells = map_cells(fmap.field, fmap.energy)
        values = fmap.values  # an uneven axis is drawn on a finer, even one (see axis_cells)
        if y.index is not None:
            values = values[y.index]
        if x.index is not None:
            values = values[:, x.index]
        with scale.quiet():  # the histogram re-reads the levels from the new image
            self.image.setImage(values, autoLevels=False, levels=(lo, hi))
            self.image.setRect(QRectF(x.start, y.start, x.span, y.span))
            scale.set_levels(lo, hi, auto_range=self._auto_levels, fit=new_data)
        self._copy_map_below()

    def clear_map(self) -> None:
        self._fmap = None
        self._cells = None
        self.image.clear()
        self.set_overlays([])
        self.set_points(None)

    # ------------------------------------------------------------------ overlays
    # Overlays are drawn in two passes, so that a map shows in full colour where no other
    # map lies under it: every map opaque, the lowest one last (at each point the lowest map
    # drawn there shows), then every map above the bottom one, the map last, translucent.
    def set_overlays(self, maps: list[FieldMap], opacity: float | None = None) -> None:
        """Draw *maps* below the map (bottom first) with its levels and colours. Where maps
        overlap, those above the lowest one are blended in with *opacity* (None keeps it);
        elsewhere every map shows as it is. Without overlays the map is drawn alone."""
        k = len(maps)
        self._resize(self._base, k + 1 if k else 0)
        self._resize(self._blend, max(k - 1, 0))
        self.image.followers = [*self._base, *self._blend]
        drawn = [_drawn(fmap) for fmap in maps]
        self._overlays = [(fmap, cells) for fmap, (_values, cells) in zip(maps, drawn, strict=True)]
        if k:
            self._copy_map_below()
            for image, layer in zip(self._base[1:], reversed(drawn), strict=True):
                self._set_layer(image, *layer)
            for image, layer in zip(self._blend, drawn[1:], strict=True):
                self._set_layer(image, *layer)
        for z, image in enumerate(self._base):
            image.setZValue(-200 + z)
        for z, image in enumerate(self._blend):
            image.setZValue(-100 + z)
        if opacity is not None:
            self._opacity = float(opacity)
        self.set_overlay_opacity(self._opacity)
        if self._fmap is not None and self._auto[0]:  # the field range fits the maps below too
            self.plot.setXRange(*self.data_extent()[0], padding=0)  # (y auto-ranges on its own)

    def set_overlay_opacity(self, opacity: float) -> None:
        """Opacity of the maps above the lowest one where maps overlap."""
        self._opacity = float(opacity)
        for image in self._base:
            image.setOpacity(1.0)
        for image in self._blend:
            image.setOpacity(self._opacity)
        self.image.setOpacity(self._opacity if self._overlays else 1.0)

    def _resize(self, images: list[pg.ImageItem], size: int) -> None:
        while len(images) > size:
            self.plot.removeItem(images.pop())
        while len(images) < size:
            image = pg.ImageItem(axisOrder="row-major")
            image.setLookupTable(self.image.lut)
            self.plot.addItem(image)
            images.append(image)

    def _set_layer(self, image: pg.ImageItem, values: np.ndarray, cells) -> None:
        x, y = cells
        levels = self.image.levels if self.image.levels is not None else (0.0, 1.0)
        image.setImage(values, autoLevels=False, levels=levels)
        image.setRect(QRectF(x.start, y.start, x.span, y.span))

    def _copy_map_below(self) -> None:
        """The opaque copy of the map under the overlays follows the map."""
        if self._base and self._cells is not None and self.image.image is not None:
            self._set_layer(self._base[0], self.image.image, self._cells)

    def overlay_maps(self) -> list[FieldMap]:
        """The maps drawn below the map, bottom first."""
        return [fmap for fmap, _ in self._overlays]

    def overlay_opacity(self) -> float:
        return self._opacity

    def data_extent(self) -> tuple[tuple[float, float], tuple[float, float]] | None:
        """(field, energy) span of the samples of the map and the maps below it (what the
        ranges fit when they fit the data), or None without a map."""
        maps = [m for m in (self._fmap, *self.overlay_maps()) if m is not None]
        if not maps:
            return None
        return (
            (
                float(min(np.nanmin(m.field) for m in maps)),
                float(max(np.nanmax(m.field) for m in maps)),
            ),
            (
                float(min(np.nanmin(m.energy) for m in maps)),
                float(max(np.nanmax(m.energy) for m in maps)),
            ),
        )

    def _on_click(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton or self.image.image is None:
            return
        pos = event.scenePos()
        if not self.plot.vb.sceneBoundingRect().contains(pos):
            return
        point = self.plot.vb.mapSceneToView(pos)
        self.pointClicked.emit(point.x(), point.y())

    # ------------------------------------------------------------------ export
    def _export_parts(self, height: float) -> list[tuple[float, PartPainter]]:
        parts = super()._export_parts(height)
        if not self.scale_visible():
            return parts
        scale = self._scale
        top, bottom = self._margins

        def paint(painter: QPainter, rect: QRectF, resolution: float) -> None:
            inner = rect.adjusted(0, top, 0, -bottom)
            if inner.height() > 0:
                scale.render(painter, inner, resolution)

        return [*parts, (float(scale.widget.width()), paint)]
