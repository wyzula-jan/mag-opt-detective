"""Colour section: colour map, level mode and levels of the map on screen, with a histogram.

Levels are kept per level key (plot kind, derivative order and axis, per unit; see
:func:`~mag_opt_detective.gui.controller.level_key`) in ``controller.view``. The section edits
the key of the map on the Map tab, or of the reference map on the Reference tab. The plot's
own colour scale (histogram or slim bar) edits the same key through the controller, so the two
always agree. Levels of per-unit energy derivatives are saved per cm^-1.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QRectF, QSignalBlocker, QSize, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QLinearGradient, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QAbstractButton,
    QButtonGroup,
    QFrame,
    QGraphicsRectItem,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from mag_opt_detective.core import colormaps
from mag_opt_detective.core.units import Unit, convert_levels
from mag_opt_detective.gui.controller import (
    AUTO_COLOURS,
    AUTO_LEVELS,
    FIXED_LEVELS,
    LEVEL_MODES,
    SYMMETRIC_LEVELS,
    AppController,
    default_level_mode,
    level_key,
    level_label,
    parse_level_key,
    symmetric_centre,
    symmetric_levels,
    user_action,
)
from mag_opt_detective.gui.inspector.view import NumberSpin, load_json, to_pair, update_sections
from mag_opt_detective.gui.kit import CollapsibleSection, SegmentedControl
from mag_opt_detective.gui.kit._common import set_style_property
from mag_opt_detective.gui.kit.range_control import ORDER_ERROR, decimals_for
from mag_opt_detective.gui.plots.colorscale import fit_range, sample_values, tails, widened
from mag_opt_detective.gui.theme import current_tokens

Pair = tuple[float, float]

COLOURMAPS = (AUTO_COLOURS, "magma", "inferno", "viridis", "plasma", "turbo", "grey", "bipolar")
AUTO_TIP = "Magma for maps, grey for derivatives"
MODES = (
    (AUTO_LEVELS, "Auto", "The 1st to 99th percentile of the map"),
    (FIXED_LEVELS, "Fixed", "Keep these levels"),
    (SYMMETRIC_LEVELS, "Symmetric", "Levels symmetric about the centre of the plot kind"),
)
AUTO_NOTE = "Auto uses the 1st to 99th percentile of the map."


def gradient(name: str, x0: float, x1: float, part: Pair = (0.0, 1.0)) -> QLinearGradient:
    """Horizontal gradient of colour map *name* from *x0* to *x1* (only its *part* of 0-1)."""
    grad = QLinearGradient(x0, 0.0, x1, 0.0)
    if part == (0.0, 1.0):
        for pos, rgb in colormaps.stops(name):
            grad.setColorAt(pos, QColor(*rgb))
        return grad
    table = colormaps.lut(name, 256)
    lo, hi = part
    for t in np.linspace(0.0, 1.0, 32):
        index = round((lo + (hi - lo) * t) * (len(table) - 1))
        grad.setColorAt(float(t), QColor(*(int(c) for c in table[index])))
    return grad


def _small(widget: QWidget, factor: float = 0.88) -> None:
    font = widget.font()
    if font.pointSizeF() > 0:
        font.setPointSizeF(font.pointSizeF() * factor)
        widget.setFont(font)


# ---------------------------------------------------------------------- colour map swatches
class Swatch(QAbstractButton):
    """A checkable colour map button: a gradient strip above the map's name."""

    STRIP = 12

    def __init__(self, name: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.name = name
        self.setCheckable(True)
        self.setText(name.capitalize())
        self.setToolTip(AUTO_TIP if name == AUTO_COLOURS else self.text())
        self.setAccessibleName(f"{self.text()} colour map")
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover)
        _small(self)

    def sizeHint(self) -> QSize:
        metrics = self.fontMetrics()
        width = max(44, metrics.horizontalAdvance(self.text()) + 12)
        return QSize(width, 4 + self.STRIP + 3 + metrics.height() + 4)

    def minimumSizeHint(self) -> QSize:
        return QSize(32, self.sizeHint().height())

    def paintEvent(self, _event) -> None:
        t = current_tokens()
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        checked = self.isChecked()
        if checked or self.hasFocus():
            border = t["accent"]
        else:
            border = t["line-strong"] if self.underMouse() else t["line"]
        width = 2.0 if checked else 1.0
        frame = QRectF(self.rect()).adjusted(width / 2, width / 2, -width / 2, -width / 2)
        p.setPen(QPen(border, width))
        p.setBrush(t["surface"])
        p.drawRoundedRect(frame, 6, 6)

        strip = QRectF(5, 5, self.width() - 10, self.STRIP)
        path = QPainterPath()
        path.addRoundedRect(strip, 3, 3)
        p.save()
        p.setClipPath(path)
        if self.name == AUTO_COLOURS:  # magma for maps over grey for derivatives
            top, bottom = strip.adjusted(0, 0, 0, -strip.height() / 2), strip
            p.fillRect(bottom, gradient("grey", strip.left(), strip.right()))
            p.fillRect(top, gradient("magma", strip.left(), strip.right()))
        else:
            p.fillRect(strip, gradient(self.name, strip.left(), strip.right()))
        p.restore()
        p.setPen(QPen(QColor(0, 0, 0, 40), 1))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRoundedRect(strip, 3, 3)

        p.setPen(t["fg"] if checked else t["muted"])
        text = QRectF(2, strip.bottom() + 2, self.width() - 4, self.height() - strip.bottom() - 4)
        elided = self.fontMetrics().elidedText(
            self.text(), Qt.TextElideMode.ElideRight, int(text.width())
        )
        p.drawText(text, Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop, elided)
        p.end()


class ColormapPicker(QWidget):
    """Colour map swatches, Auto first, four per row; one is checked.

    ``valueChanged`` fires when the choice changes (also programmatically).
    Settings protocol: the colour map name.
    """

    valueChanged = Signal(str)
    COLUMNS = 4

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        grid = QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(6)
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self.swatches: dict[str, Swatch] = {}
        for i, name in enumerate(COLOURMAPS):
            swatch = Swatch(name)
            grid.addWidget(swatch, i // self.COLUMNS, i % self.COLUMNS)
            self._group.addButton(swatch)
            self.swatches[name] = swatch
        self._value = AUTO_COLOURS
        self.swatches[AUTO_COLOURS].setChecked(True)
        self._group.buttonToggled.connect(self._on_toggled)
        self.setAccessibleName("Colour map")

    def value(self) -> str:
        return self._value

    def set_value(self, name: str) -> None:
        if name not in self.swatches:
            raise ValueError(f"unknown colour map {name!r}")
        self.swatches[name].setChecked(True)

    def _on_toggled(self, button, checked: bool) -> None:
        if checked:
            self._value = button.name
            self.valueChanged.emit(self._value)

    def settings_value(self) -> str:
        return self._value

    def set_settings_value(self, value) -> bool:
        if not isinstance(value, str) or value not in self.swatches:
            return False
        self.set_value(value)
        return True


# ---------------------------------------------------------------------- histogram
class LevelHistogram(QWidget):
    """Histogram of the map values (square-root counts) with the colour levels as a
    draggable region and the colour map, stretched over the levels, as a strip below.

    ``levelsChanging`` fires while a level is dragged, ``levelsChosen`` once it is let go;
    setting levels from the program emits neither. With a symmetric centre, dragging one end
    mirrors the other; dragging the region shifts both.

    The value range shows the bulk of the values and the levels. It stays where it is when
    only the levels change (widened just enough to show them) and is fitted again for other
    values, on :meth:`fit` and on a double-click; following the levels (auto-scale), it is
    fitted on every change.
    """

    levelsChanging = Signal(float, float)
    levelsChosen = Signal(float, float)

    BINS = 72
    HEIGHT = 78
    STRIP = (-0.2, 0.11)  # y and height of the colour strip (bars span 0 to 1)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setFixedHeight(self.HEIGHT)
        self.setAccessibleName("Histogram of the map values with the colour range")
        self._values = np.empty(0)
        self._tails: Pair = (0.0, 1.0)  # 0.5th and 99.5th percentile
        self._levels: Pair = (0.0, 1.0)
        self._range: Pair | None = None  # None: fit it at the next levels
        self._centre: float | None = None
        self._cmap = "grey"
        self._quiet = 0
        self._follow = False

        self.view = pg.GraphicsLayoutWidget()
        self.view.setFrameShape(QFrame.Shape.NoFrame)
        self.view.ci.setContentsMargins(4, 2, 4, 2)
        self.plot = self.view.addPlot()
        for axis in ("left", "bottom"):
            self.plot.hideAxis(axis)
        self.plot.setMenuEnabled(False)
        self.plot.setMouseEnabled(x=False, y=False)
        self.plot.hideButtons()
        self.plot.setToolTip("Double-click to fit")
        self.bars = pg.BarGraphItem(x0=[0.0], x1=[1.0], height=[0.0], pen=pg.mkPen(None))
        self.strip = QGraphicsRectItem()  # filled with the colour map stretched over the levels
        self.strip.setPen(QPen(Qt.PenStyle.NoPen))
        self.centre = pg.InfiniteLine(angle=90, movable=False)
        self.region = pg.LinearRegionItem((0.0, 1.0), swapMode="sort")
        self.region.setZValue(10)
        for line in self.region.lines:
            line.addMarker("v", position=0.97, size=9)
        for item in (self.bars, self.strip, self.centre, self.region):
            self.plot.addItem(item)
        self.region.sigRegionChanged.connect(self._on_changing)
        self.region.sigRegionChangeFinished.connect(self._on_finished)
        self.view.scene().sigMouseClicked.connect(self._on_click)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(1, 1, 1, 1)
        layout.addWidget(self.view)
        self._count()
        self.apply_theme()

    # --- data --------------------------------------------------------------------------
    def set_values(self, values: np.ndarray | None) -> None:
        """The map values to count (None: no map); other values fit the range again."""
        finite = sample_values(values)
        if np.array_equal(finite, self._values):
            return
        self._values = finite
        self._range = None
        if finite.size:
            self._tails = tails(finite)

    def levels(self) -> Pair:
        return self._levels

    def value_range(self) -> Pair | None:
        """The values shown, left to right (None before any levels)."""
        return self._range

    def follows_levels(self) -> bool:
        return self._follow

    def set_follow_levels(self, follow: bool) -> None:
        """Auto-scale: fit the range to the values and the levels whenever they change
        (*follow*), or keep it still while only the levels change."""
        self._follow = bool(follow)
        if self._follow:
            self.fit()

    def fit(self) -> None:
        """Fit the range to the bulk of the values and the levels."""
        self._show_range(fit_range(self._levels, self._tails))

    def set_levels(self, lo: float, hi: float, centre: float | None = None) -> None:
        """Show levels *lo* - *hi* (and the *centre* of symmetric levels) without emitting."""
        self._levels = (float(lo), float(hi))
        self._centre = centre
        if self._follow or self._range is None:
            self.fit()
        else:
            self._show_range(widened(self._range, self._levels))
        self._quiet += 1
        try:
            self.region.setRegion(self._levels)
        finally:
            self._quiet -= 1
        self.centre.setVisible(centre is not None)
        if centre is not None:
            self.centre.setValue(centre)
        self._paint_levels()

    def set_colormap(self, name: str) -> None:
        if name != self._cmap:
            self._cmap = name
            self._paint_levels()

    def _show_range(self, rng: Pair) -> None:
        if rng == self._range:
            return
        self._range = rng
        self._count()
        self.plot.setXRange(*rng, padding=0)
        self.plot.setYRange(self.STRIP[0] - 0.03, 1.08, padding=0)
        self._paint_levels()

    def _on_click(self, event) -> None:
        if event.double() and event.button() == Qt.MouseButton.LeftButton:
            self.fit()

    def _count(self) -> None:
        rng = self._range or (0.0, 1.0)
        if self._values.size:
            counts, edges = np.histogram(self._values, bins=self.BINS, range=rng)
        else:
            counts, edges = np.zeros(self.BINS), np.linspace(*rng, self.BINS + 1)
        heights = np.sqrt(counts.astype(float))
        if heights.max() > 0:
            heights /= heights.max()
        self.bars.setOpts(x0=edges[:-1], x1=edges[1:], height=heights, y0=0.0)
        self._centres = 0.5 * (edges[:-1] + edges[1:])
        y, height = self.STRIP
        self.strip.setRect(QRectF(rng[0], y, rng[1] - rng[0], height))

    def _paint_levels(self) -> None:
        lo, hi = self._levels
        strip = gradient(self._cmap, lo, hi if hi > lo else lo + 1e-12)  # ends padded beyond
        self.strip.setBrush(QBrush(strip))
        inside = (self._centres >= lo) & (self._centres <= hi)
        brushes = [self._inside_brush if i else self._outside_brush for i in inside]
        self.bars.setOpts(brushes=brushes)

    # --- user edits --------------------------------------------------------------------
    def _on_changing(self) -> None:
        if self._quiet:
            return
        lo, hi = (float(v) for v in self.region.getRegion())
        old_lo, old_hi = self._levels
        c = self._centre
        if c is not None:
            if hi == old_hi and lo != old_lo and lo < c:
                hi = 2 * c - lo
            elif lo == old_lo and hi != old_hi and hi > c:
                lo = 2 * c - hi
            self._quiet += 1
            try:
                self.region.setRegion((lo, hi))
            finally:
                self._quiet -= 1
        if (lo, hi) == self._levels:
            return
        self._levels = (lo, hi)
        self._paint_levels()
        self.levelsChanging.emit(lo, hi)

    def _on_finished(self) -> None:
        if not self._quiet and self._levels[1] > self._levels[0]:
            self.levelsChosen.emit(*self._levels)

    # --- look --------------------------------------------------------------------------
    def apply_theme(self) -> None:
        t = current_tokens()
        self.view.setBackground(t["surface"])
        inside, outside = QColor(t["muted"]), QColor(t["muted"])
        inside.setAlphaF(0.6)
        outside.setAlphaF(0.22)
        self._inside_brush, self._outside_brush = pg.mkBrush(inside), pg.mkBrush(outside)
        accent = t["accent"]
        hover = QColor(accent)
        hover.setAlphaF(0.08)
        self.region.setBrush(pg.mkBrush(None))
        self.region.setHoverBrush(pg.mkBrush(hover))
        for line in self.region.lines:
            line.setPen(pg.mkPen(accent, width=1.5))
            line.setHoverPen(pg.mkPen(accent, width=3))
        self.centre.setPen(pg.mkPen(t["muted"], width=1, style=Qt.PenStyle.DashLine))
        self._paint_levels()
        self.update()

    def paintEvent(self, _event) -> None:
        t = current_tokens()
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(QPen(t["line"], 1))
        p.setBrush(t["surface"])
        p.drawRoundedRect(QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5), 6, 6)
        p.end()


# ---------------------------------------------------------------------- level fields
def level_digits(value: float, span: float) -> int:
    """Significant digits (4 to 6) that show *value* to about 1/1000 of *span*."""
    if not (math.isfinite(value) and math.isfinite(span)) or value == 0 or span <= 0:
        return 4
    needed = decimals_for(span) + math.floor(math.log10(abs(value))) + 1
    return min(6, max(4, needed))


class LevelFields(QWidget):
    """Two number fields for the colour levels and a note; ``edited(lo, hi)`` fires when the
    user changes a field to a valid pair, a reversed pair is refused inline."""

    edited = Signal(float, float)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setProperty("kit", "range")  # the stylesheet styles the number fields
        self.lo, self.hi = NumberSpin(), NumberSpin()
        self.lo.setAccessibleName("Colour range minimum")
        self.hi.setAccessibleName("Colour range maximum")
        self.lo.valueChanged.connect(self._on_edit)
        self.hi.valueChanged.connect(self._on_edit)
        dash = QLabel("–")
        dash.setProperty("kit", "muted")
        self.note = QLabel()
        self.note.setProperty("kit", "note")
        self.note.setWordWrap(True)
        _small(self.note, 0.9)
        self._note = ""
        self._error = ""
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        row.addWidget(self.lo, stretch=1)
        row.addWidget(dash)
        row.addWidget(self.hi, stretch=1)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addLayout(row)
        layout.addWidget(self.note)

    def levels(self) -> Pair:
        return self.lo.value(), self.hi.value()

    def set_levels(self, lo: float, hi: float) -> None:
        """Show *lo* and *hi* (clears a refused edit), without emitting: four significant
        digits as the notes, more (up to six) where the range needs them."""
        step = (hi - lo) / 100 if hi > lo else None
        for field, value in ((self.lo, lo), (self.hi, hi)):
            field.digits = level_digits(value, hi - lo)
            field.set_quietly(value, step)
        self._set_error("")

    def set_note(self, text: str) -> None:
        self._note = text
        self._show_note()

    def error(self) -> str:
        return self._error

    def _on_edit(self) -> None:
        lo, hi = self.levels()
        if not lo < hi:
            self._set_error(ORDER_ERROR)
            return
        self._set_error("")
        self.edited.emit(lo, hi)

    def _set_error(self, text: str) -> None:
        self._error = text
        for spin in (self.lo, self.hi):
            set_style_property(spin, "invalid", bool(text))
        self._show_note()

    def _show_note(self) -> None:
        set_style_property(self.note, "error", bool(self._error))
        self.note.setText(self._error or self._note)


class ElidedLabel(QLabel):
    """A one-line label that shortens its text with an ellipsis instead of growing."""

    def minimumSizeHint(self) -> QSize:
        return QSize(0, super().minimumSizeHint().height())

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        text = self.fontMetrics().elidedText(
            self.text(), Qt.TextElideMode.ElideRight, self.contentsRect().width()
        )
        align = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        self.style().drawItemText(
            p, self.contentsRect(), align, self.palette(), self.isEnabled(), text,
            self.foregroundRole(),
        )  # fmt: skip
        p.end()


# ---------------------------------------------------------------------- page
class ColourPage(QWidget):
    """Colour map swatches, the level mode, a collapsible histogram and the level fields."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.picker = ColormapPicker()
        self.mode = SegmentedControl(size="sm", expand=True)
        for value, text, tip in MODES:
            self.mode.add_option(value, text, tip)
        self.mode.setAccessibleName("Colour range")
        self.histogram = LevelHistogram()
        self.hist_section = CollapsibleSection("Histogram", expanded=False, separator=False)
        self.hist_section.header.layout().setContentsMargins(0, 0, 0, 0)
        self.hist_section.body_layout().setContentsMargins(0, 6, 0, 0)
        self.hist_section.body_layout().addWidget(self.histogram)
        self.fields = LevelFields()
        self.label = ElidedLabel()  # the level key, shown in the section header
        self.label.setProperty("kit", "muted")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        for widget in (self.picker, self.mode, self.hist_section, self.fields):
            layout.addWidget(widget)

    def note_for(self, mode: str, key: str) -> str:
        label = level_label(key)
        if mode == AUTO_LEVELS:
            return AUTO_NOTE
        if mode == SYMMETRIC_LEVELS and symmetric_centre(key) is not None:
            return f"Centred on {symmetric_centre(key):g}. Remembered for {label}."
        return f"Remembered for {label}. Drag here or on the colour scale."


class ColourLevels:
    """Keeps the Colour section, the plot colour scales and ``controller.view`` in step."""

    def __init__(self, window, page: ColourPage):
        self.window = window
        self.page = page
        self.c: AppController = window.controller
        self._counted = None  # the map the histogram counts

    def target(self) -> tuple[str, str] | None:
        """(plot view, level key) the section edits; None on the Stacked tab."""
        view = self.window.plot_area.current_view()
        if view == "stacked":
            return None
        s = self.c.selection
        return view, s.level_key if view == "map" else level_key(s.reference_kind)

    def levels(self, view: str, key: str) -> Pair | None:
        """The levels plot *view* is drawn with (its kept levels without a map)."""
        fmap = self.window.shown_maps.get(view)
        if fmap is None:
            return self.c.view.levels_for(key)
        return self.c.view.levels_in_effect(key, fmap.values)

    def _histogram_shown(self) -> bool:
        section = self.window.inspector["colour"]
        return (
            not section.isHidden()
            and section.is_expanded()
            and self.page.hist_section.is_expanded()
        )

    def refresh(self) -> None:
        """Show the colour map, mode and levels of the map on screen (without emitting)."""
        target = self.target()
        if target is None or self.c.is_restoring():
            return
        view, key = target
        v, page = self.c.view, self.page
        mode, centre = v.level_mode(key), symmetric_centre(key)
        with QSignalBlocker(page.picker):
            page.picker.set_value(v.colormap if v.colormap in COLOURMAPS else AUTO_COLOURS)
        sym = page.mode.button(SYMMETRIC_LEVELS)
        page.mode.set_option_enabled(SYMMETRIC_LEVELS, centre is not None)
        sym.setToolTip(
            "Not available for raw data" if centre is None else f"Symmetric about {centre:g}"
        )
        with QSignalBlocker(page.mode):
            page.mode.set_value(mode)
        page.label.setText(level_label(key))
        page.label.setToolTip(f"Levels remembered for {level_label(key)}")
        levels = self.levels(view, key)
        if levels is not None:
            page.fields.set_levels(*levels)
        page.fields.set_note(page.note_for(mode, key))
        if not self._histogram_shown():
            return
        fmap = self.window.shown_maps.get(view)
        if fmap is not self._counted:
            self._counted = fmap
            page.histogram.set_values(None if fmap is None else fmap.values)
        page.histogram.set_colormap(v.colormap_for(parse_level_key(key).order))
        if levels is not None:
            sym_centre = centre if mode == SYMMETRIC_LEVELS else None
            page.histogram.set_levels(*levels, centre=sym_centre)

    # --- user edits --------------------------------------------------------------------
    def on_mode(self, mode: str) -> None:
        target = self.target()
        if target is None:
            return
        view, key = target
        levels = self.levels(view, key)
        centre = symmetric_centre(key)
        if mode == SYMMETRIC_LEVELS and centre is not None and levels is not None:
            levels = symmetric_levels(levels, centre)
        elif mode == AUTO_LEVELS:
            levels = None
        self.c.set_level_mode(key, mode, levels)

    def on_changing(self, lo: float, hi: float) -> None:
        """A live histogram drag: recolour the plot at once (kept when the drag ends)."""
        target = self.target()
        if target is None:
            return
        plot = self.window.plots[target[0]]
        if plot.image.image is not None:
            plot.set_levels(lo, hi)
        self.page.fields.set_levels(lo, hi)

    def on_chosen(self, lo: float, hi: float, mode: str | None = None) -> None:
        target = self.target()
        if target is not None:
            set_levels(self.window, target[1], lo, hi, mode)


@user_action("Colour range")
def set_levels(window, key: str, lo: float, hi: float, mode: str | None = None) -> None:
    window.controller.set_levels(key, lo, hi, mode)


# ---------------------------------------------------------------------- settings
class LevelsSetting:
    """Settings protocol for the colour levels: JSON ``{key: {"mode": m, "levels": [lo, hi]}}``
    with the levels of per-unit energy derivatives per cm^-1. Restored levels replace the
    kept ones once settings are restored, in the unit shown then."""

    def __init__(self, controller: AppController):
        self.controller = controller
        self.pending: tuple[dict[str, Pair], dict[str, str]] | None = None

    def settings_value(self) -> str:
        c = self.controller
        v = c.view
        stored: dict[str, dict] = {}
        for key in sorted(set(v.levels) | set(v.level_modes)):
            entry: dict[str, object] = {"mode": v.level_mode(key)}
            if key in v.levels:
                k = parse_level_key(key)
                levels = convert_levels(
                    v.levels[key], c.unit, Unit.CM1, k.order, k.axis_is_energy, k.physical
                )
                entry["levels"] = list(levels)
            stored[key] = entry
        return json.dumps(stored)

    def set_settings_value(self, value) -> bool:
        data = load_json(value)
        if data is None:
            return False
        levels: dict[str, Pair] = {}
        modes: dict[str, str] = {}
        try:
            for key, entry in data.items():
                parse_level_key(key)
                if not isinstance(entry, dict) or entry.get("mode") not in LEVEL_MODES:
                    return False
                modes[key] = entry["mode"]
                pair = to_pair(entry.get("levels"))
                if pair is not None:
                    levels[key] = pair
        except (TypeError, ValueError):
            return False
        self.pending = (levels, modes)
        return True

    def apply(self) -> None:
        if self.pending is None:
            return
        c = self.controller
        stored, modes = self.pending
        self.pending = None
        levels = {}
        for key, pair in stored.items():
            k = parse_level_key(key)
            levels[key] = convert_levels(
                pair, Unit.CM1, c.unit, k.order, k.axis_is_energy, k.physical
            )
        modes = {key: mode for key, mode in modes.items() if mode != default_level_mode(key)}
        c.set_view(levels=levels, level_modes=modes)


# ---------------------------------------------------------------------- install
def install(window) -> None:
    c = window.controller
    page = ColourPage()
    section = window.add_inspector_section("colour", "Colour", page)
    section.set_trailing(page.label)
    sync = ColourLevels(window, page)
    setting = LevelsSetting(c)
    c.restored.connect(setting.apply)  # before showing them

    page.picker.valueChanged.connect(lambda name: c.set_view(colormap=name))
    page.mode.valueChanged.connect(sync.on_mode)
    page.histogram.levelsChanging.connect(sync.on_changing)
    page.histogram.levelsChosen.connect(sync.on_chosen)
    page.fields.edited.connect(lambda lo, hi: sync.on_chosen(lo, hi, FIXED_LEVELS))

    # after the plot area's redraw (connected earlier)
    for signal in (c.resultChanged, c.selectionChanged, c.viewChanged, c.restored):
        signal.connect(sync.refresh)
    c.unitChanged.connect(lambda _old, _new: sync.refresh())
    window.plot_area.tabs.currentChanged.connect(lambda _index: sync.refresh())
    section.toggled.connect(lambda _expanded: sync.refresh())
    page.hist_section.toggled.connect(lambda _expanded: sync.refresh())

    # the inspector histogram opens with the slim bar and closes with the classic histogram
    style = window.plot_area.scale_style_button
    page.hist_section.set_expanded(style.isChecked(), animate=False)
    style.toggled.connect(page.hist_section.set_expanded)
    # auto-scale with the maps' histograms (the plot toolbar button)
    auto_scale = window.plot_area.auto_scale_button
    page.histogram.set_follow_levels(auto_scale.isChecked())
    auto_scale.toggled.connect(page.histogram.set_follow_levels)

    def on_fitted(_view: str) -> None:  # Fit to data (A) on the map or reference shown
        if sync.target() is not None:
            page.histogram.fit()

    window.plot_area.fitted.connect(on_fitted)

    def on_theme() -> None:
        page.histogram.apply_theme()
        page.update()

    window.themeChanged.connect(on_theme)
    update_sections(window)
    sync.refresh()

    if window.persistence is not None:
        window.persistence.bind("view/colormap", page.picker)
        window.persistence.bind("view/levels", setting)
