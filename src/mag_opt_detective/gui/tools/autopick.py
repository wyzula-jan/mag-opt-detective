"""Auto-pick tool (W): find a transition on the map shown and put it into the current curve.

*Track*: a click on a line follows it field by field in both directions
(:func:`~mag_opt_detective.core.picking.track`). *Detect*: a box dragged on the map finds every
line inside it (:func:`~mag_opt_detective.core.picking.detect`); the line that continues the
current curve's points is chosen, else the longest, and a click on another line chooses that one.

The lines found are a preview in the map's "preview" layer (the chosen one with hollow diamonds),
not yet in the point table. Accept records the chosen line into the current curve as one undo
step, replacing the curve's points at the same fields; Discard, Esc or another tool drop it.
Changing an option searches again from the same click or box, on the map as shown (plot kind,
derivative and display unit). Energies are kept in cm^-1 and shown in the display unit, so a
unit switch only converts the preview. Only Detect takes left-drags (for the box); the wheel
zooms and the middle button pans in both modes.
"""

from __future__ import annotations

import contextlib
import math
from collections.abc import Iterator
from dataclasses import dataclass, replace

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QEvent, QObject, QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QBrush, QColor, QKeySequence, QPen
from PySide6.QtWidgets import QApplication, QGraphicsRectItem

from mag_opt_detective.core import picking
from mag_opt_detective.core.picking import Feature, Track
from mag_opt_detective.core.processing import Axis
from mag_opt_detective.core.spectra import FieldMap
from mag_opt_detective.core.units import CM1_PER_UNIT, Unit, derivative_scale, from_cm1, to_cm1
from mag_opt_detective.gui.controller import PlotSelection
from mag_opt_detective.gui.display import UNIT_TEXT
from mag_opt_detective.gui.plot_panel import PlotClick
from mag_opt_detective.gui.points_view import OUTLINE
from mag_opt_detective.gui.tools.autopick_bar import DETECT, TRACK, AutoPickBar
from mag_opt_detective.gui.widgets import parse_float

NAME = "autopick"
LAYER = "preview"
SG_ORDER = 2  # polynomial order of the smoothing
WINDOW_PART = 0.02  # default search window: this part of the map's energy range
TRACK_MISSES, DETECT_MISSES = 2, 1  # fields without the line bridged in a row
HISTORY = 3  # points the next energy is predicted from
MIN_LENGTH = 3  # Detect drops lines with fewer points
MARKER_SIZE = 9  # px, the diamonds of the chosen line
SELECT_RADIUS, DRAG_DISTANCE = 12.0, 4.0  # px: a click this close chooses a line; a drag
BUSY_SIZE = 250_000  # map values from which a search shows the wait cursor (about 0.2 s)
DEBOUNCE_MS = 300  # typing in a field searches again after this pause
SINGULAR = {
    Feature.MAX: "maximum",
    Feature.MIN: "minimum",
    Feature.RISING: "rising inflection point",
    Feature.FALLING: "falling inflection point",
}
PLURAL = {
    Feature.MAX: "maxima",
    Feature.MIN: "minima",
    Feature.RISING: "rising inflection points",
    Feature.FALLING: "falling inflection points",
}
IDLE = {
    TRACK: "Click a line on the map: it is followed field by field in both directions.",
    DETECT: "Drag a box around the lines on the map: every line inside it is found.",
}
NO_MAP = "Nothing to pick on yet: process a sweep (or plot a library map) first."
UNDO_KEY = QKeySequence("Ctrl+Z").toString(QKeySequence.SequenceFormat.NativeText)


# ---------------------------------------------------------------------- search
@dataclass(frozen=True)
class Target:
    """Where to search: a click (*seed*, field and energy) or a *box* (field range, energy
    range); energies in cm^-1."""

    seed: tuple[float, float] | None = None
    box: tuple[tuple[float, float], tuple[float, float]] | None = None


@dataclass(frozen=True)
class Options:
    """How to search, in the display unit: *window* (energy), *smooth* (points, 0: off) and
    *prominence* (map values, per energy unit for inflections; None: automatic)."""

    feature: Feature
    window: float
    smooth: int = 0
    prominence: float | None = None


@dataclass(frozen=True, eq=False)
class Candidate:
    """A line found: fields (T), energies (cm^-1) and strengths, sorted by field."""

    field: np.ndarray
    energy: np.ndarray
    strength: np.ndarray

    @classmethod
    def from_track(cls, track: Track, unit: Unit) -> Candidate:
        return cls(track.field, to_cm1(track.energy, unit), track.strength)

    def __len__(self) -> int:
        return self.field.size


def auto_window(fmap: FieldMap) -> float:
    """The default search window: :data:`WINDOW_PART` of the energy range (two digits)."""
    energy = fmap.energy[np.isfinite(fmap.energy)]
    span = float(np.ptp(energy)) if energy.size else 0.0
    return float(f"{WINDOW_PART * span:.2g}") if span > 0 else 1.0


def _smoothing(options: Options) -> picking.Smoothing | None:
    return (options.smooth, SG_ORDER) if options.smooth else None


def _energy_range(fmap: FieldMap, target: Target) -> tuple[float, float]:
    assert target.box is not None
    lo, hi = from_cm1(np.array(target.box[1]), fmap.unit)
    return float(lo), float(hi)


def prominence_for(fmap: FieldMap, target: Target, options: Options) -> float:
    """The prominence a search uses: the typed one, else the automatic one of the map (Track)
    or of the box (Detect)."""
    if options.prominence is not None:
        return options.prominence
    smooth, feature = _smoothing(options), options.feature
    if target.box is None:
        return picking.auto_prominence(fmap, feature, smooth=smooth)
    e_range = _energy_range(fmap, target)
    return picking.auto_prominence(
        fmap, feature, smooth=smooth, b_range=target.box[0], e_range=e_range
    )


def search(fmap: FieldMap, target: Target, options: Options) -> list[Track]:
    """The lines for *target* on *fmap* (display unit); ValueError when a click finds no line
    (see :func:`core.picking.track`)."""
    smooth = _smoothing(options)
    prominence = prominence_for(fmap, target, options)
    if target.seed is not None:
        b, e_cm1 = target.seed
        e = float(from_cm1(e_cm1, fmap.unit))
        seed = (_clip(b, fmap.field), _clip(e, fmap.energy))  # a click on an edge pixel
        found = picking.track(
            fmap,
            seed,
            options.feature,
            window=options.window,
            smooth=smooth,
            prominence=prominence,
            max_misses=TRACK_MISSES,
            history=HISTORY,
        )
        return [found]
    return picking.detect(
        fmap,
        feature=options.feature,
        b_range=target.box[0],
        e_range=_energy_range(fmap, target),
        smooth=smooth,
        prominence=prominence,
        max_jump=options.window,
        min_length=MIN_LENGTH,
        max_misses=DETECT_MISSES,
        history=HISTORY,
    )


def _clip(value: float, axis: np.ndarray) -> float:
    finite = axis[np.isfinite(axis)]
    return float(np.clip(value, finite.min(), finite.max())) if finite.size else value


def default_choice(
    candidates: list[Candidate], previous: Candidate | None = None, reach: float = math.inf
) -> int:
    """The line to preselect: the one closest to *previous* (the line chosen before the search
    ran again, or the current curve's points) on their common fields, if within *reach*
    (cm^-1) on average; else the longest (the strongest of equally long ones); -1 if none."""
    if not candidates:
        return -1
    if previous is not None:
        best, best_distance = -1, reach
        for i, candidate in enumerate(candidates):
            _common, a, b = np.intersect1d(candidate.field, previous.field, return_indices=True)
            if a.size:
                distance = float(np.mean(np.abs(candidate.energy[a] - previous.energy[b])))
                if distance <= best_distance:
                    best, best_distance = i, distance
        if best >= 0:
            return best
    scores = [(len(c), float(np.mean(c.strength))) for c in candidates]
    return max(range(len(candidates)), key=lambda i: scores[i])


def prominence_scale(selection: PlotSelection, unit: Unit, feature: Feature) -> float:
    """Factor of a prominence in *unit* over the same one per cm^-1: the values of a per-unit
    energy derivative scale with the unit, and so do slopes (inflection points)."""
    s = selection
    values = derivative_scale(unit, s.order, s.axis == Axis.ENERGY, s.physical)
    return values * (CM1_PER_UNIT[Unit(unit)] if _is_slope(feature) else 1.0)


def _is_slope(feature: Feature) -> bool:
    return feature in (Feature.RISING, Feature.FALLING)


# ---------------------------------------------------------------------- the tool
class AutoPick(QObject):
    """The Auto-pick tool of one window: its options bar, preview and box drags."""

    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.c = window.controller
        self.bar = AutoPickBar(window.plot_area.plot_box, window.infobar)
        plot = window.plots.map
        self.layer = plot.layer(LAYER)
        self.box_items = (QGraphicsRectItem(), QGraphicsRectItem())  # shadow, dashed line
        for z, item in enumerate(self.box_items):
            item.setZValue(9.7 + 0.05 * z)  # below the overlay layers, above the image
            item.hide()
            plot.plot.addItem(item, ignoreBounds=True)
        self.target: Target | None = None
        self.candidates: list[Candidate] = []
        self.chosen = -1
        self.prominence_used: float | None = None
        self._message: tuple[str, str] | None = None  # (level, text) of the last action
        self._slope = _is_slope(self.feature())  # the prominence is of a slope
        self._quiet = False  # the fields are set by the tool itself
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(DEBOUNCE_MS)
        self._timer.timeout.connect(self.search_again)
        self.drag = BoxDrag(self)

    # --- options ------------------------------------------------------------------------
    def mode(self) -> str:
        return self.bar.mode.value()

    def feature(self) -> Feature:
        return Feature(self.bar.feature.value())

    def is_active(self) -> bool:
        return self.window.tools.active() == NAME

    def wants_box(self) -> bool:
        """Left-drags on the map draw a Detect box (instead of panning)."""
        return self.is_active() and self.mode() == DETECT and self.c.result is not None

    def options(self, fmap: FieldMap) -> Options:
        """The options in the bar for *fmap*; ValueError (and the field marked) if invalid."""
        bar = self.bar
        typed = bool(bar.window_edit.text().strip())
        cm1 = bar.window_edit.cm1()
        bad_window = typed and (cm1 is None or not cm1 > 0)
        bar.window_field.set_invalid(bad_window)
        text = bar.prominence_edit.text().strip()
        prominence = parse_float(text) if text else None
        bad_prominence = bool(text) and (prominence is None or prominence < 0)
        bar.prominence_field.set_invalid(bad_prominence)
        if bad_window:
            raise ValueError("Window: enter a positive energy, or leave it empty for automatic.")
        if bad_prominence:
            raise ValueError("Prominence: enter a number ≥ 0, or leave it empty for automatic.")
        window = float(from_cm1(cm1, fmap.unit)) if typed else auto_window(fmap)
        return Options(self.feature(), window, int(bar.smooth.value()), prominence)

    # --- searching ---------------------------------------------------------------------
    def shown_map(self) -> FieldMap:
        """The map on screen, in the display unit (ValueError if there is none)."""
        maps = getattr(self.window, "shown_maps", None)
        fmap = maps.get("map") if maps is not None else None
        return fmap if fmap is not None else self.c.current_map()

    def track_at(self, b: float, energy: float) -> None:
        """Follow the line nearest to a click at (*b*, *energy* in the display unit)."""
        self._start(Target(seed=(float(b), float(to_cm1(energy, self.c.unit)))))

    def detect_in(self, b_range: tuple[float, float], e_range: tuple[float, float]) -> None:
        """Find the lines in a box (*e_range* in the display unit)."""
        unit = self.c.unit
        b0, b1 = sorted(float(v) for v in b_range)
        e0, e1 = sorted(float(v) for v in to_cm1(np.array(e_range, dtype=float), unit))
        self._start(Target(box=((b0, b1), (e0, e1))))

    def _start(self, target: Target) -> None:
        self.target = target
        self.candidates, self.chosen = [], -1
        self.search_again()

    def search_again(self) -> None:
        """Search for the current target with the options in the bar (and redraw)."""
        self._timer.stop()
        self._message = None
        previous = self.chosen_candidate()
        if self.target is None or self.c.result is None:
            if self.c.result is None:
                self.target = None
            self.candidates, self.chosen = [], -1
            self.refresh()
            return
        try:
            fmap = self.shown_map()
            options = self.options(fmap)
        except ValueError as exc:
            self._fail(str(exc))
            return
        try:
            with _busy(searched_size(fmap, self.target)):
                self.prominence_used = prominence_for(fmap, self.target, options)
                options = replace(options, prominence=self.prominence_used)
                tracks = search(fmap, self.target, options)
        except ValueError as exc:
            seed = self.target.seed
            if seed is None or "within" not in str(exc):
                self._fail(str(exc))
                return
            near = f"±{options.window:.3g} {UNIT_TEXT[Unit(fmap.unit)]}"
            self._fail(
                f"No {SINGULAR[options.feature]} within {near} of the click at {seed[0]:.3g} T. "
                "Click on the line, widen the window or lower the prominence."
            )
            return
        unit = Unit(fmap.unit)
        self.candidates = [Candidate.from_track(t, unit) for t in tracks if len(t)]
        reach = float(to_cm1(options.window, unit))
        self.chosen = default_choice(self.candidates, previous or self.curve_points(), reach)
        if not self.candidates:
            what = PLURAL[options.feature]
            self._message = (
                "warn",
                f"No {what} found in the box. Lower the prominence, smooth the spectra or "
                "drag a larger box.",
            )
        self.refresh()

    def _fail(self, text: str) -> None:
        """Show why nothing was found (the click or box stays, for other options)."""
        self.candidates, self.chosen = [], -1
        self._message = ("warn", text[:1].upper() + text[1:])
        self.refresh()

    def curve_points(self) -> Candidate | None:
        """The current curve's points, as a line found (a Detect box preselects the line that
        continues them)."""
        c = self.c
        if c.points is None or c.curve not in c.points.names:
            return None
        field, energy = c.points.points(c.curve)
        return Candidate(field, energy, np.ones(field.size)) if field.size else None

    def chosen_candidate(self) -> Candidate | None:
        return self.candidates[self.chosen] if 0 <= self.chosen < len(self.candidates) else None

    def choose(self, index: int) -> None:
        if 0 <= index < len(self.candidates) and index != self.chosen:
            self.chosen = index
            self.refresh()

    def choose_at(self, b: float, energy: float) -> bool:
        """Choose the line found nearest to a click at (*b*, *energy* in the display unit),
        if it is within :data:`SELECT_RADIUS` pixels; True if one was."""
        if not self.candidates:
            return False
        dx, dy = (abs(v) or 1.0 for v in self.window.plots.map.plot.vb.viewPixelSize())
        unit = self.c.unit
        distances = [
            float(np.min(np.hypot((c.field - b) / dx, (from_cm1(c.energy, unit) - energy) / dy)))
            for c in self.candidates
        ]
        index = int(np.argmin(distances))
        if distances[index] > SELECT_RADIUS:
            return False
        self.choose(index)
        return True

    def accept(self) -> int:
        """Put the chosen line into the current curve (one undo step); returns how many."""
        candidate = self.chosen_candidate()
        if candidate is None:
            return 0
        c = self.c
        try:
            curve = c.curve_name()
            count = c.record_points(
                candidate.field,
                candidate.energy,
                unit=Unit.CM1,
                text=f"Auto-pick {len(candidate)} points on {curve}",
            )
        except ValueError as exc:
            self._message = ("warn", str(exc)[:1].upper() + str(exc)[1:])
            self.refresh()
            return 0
        self.target, self.candidates, self.chosen = None, [], -1
        self._message = ("ok", f"Added {count} points to {curve}; {UNDO_KEY} takes them back.")
        self.refresh()
        return count

    def discard(self) -> None:
        """Drop the preview (and the click or box it came from)."""
        self._timer.stop()
        self.target, self.candidates, self.chosen = None, [], -1
        self._message = None
        self.refresh()

    # --- showing -----------------------------------------------------------------------
    def refresh(self) -> None:
        self.draw()
        self.update_status()

    def replaced(self, candidate: Candidate) -> int:
        """How many points of the current curve the candidate would replace."""
        c = self.c
        table = c.points
        if table is None or c.curve not in table.names:
            return 0
        rows = np.abs(table.field[None, :] - candidate.field[:, None]).argmin(axis=1)
        return int(np.count_nonzero(~np.isnan(table.column(c.curve)[np.unique(rows)])))

    def update_status(self) -> None:
        bar, c = self.bar, self.c
        has_preview = self.chosen_candidate() is not None
        bar.accept_button.setEnabled(has_preview)
        bar.discard_button.setEnabled(has_preview or self.target is not None)
        unit_text = UNIT_TEXT[c.unit]
        bar.set_prominence_unit(_is_slope(self.feature()), unit_text)
        bar.window_field.set_unit(unit_text)
        self._update_placeholders()
        if c.result is None:
            bar.set_status(NO_MAP, "info")
            return
        if self._message is not None:
            bar.set_status(self._message[1], self._message[0])
            return
        candidate = self.chosen_candidate()
        if candidate is None:
            bar.set_status(IDLE[self.mode()], "info")
            return
        curve = c.curve or "the current curve"
        n = len(candidate)
        points = f"{n} point{'s' * (n != 1)}"
        if len(self.candidates) > 1:
            text = (
                f"{len(self.candidates)} lines found: {points} for {curve} on the one with "
                "diamonds (click another line to choose it)."
            )
        else:
            text = f"{points.capitalize()} found for {curve}."
        replaced = self.replaced(candidate)
        if replaced:
            text += f" Replaces {replaced} of its points."
        bar.set_status(text, "ok")

    def _update_placeholders(self) -> None:
        bar = self.bar
        window = prominence = "auto"
        with contextlib.suppress(ValueError):
            if self.c.result is not None:
                window = f"auto {auto_window(self.shown_map()):.3g}"
        if self.prominence_used is not None and self.target is not None:
            prominence = f"auto {self.prominence_used:.2g}"
        bar.window_edit.setPlaceholderText(window)
        bar.prominence_edit.setPlaceholderText(prominence)

    def draw(self) -> None:
        """The preview: every line found as a thin accent line, the chosen one with hollow
        diamonds on its points, and the click or the box; nothing while the tool is off."""
        layer = self.layer
        layer.clear()
        for item in self.box_items:
            item.hide()
        if not self.is_active() or self.target is None:
            return
        unit = self.c.unit
        accent, halo = self.colors()
        chosen = self.chosen_candidate()
        others = [c for c in self.candidates if c is not chosen]
        lines = [_joined(others, unit)] if others else []  # one item: hundreds draw slowly
        if chosen is not None:
            lines.append((chosen.field, from_cm1(chosen.energy, unit)))
        layer.set_curves(lines, pg.mkPen(accent, width=1.5), shadow_pen=pg.mkPen(halo, width=3.5))
        if chosen is not None:
            x, y = chosen.field, from_cm1(chosen.energy, unit)
            for pen in (pg.mkPen(halo, width=3.5), pg.mkPen(accent, width=1.6)):
                layer.add_points(x, y, symbol="d", size=MARKER_SIZE, pen=pen, brush=None)
        target = self.target
        if target.seed is not None:
            b, e = target.seed[0], float(from_cm1(target.seed[1], unit))
            for pen in (pg.mkPen(halo, width=4), pg.mkPen(accent, width=1.8)):
                layer.add_points([b], [e], symbol="+", size=15, pen=pen)
        if target.box is not None:
            (b0, b1), energies = target.box
            e0, e1 = (float(v) for v in from_cm1(np.array(energies), unit))
            self.show_box(QRectF(QPointF(b0, e0), QPointF(b1, e1)))

    def colors(self) -> tuple[QColor, QColor]:
        """The theme's accent and a halo that sets it off on any colour map: light around a
        dark accent, dark around a light one."""
        accent = QColor(self.window.theme.plot_colors()["accent"])
        halo = QColor(255, 255, 255, 225) if accent.lightnessF() < 0.5 else QColor(OUTLINE)
        return accent, halo

    def show_box(self, rect: QRectF, dragging: bool = False) -> None:
        """Draw the box (*rect* in data coordinates): a dashed accent line on its halo."""
        accent, halo = self.colors()
        shadow, line = self.box_items
        dark = QPen(halo, 2.5)
        dashed = QPen(accent, 1.2, Qt.PenStyle.DashLine)
        for pen in (dark, dashed):
            pen.setCosmetic(True)
        fill = QColor(accent)
        fill.setAlphaF(0.12 if dragging else 0.0)
        shadow.setPen(dark)
        shadow.setBrush(QBrush(fill))
        line.setPen(dashed)
        line.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        for item in self.box_items:
            item.setRect(rect.normalized())
            item.show()

    # --- tool and window events --------------------------------------------------------
    def on_activate(self) -> None:
        self.bar.show()
        self.refresh()

    def on_deactivate(self) -> None:
        self.drag.cancel()
        self.discard()
        self.bar.hide()

    def on_click(self, click: PlotClick) -> None:
        if self.mode() == TRACK:
            if self.c.result is None:
                self.refresh()
            else:
                self.track_at(click.x, click.y)
        else:
            self.choose_at(click.x, click.y)

    def on_mode(self) -> None:
        self.discard()

    def on_feature(self) -> None:
        if _is_slope(self.feature()) != self._slope:  # the prominence means something else
            self._slope = _is_slope(self.feature())
            self.clear_prominence()
        self.search_again()

    def clear_prominence(self) -> None:
        self.prominence_used = None
        with self._quietly():
            self.bar.prominence_edit.clear()
        self.bar.prominence_field.set_invalid(False)

    def on_typed(self) -> None:
        """A typed option searches again after a short pause (or when editing ends)."""
        if not self._quiet:
            self._timer.start()

    def pending(self) -> bool:
        """A typed option waits for the pause before searching again."""
        return self._timer.isActive()

    def flush(self) -> None:
        if self.pending():
            self.search_again()

    def on_unit(self, old: Unit, new: Unit) -> None:
        """Show the preview, the window and the prominence in the new unit (that of a per-unit
        derivative or of a slope scales with it); nothing is searched again."""
        s, feature = self.c.selection, self.feature()
        factor = prominence_scale(s, new, feature) / prominence_scale(s, old, feature)
        typed = parse_float(self.bar.prominence_edit.text())
        with self._quietly():
            self.bar.window_edit.set_unit(new)
            if factor != 1.0 and typed is not None:
                self.bar.prominence_edit.setText(f"{typed * factor:.6g}")
        if self.prominence_used is not None:
            self.prominence_used *= factor
        self.refresh()

    def on_selection(self) -> None:
        """Another map is shown: its values differ, so the prominence goes back to automatic,
        and the search runs again on it."""
        if self.c.is_restoring():
            return
        self.clear_prominence()
        self.search_again()

    def on_result(self) -> None:
        """New data: what was found on the old map no longer applies."""
        self.discard()

    @contextlib.contextmanager
    def _quietly(self) -> Iterator[None]:
        self._quiet = True
        try:
            yield
        finally:
            self._quiet = False


def _joined(candidates: list[Candidate], unit: Unit) -> tuple[np.ndarray, np.ndarray]:
    """The lines of *candidates* as one curve, NaN between them (a gap when drawn)."""
    gap = np.array([np.nan])
    x = np.concatenate([part for c in candidates for part in (c.field, gap)])
    y = np.concatenate([part for c in candidates for part in (from_cm1(c.energy, unit), gap)])
    return x[:-1], y[:-1]


def searched_size(fmap: FieldMap, target: Target) -> int:
    """How many map values a search for *target* goes through."""
    if target.box is None:
        return fmap.values.size
    (b0, b1), _energies = target.box
    columns = np.count_nonzero((fmap.field >= b0) & (fmap.field <= b1))
    return fmap.energy.size * columns


@contextlib.contextmanager
def _busy(size: int) -> Iterator[None]:
    """The wait cursor while a search goes through *size* map values or more."""
    if size < BUSY_SIZE:
        yield
        return
    QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
    try:
        yield
    finally:
        QApplication.restoreOverrideCursor()


class BoxDrag(QObject):
    """Left-drags on the map draw the Detect box (instead of panning); a click there chooses
    one of the lines found. Other buttons, the wheel and the moves (crosshair) pass through."""

    def __init__(self, tool: AutoPick):
        super().__init__(tool)
        self.tool = tool
        self.plot = tool.window.plots.map
        self.start: QPointF | None = None  # data coordinates of the press
        self.press: QPointF | None = None  # its pixel position
        self.plot.view.viewport().installEventFilter(self)

    def cancel(self) -> None:
        self.start = self.press = None

    def _point(self, event, inside: bool) -> QPointF | None:
        vb = self.plot.plot.vb
        scene = self.plot.view.mapToScene(event.position().toPoint())
        if inside and (not vb.sceneBoundingRect().contains(scene) or self.plot.image.image is None):
            return None
        return vb.mapSceneToView(scene)

    def _moved(self, event) -> bool:
        return (event.position() - self.press).manhattanLength() >= DRAG_DISTANCE

    def eventFilter(self, watched, event) -> bool:
        kind = event.type()
        left = Qt.MouseButton.LeftButton
        if kind in (QEvent.Type.MouseButtonPress, QEvent.Type.MouseButtonDblClick):
            if event.button() != left or not self.tool.wants_box():
                return False
            point = self._point(event, inside=True)
            if point is None:
                return False
            self.start, self.press = point, event.position()
            return True
        if self.start is None:
            return False
        if kind == QEvent.Type.MouseMove:
            if self._moved(event):
                end = self._point(event, inside=False)
                self.tool.show_box(QRectF(self.start, end), dragging=True)
            return False  # the crosshair follows
        if kind == QEvent.Type.MouseButtonRelease and event.button() == left:
            start, end = self.start, self._point(event, inside=False)
            moved = self._moved(event)
            self.cancel()
            if not self.tool.wants_box():
                return True
            if moved:
                self.tool.detect_in((start.x(), end.x()), (start.y(), end.y()))
            else:
                self.tool.draw()  # the box shown before the press
                self.tool.choose_at(start.x(), start.y())
            return True
        return False


# ---------------------------------------------------------------------- install
def install(window) -> None:
    """The Auto-pick tool (W) on the map, next to Pan, Zoom and Pick."""
    tool = AutoPick(window)
    window.autopick = tool
    window.tools.register(
        NAME,
        "wand-sparkles",
        "Auto-pick",
        "W",
        on_activate=tool.on_activate,
        on_deactivate=tool.on_deactivate,
        views=("map",),
        on_click=tool.on_click,
    )
    c, bar = window.controller, tool.bar
    bar.mode.valueChanged.connect(lambda _v: tool.on_mode())
    bar.feature.valueChanged.connect(lambda _v: tool.on_feature())
    bar.smooth.valueChanged.connect(lambda _v: tool.search_again())
    for edit in (bar.window_edit, bar.prominence_edit):
        edit.textChanged.connect(lambda _text: tool.on_typed())
        edit.editingFinished.connect(tool.flush)
    bar.accept_button.clicked.connect(tool.accept)
    bar.discard_button.clicked.connect(tool.discard)
    c.unitChanged.connect(tool.on_unit)
    c.selectionChanged.connect(tool.on_selection)
    c.resultChanged.connect(tool.on_result)
    c.pointsChanged.connect(tool.update_status)
    c.restored.connect(tool.refresh)
    window.themeChanged.connect(tool.draw)
    with tool._quietly():
        bar.window_edit.set_unit(c.unit)
    tool.refresh()

    p = window.persistence
    if p is not None:
        p.bind("autopick/mode", bar.mode)
        p.bind("autopick/feature", bar.feature)
        p.bind("autopick/smooth", bar.smooth)
        p.bind("autopick/window_cm1", bar.window_edit)
