"""View section: the field, energy and intensity ranges of the plots.

The ranges live in ``controller.view`` (:class:`ViewState`); None fits the data (Auto), a pair
is fixed. Editing a range control, panning or zooming a plot, its menu, a double-click or the
Fit tool change them through :meth:`AppController.set_ranges` /
:meth:`~AppController.fit_ranges`, which redraw nothing: this module puts the ranges on the
plots, also after every redraw, so they survive level drags and unit switches. The energy range
is shared by all three plots, the field range by the map and the reference. A plot without data
keeps nothing, and a fixed stacked intensity range fits the data again when another kind of
map (level key) is shown. This module also shows the inspector sections that belong to the
plot on screen (``window.inspector_views``) and holds the number field and map cache the other
inspector sections use.

A pan or zoom on a plot moves only that plot while it lasts: the View fields follow at most
every :data:`GESTURE_MS`, and the ranges go into the view state (and onto the other plots) once,
when the mouse button is released or :data:`GESTURE_MS` after the last wheel step
(``window.view_ranges.finish()`` does it at once). A range slider changes the view state and
the plot on screen at once; while it is dragged it keeps its extent, so the range stays under
the cursor, and the other plots catch up when it is released.
"""

from __future__ import annotations

import dataclasses
import json
import math
import re

import numpy as np
from PySide6.QtCore import QEvent, QLocale, QObject, Qt, QTimer
from PySide6.QtGui import QValidator, QWheelEvent
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QDoubleSpinBox,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from mag_opt_detective.core.spectra import FieldMap
from mag_opt_detective.core.units import Unit, convert_range
from mag_opt_detective.gui.controller import VIEW_RANGES, AppController, ViewState
from mag_opt_detective.gui.display import format_number, format_range, unit_text
from mag_opt_detective.gui.kit import RangeControl
from mag_opt_detective.gui.widgets import parse_float

Pair = tuple[float, float]

# the plot views each inspector section belongs to (sections not listed show everywhere)
SECTION_VIEWS = {
    "view": ("map", "stacked", "reference"),
    "colour": ("map", "reference"),
    "traces": ("stacked",),
    "overlays": ("map",),
}
HINT = (
    "Drag a plot to pan, scroll over it or an axis to zoom, double-click to fit. Any change "
    "fixes the range; Auto fits the data again."
)
INTENSITY_NOTE = "Follows the offset while on Auto"
NO_MAP = "Nothing to set until a sweep is processed"
NO_REFERENCE_MAP = "Nothing to set until a reference map exists"
GESTURE_MS = 100  # a pan or zoom: the fields follow at most this often; a wheel ends this after
_PARTIAL_NUMBER = re.compile(r"[+-]?(\d+\.?\d*|\.\d*)?([eE][+-]?\d*)?")


# ---------------------------------------------------------------------- shared helpers
class NumberSpin(QDoubleSpinBox):
    """Number field (C locale, no arrows) that shows six significant digits of any magnitude
    (or :attr:`digits`, as :func:`format_number`) and accepts scientific notation; the wheel
    changes it only while it has the focus."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.digits: int | None = None
        self.setLocale(QLocale.c())
        self.setDecimals(30)  # keeps tiny values (derivative levels) instead of rounding to 0
        self.setRange(-1e300, 1e300)
        self.setKeyboardTracking(False)
        self.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

    def textFromValue(self, value: float) -> str:
        return f"{value:.6g}" if self.digits is None else format_number(value, self.digits)

    def valueFromText(self, text: str) -> float:
        value = parse_float(text)
        return self.value() if value is None else value

    def validate(self, text: str, pos: int):
        if parse_float(text) is not None:
            return QValidator.State.Acceptable, text, pos
        if _PARTIAL_NUMBER.fullmatch(text.strip()):
            return QValidator.State.Intermediate, text, pos
        return QValidator.State.Invalid, text, pos

    def wheelEvent(self, event: QWheelEvent) -> None:
        if self.hasFocus():
            super().wheelEvent(event)
        else:
            event.ignore()

    def set_quietly(self, value: float, step: float | None = None) -> None:
        """Show *value* (and use *step* for the arrow keys) without emitting valueChanged."""
        self.blockSignals(True)
        try:
            if step is not None and math.isfinite(step) and step > 0:
                self.setSingleStep(step)
            self.setValue(value)
        finally:
            self.blockSignals(False)


class ShownMaps:
    """The maps on the plots ("map" and "stacked" show the same one), in the display unit.

    Computed on demand and kept until the result, the selection or the unit change; None when
    there is nothing to show (or it cannot be computed: the plot area reports that).
    """

    def __init__(self, controller: AppController):
        self.controller = controller
        self._for: tuple = (None, None, None)  # result, selection, unit of the kept maps
        self._maps: dict[str, FieldMap | None] = {}

    def get(self, view: str) -> FieldMap | None:
        c = self.controller
        if c.result is None:
            return None
        result, selection, unit = self._for
        if not (result is c.result and selection == c.selection and unit is c.unit):
            self._for, self._maps = (c.result, c.selection, c.unit), {}
        name = "reference" if view == "reference" else "map"
        if name not in self._maps:
            try:
                self._maps[name] = c.reference_map() if name == "reference" else c.current_map()
            except ValueError:
                self._maps[name] = None
        return self._maps[name]


def span(values: np.ndarray) -> Pair | None:
    """(min, max) of the finite *values* (widened when they are all equal), or None."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    lo, hi = float(finite.min()), float(finite.max())
    if lo == hi:
        pad = 0.5 * abs(lo) or 0.5
        lo, hi = lo - pad, hi + pad
    return lo, hi


def stacked_extent(fmap: FieldMap, view: ViewState, energy: Pair | None) -> Pair:
    """Intensity range of the stacked spectra (with their offsets) within *energy*, padded."""
    shown = np.arange(0, fmap.field.size, max(1, view.stacked_every))
    rows = np.ones(fmap.energy.size, dtype=bool)
    if energy is not None:
        inside = (fmap.energy >= energy[0]) & (fmap.energy <= energy[1])
        rows = inside if inside.any() else rows
    values = fmap.values[rows][:, shown] + np.arange(shown.size) * view.stacked_offset
    extent = span(values)
    if extent is None:
        return 0.0, 1.0
    pad = 0.04 * (extent[1] - extent[0])
    return extent[0] - pad, extent[1] + pad


def update_sections(window) -> None:
    """Show the inspector sections of the plot on screen (``window.inspector_views``).

    A plot with nothing to show (no processed map, no reference map) has nothing to set:
    its sections are hidden and the subtitle says what is missing.
    """
    view = window.plot_area.current_view()
    shown = window.shown_maps.get(view) is not None
    for name, section in window.inspector.items():
        views = window.inspector_views.get(name)
        section.setVisible(views is None or (shown and view in views))
    if shown:
        window.set_inspector_subtitle()
    elif view == "reference":
        window.set_inspector_subtitle(NO_REFERENCE_MAP)
    else:
        window.set_inspector_subtitle(NO_MAP)


def load_json(value) -> dict | None:
    """A stored JSON object (dict or text), else None."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return None
    return value if isinstance(value, dict) else None


def to_pair(value) -> Pair | None:
    """A stored ``[lo, hi]`` with finite lo < hi; ValueError for anything else but null."""
    if value is None:
        return None
    if not (isinstance(value, list | tuple) and len(value) == 2):
        raise ValueError(f"not a range: {value!r}")
    lo, hi = (float(v) for v in value)
    if not (math.isfinite(lo) and math.isfinite(hi) and lo < hi):
        raise ValueError(f"not a range: {value!r}")
    return lo, hi


def _same(a: Pair, b: Pair) -> bool:
    tol = 1e-9 * max(abs(a[1] - a[0]), abs(b[1] - b[0]), 1e-300)
    return abs(a[0] - b[0]) <= tol and abs(a[1] - b[1]) <= tol


# ---------------------------------------------------------------------- page
class ViewPage(QWidget):
    """Range controls for the field B, the energy E (shared) and the stacked intensity."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.field = RangeControl("Field <i>B</i>", "T", name="Field")
        self.energy = RangeControl("Energy <i>E</i>", unit_text(Unit.CM1), name="Energy")
        self.intensity = RangeControl("Intensity", "", name="Intensity")
        self.hint = QLabel(HINT)
        self.hint.setProperty("kit", "muted")
        self.hint.setWordWrap(True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(14)
        for widget in (self.field, self.energy, self.intensity, self.hint):
            layout.addWidget(widget)

    def controls(self) -> dict[str, RangeControl]:
        """Range control by ViewState field name."""
        return {
            "field_range": self.field,
            "energy_range": self.energy,
            "stacked_range": self.intensity,
        }

    def show_view(self, view: str) -> None:
        """The controls of plot *view*: B and E for maps, E and intensity for stacked."""
        self.field.setVisible(view != "stacked")
        self.intensity.setVisible(view == "stacked")


# ---------------------------------------------------------------------- ranges
class ViewRanges(QObject):
    """Keeps the plot ranges, the controls and ``controller.view`` in step.

    A pan or zoom on a plot (pyqtgraph's ``sigRangeChangedManually``, every mouse move or wheel
    step) is kept here while it lasts: only that plot moves, the View fields follow at most
    every :data:`GESTURE_MS`, and :meth:`finish` puts it into the view state once (which draws
    the shared ranges on the other plots): on the release of the mouse button, or
    :data:`GESTURE_MS` after the last step without a button held (the wheel). Should the
    release get lost, the gesture ends when the mouse leaves the plot, the plot loses the focus
    or the application is no longer active.
    """

    def __init__(self, window, page: ViewPage):
        super().__init__(page)
        self.window = window
        self.page = page
        self.c: AppController = window.controller
        self.maps: ShownMaps = window.shown_maps
        self._applied: dict[str, tuple[Pair, Pair]] = {}  # ranges put on the plots with data
        self._key = self.c.selection.level_key
        self._pending: dict[str, Pair] | None = None  # ranges of a pan or zoom in progress
        self._follow = self._timer(self.sync_controls)  # the fields follow the gesture
        self._idle = self._timer(self._on_idle)  # restarted on every step: the gesture ended
        self._plots: set[QObject] = set()  # the plot views and view ports: they end gestures
        self._held = False  # a mouse button is down on a plot (a drag)
        self._sliders: dict[QObject, RangeControl] = {}
        self._fields: set[QObject] = set()  # the number fields of the controls
        self._sliding: set[RangeControl] = set()  # controls whose slider is dragged or keyed
        self._last: dict[RangeControl, Pair] = {}  # the range each control was given last
        QApplication.instance().applicationStateChanged.connect(self._on_app_state)

    def _timer(self, slot) -> QTimer:
        timer = QTimer(self)
        timer.setSingleShot(True)
        timer.setInterval(GESTURE_MS)
        timer.timeout.connect(slot)
        return timer

    # --- what is shown -----------------------------------------------------------------
    def effective(self) -> dict[str, tuple[Pair, Pair]]:
        """(x, y) range of each plot that shows data: the fixed ranges, else the data's."""
        v = self.c.view
        out: dict[str, tuple[Pair, Pair]] = {}
        for name in ("map", "reference"):
            fmap = self.maps.get(name)
            if fmap is None:
                continue
            b = v.field_range or span(fmap.field)
            e = v.energy_range or span(fmap.energy)
            if b is None or e is None:
                continue
            out[name] = (b, e)
            if name == "map":
                out["stacked"] = (e, v.stacked_range or stacked_extent(fmap, v, e))
        return out

    def apply(self) -> None:
        """Put the ranges on the plots that show data; while a range slider is in use only on
        the plot on screen (the others follow when it is released)."""
        self._applied = self.effective()
        only = self.window.plot_area.current_view() if self._sliding else None
        for name, (x, y) in self._applied.items():
            if only is None or name == only:
                self.window.plots[name].plot.vb.setRange(xRange=x, yRange=y, padding=0)

    def shown(self) -> ViewState:
        """The view state with the ranges of a pan or zoom in progress."""
        v = self.c.view
        return v if self._pending is None else dataclasses.replace(v, **self._pending)

    def sync_controls(self) -> None:
        """Show the ranges of the plot on screen in the controls (without emitting); those of
        a pan or zoom in progress, if any."""
        c, page = self.c, self.page
        view = self.window.plot_area.current_view()
        v = self.shown()
        unit = unit_text(c.unit)
        page.energy.set_unit(unit)
        fmap = self.maps.get(view)
        shared = "Map" if view == "stacked" else "Stacked"
        b_data = span(fmap.field) if fmap is not None else None
        e_data = span(fmap.energy) if fmap is not None else None
        e_note = None
        if e_data is not None:
            e_note = f"Data {format_range(*e_data, unit)} · shared with {shared}"
        self._show(page.field, v.field_range, b_data)
        self._show(page.energy, v.energy_range, e_data, e_note)
        i_data = None
        if fmap is not None and view == "stacked":
            e = v.energy_range or e_data
            i_data = stacked_extent(fmap, v, e)
        self._show(page.intensity, v.stacked_range, i_data, INTENSITY_NOTE)

    def _show(self, control: RangeControl, fixed: Pair | None, data: Pair | None, note=None):
        _show(control, fixed, data, note, keep_extent=control in self._sliding)
        self._last[control] = control.range()

    def refresh(self) -> None:
        """Put the ranges on the plots and into the controls; a pan or zoom in progress goes
        into the view state first (after a redraw, which drew the state's ranges)."""
        if self.c.is_restoring():  # restored values are shown once settings are restored
            return
        if self._pending is not None:
            self.finish()  # refreshes
            return
        self.apply()
        self.sync_controls()

    def on_unit(self, old: Unit, new: Unit) -> None:
        """Another energy unit: a pan or zoom in progress (in the old unit) goes into the view
        state converted, as the state's ranges were."""
        pending = self._pending
        self.discard()
        if pending:
            moved = ViewState(**pending).converted(old, new, self.c.selection)
            self.c.set_ranges(**{name: getattr(moved, name) for name in pending})
        self.refresh()

    def on_ranges(self) -> None:
        """The ranges were set (controls, Fit, Auto, a pan or zoom put into the state): a pan
        or zoom still in progress gives way to them."""
        self.discard()
        self.refresh()

    def on_selection(self) -> None:
        """Another level key: the stacked spectra have other intensities, so a fixed
        intensity range fits them again."""
        old, self._key = self._key, self.c.selection.level_key
        if self._key != old and not self.c.is_restoring():
            self.finish()
            self.c.set_ranges(stacked_range=None)

    def edit(self, **ranges: Pair | None) -> None:
        """Set ranges from the controls, after a pan or zoom in progress. An end its control
        still showed from before the last steps of that gesture (the end not edited) takes
        the gesture's value."""
        controls = self.page.controls()
        for name, pair in ranges.items():
            fresh = None if self._pending is None else self._pending.get(name)
            shown = self._last.get(controls.get(name))
            if pair is None or fresh is None or shown is None:
                continue
            merged = tuple(f if p == s else p for p, s, f in zip(pair, shown, fresh, strict=True))
            if merged[0] < merged[1]:
                ranges[name] = merged
        self.finish()
        self.c.set_ranges(**ranges)

    # --- user changes on the plots -----------------------------------------------------
    def on_manual(self, view: str) -> None:
        """A pan or zoom step on plot *view*: the axes it moved become fixed (on a plot with
        data; an empty one keeps nothing), in the view state once the gesture ends."""
        applied = self._applied.get(view)
        if applied is None:
            return
        vb = self.window.plots[view].plot.vb
        moved = tuple((float(lo), float(hi)) for lo, hi in vb.viewRange())
        pending = {} if self._pending is None else self._pending
        for i, name in enumerate(VIEW_RANGES[view]):
            if name in pending or not _same(applied[i], moved[i]):
                pending[name] = moved[i]
        self._pending = pending
        self._idle.start()
        if not self._follow.isActive():
            self._follow.start()

    def on_typed(self, view: str) -> None:
        """Limits typed into the plot menu: fixed at once."""
        self.on_manual(view)
        self.finish()

    def finish(self) -> None:
        """Put a pan or zoom in progress into the view state now (this draws its ranges on
        the other plots and shows them in the fields); nothing without one."""
        pending = self._pending
        self.discard()
        if pending is None:
            return
        view = self.c.view
        if pending:
            self.c.set_ranges(**pending)  # refreshes through rangesChanged
        if self.c.view == view:  # nothing changed: show the state again
            self.refresh()

    def discard(self) -> None:
        """Forget a pan or zoom in progress (the plots keep showing it until a refresh)."""
        self._pending = None
        self._idle.stop()
        self._follow.stop()

    def in_gesture(self) -> bool:
        """True while a pan or zoom is not yet in the view state."""
        return self._pending is not None

    def _on_idle(self) -> None:
        if self._held:
            self._idle.start()  # a drag held still: it ends on the release
            return
        self.finish()

    def _on_release(self) -> None:
        if not self._held:
            self.finish()

    def _on_app_state(self, state: Qt.ApplicationState) -> None:
        if state != Qt.ApplicationState.ApplicationActive:  # a release may not come back
            self._held = False
            self.finish()

    def watch_plot(self, view: QWidget) -> None:
        """End gestures on the mouse releases in plot *view* (a GraphicsView), or when the
        mouse leaves it or it loses the focus (the release went elsewhere)."""
        for widget in (view, view.viewport()):
            self._plots.add(widget)
            widget.installEventFilter(self)

    def watch_control(self, control: RangeControl) -> None:
        """Keep the extent of *control*'s slider while it is dragged or moved with keys; a
        press on it or its number fields first puts a pan or zoom in progress into the state,
        so that they show its ranges."""
        self._sliders[control.slider] = control
        control.slider.installEventFilter(self)
        for spin in (control.lo_spin, control.hi_spin):
            for field in (spin, spin.lineEdit()):
                self._fields.add(field)
                field.installEventFilter(self)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        kind = event.type()
        if watched in self._plots:
            if kind == QEvent.Type.MouseButtonPress:
                self._held = True
            elif kind == QEvent.Type.MouseButtonRelease:
                self._held = False
                QTimer.singleShot(0, self, self._on_release)  # after pyqtgraph's last step
            elif kind in (QEvent.Type.Leave, QEvent.Type.FocusOut):  # no drag goes on
                self._held = False
                self.finish()
        elif watched in self._sliders:
            control = self._sliders[watched]
            if kind in (QEvent.Type.MouseButtonPress, QEvent.Type.KeyPress):
                self.finish()
                self._sliding.add(control)
            elif control in self._sliding and (
                kind in (QEvent.Type.MouseButtonRelease, QEvent.Type.FocusOut)
                or (kind == QEvent.Type.KeyRelease and not event.isAutoRepeat())
            ):
                self._sliding.discard(control)
                QTimer.singleShot(0, self, self.refresh)  # the extent and other plots
        elif watched in self._fields and kind in (
            QEvent.Type.MouseButtonPress,
            QEvent.Type.FocusIn,
        ):
            self.finish()
        return False

    def on_click(self, view: str, event) -> None:
        """A double-click in the data area of plot *view* fits it to the data, unless the
        active tool takes the clicks there (Pick)."""
        if not (event.double() and event.button() == Qt.MouseButton.LeftButton):
            return
        tools = self.window.tools
        tool = tools.tool(tools.active())
        if view not in self._applied or (tool.on_click is not None and view in tool.views):
            return
        vb = self.window.plots[view].plot.vb
        if vb.sceneBoundingRect().contains(event.scenePos()):
            self.c.fit_ranges(view)

    def on_menu_auto(self, view: str, axis: int) -> None:
        """Auto for one axis in the plot menu: that range fits the data again."""
        if view in self._applied:
            self.edit(**{VIEW_RANGES[view][axis]: None})
            self.apply()  # also when it was on Auto: the menu left pyqtgraph's auto-range on

    def on_menu_manual(self, view: str, axis: int) -> None:
        """Manual for one axis in the plot menu: the range shown becomes fixed."""
        if view in self._applied:
            lo, hi = self.window.plots[view].plot.vb.viewRange()[axis]
            self.edit(**{VIEW_RANGES[view][axis]: (float(lo), float(hi))})


def _show(
    control: RangeControl,
    fixed: Pair | None,
    data: Pair | None,
    note=None,
    keep_extent: bool = False,
) -> None:
    """Show *fixed* (or, on Auto, the *data* range) in *control*, with *data* (and *fixed*) as
    its extent; with *keep_extent* (its slider is in use) the extent stays."""
    shown = fixed or data
    if data is None:
        data = shown or control.extent()
        note = "No data to show yet"
    elif fixed is not None:
        data = (min(data[0], fixed[0]), max(data[1], fixed[1]))
    if not keep_extent:
        control.set_extent(*data, note=note)
    if shown is not None:
        control.set_range(*shown)
    control.set_auto(fixed is None)


# ---------------------------------------------------------------------- settings
class ViewSetting:
    """Settings protocol for the plot ranges: JSON with the field range (T), the energy range
    in cm^-1 and the stacked intensity range (per cm^-1 for per-unit energy derivatives); null
    fits the data. Restored ranges are applied in the unit shown once settings are restored.
    A pan or zoom still in progress in *ranges* goes into the view state before it is saved."""

    def __init__(self, controller: AppController, ranges: ViewRanges | None = None):
        self.controller = controller
        self.ranges = ranges
        self.pending: dict[str, Pair | None] | None = None

    def settings_value(self) -> str:
        if self.ranges is not None:
            self.ranges.finish()
        c = self.controller
        v = c.view
        scale = c.derivative_scale()
        intensity = None
        if v.stacked_range is not None:
            intensity = [v.stacked_range[0] / scale, v.stacked_range[1] / scale]
        return json.dumps(
            {
                "field": _list(v.field_range),
                "energy_cm1": _list(convert_range(v.energy_range, c.unit, Unit.CM1)),
                "intensity": intensity,
            }
        )

    def set_settings_value(self, value) -> bool:
        data = load_json(value)
        if data is None:
            return False
        try:
            self.pending = {
                "field": to_pair(data.get("field")),
                "energy_cm1": to_pair(data.get("energy_cm1")),
                "intensity": to_pair(data.get("intensity")),
            }
        except (TypeError, ValueError):
            return False
        return True

    def apply(self) -> None:
        """Put restored ranges into the view, in the display unit."""
        if self.pending is None:
            return
        c, data = self.controller, self.pending
        self.pending = None
        scale = c.derivative_scale()
        intensity = data["intensity"]
        if intensity is not None:
            intensity = (intensity[0] * scale, intensity[1] * scale)
        c.set_view(
            field_range=data["field"],
            energy_range=convert_range(data["energy_cm1"], Unit.CM1, c.unit),
            stacked_range=intensity,
        )


def _list(pair) -> list[float] | None:
    return None if pair is None else [float(pair[0]), float(pair[1])]


# ---------------------------------------------------------------------- install
def connect_menu(menu, view: str, ranges: ViewRanges) -> None:
    """Route the range entries of pyqtgraph's plot menu (*menu*: a ViewBoxMenu) through the
    View section: Auto fits that axis, Manual and typed limits fix it. Its auto-range options
    (percent, visible data, pan only) are hidden, as the View section's Auto replaces them."""
    for axis, ui in enumerate(menu.ctrl):
        for widget in (ui.autoPercentSpin, ui.visibleOnlyCheck, ui.autoPanCheck):
            widget.hide()
        ui.autoRadio.clicked.connect(lambda *_args, i=axis: ranges.on_menu_auto(view, i))
        ui.manualRadio.clicked.connect(lambda *_args, i=axis: ranges.on_menu_manual(view, i))
        for text in (ui.minText, ui.maxText):
            text.editingFinished.connect(lambda: ranges.on_typed(view))


def install(window) -> None:
    c = window.controller
    page = ViewPage()
    window.add_inspector_section("view", "View", page)
    window.inspector_views = dict(SECTION_VIEWS)
    window.shown_maps = ShownMaps(c)
    ranges = window.view_ranges = ViewRanges(window, page)

    for name, control in page.controls().items():
        control.rangeEdited.connect(lambda lo, hi, n=name: ranges.edit(**{n: (lo, hi)}))
        control.autoRequested.connect(lambda n=name: ranges.edit(**{n: None}))
        ranges.watch_control(control)

    for view, plot in window.plots.items():
        plot.plot.vb.sigRangeChangedManually.connect(lambda _mask, v=view: ranges.on_manual(v))
        ranges.watch_plot(plot.view)
        plot.plot.scene().sigMouseClicked.connect(lambda event, v=view: ranges.on_click(v, event))
        plot.plot.autoBtn.clicked.connect(lambda *_args, v=view: c.fit_ranges(v))
        plot.plot.vb.menu.viewAll.triggered.connect(lambda *_args, v=view: c.fit_ranges(v))
        connect_menu(plot.plot.vb.menu, view, ranges)

    # after the plot area's redraw (connected earlier), which draws the stored ranges
    c.selectionChanged.connect(ranges.on_selection)
    c.resultChanged.connect(lambda: update_sections(window))
    for signal in (c.resultChanged, c.selectionChanged, c.viewChanged):
        signal.connect(ranges.refresh)
    c.rangesChanged.connect(ranges.on_ranges)

    c.unitChanged.connect(ranges.on_unit)

    def on_tab(_index: int) -> None:
        view = window.plot_area.current_view()
        page.show_view(view)
        update_sections(window)
        ranges.refresh()

    window.plot_area.tabs.currentChanged.connect(on_tab)
    on_tab(window.plot_area.tabs.currentIndex())

    setting = ViewSetting(c, ranges)

    def on_restored() -> None:
        ranges.discard()
        setting.apply()
        ranges.refresh()

    c.restored.connect(on_restored)
    if window.persistence is not None:
        window.persistence.bind("view/ranges", setting)
