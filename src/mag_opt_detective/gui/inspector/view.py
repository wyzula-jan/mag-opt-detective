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
"""

from __future__ import annotations

import json
import math
import re

import numpy as np
from PySide6.QtCore import QLocale, Qt
from PySide6.QtGui import QValidator, QWheelEvent
from PySide6.QtWidgets import QAbstractSpinBox, QDoubleSpinBox, QLabel, QVBoxLayout, QWidget

from mag_opt_detective.core.spectra import FieldMap
from mag_opt_detective.core.units import Unit, convert_range
from mag_opt_detective.gui.controller import VIEW_RANGES, AppController, ViewState
from mag_opt_detective.gui.display import format_range, unit_text
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
_PARTIAL_NUMBER = re.compile(r"[+-]?(\d+\.?\d*|\.\d*)?([eE][+-]?\d*)?")


# ---------------------------------------------------------------------- shared helpers
class NumberSpin(QDoubleSpinBox):
    """Number field (C locale, no arrows) that shows six significant digits of any magnitude
    and accepts scientific notation; the wheel changes it only while it has the focus."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setLocale(QLocale.c())
        self.setDecimals(30)  # keeps tiny values (derivative levels) instead of rounding to 0
        self.setRange(-1e300, 1e300)
        self.setKeyboardTracking(False)
        self.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

    def textFromValue(self, value: float) -> str:
        return f"{value:.6g}"

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
    """Show the inspector sections of the plot on screen (``window.inspector_views``)."""
    view = window.plot_area.current_view()
    for name, section in window.inspector.items():
        views = window.inspector_views.get(name)
        section.setVisible(views is None or view in views)


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
class ViewRanges:
    """Keeps the plot ranges, the controls and ``controller.view`` in step."""

    def __init__(self, window, page: ViewPage):
        self.window = window
        self.page = page
        self.c: AppController = window.controller
        self.maps: ShownMaps = window.shown_maps
        self._applied: dict[str, tuple[Pair, Pair]] = {}  # ranges put on the plots with data
        self._key = self.c.selection.level_key

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
        """Put the ranges on the plots that show data."""
        self._applied = self.effective()
        for name, (x, y) in self._applied.items():
            self.window.plots[name].plot.vb.setRange(xRange=x, yRange=y, padding=0)

    def sync_controls(self) -> None:
        """Show the ranges of the plot on screen in the controls (without emitting)."""
        c, page = self.c, self.page
        view = self.window.plot_area.current_view()
        v = c.view
        unit = unit_text(c.unit)
        page.energy.set_unit(unit)
        fmap = self.maps.get(view)
        shared = "Map" if view == "stacked" else "Stacked"
        b_data = span(fmap.field) if fmap is not None else None
        e_data = span(fmap.energy) if fmap is not None else None
        e_note = None
        if e_data is not None:
            e_note = f"Data {format_range(*e_data, unit)} · shared with {shared}"
        _show(page.field, v.field_range, b_data)
        _show(page.energy, v.energy_range, e_data, e_note)
        i_data = None
        if fmap is not None and view == "stacked":
            e = v.energy_range or e_data
            i_data = stacked_extent(fmap, v, e)
        _show(page.intensity, v.stacked_range, i_data, INTENSITY_NOTE)

    def refresh(self) -> None:
        if self.c.is_restoring():  # restored values are shown once settings are restored
            return
        self.apply()
        self.sync_controls()

    def on_selection(self) -> None:
        """Another level key: the stacked spectra have other intensities, so a fixed
        intensity range fits them again."""
        old, self._key = self._key, self.c.selection.level_key
        if self._key != old and not self.c.is_restoring():
            self.c.set_ranges(stacked_range=None)

    # --- user changes on the plots -----------------------------------------------------
    def on_manual(self, view: str) -> None:
        """A pan or zoom on plot *view*: the axes it moved become fixed (on a plot with data;
        an empty one keeps nothing)."""
        applied = self._applied.get(view)
        if applied is None:
            return
        vb = self.window.plots[view].plot.vb
        moved = tuple((float(lo), float(hi)) for lo, hi in vb.viewRange())
        changes = {}
        for i, name in enumerate(VIEW_RANGES[view]):
            if not _same(applied[i], moved[i]):
                changes[name] = moved[i]
        if changes:
            self.c.set_ranges(**changes)

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
            self.c.set_ranges(**{VIEW_RANGES[view][axis]: None})
            self.apply()  # also when it was on Auto: the menu left pyqtgraph's auto-range on

    def on_menu_manual(self, view: str, axis: int) -> None:
        """Manual for one axis in the plot menu: the range shown becomes fixed."""
        if view in self._applied:
            lo, hi = self.window.plots[view].plot.vb.viewRange()[axis]
            self.c.set_ranges(**{VIEW_RANGES[view][axis]: (float(lo), float(hi))})


def _show(control: RangeControl, fixed: Pair | None, data: Pair | None, note=None) -> None:
    """Show *fixed* (or, on Auto, the *data* range) in *control*, with *data* as its extent."""
    shown = fixed or data
    if data is None:
        data = shown or control.extent()
        note = "No data to show yet"
    elif fixed is not None:
        data = (min(data[0], fixed[0]), max(data[1], fixed[1]))
    control.set_extent(*data, note=note)
    if shown is not None:
        control.set_range(*shown)
    control.set_auto(fixed is None)


# ---------------------------------------------------------------------- settings
class ViewSetting:
    """Settings protocol for the plot ranges: JSON with the field range (T), the energy range
    in cm^-1 and the stacked intensity range (per cm^-1 for per-unit energy derivatives); null
    fits the data. Restored ranges are applied in the unit shown once settings are restored."""

    def __init__(self, controller: AppController):
        self.controller = controller
        self.pending: dict[str, Pair | None] | None = None

    def settings_value(self) -> str:
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
            text.editingFinished.connect(lambda: ranges.on_manual(view))


def install(window) -> None:
    c = window.controller
    page = ViewPage()
    window.add_inspector_section("view", "View", page)
    window.inspector_views = dict(SECTION_VIEWS)
    window.shown_maps = ShownMaps(c)
    ranges = ViewRanges(window, page)

    for name, control in page.controls().items():
        control.rangeEdited.connect(lambda lo, hi, n=name: c.set_ranges(**{n: (lo, hi)}))
        control.autoRequested.connect(lambda n=name: c.set_ranges(**{n: None}))

    for view, plot in window.plots.items():
        plot.plot.vb.sigRangeChangedManually.connect(lambda _mask, v=view: ranges.on_manual(v))
        plot.plot.scene().sigMouseClicked.connect(lambda event, v=view: ranges.on_click(v, event))
        plot.plot.autoBtn.clicked.connect(lambda *_args, v=view: c.fit_ranges(v))
        plot.plot.vb.menu.viewAll.triggered.connect(lambda *_args, v=view: c.fit_ranges(v))
        connect_menu(plot.plot.vb.menu, view, ranges)

    # after the plot area's redraw (connected earlier), which draws the stored ranges
    c.selectionChanged.connect(ranges.on_selection)
    for signal in (c.resultChanged, c.selectionChanged, c.viewChanged, c.rangesChanged):
        signal.connect(ranges.refresh)
    c.unitChanged.connect(lambda _old, _new: ranges.refresh())

    def on_tab(_index: int) -> None:
        view = window.plot_area.current_view()
        page.show_view(view)
        update_sections(window)
        ranges.sync_controls()

    window.plot_area.tabs.currentChanged.connect(on_tab)
    on_tab(window.plot_area.tabs.currentIndex())

    setting = ViewSetting(c)

    def on_restored() -> None:
        setting.apply()
        ranges.refresh()

    c.restored.connect(on_restored)
    if window.persistence is not None:
        window.persistence.bind("view/ranges", setting)
