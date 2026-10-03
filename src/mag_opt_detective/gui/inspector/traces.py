"""Traces section of the stacked plot: offset between spectra, every n-th spectrum and colour
by field. Every change is kept in ``controller.view`` and drawn at once.

The offset is an intensity: for a per-unit energy derivative it is converted with the unit,
like the intensity range, and saved per cm^-1.
"""

from __future__ import annotations

import json
import math

import numpy as np
from PySide6.QtCore import QRectF, QSignalBlocker, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QPainter, QPainterPath
from PySide6.QtWidgets import QHBoxLayout, QLabel, QSlider, QVBoxLayout, QWidget

from mag_opt_detective.gui.controller import AppController
from mag_opt_detective.gui.inspector.colour import gradient
from mag_opt_detective.gui.inspector.view import NumberSpin, load_json, update_sections
from mag_opt_detective.gui.kit import SegmentedControl, Switch
from mag_opt_detective.gui.plots.base import robust_levels
from mag_opt_detective.gui.theme import current_tokens

EVERY = (
    ("1", "All", "Every spectrum"),
    ("2", "Every 2nd", "Every second spectrum"),
    ("4", "Every 4th", "Every fourth spectrum"),
)
FIELD_PART = (0.1, 0.82)  # the part of the colour map the stacked plot uses for the fields


def nice_ceil(value: float) -> float:
    """The smallest 1, 2 or 5 times a power of ten that is at least *value* (> 0)."""
    if not (math.isfinite(value) and value > 0):
        return 1.0
    power = 10.0 ** math.floor(math.log10(value))
    for factor in (1, 2, 5, 10):
        if factor * power >= value * (1 - 1e-12):
            return factor * power
    return 10 * power


class OffsetControl(QWidget):
    """A slider for quick, live changes and a number field for the offset (>= 0).

    ``valueChanged`` fires on user edits only; slider moves that arrive while the plot is
    still drawing are merged into one.
    """

    valueChanged = Signal(float)
    STEPS = 1000

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setProperty("kit", "range")  # the stylesheet styles the number field
        self._maximum = 0.05
        self._value = 0.0
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, self.STEPS)
        self.slider.setAccessibleName("Offset between spectra")
        self.spin = NumberSpin()
        self.spin.setMinimum(0.0)
        self.spin.setAccessibleName("Offset value")
        self.spin.setFixedWidth(90)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(self.slider, stretch=1)
        layout.addWidget(self.spin)
        self._merge = QTimer(self)  # one redraw for the slider moves queued meanwhile
        self._merge.setSingleShot(True)
        self._merge.setInterval(0)
        self._merge.timeout.connect(lambda: self.valueChanged.emit(self._value))
        self.slider.valueChanged.connect(self._on_slider)
        self.spin.valueChanged.connect(self._on_spin)

    def value(self) -> float:
        return self._value

    def maximum(self) -> float:
        return self._maximum

    def set_value(self, value: float, maximum: float | None = None) -> None:
        """Show *value*; *maximum* is the slider's end (widened to hold the value)."""
        self._value = float(value)
        if maximum is not None:
            self._maximum = nice_ceil(max(maximum, self._value))
        elif self._value > self._maximum:
            self._maximum = nice_ceil(self._value)
        self.spin.set_quietly(self._value, self._maximum / 100)
        with QSignalBlocker(self.slider):
            self.slider.setValue(round(self._value / self._maximum * self.STEPS))

    def _on_slider(self, position: int) -> None:
        self._value = position * self._maximum / self.STEPS
        self.spin.set_quietly(self._value)
        self._merge.start()

    def _on_spin(self, value: float) -> None:
        if value < 0 or value == self._value:
            return
        self._merge.stop()
        self.set_value(value)
        self.valueChanged.emit(self._value)


class FieldLegend(QWidget):
    """The field colours of the stacked spectra: a gradient between the lowest and highest
    field."""

    BAR = 8

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._cmap = "viridis"
        self._fields: tuple[float, float] | None = None
        self.setAccessibleName("Field colours")

    def set_colormap(self, name: str) -> None:
        self._cmap = name
        self.update()

    def set_fields(self, fields: tuple[float, float] | None) -> None:
        self._fields = fields
        self.update()

    def labels(self) -> tuple[str, str]:
        if self._fields is None:
            return "", ""
        return f"{self._fields[0]:g} T", f"{self._fields[1]:g} T"

    def sizeHint(self) -> QSize:
        return QSize(120, self.BAR + 4 + self.fontMetrics().height())

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        bar = QRectF(0, 0, self.width(), self.BAR)
        path = QPainterPath()
        path.addRoundedRect(bar, 3, 3)
        p.fillPath(path, gradient(self._cmap, bar.left(), bar.right(), FIELD_PART))
        p.setPen(current_tokens()["muted"])
        text = QRectF(0, self.BAR + 3, self.width(), self.height() - self.BAR - 3)
        low, high = self.labels()
        p.drawText(text, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop, low)
        p.drawText(text, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop, high)
        p.end()


class TracesPage(QWidget):
    """Offset, every n-th spectrum and colour by field."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        offset_label = QLabel("Offset between spectra")
        offset_label.setProperty("kit", "muted")
        self.offset = OffsetControl()
        show_label = QLabel("Show")
        show_label.setProperty("kit", "muted")
        self.every = SegmentedControl(size="sm", expand=True)
        for value, text, tip in EVERY:
            self.every.add_option(value, text, tip)
        self.every.setAccessibleName("Spectra shown")
        self.by_field = Switch()
        self.by_field.setAccessibleName("Colour by field")
        title = QLabel("<b>Colour by field</b>")
        detail = QLabel("Uses the colour map; Auto uses viridis")
        detail.setProperty("kit", "muted")
        detail.setWordWrap(True)
        title.setBuddy(self.by_field)
        text = QVBoxLayout()
        text.setSpacing(1)
        text.addWidget(title)
        text.addWidget(detail)
        switch_row = QHBoxLayout()
        switch_row.addLayout(text, stretch=1)
        switch_row.addWidget(self.by_field, alignment=Qt.AlignmentFlag.AlignVCenter)
        self.legend = FieldLegend()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(offset_label)
        layout.addWidget(self.offset)
        layout.addSpacing(8)
        layout.addWidget(show_label)
        layout.addWidget(self.every)
        layout.addSpacing(8)
        layout.addLayout(switch_row)
        layout.addWidget(self.legend)


class Traces:
    """Keeps the Traces section, the stacked plot and ``controller.view`` in step."""

    def __init__(self, window, page: TracesPage):
        self.window = window
        self.page = page
        self.c: AppController = window.controller
        self._options: tuple | None = None

    def options(self) -> tuple[int, bool, str]:
        v = self.c.view
        return v.stacked_every, v.stacked_by_field, v.trace_colormap()

    def apply(self) -> None:
        """Draw the stacked plot with the trace options (once per change)."""
        options = self.options()
        if options != self._options:
            self._options = options
            every, by_field, cmap = options
            self.window.plots.stacked.set_trace_options(every, by_field, cmap)

    def refresh(self) -> None:
        if self.c.is_restoring():
            return
        self.apply()
        v, page = self.c.view, self.page
        fmap = self.window.shown_maps.get("stacked")
        maximum = None
        if fmap is not None:
            lo, hi = robust_levels(fmap.values)
            maximum = (hi - lo) / 2
        page.offset.set_value(v.stacked_offset, maximum)
        if str(v.stacked_every) in page.every.options():
            with QSignalBlocker(page.every):
                page.every.set_value(str(v.stacked_every))
        with QSignalBlocker(page.by_field):
            page.by_field.setChecked(v.stacked_by_field)
        page.legend.setVisible(v.stacked_by_field)
        page.legend.set_colormap(v.trace_colormap())
        fields = None
        if fmap is not None and fmap.field.size:
            fields = (float(np.nanmin(fmap.field)), float(np.nanmax(fmap.field)))
        page.legend.set_fields(fields)


class TracesSetting:
    """Settings protocol for the trace options: JSON with the offset (per cm^-1 for per-unit
    energy derivatives), every n-th spectrum and colour by field; applied once settings are
    restored."""

    def __init__(self, controller: AppController):
        self.controller = controller
        self.pending: dict | None = None

    def settings_value(self) -> str:
        c = self.controller
        v = c.view
        return json.dumps(
            {
                "offset": v.stacked_offset / c.derivative_scale(),
                "every": v.stacked_every,
                "by_field": v.stacked_by_field,
            }
        )

    def set_settings_value(self, value) -> bool:
        data = load_json(value)
        if data is None:
            return False
        offset, every, by_field = data.get("offset"), data.get("every"), data.get("by_field")
        if isinstance(offset, bool) or not isinstance(offset, int | float):
            return False
        if not (math.isfinite(offset) and offset >= 0):
            return False
        if every not in (1, 2, 4) or isinstance(every, bool) or not isinstance(by_field, bool):
            return False
        self.pending = {"offset": float(offset), "every": every, "by_field": by_field}
        return True

    def apply(self) -> None:
        if self.pending is None:
            return
        c, data = self.controller, self.pending
        self.pending = None
        c.set_view(
            stacked_offset=data["offset"] * c.derivative_scale(),
            stacked_every=data["every"],
            stacked_by_field=data["by_field"],
        )


def install(window) -> None:
    c = window.controller
    page = TracesPage()
    window.add_inspector_section("traces", "Traces", page)
    traces = Traces(window, page)
    setting = TracesSetting(c)
    c.restored.connect(setting.apply)  # before showing them

    page.offset.valueChanged.connect(lambda value: c.set_view(stacked_offset=value))
    page.every.valueChanged.connect(lambda value: c.set_view(stacked_every=int(value)))
    page.by_field.toggled.connect(lambda on: c.set_view(stacked_by_field=on))

    for signal in (c.resultChanged, c.selectionChanged, c.viewChanged, c.restored):
        signal.connect(traces.refresh)
    c.unitChanged.connect(lambda _old, _new: traces.refresh())
    window.themeChanged.connect(page.legend.update)
    traces.apply()  # before the first drawing
    traces.refresh()
    update_sections(window)

    if window.persistence is not None:
        window.persistence.bind("view/traces", setting)
