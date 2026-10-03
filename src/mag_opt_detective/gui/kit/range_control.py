"""Editor for one axis range: Auto/Fixed, a range slider, two number fields and a note."""

from __future__ import annotations

import json
import math
import re

from PySide6.QtCore import QLocale, QSignalBlocker, Signal
from PySide6.QtGui import QWheelEvent
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from mag_opt_detective.gui.display import format_number
from mag_opt_detective.gui.kit._common import (
    json_object,
    set_style_property,
    to_bool,
    to_float,
)
from mag_opt_detective.gui.kit.range_slider import RangeSlider
from mag_opt_detective.gui.kit.segmented import SegmentedControl

ORDER_ERROR = "The lower limit must be below the upper limit."


class _Spin(QDoubleSpinBox):
    """Number field that ignores the wheel unless it has focus (scrolling past it is safe).

    It shows its value as the notes do (four significant digits, no trailing zeros), with
    at least the decimals that resolve the range (:attr:`resolution`).
    """

    resolution = 0

    def textFromValue(self, value: float) -> str:
        shown = min(self.decimals(), max(self.resolution, significant_decimals(value)))
        text = f"{value:.{shown}f}"
        return text.rstrip("0").rstrip(".") if "." in text else text

    def wheelEvent(self, event: QWheelEvent) -> None:
        if self.hasFocus():
            super().wheelEvent(event)
        else:
            event.ignore()


def decimals_for(width: float) -> int:
    """Decimals that resolve about 1/1000 of *width*."""
    if not math.isfinite(width) or width <= 0:
        return 3
    return min(8, max(0, 3 - math.floor(math.log10(width))))


def significant_decimals(value: float, digits: int = 4) -> int:
    """Decimals that show *value* with *digits* significant digits (at most 8)."""
    if not math.isfinite(value) or value == 0:
        return 0
    return min(8, max(0, digits - 1 - math.floor(math.log10(abs(value)))))


class RangeControl(QWidget):
    """Range editor as in the inspector mockup.

    ``rangeEdited(lo, hi)`` fires when the user changes the range with the slider or the number
    fields, or picks Fixed; the control then switches itself to Fixed. ``autoRequested`` fires
    when the user picks Auto; the owner should then call :meth:`set_range` with the data range.
    Programmatic setters are silent. A low limit at or above the high one is refused with an
    inline message. Settings protocol: JSON ``{"auto": bool, "lo": float, "hi": float}``, with
    lo/hi multiplied by :meth:`set_settings_scale` (e.g. cm⁻¹ per displayed energy unit).
    """

    rangeEdited = Signal(float, float)
    autoRequested = Signal()

    def __init__(
        self,
        title: str = "",
        unit: str = "",
        parent: QWidget | None = None,
        *,
        name: str | None = None,
    ):
        super().__init__(parent)
        self._lo, self._hi = 0.0, 1.0
        self._a, self._b = 0.0, 1.0
        self._unit = unit
        self._note: str | None = None
        self._error = ""
        self._auto = True
        self._scale = 1.0
        plain = name or re.sub(r"<[^>]+>", "", title).strip() or "Range"

        self.title = QLabel(title)
        font = self.title.font()
        font.setBold(True)
        self.title.setFont(font)
        self.mode = SegmentedControl(size="xs")
        self.mode.add_option("auto", "Auto", "Fit the range to the data")
        self.mode.add_option("fixed", "Fixed", "Keep this range")
        self.mode.setAccessibleName(f"{plain} range mode")
        self.mode.valueChanged.connect(self._on_mode)

        self.slider = RangeSlider()
        self.slider.setAccessibleName(f"{plain} range")
        self.slider.set_muted(True)
        self.slider.valuesChanged.connect(self._on_slider)

        self.lo_spin, self.hi_spin = _Spin(), _Spin()
        for spin, end in ((self.lo_spin, "minimum"), (self.hi_spin, "maximum")):
            spin.setLocale(QLocale.c())
            spin.setKeyboardTracking(False)
            spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
            spin.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            spin.setAccessibleName(f"{plain} {end}")
        self.lo_spin.valueChanged.connect(lambda _v: self._on_spin(low=True))
        self.hi_spin.valueChanged.connect(lambda _v: self._on_spin(low=False))
        dash = QLabel("–")
        dash.setProperty("kit", "muted")

        self.note = QLabel()
        self.note.setProperty("kit", "note")
        self.note.setWordWrap(True)
        small = self.note.font()
        if small.pointSizeF() > 0:
            small.setPointSizeF(small.pointSizeF() * 0.9)
            self.note.setFont(small)

        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.addWidget(self.title, stretch=1)
        head.addWidget(self.mode)
        inputs = QHBoxLayout()
        inputs.setContentsMargins(0, 0, 0, 0)
        inputs.setSpacing(6)
        inputs.addWidget(self.lo_spin, stretch=1)
        inputs.addWidget(dash)
        inputs.addWidget(self.hi_spin, stretch=1)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addLayout(head)
        layout.addWidget(self.slider)
        layout.addLayout(inputs)
        layout.addWidget(self.note)

        self.setProperty("kit", "range")
        self.setAccessibleName(plain)
        self._sync()

    # --- public API --------------------------------------------------------------------
    def set_extent(self, a: float, b: float, note: str | None = None) -> None:
        """The data range (slider extent); *note* replaces the default "Data a – b unit"."""
        self._a, self._b = sorted((float(a), float(b)))
        self._note = note
        self.slider.set_extent(self._a, self._b)
        self._sync()

    def extent(self) -> tuple[float, float]:
        return self._a, self._b

    def set_range(self, lo: float, hi: float) -> None:
        lo, hi = float(lo), float(hi)
        if not (math.isfinite(lo) and math.isfinite(hi)):
            raise ValueError("range limits must be finite")
        self._lo, self._hi = min(lo, hi), max(lo, hi)
        self._sync()

    def range(self) -> tuple[float, float]:
        return self._lo, self._hi

    def set_auto(self, auto: bool) -> None:
        self._auto = bool(auto)
        self._sync()

    def is_auto(self) -> bool:
        return self._auto

    def set_unit(self, text: str) -> None:
        self._unit = text
        self._sync()

    def unit(self) -> str:
        return self._unit

    def error(self) -> str:
        """The inline validation message, or "" when the fields are valid."""
        return self._error

    def set_settings_scale(self, scale: float) -> None:
        """Stored lo/hi = shown values times *scale* (default 1)."""
        if not (math.isfinite(scale) and scale > 0):
            raise ValueError("scale must be a positive number")
        self._scale = float(scale)

    # --- user edits --------------------------------------------------------------------
    def _on_mode(self, value: str) -> None:
        self._auto = value == "auto"
        self._sync()
        if self._auto:
            self.autoRequested.emit()
        else:
            self.rangeEdited.emit(self._lo, self._hi)

    def _on_slider(self, lo: float, hi: float) -> None:
        self._lo, self._hi = lo, hi
        self._auto = False
        self._sync()
        self.rangeEdited.emit(lo, hi)

    def _on_spin(self, low: bool) -> None:
        # the other field keeps the precise stored value unless it holds a refused edit
        lo = self.lo_spin.value() if low or self._error else self._lo
        hi = self.hi_spin.value() if not low or self._error else self._hi
        if lo >= hi:
            self._error = ORDER_ERROR
            self._sync_state()
            return
        self._lo, self._hi = lo, hi
        self._auto = False
        self._sync()
        self.rangeEdited.emit(lo, hi)

    # --- display -----------------------------------------------------------------------
    def _sync(self) -> None:
        """Show the stored (valid) state in the child widgets, without emitting."""
        self._error = ""  # the fields get the stored values back
        width = self._b - self._a
        resolution = decimals_for(width if width > 0 else abs(self._hi - self._lo))
        decimals = max(resolution, *(significant_decimals(v) for v in (self._lo, self._hi)))
        span = max(width, self._hi - self._lo, 1e-12)
        step = 10.0 ** (math.floor(math.log10(span)) - 2)
        low = min(self._a - 10 * span, self._lo)
        high = max(self._b + 10 * span, self._hi)
        suffix = f" {self._unit}" if self._unit else ""
        for spin, value in ((self.lo_spin, self._lo), (self.hi_spin, self._hi)):
            with QSignalBlocker(spin):
                spin.resolution = resolution
                spin.setDecimals(decimals)
                spin.setRange(low, high)
                spin.setSingleStep(step)
                spin.setSuffix(suffix)
                spin.setValue(value)
        with QSignalBlocker(self.mode):
            self.mode.set_value("auto" if self._auto else "fixed")
        self.slider.set_values(self._lo, self._hi)
        self.slider.set_muted(self._auto)
        self._sync_state()

    def _sync_state(self) -> None:
        invalid = bool(self._error)
        for spin in (self.lo_spin, self.hi_spin):
            set_style_property(spin, "invalid", invalid)
        set_style_property(self.note, "error", invalid)
        self.note.setText(self._error or self._note_text())

    def _note_text(self) -> str:
        if self._note is not None:
            return self._note
        unit = f" {self._unit}" if self._unit else ""
        return f"Data {format_number(self._a)} – {format_number(self._b)}{unit}"

    # --- settings protocol -------------------------------------------------------------
    def settings_value(self) -> str:
        return json.dumps(
            {"auto": self._auto, "lo": self._lo * self._scale, "hi": self._hi * self._scale}
        )

    def set_settings_value(self, value) -> bool:
        data = json_object(value)
        if data is None:
            return False
        auto = to_bool(data.get("auto"))
        lo, hi = to_float(data.get("lo")), to_float(data.get("hi"))
        if auto is None or lo is None or hi is None or lo >= hi:
            return False
        self._auto = auto
        self.set_range(lo / self._scale, hi / self._scale)
        return True
