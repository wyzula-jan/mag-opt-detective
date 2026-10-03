"""The baseline chip: whether the map on screen is baseline-corrected, and over which region.

The chip shows what the map shown has, not what the Processing panel is set to: the region
applied (in the display unit), a "Live" tag while Live applies the panel's region at once, and
the warning dot (as on the Process button) when the panel's region differs from the applied one,
so the next Process changes it. A library map shows the region it was plotted with. Without a
baseline on the map shown the chip is hidden. :func:`baseline_mark` derives this from the
controller; :class:`BaselineChip` draws it and gets narrower (and then empty) when its place is.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from PySide6.QtCore import QRectF, QSize, Qt
from PySide6.QtGui import QFont, QFontMetrics, QPainter, QPen
from PySide6.QtWidgets import QAbstractButton, QSizePolicy

from mag_opt_detective.core.units import Unit, from_cm1
from mag_opt_detective.gui import icons
from mag_opt_detective.gui.display import format_range, process_key, unit_text
from mag_opt_detective.gui.theme import current_tokens

ICON = "sliders-horizontal"  # the Processing panel's icon: the chip opens it
WHAT = "each spectrum is shifted so that this region averages 1"
NUMBER = re.compile(r"[0-9]+(?:\.[0-9]+)?")


@dataclass(frozen=True)
class BaselineMark:
    """The baseline of the map shown, as the chip says it (energies in *unit*)."""

    region: tuple[float, float]  # applied to the map shown
    unit: Unit
    live: bool = False  # Live is on and applies the panel's region to this map at once
    pending: bool = False  # the panel's region differs from the applied one
    library: bool = False  # a library map: it keeps the region it was plotted with
    wanted: tuple[float | None, float | None] | None = None  # the panel's region (None: off)

    def range_text(self) -> str:
        return format_range(*self.region, unit_text(self.unit))

    def text(self) -> str:
        return f"Baseline {self.range_text()}"

    def accessible_name(self) -> str:
        states = [self.text()]
        if self.library:
            states.append("library map")
        if self.live:
            states.append("live")
        if self.pending:
            states.append("changed, process again to apply")
        return ", ".join(states)

    def tooltip(self) -> str:
        if self.library:
            lines = [
                f"The library map shown was plotted with the baseline {self.range_text()} "
                f"({WHAT}).",
                "It keeps it: the region of the Processing panel applies when you plot a map "
                "again or process.",
            ]
        else:
            lines = [f"The map shown is baseline-corrected over {self.range_text()}: {WHAT}."]
            if self.live:
                lines.append("Live: changes to the region apply at once.")
            if self.pending:
                lines.append(f"{self._wanted_text()} Process ({process_key()}) to apply it.")
        lines.append("Click to edit the region in the Processing panel.")
        return "\n".join(lines)

    def _wanted_text(self) -> str:
        wanted = self.wanted
        if wanted is None:
            return "The Processing panel has the baseline off now."
        if None in wanted:
            return "The Processing panel's region is incomplete."
        unit = unit_text(self.unit)
        if wanted[0] >= wanted[1]:
            return "The Processing panel's region is reversed."
        return f"The Processing panel says {format_range(*wanted, unit)} now."


def baseline_mark(controller) -> BaselineMark | None:
    """The baseline of the map *controller* shows, or None when it has none (or no map)."""
    c = controller
    result = c.result
    if result is None or result.baseline_region is None:
        return None
    unit = c.unit
    region = shown(result.baseline_region, unit)
    if c.result_source == "library":
        return BaselineMark(region, unit, library=True)
    live = c.live_baseline() and c.can_apply_baseline()
    wanted = c.processing.baseline
    pending = wanted != result.baseline_region and not (live and live_applies(c))
    if not pending or wanted is None:  # (only a pending mark says what the panel has)
        return BaselineMark(region, unit, live=live, pending=pending)
    return BaselineMark(region, unit, live=live, pending=True, wanted=shown(wanted, unit))


def shown(region, unit: Unit) -> tuple:
    """A region in cm^-1 (an end may be None) in *unit*."""
    return tuple(None if v is None else float(from_cm1(float(v), unit)) for v in region)


def live_applies(controller) -> bool:
    """Live can apply the panel's region to the map shown: it is off, or complete, in order and
    holds data. (A region that is not waits for a fix and counts as changed.)"""
    c = controller
    region = c.processing.baseline
    if region is None:
        return True
    if None in region:
        return False
    try:
        c.check_energy_range("baseline region", region, c.result.ratio.energy, "processing")
    except ValueError:
        return False
    return True


def widest(text: str) -> str:
    """*text* with each number as wide as one digit more can make it (for reserving room)."""
    return NUMBER.sub(lambda m: "8" + re.sub("[0-9]", "8", m.group()), text)


class BaselineChip(QAbstractButton):
    """A chip with the Processing icon, "Baseline 500 – 880 cm⁻¹", a "Live" tag and the
    warning dot (see :class:`BaselineMark`); a click is for opening the Processing panel.

    Its width may be anything down to zero: given less than it asks for it drops the word
    "Baseline", then shows the icon (and the dot) alone, then nothing. While held
    (:meth:`set_held`, e.g. during a drag of the region) it asks for no less width than it
    had, and room for one more digit at each end, so the widgets beside it stay where they are
    while the numbers change.
    """

    HEIGHT = 22  # the chip; the widget adds MARGIN above and below for the dot
    MARGIN = 3
    PAD = 7  # inside the chip, left and right
    ICON = 14
    GAP = 5
    DOT = 9.0

    def __init__(self, height: int = HEIGHT, parent=None):
        super().__init__(parent)
        self._height = height
        self._mark: BaselineMark | None = None
        self._holding = False
        self._held = 0  # the width asked for at least while held
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self.hide()

    def mark(self) -> BaselineMark | None:
        return self._mark

    def set_mark(self, mark: BaselineMark | None) -> None:
        """Show *mark*, or hide the chip (None)."""
        if mark == self._mark:
            return
        old, self._mark = self._mark, mark
        if mark is not None:
            self.setText(mark.text())
            self.setToolTip(mark.tooltip())
            self.setAccessibleName(mark.accessible_name())
            self.setAccessibleDescription(mark.tooltip())
        resized = old is None or mark is None or self.widths(old) != self.widths(mark)
        if self._holding:
            resized = self._hold(self._held) or resized
        self.setVisible(mark is not None)
        if resized:
            self._relayout()
        self.update()

    def set_held(self, held: bool) -> None:
        """Keep (True) or release (False) the width the chip asks for. Held, it starts with
        room for one more digit at each end of the region shown."""
        if held == self._holding:
            return
        self._holding = held
        if held:
            mark, reserve = self._mark, 0
            if mark is not None and mark.region:
                texts = [("Baseline ", "muted"), (widest(mark.range_text()), "fg")]
                reserve = self._width(texts, mark.live) + self.MARGIN
            changed = self._hold(reserve)
        else:
            changed, self._held = self._held != 0, 0
        if changed:
            self._relayout()

    def _hold(self, width: int) -> bool:
        """Ask for at least *width* and the width the mark shown needs; True if that is more
        than before."""
        width = max(width, self.sizeHint().width())
        changed, self._held = width != self._held, width
        return changed

    def _relayout(self) -> None:
        """Take the new size at once: painted at the old one, the chip would drop to a
        narrower form for a frame."""
        self.updateGeometry()
        parent = self.parentWidget()
        if self.isVisible() and parent is not None and parent.layout() is not None:
            parent.layout().activate()

    # --- drawing -------------------------------------------------------------------------
    def _fonts(self) -> tuple[QFont, QFont]:
        label = QFont(self.font())
        label.setWeight(QFont.Weight.Medium)
        tag = QFont(self.font())
        if tag.pointSizeF() > 0:
            tag.setPointSizeF(tag.pointSizeF() * 0.82)
        tag.setWeight(QFont.Weight.DemiBold)
        return label, tag

    def _tag_width(self) -> int:
        return QFontMetrics(self._fonts()[1]).horizontalAdvance("Live") + 10

    @staticmethod
    def _texts(mark: BaselineMark, form: int) -> list[tuple[str, str]]:
        """The texts of *form* (0 all, 1 without "Baseline", 2 the icon alone) and their
        colour tokens."""
        texts = [("Baseline ", "muted"), (mark.range_text(), "fg")]
        return texts[form:] if form < 2 else []

    def _width(self, texts: list[tuple[str, str]], live: bool) -> int:
        width = self.PAD + self.ICON + self.PAD
        if texts:
            label = QFontMetrics(self._fonts()[0])
            width += self.GAP + sum(label.horizontalAdvance(text) for text, _ in texts)
            if live:
                width += self.GAP + self._tag_width()
        return width

    def widths(self, mark: BaselineMark) -> tuple[int, ...]:
        """The chip's width in each form of :meth:`form` (without the margin for the dot)."""
        return tuple(self._width(self._texts(mark, form), mark.live) for form in range(3))

    def sizeHint(self) -> QSize:
        width = self.widths(self._mark)[0] + self.MARGIN if self._mark is not None else 0
        return QSize(max(width, self._held), self._height + 2 * self.MARGIN)

    def minimumSizeHint(self) -> QSize:
        return QSize(0, self._height + 2 * self.MARGIN)

    def form(self) -> int | None:
        """What fits the width the chip has: 0 all, 1 without "Baseline", 2 the icon, None
        nothing (or no mark)."""
        if self._mark is None:
            return None
        room = self.width() - self.MARGIN
        for form, width in enumerate(self.widths(self._mark)):
            if width <= room:
                return form
        return None

    def paintEvent(self, event) -> None:
        mark, form = self._mark, self.form()
        if mark is None or form is None:
            return
        tokens = current_tokens()
        label_font, tag_font = self._fonts()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        frame = QRectF(0.5, self.MARGIN + 0.5, self.widths(mark)[form] - 1, self._height - 1)
        painter.setPen(QPen(tokens["accent" if self.hasFocus() else "line-strong"], 1))
        painter.setBrush(tokens["hover" if self.underMouse() or self.isDown() else "surface"])
        painter.drawRoundedRect(frame, 6, 6)
        middle = frame.center().y()
        x = self.PAD
        icon = icons.icon(ICON, "muted")
        icon.paint(painter, x, round(middle - self.ICON / 2), self.ICON, self.ICON)
        x += self.ICON + self.GAP
        painter.setFont(label_font)
        metrics = QFontMetrics(label_font)
        flags = Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft
        for text, color in self._texts(mark, form):
            painter.setPen(tokens[color])
            advance = metrics.horizontalAdvance(text)
            painter.drawText(QRectF(x, frame.top(), advance + 2, frame.height()), flags, text)
            x += advance
        if mark.live and form < 2:
            tag = QRectF(x + self.GAP, middle - 8, self._tag_width(), 16)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(tokens["accent-soft"])
            painter.drawRoundedRect(tag, 4, 4)
            painter.setFont(tag_font)
            painter.setPen(tokens["accent"])
            painter.drawText(tag, Qt.AlignmentFlag.AlignCenter, "Live")
        if mark.pending:  # the Process button's dot, over the chip's top-right corner
            dot = QRectF(frame.right() - self.DOT + 3, frame.top() - 3, self.DOT, self.DOT)
            painter.setPen(QPen(tokens["win"], 1.5))
            painter.setBrush(tokens["warn"])
            painter.drawEllipse(dot)
        painter.end()

    def enterEvent(self, event) -> None:
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:
        self.update()
        super().leaveEvent(event)
