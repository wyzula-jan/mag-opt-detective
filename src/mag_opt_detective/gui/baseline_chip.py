"""The baseline chip: whether the map on screen is baseline-corrected, and over which region.

The chip shows what the map shown has, not what the Processing panel is set to: the region
applied (in the display unit, with the digits of the panel's fields), a "Live" tag while Live
applies the panel's region at once, and the warning dot (as on the Process button) when the
panel's region differs from the applied one. A processed map without a baseline shows "No
baseline" with the dot while the panel has a region that the next Process would apply; else it
shows nothing. A library map shows the region it was plotted with. :func:`baseline_mark`
derives this from the controller; :class:`BaselineChip` draws it and gets narrower (and then
empty) when its place is.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from PySide6.QtCore import QRectF, QSize, Qt
from PySide6.QtGui import QFont, QFontMetrics, QPainter, QPen
from PySide6.QtWidgets import QAbstractButton, QSizePolicy

from mag_opt_detective.core.units import Unit, from_cm1
from mag_opt_detective.gui import icons
from mag_opt_detective.gui.display import process_key, unit_text
from mag_opt_detective.gui.theme import current_tokens

PROCESSING_ICON = "sliders-horizontal"  # the Processing panel's icon: the chip opens it
WHAT = "each spectrum is shifted so that this region averages 1"
DIGITS = 6  # significant digits, as in the Processing panel's fields
NUMBER = re.compile(r"[0-9]+(?:\.[0-9]+)?")

Region = tuple[float | None, float | None]


def ends_text(region: Region, digits: int = DIGITS) -> tuple[str, str]:
    """The ends of *region* with *digits* significant digits ("open" for None)."""
    return tuple("open" if v is None else f"{v:.{digits}g}" for v in region)


def range_text(region: Region, unit: Unit, digits: int = DIGITS) -> str:
    """``500 – 880 cm⁻¹``: a region in *unit* with *digits* significant digits."""
    lo, hi = ends_text(region, digits)
    return f"{lo} – {hi} {unit_text(unit)}"


@dataclass(frozen=True)
class BaselineMark:
    """The baseline of the map shown, as the chip says it (energies in *unit*)."""

    region: tuple[float, float] | None  # applied to the map shown; None: no baseline
    unit: Unit
    live: bool = False  # Live is on and applies the panel's region to this map at once
    pending: bool = False  # the panel's region differs from the applied one
    library: bool = False  # a library map: it keeps the region it was plotted with
    wanted: Region | None = None  # the panel's region (None: off), when pending
    problem: str = ""  # why Process cannot apply the panel's region ("" when it can)
    digits: int = DIGITS  # enough to tell the applied region from the panel's

    def range_text(self) -> str:
        return range_text(self.region, self.unit, self.digits) if self.region else ""

    def text(self) -> str:
        return f"Baseline {self.range_text()}" if self.region else "No baseline"

    def accessible_name(self) -> str:
        states = [self.text()]
        if self.library:
            states.append("library map")
        if self.live:
            states.append("live")
        if self.pending:
            states.append("changed, process again to apply" if not self.problem else "changed")
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
            if self.region:
                lines = [f"The map shown is baseline-corrected over {self.range_text()}: {WHAT}."]
            else:
                lines = ["The map shown has no baseline correction."]
            if self.live:
                lines.append("Live: changes to the region apply at once.")
            if self.pending:
                lines.append(self._pending_text())
        lines.append("Click to edit the region in the Processing panel.")
        return "\n".join(lines)

    def _pending_text(self) -> str:
        key = process_key()
        if self.problem:
            return f"The Processing panel's region cannot be applied: {self.problem}."
        if self.wanted is None:
            return f"The Processing panel has the baseline off. Process ({key}) to remove it."
        wanted = range_text(self.wanted, self.unit, self.digits)
        return f"The Processing panel has {wanted}. Process ({key}) to apply it."


def baseline_mark(controller) -> BaselineMark | None:
    """The baseline of the map *controller* shows, or None when the chip shows nothing."""
    c = controller
    result = c.result
    if result is None:
        return None
    unit = c.unit
    applied = shown(result.baseline_region, unit) if result.baseline_region else None
    if c.result_source == "library":
        return BaselineMark(applied, unit, library=True) if applied else None
    live = c.live_baseline() and c.can_apply_baseline()
    wanted_cm1 = c.processing.baseline
    pending = wanted_cm1 != result.baseline_region and not c.baseline_is_live()
    if not pending:
        return BaselineMark(applied, unit, live=live) if applied else None
    wanted = None if wanted_cm1 is None else shown(wanted_cm1, unit)
    problem = region_problem(c, wanted_cm1)
    if applied is None and (wanted is None or problem):  # nothing the next Process would apply
        return None
    return BaselineMark(
        applied,
        unit,
        live=live,
        pending=True,
        wanted=wanted,
        problem=problem,
        digits=distinct_digits(applied, wanted),
    )


def shown(region, unit: Unit) -> tuple:
    """A region in cm^-1 (an end may be None) in *unit*."""
    return tuple(None if v is None else float(from_cm1(float(v), unit)) for v in region)


def region_problem(controller, region) -> str:
    """Why the next Process cannot apply the baseline *region* (cm^-1; None: off) to the
    data of the map shown, or "" when it can (an approximation: Process checks the data it
    loads, cut to the energy window)."""
    c = controller
    if region is None:
        return ""
    unit = c.unit
    if None in region:
        return "it is incomplete; enter both limits"
    text = range_text(shown(region, unit), unit)
    if region[0] >= region[1]:
        return f"{text} is reversed; the first value must be below the second"
    energy = c.result.ratio.energy
    try:
        c.check_energy_range("baseline region", region, energy, "processing")
    except ValueError:
        span = range_text(shown((energy.min(), energy.max()), unit), unit)
        return f"{text} holds no data (the map spans {span})"
    return ""


def distinct_digits(applied, wanted) -> int:
    """The fewest significant digits (at least :data:`DIGITS`) that write the regions *applied*
    and *wanted* apart when they differ."""
    if applied is None or wanted is None or None in wanted or applied == wanted:
        return DIGITS
    for digits in range(DIGITS, 17):
        if ends_text(applied, digits) != ends_text(wanted, digits):
            return digits
    return 17  # (enough for any two floats)


def widest(text: str) -> str:
    """*text* with each number as wide as one digit more can make it (for reserving room)."""
    return NUMBER.sub(lambda m: "8" + re.sub("[0-9]", "8", m.group()), text)


class BaselineChip(QAbstractButton):
    """A chip with the Processing icon, "Baseline 500 – 880 cm⁻¹" (or "No baseline"), a "Live"
    tag and the warning dot (see :class:`BaselineMark`); a click is for opening the Processing
    panel.

    Its width may be anything down to zero: given less than it asks for it drops the word
    "Baseline", then shows the icon (and the dot) alone, then nothing. While held
    (:meth:`set_held`, e.g. during a drag of the region) it asks for no less width than it
    had, and room for one more digit at each end, so the widgets beside it stay where they are
    while the numbers change.
    """

    HEIGHT = 22  # the chip; the widget adds MARGIN above and below for the dot
    MARGIN = 3
    PAD = 7  # inside the chip, left and right
    ICON_SIZE = 14
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
        if form == 2:
            return []
        if not mark.region:
            return [("No baseline", "muted")]
        return [("Baseline ", "muted"), (mark.range_text(), "fg")][form:]

    def _width(self, texts: list[tuple[str, str]], live: bool) -> int:
        width = self.PAD + self.ICON_SIZE + self.PAD
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
        size = self.ICON_SIZE
        icons.icon(PROCESSING_ICON, "muted").paint(painter, x, round(middle - size / 2), size, size)
        x += size + self.GAP
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
