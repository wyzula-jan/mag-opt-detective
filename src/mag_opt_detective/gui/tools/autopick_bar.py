"""The options bar of the Auto-pick tool, floating over the top of the map.

A card in the window colour with an accent border, like the pick hint chip: the mode, the
feature, the search window, the smoothing and the prominence, then a status line with Discard
and Accept. The controls wrap onto more rows when the plot is narrow.
"""

from __future__ import annotations

from PySide6.QtCore import QEvent, QObject, QRectF, QSize, Qt
from PySide6.QtGui import QPainter, QPen
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from mag_opt_detective.core.picking import Feature
from mag_opt_detective.gui import icons
from mag_opt_detective.gui.kit import SegmentedControl
from mag_opt_detective.gui.panels.common import Note, UnitField, scaled_font
from mag_opt_detective.gui.theme import current_tokens
from mag_opt_detective.gui.widgets import EnergyEdit, FloatEdit, FlowLayout

SPACING = 12  # between the groups of controls
TRACK, DETECT = "track", "detect"
MODES = (  # value, label, tooltip
    (TRACK, "Track", "Click a line on the map: it is followed field by field both ways"),
    (DETECT, "Detect", "Drag a box on the map: every line inside it is found"),
)
FEATURES = (
    (Feature.MAX, "Max", "Maxima of the map values (peaks)"),
    (Feature.MIN, "Min", "Minima of the map values (dips, e.g. absorption in R(B)/R(0))"),
    (
        Feature.RISING,
        "Rising",
        "Rising inflection points: where the values grow fastest with energy",
    ),
    (
        Feature.FALLING,
        "Falling",
        "Falling inflection points: where the values drop fastest with energy",
    ),
)
SMOOTHING = (
    ("0", "Off", "Search the spectra as they are"),
    ("5", "5", "Savitzky–Golay smoothing over 5 points first"),
    ("9", "9", "Savitzky–Golay smoothing over 9 points first"),
    ("15", "15", "Savitzky–Golay smoothing over 15 points first"),
)
WINDOW_TIP = (
    "How far a line may move from one field to the next (Track: from the predicted energy; "
    "Detect: between neighbouring fields). Empty: 2 % of the energy range."
)
PROMINENCE_TIPS = {
    "value": (
        "Least prominence of a maximum or minimum, in the values of the map shown "
        "(0.01 is 1 % of R(B)/R(0)). Empty: automatic, above the noise."
    ),
    "slope": (
        "Least prominence of the slope at an inflection point, in map values per {unit}. "
        "Empty: automatic, above the noise."
    ),
}


def _label(text: str) -> QLabel:
    label = QLabel(text)
    label.setProperty("kit", "muted")
    label.setFont(scaled_font(label, 0.9))
    return label


def _segmented(options, name: str) -> SegmentedControl:
    control = SegmentedControl(size="xs")
    for value, text, tooltip in options:
        control.add_option(str(value), text, tooltip)
    control.setAccessibleName(name)
    return control


def _group(*widgets: QWidget) -> QWidget:
    """Widgets kept on one row of the bar (a label and its control)."""
    box = QWidget()
    row = QHBoxLayout(box)
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(5)
    for widget in widgets:
        row.addWidget(widget)
    return box


class AutoPickBar(QFrame):
    """Options, status and Accept/Discard of the Auto-pick tool.

    It sits at the top centre of *host* (the plot box), below *avoid* (the error bar) while
    that is shown, and is as wide as its controls in one row or as the host allows.
    """

    MARGIN = 10

    def __init__(self, host: QWidget, avoid: QWidget | None = None):
        super().__init__(host)
        self._host = host
        self._avoid = avoid
        self.setAccessibleName("Auto-pick options")
        self.mode = _segmented(MODES, "Auto-pick mode")
        self.feature = _segmented(FEATURES, "Feature")
        self.window_edit = EnergyEdit(name="search window")
        self.window_edit.setAccessibleName("Search window")
        self.window_edit.setToolTip(WINDOW_TIP)
        self.window_field = UnitField(self.window_edit, "cm⁻¹")
        self.window_field.setToolTip(WINDOW_TIP)
        self.smooth = _segmented(SMOOTHING, "Smoothing")
        self.prominence_edit = FloatEdit(name="prominence")
        self.prominence_edit.setAccessibleName("Prominence")
        self.prominence_field = UnitField(self.prominence_edit, "")
        for edit in (self.window_edit, self.prominence_edit):
            edit.setFixedWidth(edit.fontMetrics().horizontalAdvance("auto 0.0000") + 6)
            edit.setPlaceholderText("auto")
        for field in (self.window_field, self.prominence_field):
            field.setFixedHeight(24)

        self.status = Note("", "info")
        self.status.setAccessibleName("Auto-pick status")
        self.discard_button = QPushButton("Discard")
        self.discard_button.setToolTip("Drop the points found (Esc leaves the tool)")
        icons.set_icon(self.discard_button, "x")
        self.accept_button = QPushButton("Accept")
        self.accept_button.setProperty("kit", "primary")
        self.accept_button.setToolTip("Put the points found into the current curve (one undo step)")
        icons.set_icon(self.accept_button, "check", "accent-fg")
        for button in (self.discard_button, self.accept_button):
            button.setIconSize(QSize(14, 14))
            button.setFixedHeight(24)

        self.controls = QWidget()
        self._flow = FlowLayout(self.controls, spacing=SPACING, row_spacing=5)
        self._flow.addWidget(self.mode)
        self._flow.addWidget(self.feature)
        self.window_label = _label("Window")
        self.smooth_label = _label("Smooth")
        self.prominence_label = _label("Prominence")
        self.window_label.setBuddy(self.window_edit)
        self.prominence_label.setBuddy(self.prominence_edit)
        self._flow.addWidget(_group(self.window_label, self.window_field))
        self._flow.addWidget(_group(self.smooth_label, self.smooth))
        self._flow.addWidget(_group(self.prominence_label, self.prominence_field))

        footer = QHBoxLayout()
        footer.setSpacing(8)
        footer.addWidget(self.status, 1)
        footer.addWidget(self.discard_button, 0, Qt.AlignmentFlag.AlignTop)
        footer.addWidget(self.accept_button, 0, Qt.AlignmentFlag.AlignTop)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 6, 10, 6)
        layout.setSpacing(6)
        layout.addWidget(self.controls)
        layout.addLayout(footer)

        host.installEventFilter(self)
        if avoid is not None:
            avoid.installEventFilter(self)
        self.hide()

    # --- content -----------------------------------------------------------------------
    def set_prominence_unit(self, slope: bool, unit_text: str) -> None:
        """Label the prominence: map values (Max/Min) or values per energy unit (slopes)."""
        self.prominence_field.set_unit(f"/{unit_text}" if slope else "")
        tip = PROMINENCE_TIPS["slope" if slope else "value"].format(unit=unit_text)
        self.prominence_edit.setToolTip(tip)
        self.prominence_field.setToolTip(tip)
        self.prominence_label.setToolTip(tip)
        self.place()

    def set_status(self, text: str, level: str = "info") -> None:
        self.status.set_text(text, level)
        self.status.setToolTip(text)
        self.place()

    def status_text(self) -> str:
        return self.status.text()

    # --- placement ---------------------------------------------------------------------
    def _row_width(self) -> int:
        """Width of the controls in one row (with the margins)."""
        hints = [self._flow.itemAt(i).sizeHint().width() for i in range(self._flow.count())]
        margins = self.layout().contentsMargins()
        spacing = SPACING * (len(hints) - 1)
        return sum(hints) + spacing + margins.left() + margins.right()

    def place(self) -> None:
        if self.isHidden():
            return
        area = self._host.rect()
        width = max(160, min(self._row_width(), area.width() - 2 * self.MARGIN))
        height = max(self.heightForWidth(width), self.minimumSizeHint().height())
        top = self.MARGIN
        avoid = self._avoid
        if avoid is not None and avoid.isVisible() and avoid.parentWidget() is self._host:
            top = avoid.geometry().bottom() + 8
        self.setGeometry((area.width() - width) // 2, top, width, height)
        self.raise_()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self.place()

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        kinds = (QEvent.Type.Resize, QEvent.Type.Move, QEvent.Type.Show, QEvent.Type.Hide)
        if event.type() in kinds:
            self.place()
        return False

    def paintEvent(self, event) -> None:
        tokens = current_tokens()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(tokens["accent"], 1))
        painter.setBrush(tokens["win"])
        painter.drawRoundedRect(QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5), 8, 8)
        painter.end()
