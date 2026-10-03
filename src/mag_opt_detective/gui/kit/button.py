"""The mockup's small button (``.btn.sm``): a 14 px icon, a 6 px gap and the text."""

from __future__ import annotations

from PySide6.QtCore import QRect, QSize, Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QStyle, QStyleOptionToolButton, QStylePainter, QWidget

from mag_opt_detective.gui import icons
from mag_opt_detective.gui.kit._common import TightToolButton
from mag_opt_detective.gui.theme import current_tokens

ICON_GAP = 6  # between the icon and the text (Qt's tool button leaves about 2 px)
ICON_SIZE = 14
QT_ICON_SPACING = 4  # what QToolButton's size hint puts between icon and text
FONT_SCALE = 0.94


class SmallButton(TightToolButton):
    """A compact kit button (``kit="button"``) with an optional icon before its text."""

    def __init__(
        self, text: str, icon: str | None = None, tooltip: str = "", parent: QWidget | None = None
    ):
        super().__init__(parent)
        self.setProperty("kit", "button")
        self.setText(text)
        self.setToolTip(tooltip)
        font = self.font()
        if font.pointSizeF() > 0:
            font.setPointSizeF(font.pointSizeF() * FONT_SCALE)
        self.setFont(font)
        self.setIconSize(QSize(ICON_SIZE, ICON_SIZE))
        if icon is not None:
            icons.set_icon(self, icon)
            self.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        else:
            self.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)

    def _icon_and_text(self) -> bool:
        beside = self.toolButtonStyle() == Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        return beside and bool(self.text()) and not self.icon().isNull()

    def sizeHint(self) -> QSize:
        hint = super().sizeHint()
        if self._icon_and_text():
            hint.setWidth(hint.width() + ICON_GAP - QT_ICON_SPACING)
        return hint

    def minimumSizeHint(self) -> QSize:
        return self.sizeHint()

    def paintEvent(self, event) -> None:
        if not self._icon_and_text():
            super().paintEvent(event)
            return
        option = QStyleOptionToolButton()
        self.initStyleOption(option)
        option.text = ""
        option.icon = QIcon()
        painter = QStylePainter(self)
        painter.drawComplexControl(QStyle.ComplexControl.CC_ToolButton, option)
        tokens = current_tokens()  # the colours of the stylesheet's kit button
        if not self.isEnabled():
            color = tokens["faint"]
        else:
            color = tokens["accent"] if self.isChecked() else tokens["fg"]
        text_width = self.fontMetrics().horizontalAdvance(self.text())
        x = (self.width() - (ICON_SIZE + ICON_GAP + text_width)) // 2
        middle = self.height() // 2
        mode = QIcon.Mode.Normal if self.isEnabled() else QIcon.Mode.Disabled
        state = QIcon.State.On if self.isChecked() else QIcon.State.Off
        box = QRect(x, middle - ICON_SIZE // 2, ICON_SIZE, ICON_SIZE)
        self.icon().paint(painter, box, Qt.AlignmentFlag.AlignCenter, mode, state)
        painter.setPen(color)
        x += ICON_SIZE + ICON_GAP
        text_box = QRect(x, 0, text_width + 1, self.height())
        painter.drawText(text_box, Qt.AlignmentFlag.AlignVCenter, self.text())
        painter.end()
