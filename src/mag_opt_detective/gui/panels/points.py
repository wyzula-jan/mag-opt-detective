"""Points panel: the curves of picked points, their table, the Pick tool and Undo/Redo.

Picked energies are kept in cm^-1 by the controller; the table, the markers and the files use
the display unit. The Pick tool (P) works on the map, where a click records the current
curve's point at the nearest field and Alt-click removes the nearest point, and on the stacked
plot, where a click near a trace records at that trace's field. Every point edit is a step of
``controller.points_undo``, undone with Edit > Undo (Ctrl+Z) and redone with Ctrl+Shift+Z.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PySide6.QtCore import QRect, QRectF, QRegularExpression, QSignalBlocker, QSize, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QFontDatabase,
    QFontMetrics,
    QIcon,
    QKeySequence,
    QPainter,
    QPen,
    QRegularExpressionValidator,
)
from PySide6.QtWidgets import (
    QAbstractButton,
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QStackedWidget,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionButton,
    QStylePainter,
    QTableView,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from mag_opt_detective.core.units import from_cm1
from mag_opt_detective.gui import icons
from mag_opt_detective.gui.controller import CURVE_NAME, user_action
from mag_opt_detective.gui.kit import SegmentedControl, Switch
from mag_opt_detective.gui.kit._common import set_style_property
from mag_opt_detective.gui.plot_panel import PlotClick
from mag_opt_detective.gui.points_model import CurvePointsModel
from mag_opt_detective.gui.points_view import (
    MAP_SIZE,
    SHOW_ALL,
    SHOW_MODES,
    STACKED_SIZE,
    PickHint,
    curve_color,
    draw_markers,
    map_markers,
    stacked_field,
    stacked_markers,
)
from mag_opt_detective.gui.theme import current_tokens
from mag_opt_detective.gui.widgets import FlowLayout, open_file, save_file

PICK = "pick"
ALT = Qt.KeyboardModifier.AltModifier
ALT_TEXT = "⌥" if sys.platform == "darwin" else "Alt"
REMOVE_RADIUS = 12.0  # px: Alt-click removes the current curve's point this close to it
EMPTY = "No points yet. Turn on picking and click the map."
NO_TABLE = "No point table yet. Process a sweep (or import points), then pick."


def _hint(text: str) -> QLabel:
    label = QLabel(text)
    label.setProperty("kit", "muted")
    label.setWordWrap(True)
    return label


def _title(text: str) -> QLabel:
    """A small upper-case block title."""
    label = QLabel(text.upper())
    label.setProperty("kit", "muted")
    font = QFont(label.font())
    font.setBold(True)
    if font.pointSizeF() > 0:
        font.setPointSizeF(font.pointSizeF() * 0.85)
    font.setLetterSpacing(QFont.SpacingType.PercentageSpacing, 106)
    label.setFont(font)
    return label


def _keys(first: str, standard: QKeySequence.StandardKey) -> list[QKeySequence]:
    """*first* (shown in menus), then the platform's other keys for *standard*."""
    keys = [QKeySequence(first)]
    keys += [key for key in QKeySequence.keyBindings(standard) if key not in keys]
    return keys


# ---------------------------------------------------------------------- widgets
class PickButton(QPushButton):
    """A full-width toggle showing its shortcut key; filled with the accent colour while on."""

    ICON, GAP = 16, 6

    def __init__(self, text: str, key: str, parent=None):
        super().__init__(text, parent)
        self._key = key
        self.setCheckable(True)
        self.setMinimumHeight(30)
        icons.set_icon(self, "crosshair", None, on_color="accent-fg")
        self.toggled.connect(lambda on: set_style_property(self, "kit", "primary" if on else None))

    def _key_size(self) -> QSize:
        metrics = self.fontMetrics()
        return QSize(metrics.horizontalAdvance(self._key) + 10, metrics.height() + 2)

    def _content_width(self) -> int:
        """Icon, text and key cap side by side."""
        text = self.fontMetrics().horizontalAdvance(self.text())
        return self.ICON + self.GAP + text + 2 * self.GAP + self._key_size().width()

    def sizeHint(self) -> QSize:
        hint = super().sizeHint()
        return QSize(max(hint.width(), self._content_width() + 24), max(hint.height(), 30))

    def paintEvent(self, event) -> None:
        option = QStyleOptionButton()
        self.initStyleOption(option)
        option.text = ""
        option.icon = QIcon()
        painter = QStylePainter(self)
        painter.drawControl(QStyle.ControlElement.CE_PushButton, option)
        tokens = current_tokens()
        on = self.isChecked()
        color = tokens["faint"] if not self.isEnabled() else tokens["accent-fg" if on else "fg"]
        x = (self.width() - self._content_width()) // 2
        middle = self.height() // 2
        mode = QIcon.Mode.Normal if self.isEnabled() else QIcon.Mode.Disabled
        state = QIcon.State.On if on else QIcon.State.Off
        icon_rect = QRect(x, middle - self.ICON // 2, self.ICON, self.ICON)
        self.icon().paint(painter, icon_rect, Qt.AlignmentFlag.AlignCenter, mode, state)
        x += self.ICON + self.GAP
        text_width = self.fontMetrics().horizontalAdvance(self.text())
        painter.setPen(color)
        flags = Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft
        painter.drawText(QRect(x, 0, text_width + 2, self.height()), flags, self.text())
        key = self._key_size()
        left = x + text_width + 2 * self.GAP
        cap = QRectF(left, middle - key.height() / 2, key.width(), key.height())
        border = QColor(color)
        border.setAlphaF(0.45)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(border, 1))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(cap.adjusted(0.5, 0.5, -0.5, -0.5), 4, 4)
        painter.setPen(color)
        painter.drawText(cap, Qt.AlignmentFlag.AlignCenter, self._key)


class CurveChip(QAbstractButton):
    """A pill with a colour dot (or an icon), a name and a count; outlined when current."""

    HEIGHT = 26

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.color: QColor | None = None
        self.icon_name: str | None = None
        self.count = ""
        self.current = False

    def set_curve(self, name: str, count: int, color: QColor, current: bool) -> None:
        self.setText(name)
        self.count, self.color, self.current = str(count), QColor(color), current
        state = ", current curve" if current else ""
        self.setAccessibleName(f"{name}, {count} point{'s' * (count != 1)}{state}")
        self.setToolTip(f"{name}: {count} point{'s' * (count != 1)}")
        self.updateGeometry()
        self.update()

    def set_label(self, text: str, icon: str) -> None:
        self.setText(text)
        self.setAccessibleName(text)
        self.icon_name = icon
        self.updateGeometry()

    def _fonts(self) -> tuple[QFont, QFont]:
        name = QFont(self.font())
        name.setWeight(QFont.Weight.DemiBold if self.current else QFont.Weight.Medium)
        small = QFont(self.font())
        if small.pointSizeF() > 0:
            small.setPointSizeF(small.pointSizeF() * 0.9)
        return name, small

    def sizeHint(self) -> QSize:
        name, small = self._fonts()
        width = 10 + 16 + QFontMetrics(name).horizontalAdvance(self.text()) + 10
        if self.count:
            width += 6 + QFontMetrics(small).horizontalAdvance(self.count)
        return QSize(width, self.HEIGHT)

    def minimumSizeHint(self) -> QSize:
        return self.sizeHint()

    def enterEvent(self, event) -> None:
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:
        self.update()
        super().leaveEvent(event)

    def paintEvent(self, event) -> None:
        tokens = current_tokens()
        name_font, small_font = self._fonts()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        if not self.isEnabled():
            painter.setOpacity(0.45)
        width = 2.0 if self.current else 1.0
        rect = QRectF(self.rect()).adjusted(width / 2, width / 2, -width / 2, -width / 2)
        accent = self.current or self.hasFocus()
        painter.setPen(QPen(tokens["accent" if accent else "line-strong"], width))
        hover = self.underMouse() and self.isEnabled()
        painter.setBrush(tokens["hover" if hover else "surface"])
        painter.drawRoundedRect(rect, rect.height() / 2, rect.height() / 2)
        x, middle = 10, self.height() / 2
        if self.color is not None:
            painter.setPen(QPen(QColor(0, 0, 0, 115), 1))
            painter.setBrush(self.color)
            painter.drawEllipse(QRectF(x, middle - 5, 10, 10))
        elif self.icon_name:
            icon = icons.icon(self.icon_name, "muted")
            icon.paint(painter, QRect(x - 1, int(middle) - 7, 14, 14))
        x += 16
        painter.setFont(name_font)
        painter.setPen(tokens["fg"])
        text_width = QFontMetrics(name_font).horizontalAdvance(self.text())
        flags = Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft
        painter.drawText(QRectF(x, 0, text_width + 2, self.height()), flags, self.text())
        if self.count:
            painter.setFont(small_font)
            painter.setPen(tokens["muted"])
            painter.drawText(QRectF(x + text_width + 6, 0, 40, self.height()), flags, self.count)
        painter.end()


class CurveChips(QWidget):
    """The curves as chips, then a "New" chip; a click selects a curve or adds one."""

    curveClicked = Signal(str)
    newClicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._flow = FlowLayout(self, spacing=6, row_spacing=6)
        self._chips: list[CurveChip] = []
        self.new_chip = CurveChip()
        self.new_chip.set_label("New", "plus")
        self.new_chip.setToolTip("Start a new curve")
        self.new_chip.clicked.connect(self.newClicked)
        self._flow.addWidget(self.new_chip)

    def chips(self) -> list[CurveChip]:
        """The chips of the curves shown (without "New")."""
        return [chip for chip in self._chips if not chip.isHidden()]

    def set_curves(self, curves: list[tuple[str, int, QColor]], current: str) -> None:
        """Show ``(name, count, colour)`` per curve; *current* is outlined."""
        while len(self._chips) < len(curves):
            chip = CurveChip(self)
            chip.clicked.connect(lambda _checked=False, c=chip: self.curveClicked.emit(c.text()))
            self._flow.removeWidget(self.new_chip)
            self._flow.addWidget(chip)
            self._flow.addWidget(self.new_chip)
            self._chips.append(chip)
        for i, chip in enumerate(self._chips):
            if i < len(curves):
                name, count, color = curves[i]
                chip.set_curve(name, count, color, name == current)
            chip.setVisible(i < len(curves))
        self._flow.invalidate()


class _RemoveDelegate(QStyledItemDelegate):
    """The remove button of a row: an x, red under the mouse."""

    def paint(self, painter, option, index) -> None:
        super().paint(painter, option, index)
        hovered = bool(option.state & QStyle.StateFlag.State_MouseOver)
        center = option.rect.center()
        icon = icons.icon("x", "err" if hovered else "faint")
        icon.paint(painter, QRect(center.x() - 6, center.y() - 6, 13, 13))


class PointsView(QTableView):
    """The points of the current curve; the x of a row, or Delete, removes points."""

    removeRequested = Signal(list)  # fields (T) of the points to remove

    def __init__(self, model: CurvePointsModel, parent=None):
        super().__init__(parent)
        self.setModel(model)
        self.setItemDelegateForColumn(model.REMOVE, _RemoveDelegate(self))
        self.setShowGrid(False)
        self.setAlternatingRowColors(True)
        self.setMouseTracking(True)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        font = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
        if self.font().pointSizeF() > 0:
            font.setPointSizeF(self.font().pointSizeF() * 0.95)
        self.setFont(font)
        self.verticalHeader().hide()
        self.verticalHeader().setDefaultSectionSize(QFontMetrics(font).height() + 6)
        header = self.horizontalHeader()
        header.setHighlightSections(False)
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(model.REMOVE, QHeaderView.ResizeMode.Fixed)
        header.resizeSection(model.REMOVE, 30)
        self.clicked.connect(self._on_click)

    def _on_click(self, index) -> None:
        if index.column() == CurvePointsModel.REMOVE:
            self.removeRequested.emit([self.model().field_at(index.row())])

    def keyPressEvent(self, event) -> None:
        if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            rows = sorted({index.row() for index in self.selectionModel().selectedRows()})
            if rows:
                self.removeRequested.emit([self.model().field_at(row) for row in rows])
                return
        super().keyPressEvent(event)


class SwitchRow(QWidget):
    """A title and a hint (both wrapped) left of a :class:`Switch`; a click on them toggles."""

    def __init__(self, title: str, hint: str, parent=None):
        super().__init__(parent)
        self.switch = Switch()
        self.switch.setAccessibleName(title)
        label = QLabel(title)
        label.setWordWrap(True)
        font = QFont(label.font())
        font.setBold(True)
        label.setFont(font)
        text = QVBoxLayout()
        text.setSpacing(1)
        text.addWidget(label)
        text.addWidget(_hint(hint))
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(10)
        row.addLayout(text, stretch=1)
        row.addWidget(self.switch, 0, Qt.AlignmentFlag.AlignTop)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self.switch.isEnabled():
            self.switch.click()
        super().mousePressEvent(event)


class PointsPanel(QWidget):
    """Pick toggle, curve chips and name, the current curve's points, markers and files."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.pick_button = PickButton("Pick on the map", "P")
        self.pick_button.setToolTip("Pick points: a click records, Alt-click removes (P)")
        self.pick_button.setAccessibleName("Pick on the map")

        self.chips = CurveChips()
        self.column_name = QLineEdit()
        self.column_name.setValidator(
            QRegularExpressionValidator(QRegularExpression(CURVE_NAME.pattern), self)
        )
        self.column_name.setMaxLength(32)
        self.column_name.setAccessibleName("Curve name")
        self.column_name.setToolTip("Name of the current curve; type to rename it")
        self.delete_button = QToolButton()
        self.delete_button.setProperty("kit", "tool")
        self.delete_button.setToolTip("Delete curve")
        self.delete_button.setAccessibleName("Delete curve")
        icons.set_icon(self.delete_button, "trash-2", "muted")
        self.name_note = QLabel()
        self.name_note.setProperty("kit", "note")
        self.name_note.setProperty("error", True)
        self.name_note.setWordWrap(True)
        self.name_note.hide()

        self.count_label = QLabel()
        self.count_label.setProperty("kit", "muted")
        self.model = CurvePointsModel(self)
        self.table = PointsView(self.model)
        self.table.setMinimumHeight(130)
        self.empty_label = _hint(NO_TABLE)
        self.empty_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.table_stack = QStackedWidget()
        self.table_stack.addWidget(self.table)
        self.table_stack.addWidget(self.empty_label)
        self.table_stack.setCurrentWidget(self.empty_label)

        self.markers = SegmentedControl(size="sm", expand=True)
        for value, text, tooltip in SHOW_MODES:
            self.markers.add_option(value, text, tooltip)
        self.markers.set_value(SHOW_ALL)
        self.markers.setAccessibleName("Markers on the plots")

        self.new_table_row = SwitchRow(
            "Start a new table on the next Process", "Off keeps the picked points across runs."
        )
        self.new_table = self.new_table_row.switch
        self.import_button = QPushButton("Import…")
        self.import_button.setToolTip("Read a point table (replaces the curves; undo restores)")
        icons.set_icon(self.import_button, "upload")
        self.export_button = QPushButton("Export…")
        self.export_button.setToolTip("Save the table: one column per curve, one row per field")
        icons.set_icon(self.export_button, "download")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 16)
        layout.setSpacing(16)
        layout.addWidget(self.pick_button)

        curves = QVBoxLayout()
        curves.setSpacing(7)
        curves.addWidget(_title("Curves"))
        curves.addWidget(self.chips)
        name_row = QHBoxLayout()
        name_row.setSpacing(6)
        name_row.addWidget(self.column_name, stretch=1)
        name_row.addWidget(self.delete_button)
        curves.addLayout(name_row)
        curves.addWidget(self.name_note)
        layout.addLayout(curves)

        points = QVBoxLayout()
        points.setSpacing(7)
        head = QHBoxLayout()
        head.addWidget(_title("Points"))
        head.addWidget(self.count_label, stretch=1)
        points.addLayout(head)
        points.addWidget(self.table_stack, stretch=1)
        layout.addLayout(points, stretch=1)

        shown = QVBoxLayout()
        shown.setSpacing(7)
        shown.addWidget(_title("Markers on the plots"))
        shown.addWidget(self.markers)
        layout.addLayout(shown)

        layout.addWidget(self.new_table_row)

        files = QHBoxLayout()
        files.setSpacing(8)
        files.addWidget(self.import_button)
        files.addWidget(self.export_button)
        files.addStretch(1)
        layout.addLayout(files)

    def show_name_problem(self, problem: str | None) -> None:
        self.name_note.setText(problem[:1].upper() + problem[1:] + "." if problem else "")
        self.name_note.setVisible(bool(problem))
        set_style_property(self.column_name, "invalid", bool(problem))


class CurveSetting:
    """Settings protocol for the current curve's name (used for the first table)."""

    def __init__(self, controller):
        self.controller = controller

    def settings_value(self) -> str:
        return self.controller.curve

    def set_settings_value(self, value) -> bool:
        c = self.controller
        if not isinstance(value, str) or not CURVE_NAME.fullmatch(value.strip()):
            return False
        if c.points is None or value.strip() in c.points.names:
            c.set_curve(value)
        return True


# ---------------------------------------------------------------------- picking
def nearest_point(window, b: float, energy: float) -> float | None:
    """Field of the current curve's point an Alt-click on the map at (*b*, *energy*) removes.

    That is the point nearest to the click within :data:`REMOVE_RADIUS` pixels, else the one
    in the clicked field row (None if there is neither).
    """
    c = window.controller
    table = c.points
    if table is None or c.curve not in table.names:
        return None
    fields, energies = table.points(c.curve)
    if fields.size == 0:
        return None
    pixel_w, pixel_h = (abs(v) for v in window.plots.map.plot.vb.viewPixelSize())
    if pixel_w > 0 and pixel_h > 0:
        distance = np.hypot((fields - b) / pixel_w, (from_cm1(energies, c.unit) - energy) / pixel_h)
        k = int(distance.argmin())
        if distance[k] <= REMOVE_RADIUS:
            return float(fields[k])
    row_field = table.field[table.nearest_row(b)]
    return float(row_field) if np.any(fields == row_field) else None


def trace_field(window, energy: float, y: float) -> float | None:
    """Field of the stacked trace a click at (*energy*, *y*) is on or near, or None.

    The nearest trace counts if the click is within one offset (or a few pixels) of it.
    """
    c, stacked = window.controller, window.plots.stacked
    j, field = stacked.trace_at(energy, y), stacked_field(c)
    if j is None or field is None or j >= field.size:
        return None
    trace_y = stacked.trace_y(j, energy)
    pixel_h = abs(stacked.plot.vb.viewPixelSize()[1])
    reach = max(abs(c.view.stacked_offset), REMOVE_RADIUS * pixel_h)
    if trace_y is None or abs(trace_y - y) > reach:
        return None
    return float(field[j])


@user_action("Pick point")
def on_pick(window, click: PlotClick) -> None:
    """Record the clicked point of the current curve; Alt-click removes one."""
    c = window.controller
    remove = bool(click.modifiers & ALT)
    if click.view == "stacked":
        b = trace_field(window, click.x, click.y)
        if b is not None and remove:
            c.remove_point(b)
        elif b is not None:
            c.record_point(b, click.x)
    elif remove:
        b = nearest_point(window, click.x, click.y)
        if b is not None:
            c.remove_point(b)
    else:
        c.record_point(click.x, click.y)


# ---------------------------------------------------------------------- actions
@user_action("New curve")
def add_curve(window) -> None:
    window.controller.add_curve()


@user_action("Rename curve")
def rename_curve(window, name: str, session: int) -> None:
    window.controller.rename_curve(name, merge=("rename", session))


@user_action("Delete curve")
def drop_curve(window) -> None:
    window.controller.drop_curve()


@user_action("Remove points")
def remove_points(window, fields: list[float]) -> None:
    c = window.controller
    many = f"Remove {len(fields)} points from {c.curve}"
    with c.point_edit(many if len(fields) > 1 else f"Remove point from {c.curve}"):
        for b in fields:
            c.remove_point(b)


@user_action("Import points")
def load_points(window) -> None:
    path = open_file(window, "Import points")
    if path:
        window.controller.load_points(path)


@user_action("Export points")
def export_points(window) -> None:
    if window.controller.points is None:
        raise ValueError("no points to export")
    path = save_file(window, "Export points")
    if not path:
        return
    out = Path(path) if Path(path).suffix else Path(path).with_suffix(".csv")
    window.controller.save_points(out)


def install_undo(window) -> None:
    """Undo and Redo of the point edits in the Edit menu (Ctrl+Z, Ctrl+Shift+Z).

    Text fields keep these keys for their own undo while they have the focus.
    """
    stack = window.controller.points_undo
    undo = stack.createUndoAction(window, "Undo")
    redo = stack.createRedoAction(window, "Redo")
    undo.setShortcuts(_keys("Ctrl+Z", QKeySequence.StandardKey.Undo))
    redo.setShortcuts(_keys("Ctrl+Shift+Z", QKeySequence.StandardKey.Redo))
    icons.set_icon(undo, "undo-2")
    icons.set_icon(redo, "redo-2")
    for name, action in (("undo", undo), ("redo", redo)):
        window.addAction(action)
        window.edit_menu.addAction(action)
        window.commands[name] = action
    window.points_undo = stack


# ---------------------------------------------------------------------- install
def install(window) -> None:
    c, tools = window.controller, window.tools
    panel = PointsPanel()
    window.add_panel(
        "points", "Points", "chart-scatter", "Picked points", panel,
        "Transition energies picked on the map, one per field and curve.",
    )  # fmt: skip
    tools.register(
        PICK,
        "crosshair",
        "Pick points: click records, Alt-click removes",
        "P",
        views=("map", "stacked"),
        on_click=lambda click: on_pick(window, click),
    )
    install_undo(window)
    hint = PickHint(window.plot_area.plot_box, window.infobar)

    # the pick toggle <-> the tool (on the Reference tab it shows the map first)
    def on_pick_button(checked: bool) -> None:
        if checked:
            if tools.view() not in tools.tool(PICK).views:
                window.plot_area.set_current_view("map")
            tools.set_active(PICK)
        elif tools.active() == PICK:
            tools.set_active(tools.default())
        sync_tool()

    def sync_tool(*_args) -> None:
        picking = tools.active() == PICK
        panel.pick_button.setChecked(picking)
        if picking:
            where = "click a trace to record" if tools.view() == "stacked" else "click to record"
            curve = c.curve or "(no curve)"
            hint.set_text(f"Picking {curve} · {where} · {ALT_TEXT}-click to remove · Esc to stop")
        hint.setVisible(picking)

    panel.pick_button.clicked.connect(on_pick_button)
    tools.toolChanged.connect(sync_tool)
    window.plot_area.tabs.currentChanged.connect(sync_tool)

    # markers
    def draw_map() -> None:
        sets = map_markers(c, panel.markers.value())
        draw_markers(window.plots.map.layer("points"), sets, MAP_SIZE)

    def draw_stacked() -> None:
        stacked = window.plots.stacked
        sets = stacked_markers(c, stacked, panel.markers.value())
        draw_markers(stacked.layer("points"), sets, STACKED_SIZE)

    window.plots.stacked.tracesChanged.connect(draw_stacked)
    panel.markers.valueChanged.connect(lambda _value: (draw_map(), draw_stacked()))

    # state -> widgets
    def points_of(name: str) -> tuple[np.ndarray, np.ndarray]:
        table = c.points
        if table is None or name not in table.names:
            return np.array([]), np.array([])
        return table.points(name)

    def sync() -> None:
        table = c.points
        curves = [
            (name, points_of(name)[0].size, curve_color(i))
            for i, name in enumerate(c.curve_names())
        ]
        panel.chips.set_curves(curves, c.curve)
        panel.chips.new_chip.setEnabled(table is not None)
        panel.delete_button.setEnabled(table is not None)
        name = panel.column_name  # follows the curve, but keeps a refused name being typed
        typing = name.hasFocus() and c.curve_name_problem(name.text(), c.curve) is not None
        if name.text().strip() != c.curve and not typing:
            with QSignalBlocker(name):
                name.setText(c.curve)
            panel.show_name_problem(None)
        b, e = points_of(c.curve)
        panel.model.set_points(b, e)
        panel.count_label.setText("" if table is None else f"{b.size} of {table.field.size} fields")
        panel.empty_label.setText(NO_TABLE if table is None else EMPTY)
        panel.table_stack.setCurrentWidget(panel.table if b.size else panel.empty_label)
        with QSignalBlocker(panel.new_table):
            panel.new_table.setChecked(c.new_table)
        draw_map()
        draw_stacked()
        sync_tool()

    # widgets -> state
    session = 0

    def on_name(text: str) -> None:
        problem = c.curve_name_problem(text, c.curve)
        panel.show_name_problem(problem)
        if problem is None and text.strip() != c.curve:
            rename_curve(window, text, session)

    def on_name_done() -> None:
        nonlocal session
        session += 1  # the next typed name is a new undo step
        if panel.column_name.text().strip() != c.curve:
            with QSignalBlocker(panel.column_name):
                panel.column_name.setText(c.curve)
        panel.show_name_problem(None)

    panel.column_name.textChanged.connect(on_name)
    panel.column_name.editingFinished.connect(on_name_done)
    panel.chips.curveClicked.connect(c.set_curve)
    panel.chips.newClicked.connect(lambda: add_curve(window))
    panel.delete_button.clicked.connect(lambda: drop_curve(window))
    panel.table.removeRequested.connect(lambda fields: remove_points(window, fields))
    panel.new_table.toggled.connect(c.set_new_table)
    panel.import_button.clicked.connect(lambda: load_points(window))
    panel.export_button.clicked.connect(lambda: export_points(window))
    c.pointsChanged.connect(sync)

    def on_unit(_old, new) -> None:
        panel.model.set_unit(new)
        draw_map()  # the stacked markers follow the redrawn traces

    c.unitChanged.connect(on_unit)
    panel.model.set_unit(c.unit)
    sync()

    p = window.persistence
    if p is not None:
        p.bind("points/column_name", CurveSetting(c))
        p.bind("points/markers", panel.markers)
