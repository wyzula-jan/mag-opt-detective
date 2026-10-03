"""Main window: the frame (toolbar, rail, panels, plot area, inspector, log, status bar).

The window only builds the frame; the area modules (``plot_panel``, ``console``,
``panels.*``, ``inspector.*``) put their widgets into it, wire them to the
:class:`~mag_opt_detective.gui.controller.AppController` and bind their own settings.
"""

from __future__ import annotations

import itertools
import logging
from pathlib import Path

from PySide6.QtCore import QEvent, QPointF, QRect, QRectF, QSettings, QSize, Qt, Signal
from PySide6.QtGui import (
    QAction,
    QActionGroup,
    QColor,
    QFont,
    QFontMetrics,
    QIcon,
    QKeySequence,
    QPainter,
    QPalette,
    QPen,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QSplitterHandle,
    QStackedWidget,
    QStyle,
    QStyleOptionButton,
    QStylePainter,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from mag_opt_detective import __version__
from mag_opt_detective.core.processing import Axis
from mag_opt_detective.core.units import Unit
from mag_opt_detective.gui import console, export_menu, icons, licences, links, plot_panel, updates
from mag_opt_detective.gui.controller import DATA_CHANGED, DATA_CURRENT, AppController, user_action
from mag_opt_detective.gui.display import format_range, process_key, unit_text
from mag_opt_detective.gui.inspector import colour, models, traces, view
from mag_opt_detective.gui.kit import (
    CollapsibleSection,
    InfoBar,
    SegmentedControl,
    SlidePanel,
    TightToolButton,
)
from mag_opt_detective.gui.panels import PanelPage, library, points, processing, reference, sample
from mag_opt_detective.gui.settings import Persistence
from mag_opt_detective.gui.theme import SCHEMES, Theme, current_theme, current_tokens
from mag_opt_detective.gui.tools import autopick
from mag_opt_detective.gui.widgets import FlowLayout, Separator, last_dir, set_last_dir

logger = logging.getLogger("mag_opt_detective")

SIDE_WIDTH, INSPECTOR_WIDTH, LOG_HEIGHT = 292, 300, 180
DIRECT = Qt.FindChildOption.FindDirectChildrenOnly
# narrowest side panel and inspector (the mockup's narrow-window inspector): the plot gives way
# first, so the window stays usable down to about 1100 px with both open
SIDE_MIN_WIDTH, INSPECTOR_MIN_WIDTH = 240, 280
INSPECTOR_SUBTITLE = "Settings for the plot on screen"

SHORTCUTS = [
    ("Ctrl+Return (or Ctrl+F)", "Process"),
    ("Ctrl+E", "Export the shown data as a table"),
    ("Ctrl+Shift+E", "Journal figure (PDF, SVG, EPS, PNG, TIFF)"),
    ("Ctrl+L / Ctrl+Shift+L", "Open sample field / zero-field files"),
    ("Ctrl+R / Ctrl+Shift+R", "Load reference field / zero-field files"),
    ("Ctrl+1 / 2 / 3 / 4", "Plot R(B)/R(0) / Data / R(B)/R(B-AVR) / R(B)/R(B-ΔB)"),
    ("Alt+1 / 2 / 3", "No / 1st / 2nd derivative"),
    ("V / Z / P", "Pan and zoom / box zoom / pick points"),
    ("W", "Auto-pick: follow a clicked line, or find the lines in a drawn region"),
    ("Ctrl+Z / Ctrl+Shift+Z", "Undo / redo a point edit (Alt-click removes a point)"),
    ("A", "Fit the plot to the data"),
    ("F1", "Open the documentation (on macOS also ⌘?)"),
]
KINDS = (
    ("Ratio", "R(B)/R(0)", "Ctrl+1"),
    ("Data", "Data", "Ctrl+2"),
    ("Ratio_AVR", "R(B)/R(B-AVR)", "Ctrl+3"),
    ("Ratio_Step", "R(B)/R(B-ΔB)", "Ctrl+4"),
)
ORDERS = (("0", "Off", "Alt+1"), ("1", "1st", "Alt+2"), ("2", "2nd", "Alt+3"))
UNITS = tuple((unit, unit_text(unit)) for unit in (Unit.CM1, Unit.MEV, Unit.THZ))
SCHEME_ICONS = {"system": "contrast", "light": "sun", "dark": "moon"}
# how the error bar names a failed action ("Can't process: …"); others: the title, lowercased
ACTIONS = {"Colour range": "set the colour range", "New curve": "add a curve"}
# what usually fixes an error found in a panel (the bar's text; its button opens the panel)
PANEL_HINTS = {
    "sample": "Check the files and their fields in the Sample panel.",
    "reference": "Check the reference sweep in the Reference panel, or set Reference to None.",
    "processing": "Check the energy window and the baseline in the Processing panel.",
}


def _types_text(event) -> bool:
    """A key event that types a character (no modifier other than Shift)."""
    modifiers = event.modifiers() & ~(
        Qt.KeyboardModifier.ShiftModifier | Qt.KeyboardModifier.KeypadModifier
    )
    text = event.text()
    return modifiers == Qt.KeyboardModifier.NoModifier and bool(text) and text.isprintable()


def _paint_dot(widget: QWidget) -> None:
    """A status dot in the top-right corner of *widget*."""
    tokens = current_tokens()
    painter = QPainter(widget)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    size = 9.0
    rect = QRectF(widget.width() - size - 2, 2, size, size)
    painter.setPen(tokens["win"])
    painter.setBrush(tokens["warn"])
    painter.drawEllipse(rect)
    painter.end()


class _Button(QPushButton):
    """A toolbar button as in the mockup: icon, a 6 px gap and the text, then an optional
    key hint (``kbd``) or a chevron (a button with a menu)."""

    GAP = 6
    ICON = 16
    PADDING = 11  # the stylesheet's 10 px padding and 1 px border, on each side

    def __init__(self, text: str, icon: str, color: str | None = None, key: str = "", parent=None):
        super().__init__(text, parent)
        self.setProperty("kit", "button")
        icons.set_icon(self, icon, color)
        self.setIconSize(QSize(self.ICON, self.ICON))
        self._key = key
        self._key_shown = bool(key)

    def key_text(self) -> str:
        return self._key

    def key_shown(self) -> bool:
        return self._key_shown

    def show_key(self, shown: bool) -> None:
        """Show or leave out the key hint (a narrow toolbar leaves it out)."""
        shown = shown and bool(self._key)
        if shown != self._key_shown:
            self._key_shown = shown
            self.updateGeometry()
            self.update()

    def key_width(self) -> int:
        """The width the key hint adds to the button."""
        return self.GAP + self._key_size().width() if self._key else 0

    def _key_font(self) -> QFont:
        font = QFont(self.font())
        font.setBold(False)
        if font.pointSizeF() > 0:
            font.setPointSizeF(font.pointSizeF() * 0.8)
        return font

    def _key_size(self) -> QSize:
        metrics = QFontMetrics(self._key_font())
        return QSize(metrics.horizontalAdvance(self._key) + 10, metrics.height() + 2)

    def _trailing_width(self) -> int:
        if self._key_shown:
            return self.key_width()
        return self.GAP + 12 if self.menu() is not None else 0

    def sizeHint(self) -> QSize:
        """What is painted inside the padding; Qt's own width also counts a menu indicator
        (drawn here as the chevron) and its icon spacing."""
        text = self.fontMetrics().horizontalAdvance(self.text())
        width = 2 * self.PADDING + self.ICON + self.GAP + text + self._trailing_width()
        return QSize(width, max(super().sizeHint().height(), 28))

    def minimumSizeHint(self) -> QSize:
        return self.sizeHint()

    def paintEvent(self, event) -> None:
        option = QStyleOptionButton()
        self.initStyleOption(option)
        option.text = ""
        option.icon = QIcon()
        option.features &= ~QStyleOptionButton.ButtonFeature.HasMenu
        painter = QStylePainter(self)
        painter.drawControl(QStyle.ControlElement.CE_PushButton, option)
        color = option.palette.color(QPalette.ColorRole.ButtonText)
        metrics = self.fontMetrics()
        text_width = metrics.horizontalAdvance(self.text())
        width = self.ICON + self.GAP + text_width + self._trailing_width()
        x = (self.width() - width) // 2
        middle = self.height() // 2
        mode = QIcon.Mode.Normal if self.isEnabled() else QIcon.Mode.Disabled
        self.icon().paint(
            painter, QRect(x, middle - self.ICON // 2, self.ICON, self.ICON), mode=mode
        )
        x += self.ICON + self.GAP
        painter.setPen(color)
        painter.drawText(
            QRect(x, 0, text_width + 1, self.height()), Qt.AlignmentFlag.AlignVCenter, self.text()
        )
        x += text_width + self.GAP
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        if self._key_shown:
            size = self._key_size()
            box = QRectF(x, middle - size.height() / 2, size.width(), size.height())
            faded = QColor(color)
            faded.setAlphaF(0.7)
            painter.setPen(QPen(faded, 1))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(box.adjusted(0.5, 0.5, -0.5, -0.5), 4, 4)
            painter.setFont(self._key_font())
            painter.drawText(box, Qt.AlignmentFlag.AlignCenter, self._key)
        elif self.menu() is not None:
            chevron = icons.icon("chevron-down", color)
            chevron.paint(painter, QRect(x, middle - 6, 12, 12), mode=mode)
        painter.end()


class _ProcessButton(_Button):
    """The primary Process button with its shortcut; a dot marks settings changed since the
    last run."""

    def __init__(self, parent=None):
        super().__init__("Process", "play", "accent-fg", key=process_key(), parent=parent)
        self.setProperty("kit", "primary")
        self.dot = False

    def set_dot(self, dot: bool) -> None:
        self.dot = dot
        self.update()

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        if self.dot:
            _paint_dot(self)


class _RailButton(QToolButton):
    """A rail button: icon above a short label; a dot marks a panel that needs attention.

    *name* (default *text*) is the accessible name: the panel's full title under a short
    label. A panel with data shows their state (:meth:`set_data_state`).
    """

    ICON_TOP = 9  # top of the icon: the stylesheet's 7 px padding and the style's 2 px margin

    def __init__(self, icon: str, text: str, tooltip: str, name: str = "", parent=None):
        super().__init__(parent)
        self.setProperty("kit", "rail")
        self.setCheckable(True)
        self.setText(text)
        self.setToolTip(tooltip)
        self.setAccessibleName(name or text)
        self.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
        self.setIconSize(QSize(20, 20))
        self.setFixedWidth(58)
        font = self.font()
        if font.pointSizeF() > 0:
            font.setPointSizeF(font.pointSizeF() * 0.82)
        self.setFont(font)
        self._icon_name = icon
        self._tooltip = tooltip
        icons.set_icon(self, icon, "muted", on_color="accent")
        self.badge = False
        self.data_state = ""  # none shown

    def set_badge(self, badge: bool) -> None:
        self.badge = badge
        self.update()

    def set_data_state(self, state: str, text: str = "") -> None:
        """Show the state of the panel's data (``AppController.data_state``): the outline
        icon when ``"empty"``, the filled icon (the glyph on an accent tile) when
        ``"current"``, the outline icon with the dot when ``"changed"``. *text* says it in
        the tooltip and the accessible description ("<tooltip> – <text>")."""
        self.data_state = state
        if state == DATA_CURRENT:
            icons.set_icon(self, self._icon_name, "accent-fg", fill="accent")
        else:
            icons.set_icon(self, self._icon_name, "muted", on_color="accent")
        self.set_badge(state == DATA_CHANGED)
        tooltip = f"{self._tooltip} – {text}" if text else self._tooltip
        self.setToolTip(tooltip)
        self.setAccessibleDescription(tooltip)

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        if self.badge:
            self._paint_badge()

    def _paint_badge(self) -> None:
        """The dot on the icon's top-right corner, ringed in the button's background so it
        stands off the glyph (the mockup's ``.rail .badge``)."""
        tokens = current_tokens()
        size, ring = 8.0, 2.0  # the dot inside its ring
        centre = QPointF(self.width() / 2 + self.iconSize().width() / 2, self.ICON_TOP + 2)
        side = size + ring  # the ring's pen is centred on the outline
        rect = QRectF(centre.x() - side / 2, centre.y() - side / 2, side, side)
        background = "accent-soft" if self.isChecked() else "hover" if self.underMouse() else ""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(tokens[background or "sunken"], ring))
        painter.setBrush(tokens["warn"])
        painter.drawEllipse(rect)
        painter.end()


class _Pane(QWidget):
    """A plain widget filled with a theme colour (rail, panels)."""

    def __init__(self, token: str = "win", parent=None):
        super().__init__(parent)
        self._token = token

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), current_tokens()[self._token])
        painter.end()


class _LineHandle(QSplitterHandle):
    def paintEvent(self, event) -> None:
        tokens = current_tokens()
        painter = QPainter(self)
        painter.fillRect(self.rect(), tokens["win"])
        rect = self.rect()
        if self.orientation() == Qt.Orientation.Horizontal:
            painter.fillRect(rect.center().x(), 0, 1, rect.height(), tokens["line"])
        else:
            painter.fillRect(0, rect.center().y(), rect.width(), 1, tokens["line"])
        painter.end()


class _Splitter(QSplitter):
    """Splitter whose handles are drawn as one-pixel lines."""

    def __init__(self, orientation, parent=None):
        super().__init__(orientation, parent)
        self.setHandleWidth(5)

    def createHandle(self) -> QSplitterHandle:
        return _LineHandle(self.orientation(), self)


class _Group(QWidget):
    """Toolbar controls kept on one row; *separated* draws a line before them (the mockup's
    ``.tb-sep``), except where the group starts a row of the wrapped toolbar."""

    SEPARATION = 7  # space on each side of the line

    def __init__(self, separated: bool = False, parent=None):
        super().__init__(parent)
        self.separated = separated

    def paintEvent(self, event) -> None:
        if self.separated and self.x() > 0:
            painter = QPainter(self)
            painter.fillRect(0, 2, 1, self.height() - 4, current_tokens()["line"])
            painter.end()


def _group(*widgets: QWidget, label: str | None = None, separated: bool = False) -> QWidget:
    box = _Group(separated)
    row = QHBoxLayout(box)
    row.setContentsMargins(_Group.SEPARATION + 1 if separated else 0, 0, 0, 0)
    row.setSpacing(5)
    if label:
        text = QLabel(label)
        text.setProperty("kit", "muted")
        row.addWidget(text)
    for widget in widgets:
        row.addWidget(widget)
    return box


def _layout_children(widget: QWidget) -> list[QWidget]:
    """Child widgets of *widget*: those in its layout in layout order (nested layouts too),
    then the others (pages, scroll contents, floating children) in creation order."""
    ordered: list[QWidget] = []

    def walk(layout) -> None:
        for i in range(layout.count()):
            item = layout.itemAt(i)
            if item.widget() is not None:
                ordered.append(item.widget())
            elif item.layout() is not None:
                walk(item.layout())

    if widget.layout() is not None:
        walk(widget.layout())
    if isinstance(widget, QSplitter):
        ordered += [widget.widget(i) for i in range(widget.count())]
    if isinstance(widget, QScrollArea) and widget.widget() is not None:
        ordered.append(widget.widget())
    seen = set(ordered)
    others = [w for w in widget.findChildren(QWidget, options=DIRECT) if w not in seen]
    return [w for w in ordered + others if w.parentWidget() is not None and not w.isWindow()]


def focus_order(root: QWidget) -> list[QWidget]:
    """The widgets under *root* that take Tab focus, in layout order (as they are seen).

    A focusable widget counts as one stop (a spin box, a table, a plot); a scroll area is no
    stop of its own, only its content (which is seen once, though also its viewport's child).
    """
    found: list[QWidget] = []
    seen: set[QWidget] = set()

    def visit(widget: QWidget) -> None:
        if widget in seen:
            return
        seen.add(widget)
        takes_tab = bool(widget.focusPolicy() & Qt.FocusPolicy.TabFocus)
        if takes_tab and widget is not root and not isinstance(widget, QScrollArea):
            found.append(widget)
            return
        for child in _layout_children(widget):
            visit(child)

    visit(root)
    return found


def _segmented(options, name: str) -> SegmentedControl:
    control = SegmentedControl(size="sm")
    for value, text, tooltip in options:
        control.add_option(value, text, tooltip)
    control.setAccessibleName(name)
    return control


class MainToolbar(QWidget):
    """Open, Process, the plot selection, the energy unit, Export and Appearance.

    The groups wrap onto a second row when the window is narrow.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.open_button = _Button("Open sweep…", "folder-open")
        self.open_button.setToolTip("Open the in-field files of a sweep (Ctrl+L)")
        self.process_button = _ProcessButton()
        self.kind = _segmented([(v, t, f"{t} ({k})") for v, t, k in KINDS], "Plot")
        self.order = _segmented([(v, t, k) for v, t, k in ORDERS], "Derivative")
        self.axis = _segmented(
            [("E", "d/dE", "Along energy"), ("B", "d/dB", "Along field")], "Derivative axis"
        )
        self.per_unit = TightToolButton()
        self.per_unit.setProperty("kit", "chip")
        self.per_unit.setCheckable(True)
        self.per_unit.setText("per unit")
        self.per_unit.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self.per_unit.setToolTip(
            "Divide by the real step: d/dE per energy unit or d/dB per tesla.\nOff: per data point."
        )
        self.unit = _segmented([(u.value, t, f"Show energies in {t}") for u, t in UNITS], "Unit")
        self.unit.setToolTip("Energy unit of the plots, ranges, points and exports")

        self.export_button = _Button("Export", "download")
        self.export_button.setToolTip(
            "Export the data as a table, a quick image or a journal figure"
        )
        self.export_menu = QMenu(self.export_button)
        self.export_button.setMenu(self.export_menu)
        self.appearance = QToolButton()  # cycles System -> Light -> Dark
        self.appearance.setProperty("kit", "tool")
        self.appearance.setIconSize(QSize(18, 18))

        self.groups = [  # wrap as wholes
            _group(self.open_button, self.process_button),
            _group(self.kind, label="Plot", separated=True),
            _group(self.order, self.axis, self.per_unit, label="Derivative", separated=True),
            _group(self.unit, label="Unit", separated=True),
        ]
        flow_box = QWidget()
        flow = FlowLayout(flow_box, spacing=_Group.SEPARATION, row_spacing=8)
        for group in self.groups:
            flow.addWidget(group)
        flow_box.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self._end = _group(self.export_button, self.appearance)
        row = QHBoxLayout(self)
        row.setContentsMargins(10, 8, 10, 8)
        row.setSpacing(8)
        row.addWidget(flow_box, stretch=1)
        row.addWidget(self._end, 0, Qt.AlignmentFlag.AlignTop)

    def one_row_width(self) -> int:
        """The toolbar width that holds every group on one row, with Process's key hint."""
        widths = [group.sizeHint().width() for group in self.groups]
        if not self.process_button.key_shown():
            widths[0] += self.process_button.key_width()
        row, margins = self.layout(), self.layout().contentsMargins()
        flow = sum(widths) + _Group.SEPARATION * (len(widths) - 1)  # the flow's spacing
        end = row.spacing() + self._end.sizeHint().width()
        return margins.left() + flow + end + margins.right()

    def resizeEvent(self, event) -> None:
        """Process leaves out its key hint where that keeps the toolbar on one row."""
        full = self.one_row_width()
        short = full - self.process_button.key_width()
        self.process_button.show_key(not short <= self.width() < full)
        super().resizeEvent(event)


class AppearanceSetting:
    """Settings protocol for the appearance (the theme's scheme)."""

    def __init__(self, window: MainWindow):
        self.window = window

    def settings_value(self) -> str:
        return self.window.theme.scheme()

    def set_settings_value(self, value) -> bool:
        if value not in SCHEMES:
            return False
        self.window.set_appearance(value)
        return True


class PanelSetting:
    """Settings protocol for the panel shown in the side panel (by name)."""

    def __init__(self, window: MainWindow):
        self.window = window

    def settings_value(self) -> str:
        return self.window.current_panel()

    def set_settings_value(self, value) -> bool:
        if value not in self.window.panels:
            return False
        self.window.show_panel(value, open=None)
        return True


class MainWindow(QMainWindow):
    """The application window.

    *settings*: where choices are remembered between sessions; None keeps nothing (tests).
    *theme*: the application's :class:`Theme`; without one the window uses the applied theme,
    or makes its own (not applied to the application) for the plot colours and Appearance.

    Accessors for the area modules: ``controller``, ``persistence``, ``theme``, ``tools``,
    ``panels`` (name -> panel widget), ``inspector`` (name -> section), ``plots`` (map,
    stacked, reference), ``infobar``, ``commands`` (name -> QAction), ``toolbar``, ``rail``
    and the slide panels ``side_panel``, ``inspector_panel`` and ``log_panel``.

    ``themeChanged`` relays ``theme.changed``: area modules connect to it instead of the
    (application-wide) theme, so their connections end with the window.

    Tab moves through the areas as they are laid out (:meth:`update_tab_order`), not in the
    order their widgets were made.
    """

    themeChanged = Signal()

    def __init__(self, parent=None, settings: QSettings | None = None, theme: Theme | None = None):
        super().__init__(parent)
        self.setWindowTitle(f"Magneto-Optical Detective {__version__}")
        self.resize(1400, 900)
        self.controller = AppController(self)
        self.persistence = Persistence(settings) if settings is not None else None
        self.theme = theme or current_theme() or Theme("system", self)
        self.theme.changed.connect(self.themeChanged)
        self.geometry_restored = False
        self._shown = False
        self.commands: dict[str, QAction] = {}
        self.panels: dict[str, QWidget] = {}
        self.panel_pages: dict[str, PanelPage] = {}
        self.inspector: dict[str, CollapsibleSection] = {}
        self.splitters: dict[str, QSplitter] = {}
        self._rail_buttons: dict[str, _RailButton] = {}
        self._panel = ""
        self.infobar = InfoBar()

        self._create_actions()
        self._build_frame()
        self._create_menus()
        plot_panel.install(self)
        console.install(self)
        for module in (sample, reference, processing, library, points):
            module.install(self)
        for module in (view, colour, traces, models):
            module.install(self)
        autopick.install(self)

        export_menu.install(self)
        updates.install(self)
        self._wire_frame()
        self.show_panel(next(iter(self.panels)), open=None)
        if self.persistence is not None:
            self._bind_settings()
            self.restore_settings()

    # ------------------------------------------------------------------ frame
    def _action(self, name: str, text: str, shortcuts=(), slot=None) -> QAction:
        action = QAction(text, self)
        action.setShortcuts([QKeySequence(s) for s in shortcuts])
        if slot is not None:
            action.triggered.connect(lambda _checked=False: slot())
        self.addAction(action)
        self.commands[name] = action
        return action

    def _create_actions(self) -> None:
        process = self._action("process", "&Process", ("Ctrl+Return", "Ctrl+Enter", "Ctrl+F"))
        process.setToolTip("Process the loaded files (Ctrl+Return)")
        table = self._action("export_table", "Data table…", ("Ctrl+E",))
        icons.set_icon(table, "file-text")
        image = self._action("export_image", "Quick image (PNG/SVG)…")
        icons.set_icon(image, "image")
        suffix = self._action("export_suffix", "Add plot type to the file name")
        suffix.setCheckable(True)
        suffix.setChecked(True)
        self._action("quit", "&Quit", (QKeySequence.StandardKey.Quit,), self.close)

    def _build_frame(self) -> None:
        self.toolbar = MainToolbar()
        tb = self.toolbar
        tb.process_button.clicked.connect(self.commands["process"].trigger)
        tb.export_menu.addAction(self.commands["export_table"])
        tb.export_menu.addAction(self.commands["export_image"])
        tb.export_menu.addSeparator()
        tb.export_menu.addAction(self.commands["export_suffix"])
        group = QActionGroup(self)
        self._scheme_actions: dict[str, QAction] = {}
        for scheme in SCHEMES:
            action = QAction(scheme.capitalize(), self, checkable=True)
            icons.set_icon(action, SCHEME_ICONS[scheme])
            action.triggered.connect(lambda _checked=False, s=scheme: self.set_appearance(s))
            group.addAction(action)
            self._scheme_actions[scheme] = action
        tb.appearance.clicked.connect(self.cycle_appearance)
        self._sync_appearance()

        self.rail = _Pane("sunken")
        self.rail.setFixedWidth(64)
        self._rail_layout = QVBoxLayout(self.rail)
        self._rail_layout.setContentsMargins(3, 8, 3, 8)
        self._rail_layout.setSpacing(4)
        self._rail_layout.addStretch(1)

        self._side_stack = QStackedWidget()
        side = _Pane("win")
        QVBoxLayout(side).setContentsMargins(0, 0, 0, 0)
        side.layout().addWidget(self._side_stack)
        side.setMinimumWidth(SIDE_MIN_WIDTH)
        self.side_panel = SlidePanel(side, SIDE_WIDTH)

        self.stage = QWidget()
        stage_layout = QVBoxLayout(self.stage)
        stage_layout.setContentsMargins(0, 0, 0, 0)
        stage_layout.setSpacing(0)
        log = _Pane("win")
        log_layout = QVBoxLayout(log)
        log_layout.setContentsMargins(0, 0, 0, 0)
        log_layout.setSpacing(0)
        self.log_panel = SlidePanel(log, LOG_HEIGHT)
        self.log_panel.set_open(False, animate=False)
        self.stage_splitter = _Splitter(Qt.Orientation.Vertical)
        self.stage_splitter.addWidget(self.stage)
        self.stage_splitter.addWidget(self.log_panel)
        self.stage_splitter.setStretchFactor(0, 1)
        self.stage_splitter.setCollapsible(0, False)

        inspector = QWidget()
        inspector_layout = QVBoxLayout(inspector)
        inspector_layout.setContentsMargins(0, 0, 0, 0)
        inspector_layout.setSpacing(0)
        head = QWidget()
        head_layout = QVBoxLayout(head)
        head_layout.setContentsMargins(14, 12, 14, 10)
        head_layout.setSpacing(2)
        self._inspector_title = QLabel("Map")
        font = self._inspector_title.font()
        font.setBold(True)
        self._inspector_title.setFont(font)
        self._inspector_subtitle = QLabel(INSPECTOR_SUBTITLE)
        self._inspector_subtitle.setProperty("kit", "muted")
        self._inspector_subtitle.setWordWrap(True)
        head_layout.addWidget(self._inspector_title)
        head_layout.addWidget(self._inspector_subtitle)
        inspector_layout.addWidget(head)
        inspector_layout.addWidget(Separator())
        self._inspector_layout = QVBoxLayout()
        self._inspector_layout.setSpacing(0)
        inspector_layout.addLayout(self._inspector_layout)
        inspector_layout.addStretch(1)
        scroll = QScrollArea()
        scroll.setFocusPolicy(Qt.FocusPolicy.NoFocus)  # Tab goes to the controls inside
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(inspector)
        scroll.setMinimumWidth(INSPECTOR_MIN_WIDTH)
        self.inspector_panel = SlidePanel(scroll, INSPECTOR_WIDTH)

        self.body_splitter = _Splitter(Qt.Orientation.Horizontal)
        self.body_splitter.addWidget(self.side_panel)
        self.body_splitter.addWidget(self.stage_splitter)
        self.body_splitter.addWidget(self.inspector_panel)
        self.body_splitter.setStretchFactor(1, 1)
        self.body_splitter.setCollapsible(1, False)
        self.add_splitter("body", self.body_splitter)
        self.add_splitter("stage", self.stage_splitter)

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.toolbar)
        layout.addWidget(Separator())
        row = QHBoxLayout()
        row.setSpacing(0)
        row.addWidget(self.rail)
        row.addWidget(Separator(Qt.Orientation.Vertical))
        row.addWidget(self.body_splitter, stretch=1)
        layout.addLayout(row, stretch=1)
        self.setCentralWidget(central)

        status = self.statusBar()
        status.setContentsMargins(12, 0, 8, 0)  # the state dot clear of the window edge
        self._state_label = QLabel()
        self._summary_label = QLabel()
        self._summary_label.setProperty("kit", "muted")
        self._summary_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self._cursor_label = QLabel()
        self._cursor_label.setProperty("kit", "muted")
        status.addWidget(self._state_label)
        status.addWidget(self._summary_label, stretch=1)
        status.addPermanentWidget(self._cursor_label)

    def _create_menus(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        self._file_menu = file_menu
        self._file_anchor = file_menu.addSeparator()
        file_menu.addAction(self.commands["process"])
        export = file_menu.addMenu(self.toolbar.export_menu)  # the toolbar's Export menu
        export.setText("&Export")
        icons.set_icon(export, "download")
        file_menu.addSeparator()
        file_menu.addAction(self.commands["quit"])
        self.edit_menu = self.menuBar().addMenu("&Edit")  # filled by the area modules

        view_menu = self.menuBar().addMenu("&View")
        appearance = view_menu.addMenu("&Appearance")
        for action in self._scheme_actions.values():
            appearance.addAction(action)
        center = QAction("&Center window", self)
        center.triggered.connect(self.center_on_screen)
        view_menu.addAction(center)
        reset = QAction("&Reset settings", self)
        reset.triggered.connect(self.reset_settings)
        reset.setEnabled(self.persistence is not None)
        view_menu.addAction(reset)

        # Help: the web links (gui/links.py), the shortcuts, then the group of the app itself,
        # About (on macOS Qt moves it into the application menu); commands[...] names them
        self.help_menu = self.menuBar().addMenu("&Help")
        # the system's help keys (the menu shows the first: F1, on macOS Cmd+?) and F1, which
        # needs fn on Mac laptops
        keys = QKeySequence.keyBindings(QKeySequence.StandardKey.HelpContents)
        keys += [QKeySequence("F1")] if QKeySequence("F1") not in keys else []
        docs = self._action("documentation", "&Documentation", keys, self.open_documentation)
        docs.setStatusTip("Open the user guide in the web browser")
        icons.set_icon(docs, "file-text")
        feature = self._action("request_feature", "Request a &feature…", (), self.request_feature)
        feature.setStatusTip("Suggest a feature: opens the feature request form on GitHub")
        icons.set_icon(feature, "plus")
        bug = self._action("report_bug", "Report a &bug…", (), self.report_bug)
        bug.setStatusTip(
            "Report a problem: opens the bug form on GitHub, with the app version and the "
            "system filled in"
        )
        icons.set_icon(bug, "triangle-alert")
        shortcuts = self._action("shortcuts", "&Shortcuts", (), self.show_shortcuts)
        shortcuts.setStatusTip("List the keyboard shortcuts")
        about = self._action("about", "&About", (), self.show_about)
        about.setStatusTip("The version, how to cite the app and the licences")
        self.help_menu.addActions([docs, feature, bug])
        self.help_menu.addSeparator()
        self.help_menu.addAction(shortcuts)
        self.help_menu.addSeparator()
        self.help_menu.addAction(about)

    def _wire_frame(self) -> None:
        c, tb = self.controller, self.toolbar
        # The toolbar sets the unit and the plot selection (plot_panel) and follows the
        # controller back, so a change made through ``window.controller`` shows here too
        # (set_value / setChecked with the current value emit nothing: no loop).
        tb.unit.valueChanged.connect(c.set_unit)
        c.unitChanged.connect(lambda _old, new: tb.unit.set_value(new.value))

        for value, _text, shortcut in KINDS:
            self._shortcut(shortcut, lambda v=value: tb.kind.set_value(v))
        for value, _text, shortcut in ORDERS:
            self._shortcut(shortcut, lambda v=value: tb.order.set_value(v))

        def on_order(value: str) -> None:
            derivative = value != "0"
            for option in tb.axis.options():
                tb.axis.set_option_enabled(option, derivative)
            tb.per_unit.setEnabled(derivative)

        def follow_selection() -> None:
            s = c.selection
            tb.kind.set_value(s.kind.value)
            tb.order.set_value(str(s.order))
            tb.axis.set_value("B" if s.axis == Axis.FIELD else "E")
            tb.per_unit.setChecked(s.physical)
            on_order(tb.order.value())

        tb.order.valueChanged.connect(on_order)
        c.selectionChanged.connect(follow_selection)
        on_order(tb.order.value())

        c.changedSinceProcess.connect(self._sync_state)
        c.resultChanged.connect(self._sync_state)
        c.unitChanged.connect(lambda _old, _new: self._sync_state())
        self.themeChanged.connect(self._sync_state)
        self.themeChanged.connect(self._sync_appearance)
        self._sync_state()
        self.side_panel.openChanged.connect(lambda _open: self._sync_rail())

    def _shortcut(self, key: str, slot) -> None:
        action = QAction(self)
        action.setShortcut(QKeySequence(key))
        action.triggered.connect(lambda _checked=False: slot())
        self.addAction(action)

    def _bind_settings(self) -> None:
        bind, tb = self.persistence.bind, self.toolbar
        bind("window/appearance", AppearanceSetting(self))
        bind("window/panel", PanelSetting(self))
        bind("view/unit", tb.unit)
        bind("plot/kind", tb.kind)
        bind("plot/order", tb.order)
        bind("plot/axis", tb.axis)
        bind("plot/per_unit", tb.per_unit)

    # ------------------------------------------------------------------ areas
    def add_panel(
        self,
        name: str,
        title: str,
        icon: str,
        tooltip: str,
        widget: QWidget,
        subtitle: str = "",
        rail_text: str | None = None,
    ) -> PanelPage:
        """Add a rail button and its panel; returns the page (header and scrolling content).

        The button reads *rail_text* (default *title*); its accessible name is *title*.
        """
        page = PanelPage(title, subtitle, widget)
        self._side_stack.addWidget(page)
        button = _RailButton(icon, rail_text or title, tooltip, title)
        button.clicked.connect(lambda _checked=False, n=name: self._on_rail(n))
        self._rail_layout.insertWidget(self._rail_layout.count() - 1, button)
        self.panels[name] = widget
        self.panel_pages[name] = page
        self._rail_buttons[name] = button
        return page

    def current_panel(self) -> str:
        return self._panel

    def rail_button(self, name: str) -> QToolButton:
        """The rail button of panel *name*; its ``badge`` marks a panel that needs attention,
        its ``set_data_state(state, text)`` shows the state of the panel's data."""
        return self._rail_buttons[name]

    def cursor_text(self) -> str:
        """The cursor read-out in the status bar."""
        return self._cursor_label.text()

    def show_panel(self, name: str, open: bool | None = True) -> None:
        """Show panel *name* in the side panel; *open*: also open (True) or leave it (None)."""
        self._side_stack.setCurrentWidget(self.panel_pages[name])
        self._panel = name
        if open:
            self.side_panel.set_open(True)
        self._sync_rail()

    def _on_rail(self, name: str) -> None:
        if name == self._panel and self.side_panel.is_open():
            self.side_panel.set_open(False)
        else:
            self.show_panel(name)
        self._sync_rail()

    def _sync_rail(self) -> None:
        for name, button in self._rail_buttons.items():
            button.setChecked(self.side_panel.is_open() and name == self._panel)

    def add_inspector_section(
        self, name: str, title: str, widget: QWidget, expanded: bool = True
    ) -> CollapsibleSection:
        section = CollapsibleSection(title, expanded=expanded)
        section.body_layout().addWidget(widget)
        self._inspector_layout.addWidget(section)
        self.inspector[name] = section
        if self.persistence is not None:
            self.persistence.bind(f"inspector/{name}_expanded", section)
        return section

    def set_inspector_title(self, text: str) -> None:
        self._inspector_title.setText(text)

    def set_inspector_subtitle(self, text: str = INSPECTOR_SUBTITLE) -> None:
        self._inspector_subtitle.setText(text)

    def inspector_subtitle(self) -> str:
        return self._inspector_subtitle.text()

    def add_splitter(self, key: str, splitter: QSplitter) -> None:
        """Remember *splitter*'s layout between sessions (saveState under *key*)."""
        self.splitters[key] = splitter

    def add_file_action(self, action: QAction) -> None:
        """Add a file action (e.g. a load dialog) to the File menu, before Process."""
        self._file_menu.insertAction(self._file_anchor, action)
        self.addAction(action)

    def set_cursor_text(self, text: str) -> None:
        self._cursor_label.setText(text)

    # ------------------------------------------------------------------ state
    def _sync_state(self, *_args) -> None:
        c = self.controller
        tokens = current_tokens()
        changed = c.changed_since_process()
        self.toolbar.process_button.set_dot(changed)
        if (button := self._rail_buttons.get("processing")) is not None:
            button.set_badge(changed)
        if c.result is None:
            color, text = tokens["faint"], "Not processed yet"
        elif changed:
            color, text = tokens["warn"], f"Settings changed · process again ({process_key()})"
        elif c.result_source == "library":
            color, text = tokens["accent"], "Showing a library map"
        else:
            stamp = c.processed_at.strftime("%H:%M") if c.processed_at else ""
            color, text = tokens["ok"], f"Processed {stamp}"
        self._state_label.setText(
            f"<span style='color:{color.name()}'>●</span>&nbsp;"
            f"<span style='color:{(tokens['warn'] if changed else tokens['fg']).name()}'>"
            f"{text}</span>"
        )
        self.toolbar.process_button.setToolTip(
            "Settings changed since the last run. Process again (Ctrl+Return)"
            if changed
            else "Process the loaded files (Ctrl+Return)"
        )
        self._summary_label.setText(self._summary())
        self.setWindowTitle(self.window_title())

    def window_title(self) -> str:
        """The app and version, and the name of the map shown (as in the mockup)."""
        title = f"Magneto-Optical Detective {__version__}"
        if self.controller.result is not None:
            title += f" · {self.controller.result_name()}"
        return title

    def _summary(self) -> str:
        c = self.controller
        if c.result is None:
            return ""
        fmap = c.result.ratio.to_unit(c.unit)
        b, e = fmap.field, fmap.energy
        return (
            f"{b.size} spectra · B {format_range(b.min(), b.max(), 'T')} · "
            f"E {format_range(e.min(), e.max(), unit_text(c.unit))}"
        )

    def summary_text(self) -> str:
        return self._summary_label.text()

    def state_text(self) -> str:
        return self._state_label.text()

    # ------------------------------------------------------------------ appearance
    def set_appearance(self, scheme: str) -> None:
        self.theme.set_scheme(scheme)
        self._sync_appearance()

    def cycle_appearance(self) -> None:
        """System -> Light -> Dark -> System (the toolbar button)."""
        self.set_appearance(SCHEMES[(SCHEMES.index(self.theme.scheme()) + 1) % len(SCHEMES)])

    def _sync_appearance(self) -> None:
        scheme = self.theme.scheme()
        self._scheme_actions[scheme].setChecked(True)
        button = self.toolbar.appearance
        icons.set_icon(button, SCHEME_ICONS[scheme], "muted")
        button.setToolTip(f"Appearance: {scheme.capitalize()} (click to change)")
        button.setAccessibleName(f"Appearance: {scheme.capitalize()}")

    # ------------------------------------------------------------------ errors
    def report_error(
        self,
        title: str,
        message: str,
        panel: str | None = None,
        expected: bool = True,
        hint: str | None = None,
    ) -> None:
        """Log an error; expected ones show in the bar above the plot, others in a dialog.

        The bar reads "Can't <action>: <message>." with *hint* (or the usual remedy of
        *panel*) below it, as in the mockup; a message "<what>: <detail>" heads with *what*
        and puts the detail before the remedy. Without a remedy, the message goes below.
        """
        logger.error("%s: %s", title, message)
        if not expected:
            QMessageBox.warning(self, title, message)
            return
        action = ACTIONS.get(title, title[:1].lower() + title[1:])
        hint = hint or PANEL_HINTS.get(panel or "")
        what, _, detail = message.rstrip(".").partition(": ")
        if hint and detail:
            head = f"Can't {action}: {what}."
            text = f"{detail[:1].upper()}{detail[1:]}. {hint}"
        elif hint:
            head, text = f"Can't {action}: {message.rstrip('.')}.", hint
        else:
            head, text = f"Can't {action}", message[:1].upper() + message[1:]
        if panel in self.panels:
            title_of = self.panel_pages[panel].title.text()
            self.infobar.show_message(
                "error", head, text, f"Open {title_of}", lambda: self.show_panel(panel)
            )
        else:
            self.infobar.show_message("error", head, text)

    # ------------------------------------------------------------------ settings
    def restore_settings(self) -> None:
        p = self.persistence
        if p is None:
            return
        with self.controller.restoring():
            p.restore()
        if (geometry := p.bytes_value("window/geometry")) is not None:
            self.geometry_restored = self.restoreGeometry(geometry)
        for key, splitter in self.splitters.items():
            p.restore_splitter(key, splitter)
        folder = p.value("files/last_dir")
        if isinstance(folder, str) and Path(folder).is_dir():
            set_last_dir(folder)

    def save_settings(self) -> None:
        p = self.persistence
        if p is None:
            return
        p.set_value("window/geometry", self.saveGeometry())
        if self._shown:  # a window never laid out has no meaningful sizes
            for key, splitter in self.splitters.items():
                p.save_splitter(key, splitter)
        p.set_value("files/last_dir", last_dir())
        p.save()

    def reset_settings(self) -> None:
        if self.persistence is None:
            return
        with self.controller.restoring():
            self.persistence.reset()
        self.side_panel.set_open(True, animate=False)
        self.inspector_panel.set_open(True, animate=False)
        self.log_panel.set_open(False, animate=False)
        for panel in self.plot_area.scale_panels().values():
            panel.set_open(True, animate=False)
        self.resize(1400, 900)
        self.center_on_screen()
        logger.info("Settings reset to the defaults.")

    # ------------------------------------------------------------------ window
    def center_on_screen(self) -> None:
        screen = self.screen()
        if screen is None:
            return
        frame = self.frameGeometry()
        frame.moveCenter(screen.availableGeometry().center())
        self.move(frame.topLeft())

    def show_shortcuts(self) -> None:
        rows = "".join(f"<tr><td><b>{k}</b></td><td>{v}</td></tr>" for k, v in SHORTCUTS)
        QMessageBox.information(self, "Shortcuts", f"<table cellspacing='6'>{rows}</table>")

    def show_about(self) -> None:
        v = links.versions()  # the bug form gets the same
        licences.about_box(
            self,
            f"<b>Magneto-Optical Detective {__version__}</b><br>"
            f"Python {v['python']}, {v['qt']}<br>{v['libraries']}<br><br>"
            "Free software under the GNU GPL v3; commercial licences on request. If you use it "
            "for an analysis in a publication, please cite "
            "it with this version (see CITATION.cff in the repository).",
        ).exec()

    # ------------------------------------------------------------------ web links
    @user_action("Open the documentation")
    def open_documentation(self) -> None:
        self._open_link("Open the documentation", links.DOCS_URL)

    @user_action("Request a feature")
    def request_feature(self) -> None:
        self._open_link("Request a feature", links.feature_request_url())

    @user_action("Report a bug")
    def report_bug(self) -> None:
        """Open the bug form, filled in with :func:`links.environment` and the plot shown."""
        url = links.bug_report_url(links.environment(self.plot_summary()))
        self._open_link("Report a bug", url)

    def _open_link(self, title: str, url: str) -> None:
        """Open *url* in the web browser; without one, the error bar offers to copy it."""
        try:
            links.open_url(url)
        except OSError as exc:
            self.report_error(title, str(exc))
            bar = self.infobar  # the same message, with the address to copy
            bar.show_message(
                "error",
                bar.title_label.text(),
                bar.text_label.text(),
                "Copy address",
                lambda: QApplication.clipboard().setText(url),
            )

    def plot_summary(self) -> str:
        """The plot shown in a few words, e.g. ``"Map: R(B)/R(0), meV"`` (no names or data)."""
        c = self.controller
        if c.result is None:
            return "Nothing processed"
        view = self.plot_area.current_view()
        what = plot_panel.value_label(self, view) if view == "reference" else c.description()
        library = ", a library map" if c.result_source == "library" else ""
        return f"{view.capitalize()}: {what}, {unit_text(c.unit)}{library}"

    def event(self, event) -> bool:
        # The tool shortcuts are single letters for the whole window; a list, table or combo
        # box with the focus keeps typed letters (keyboard search, starting an edit).
        if (
            event.type() == QEvent.Type.ShortcutOverride
            and _types_text(event)
            and isinstance(QApplication.focusWidget(), QAbstractItemView | QComboBox)
        ):
            event.accept()
            return True
        return super().event(event)

    def focus_areas(self) -> list[QWidget]:
        """The areas Tab visits, in order."""
        return [
            self.toolbar,
            self.rail,
            self.side_panel,
            self.stage_splitter,
            self.inspector_panel,
            self.statusBar(),
        ]

    def update_tab_order(self) -> list[QWidget]:
        """Chain the focusable widgets of :meth:`focus_areas` in order; returns the chain."""
        chain = [w for area in self.focus_areas() for w in focus_order(area)]
        for first, second in itertools.pairwise(chain):
            QWidget.setTabOrder(first, second)
        return chain

    def focusNextPrevChild(self, next: bool) -> bool:
        """Tab steps along the chain of :meth:`update_tab_order` and wraps at its ends.

        The chain is made again, so widgets made since (file rows, model cards, tools) join
        where they show. Qt's own step would also stop between the last and the first at
        widgets left out of the chain (the frame of a segmented control).
        """
        chain = self.update_tab_order()
        current = QApplication.focusWidget()
        if current not in chain:
            return super().focusNextPrevChild(next)
        stops = [w for w in chain if w is current or (w.isVisibleTo(self) and w.isEnabled())]
        target = stops[(stops.index(current) + (1 if next else -1)) % len(stops)]
        target.setFocus(
            Qt.FocusReason.TabFocusReason if next else Qt.FocusReason.BacktabFocusReason
        )
        return True

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if not self._shown:  # start on the plot, not with a focus ring on the first button
            self.plots.map.view.setFocus(Qt.FocusReason.OtherFocusReason)
        self._shown = True

    def closeEvent(self, event) -> None:
        self.save_settings()
        logger.removeHandler(self.log_handler)
        super().closeEvent(event)
