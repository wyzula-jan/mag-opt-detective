"""Main window: the frame (toolbar, rail, panels, plot area, inspector, log, status bar).

The window only builds the frame; the area modules (``plot_panel``, ``console``,
``panels.*``, ``inspector.*``) put their widgets into it, wire them to the
:class:`~mag_opt_detective.gui.controller.AppController` and bind their own settings.
"""

from __future__ import annotations

import logging
import platform
from pathlib import Path

import numpy as np
import pyqtgraph as pg
import scipy
from PySide6 import __version__ as pyside_version
from PySide6.QtCore import QEvent, QRectF, QSettings, QSize, Qt, Signal, qVersion
from PySide6.QtGui import QAction, QActionGroup, QKeySequence, QPainter
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
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from mag_opt_detective import __version__
from mag_opt_detective.core.processing import Axis
from mag_opt_detective.core.units import Unit
from mag_opt_detective.gui import console, export_menu, icons, plot_panel
from mag_opt_detective.gui.controller import AppController
from mag_opt_detective.gui.inspector import colour, models, traces, view
from mag_opt_detective.gui.kit import CollapsibleSection, InfoBar, SegmentedControl, SlidePanel
from mag_opt_detective.gui.panels import PanelPage, library, points, processing, reference, sample
from mag_opt_detective.gui.settings import Persistence
from mag_opt_detective.gui.theme import SCHEMES, Theme, current_theme, current_tokens
from mag_opt_detective.gui.tools import autopick
from mag_opt_detective.gui.widgets import FlowLayout, Separator, last_dir, set_last_dir

logger = logging.getLogger("mag_opt_detective")

SIDE_WIDTH, INSPECTOR_WIDTH, LOG_HEIGHT = 292, 300, 180

SHORTCUTS = [
    ("Ctrl+Return (or Ctrl+F)", "Process"),
    ("Ctrl+E", "Export the shown data as a table"),
    ("Ctrl+Shift+E", "Export a journal figure (PDF, SVG, EPS, PNG, TIFF)"),
    ("Ctrl+L / Ctrl+Shift+L", "Open sample field / zero-field files"),
    ("Ctrl+R / Ctrl+Shift+R", "Load reference field / zero-field files"),
    ("Ctrl+1 / 2 / 3 / 4", "Plot R(B)/R(0) / Data / R(B)/R(B-AVR) / R(B)/R(B-ΔB)"),
    ("Alt+1 / 2 / 3", "No / 1st / 2nd derivative"),
    ("V / Z / P", "Pan and zoom / box zoom / pick points"),
    ("W", "Auto-pick: follow a clicked line, or find the lines in a dragged box"),
    ("Ctrl+Z / Ctrl+Shift+Z", "Undo / redo a point edit (Alt-click removes a point)"),
    ("A", "Fit the plot to the data"),
]
KINDS = (
    ("Ratio", "R(B)/R(0)", "Ctrl+1"),
    ("Data", "Data", "Ctrl+2"),
    ("Ratio_AVR", "R(B)/R(B-AVR)", "Ctrl+3"),
    ("Ratio_Step", "R(B)/R(B-ΔB)", "Ctrl+4"),
)
ORDERS = (("0", "Off", "Alt+1"), ("1", "1st", "Alt+2"), ("2", "2nd", "Alt+3"))
UNITS = ((Unit.CM1, "cm⁻¹"), (Unit.MEV, "meV"), (Unit.THZ, "THz"))
SCHEME_ICONS = {"system": "contrast", "light": "sun", "dark": "moon"}


def _types_text(event) -> bool:
    """A key event that types a character (no modifier other than Shift)."""
    modifiers = event.modifiers() & ~(
        Qt.KeyboardModifier.ShiftModifier | Qt.KeyboardModifier.KeypadModifier
    )
    text = event.text()
    return modifiers == Qt.KeyboardModifier.NoModifier and bool(text) and text.isprintable()


def _paint_dot(widget: QWidget, token: str = "warn", size: float = 9.0) -> None:
    """A status dot in the top-right corner of *widget*."""
    tokens = current_tokens()
    painter = QPainter(widget)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    rect = QRectF(widget.width() - size - 2, 2, size, size)
    painter.setPen(tokens["win"])
    painter.setBrush(tokens[token])
    painter.drawEllipse(rect)
    painter.end()


class _ProcessButton(QPushButton):
    """The primary Process button; a dot marks settings changed since the last run."""

    def __init__(self, parent=None):
        super().__init__("Process", parent)
        self.setProperty("kit", "primary")
        icons.set_icon(self, "play", "accent-fg")
        self.dot = False

    def set_dot(self, dot: bool) -> None:
        self.dot = dot
        self.update()

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        if self.dot:
            _paint_dot(self)


class _RailButton(QToolButton):
    """A rail button: icon above a short label; a dot marks a panel that needs attention."""

    def __init__(self, icon: str, text: str, tooltip: str, parent=None):
        super().__init__(parent)
        self.setProperty("kit", "rail")
        self.setCheckable(True)
        self.setText(text)
        self.setToolTip(tooltip)
        self.setAccessibleName(text)
        self.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
        self.setIconSize(QSize(20, 20))
        self.setFixedWidth(58)
        font = self.font()
        if font.pointSizeF() > 0:
            font.setPointSizeF(font.pointSizeF() * 0.82)
        self.setFont(font)
        icons.set_icon(self, icon, "muted", on_color="accent")
        self.badge = False

    def set_badge(self, badge: bool) -> None:
        self.badge = badge
        self.update()

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        if self.badge:
            _paint_dot(self, size=8.0)


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


def _group(*widgets: QWidget, label: str | None = None) -> QWidget:
    box = QWidget()
    row = QHBoxLayout(box)
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(5)
    if label:
        text = QLabel(label)
        text.setProperty("kit", "muted")
        row.addWidget(text)
    for widget in widgets:
        row.addWidget(widget)
    return box


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
        self.open_button = QPushButton("Open sweep…")
        self.open_button.setToolTip("Open the in-field files of a sweep (Ctrl+L)")
        icons.set_icon(self.open_button, "folder-open")
        self.process_button = _ProcessButton()
        self.kind = _segmented([(v, t, f"{t} ({k})") for v, t, k in KINDS], "Plot")
        self.order = _segmented([(v, t, k) for v, t, k in ORDERS], "Derivative")
        self.axis = _segmented(
            [("E", "d/dE", "Along energy"), ("B", "d/dB", "Along field")], "Derivative axis"
        )
        self.per_unit = QToolButton()
        self.per_unit.setProperty("kit", "tool")
        self.per_unit.setCheckable(True)
        self.per_unit.setText("per unit")
        self.per_unit.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self.per_unit.setToolTip(
            "Divide by the real step: d/dE per energy unit or d/dB per tesla.\n"
            "Off: per data point, as in the old versions."
        )
        self.unit = _segmented([(u.value, t, f"Show energies in {t}") for u, t in UNITS], "Unit")
        self.unit.setToolTip("Energy unit of the plots, ranges, points and exports")

        self.export_button = QToolButton()
        self.export_button.setText("Export")
        self.export_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.export_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        icons.set_icon(self.export_button, "download")
        self.export_menu = QMenu(self.export_button)
        self.export_button.setMenu(self.export_menu)
        self.appearance = QToolButton()  # cycles System -> Light -> Dark
        self.appearance.setProperty("kit", "tool")
        self.appearance.setIconSize(QSize(18, 18))

        flow_box = QWidget()
        flow = FlowLayout(flow_box, spacing=8, row_spacing=8)
        flow.addWidget(_group(self.open_button, self.process_button))
        flow.addWidget(_group(self.kind, label="Plot"))
        flow.addWidget(_group(self.order, self.axis, self.per_unit, label="Derivative"))
        flow.addWidget(_group(self.unit, label="Unit"))
        flow_box.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        row = QHBoxLayout(self)
        row.setContentsMargins(10, 8, 10, 8)
        row.setSpacing(8)
        row.addWidget(flow_box, stretch=1)
        row.addWidget(_group(self.export_button, self.appearance), 0, Qt.AlignmentFlag.AlignTop)


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
        hint = QLabel("Settings for the plot on screen")
        hint.setProperty("kit", "muted")
        head_layout.addWidget(self._inspector_title)
        head_layout.addWidget(hint)
        inspector_layout.addWidget(head)
        inspector_layout.addWidget(Separator())
        self._inspector_layout = QVBoxLayout()
        self._inspector_layout.setSpacing(0)
        inspector_layout.addLayout(self._inspector_layout)
        inspector_layout.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(inspector)
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
        file_menu.addAction(self.commands["export_table"])
        file_menu.addAction(self.commands["export_image"])
        file_menu.addSeparator()
        file_menu.addAction(self.commands["quit"])
        self.edit_menu = self.menuBar().addMenu("&Edit")  # filled by the area modules

        view_menu = self.menuBar().addMenu("&View")
        appearance = view_menu.addMenu("&Appearance")
        for action in self._scheme_actions.values():
            appearance.addAction(action)
        center = QAction("&Center Window", self)
        center.triggered.connect(self.center_on_screen)
        view_menu.addAction(center)
        reset = QAction("&Reset Settings", self)
        reset.triggered.connect(self.reset_settings)
        reset.setEnabled(self.persistence is not None)
        view_menu.addAction(reset)

        help_menu = self.menuBar().addMenu("&Help")
        shortcuts = QAction("&Shortcuts", self)
        shortcuts.triggered.connect(self.show_shortcuts)
        about = QAction("&About", self)
        about.triggered.connect(self.show_about)
        help_menu.addAction(shortcuts)
        help_menu.addAction(about)

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
        """Add a rail button and its panel; returns the page (header and scrolling content)."""
        page = PanelPage(title, subtitle, widget)
        self._side_stack.addWidget(page)
        button = _RailButton(icon, rail_text or title, tooltip)
        button.clicked.connect(lambda _checked=False, n=name: self._on_rail(n))
        self._rail_layout.insertWidget(self._rail_layout.count() - 1, button)
        self.panels[name] = widget
        self.panel_pages[name] = page
        self._rail_buttons[name] = button
        return page

    def current_panel(self) -> str:
        return self._panel

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
            color, text = tokens["warn"], "Settings changed - process again"
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

    def _summary(self) -> str:
        c = self.controller
        if c.result is None:
            return ""
        fmap = c.result.ratio.to_unit(c.unit)
        b, e = fmap.field, fmap.energy
        return (
            f"{b.size} spectra · B {b.min():.4g} – {b.max():.4g} T · "
            f"E {e.min():.4g} – {e.max():.4g} {c.unit}"
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
        self, title: str, message: str, panel: str | None = None, expected: bool = True
    ) -> None:
        """Log an error; expected ones show in the bar above the plot, others in a dialog."""
        logger.error("%s: %s", title, message)
        if not expected:
            QMessageBox.warning(self, title, message)
            return
        text = message[:1].upper() + message[1:]
        if panel in self.panels:
            title_of = self.panel_pages[panel].title.text()
            self.infobar.show_message(
                "error", f"{title} failed", text, f"Open {title_of}", lambda: self.show_panel(panel)
            )
        else:
            self.infobar.show_message("error", f"{title} failed", text)

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
        QMessageBox.about(
            self,
            "About",
            f"<b>Magneto-Optical Detective {__version__}</b><br>"
            f"Python {platform.python_version()}, Qt {qVersion()}, PySide6 {pyside_version}<br>"
            f"numpy {np.__version__}, scipy {scipy.__version__}, pyqtgraph {pg.__version__}",
        )

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

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._shown = True

    def closeEvent(self, event) -> None:
        self.save_settings()
        logger.removeHandler(self.log_handler)
        super().closeEvent(event)
