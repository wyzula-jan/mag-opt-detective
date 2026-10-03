"""The plot area: Map / Stacked / Reference tabs, the plot toolbar and its tool modes.

Each map's colour scale sits in a :class:`SlidePanel` right of the plot; the plot toolbar
switches the scale style for every map and shows or hides all scales at once. Plot clicks go
to the active tool of :class:`ToolRegistry`, with the keyboard modifiers.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import pyqtgraph as pg
from PySide6.QtCore import QEvent, QObject, QPointF, QRect, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QKeySequence, QPainter, QShortcut
from PySide6.QtWidgets import (
    QButtonGroup,
    QHBoxLayout,
    QLabel,
    QRadioButton,
    QSizePolicy,
    QSplitter,
    QSplitterHandle,
    QStackedWidget,
    QTabBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from mag_opt_detective.core.pipeline import PlotKind
from mag_opt_detective.core.processing import Axis
from mag_opt_detective.core.spectra import save_tsv
from mag_opt_detective.core.units import Unit
from mag_opt_detective.gui import icons
from mag_opt_detective.gui.controller import KIND_LABELS, level_key, level_label, user_action
from mag_opt_detective.gui.kit import SlidePanel
from mag_opt_detective.gui.plots import ColorMapPlot, PlotColors, StackedPlot, robust_levels
from mag_opt_detective.gui.theme import current_tokens
from mag_opt_detective.gui.widgets import IMAGE_FILTER, CheckableSetting, Separator, save_file

logger = logging.getLogger("mag_opt_detective")

VIEWS = ("map", "stacked", "reference")
TAB_TITLES = ("Map", "Stacked", "Reference")
DEFAULT_SCALE_STYLE = "histogram"


@dataclass(frozen=True)
class PlotViews:
    """The three plot views, by attribute or by name."""

    map: ColorMapPlot
    stacked: StackedPlot
    reference: ColorMapPlot

    def __getitem__(self, name: str):
        return getattr(self, name)

    def items(self):
        return [(name, self[name]) for name in VIEWS]


# ---------------------------------------------------------------------- tool modes
@dataclass(frozen=True)
class PlotClick:
    """A click on a plot, in data coordinates (map: field, energy; stacked: energy, y)."""

    view: str
    x: float
    y: float
    modifiers: Qt.KeyboardModifier = Qt.KeyboardModifier.NoModifier
    button: Qt.MouseButton = Qt.MouseButton.LeftButton


@dataclass
class Tool:
    name: str
    button: QToolButton
    views: tuple[str, ...]
    on_activate: Callable[[], object] | None
    on_deactivate: Callable[[], object] | None
    on_click: Callable[[PlotClick], object] | None


class ToolRegistry(QObject):
    """Exclusive tool modes of the plot toolbar; the first one registered is the default.

    Plot clicks are routed to the active tool when it works on that view (:meth:`click`).
    A tool's shortcut toggles between it and the default tool.
    """

    toolChanged = Signal(str)

    def __init__(self, bar: QHBoxLayout, shortcut_parent: QWidget, parent: QObject | None = None):
        super().__init__(parent)
        self._bar = bar
        self._shortcut_parent = shortcut_parent
        self._tools: dict[str, Tool] = {}
        self._active = ""
        self._view = VIEWS[0]

    def register(
        self,
        name: str,
        icon: str,
        tooltip: str,
        shortcut: str | None = None,
        on_activate: Callable[[], object] | None = None,
        on_deactivate: Callable[[], object] | None = None,
        views: Sequence[str] = ("map", "stacked"),
        on_click: Callable[[PlotClick], object] | None = None,
    ) -> QToolButton:
        """Add a tool button; *on_click* receives the :class:`PlotClick` on its *views*."""
        if name in self._tools:
            raise ValueError(f"tool {name!r} is already registered")
        button = QToolButton()
        button.setProperty("kit", "tool")
        button.setCheckable(True)
        button.setIconSize(QSize(16, 16))
        text = f"{tooltip} ({shortcut})" if shortcut else tooltip
        button.setToolTip(text)
        button.setAccessibleName(tooltip)
        icons.set_icon(button, icon, "muted", on_color="accent")
        button.clicked.connect(lambda _checked=False, n=name: self.set_active(n))
        self._bar.addWidget(button)
        tool = Tool(name, button, tuple(views), on_activate, on_deactivate, on_click)
        self._tools[name] = tool
        if shortcut:
            key = QShortcut(QKeySequence(shortcut), self._shortcut_parent)
            key.activated.connect(lambda n=name: self.toggle(n))
        if not self._active:
            self._active = name
            button.setChecked(True)
            if on_activate is not None:
                on_activate()
        self._sync()
        return button

    def names(self) -> list[str]:
        return list(self._tools)

    def tool(self, name: str) -> Tool:
        return self._tools[name]

    def default(self) -> str:
        return next(iter(self._tools), "")

    def active(self) -> str:
        return self._active

    def set_active(self, name: str) -> bool:
        """Make *name* the active tool; False (nothing changes) if it does not work on the
        view shown - show one of its views first."""
        if name not in self._tools:
            raise KeyError(f"no tool named {name!r}")
        if name == self._active or self._view not in self._tools[name].views:
            self._sync()
            return name == self._active
        old = self._tools.get(self._active)
        self._active = name
        if old is not None and old.on_deactivate is not None:
            old.on_deactivate()
        new = self._tools[name]
        if new.on_activate is not None:
            new.on_activate()
        self._sync()
        self.toolChanged.emit(name)
        return True

    def toggle(self, name: str) -> None:
        """Activate *name*, or go back to the default tool if it is active already."""
        if name == self._active and name != self.default():
            self.set_active(self.default())
        else:
            self.set_active(name)

    def view(self) -> str:
        return self._view

    def set_view(self, view: str) -> None:
        """The view on screen: tools for other views are disabled (and left)."""
        self._view = view
        tool = self._tools.get(self._active)
        if tool is not None and view not in tool.views:
            self.set_active(self.default())
        self._sync()

    def click(
        self,
        view: str,
        x: float,
        y: float,
        modifiers: Qt.KeyboardModifier = Qt.KeyboardModifier.NoModifier,
        button: Qt.MouseButton = Qt.MouseButton.LeftButton,
    ) -> bool:
        """Route a plot click to the active tool; True if a tool took it."""
        tool = self._tools.get(self._active)
        if tool is None or tool.on_click is None or view not in tool.views:
            return False
        tool.on_click(PlotClick(view, float(x), float(y), modifiers, button))
        return True

    def _sync(self) -> None:
        for name, tool in self._tools.items():
            tool.button.setChecked(name == self._active)
            tool.button.setEnabled(self._view in tool.views)


# ---------------------------------------------------------------------- widgets
class _Tabs(QTabBar):
    """Flat tabs with an accent underline on the current one (as in the mockup)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDrawBase(False)
        self.setExpanding(False)
        self.setDocumentMode(True)
        self.setUsesScrollButtons(False)

    def tabSizeHint(self, index: int) -> QSize:
        width = self.fontMetrics().horizontalAdvance(self.tabText(index)) + 24
        return QSize(width, 38)

    def minimumTabSizeHint(self, index: int) -> QSize:
        return self.tabSizeHint(index)

    def paintEvent(self, event) -> None:
        tokens = current_tokens()
        painter = QPainter(self)
        for i in range(self.count()):
            rect = self.tabRect(i)
            current = i == self.currentIndex()
            painter.setPen(tokens["fg"] if current else tokens["muted"])
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, self.tabText(i))
            if current:
                painter.fillRect(
                    QRect(rect.left(), rect.bottom() - 1, rect.width(), 2), tokens["accent"]
                )
        painter.end()


class _ToggleHandle(QSplitterHandle):
    """A splitter handle that also opens or closes the SlidePanel after it on a click."""

    def __init__(self, orientation, parent):
        super().__init__(orientation, parent)
        self._press: QPointF | None = None
        self.setToolTip("Click to show or hide the colour scale")

    def mousePressEvent(self, event) -> None:
        self._press = event.position()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        super().mouseReleaseEvent(event)
        press, self._press = self._press, None
        if press is None or (event.position() - press).manhattanLength() > 3:
            return
        splitter = self.splitter()
        panel = splitter.widget(splitter.indexOf(self))
        if isinstance(panel, SlidePanel):
            panel.toggle()

    def paintEvent(self, event) -> None:
        tokens = current_tokens()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(self.rect())
        painter.fillRect(rect, tokens["plot-bg"])
        grip = QRectF(rect.center().x() - 1.5, rect.center().y() - 14, 3, 28)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(tokens["line-strong"])
        painter.drawRoundedRect(grip, 1.5, 1.5)
        painter.end()


class ScaleSplitter(QSplitter):
    """[plot | colour scale panel]; a click on the handle toggles the panel."""

    def __init__(self, parent=None):
        super().__init__(Qt.Orientation.Horizontal, parent)
        self.setHandleWidth(7)
        self.setChildrenCollapsible(True)

    def createHandle(self) -> QSplitterHandle:
        return _ToggleHandle(self.orientation(), self)


class _Overlay(QObject):
    """Keeps a floating child (the error bar) at the top centre of its parent."""

    def __init__(self, child: QWidget, parent: QWidget):
        super().__init__(parent)
        self._child = child
        self._parent = parent
        child.setParent(parent)
        parent.installEventFilter(self)
        child.installEventFilter(self)

    def eventFilter(self, watched, event) -> bool:
        if event.type() in (QEvent.Type.Resize, QEvent.Type.Show, QEvent.Type.LayoutRequest):
            self.place()
        return False

    def place(self) -> None:
        child, area = self._child, self._parent.rect()
        width = max(120, min(560, area.width() - 20))
        height = child.heightForWidth(width) if child.hasHeightForWidth() else -1
        height = max(height, child.sizeHint().height())
        child.setGeometry((area.width() - width) // 2, 10, width, height)
        child.raise_()


def scale_width(plot: ColorMapPlot) -> int:
    """The (fixed) width of the plot's colour scale."""
    widget = plot.scale.widget
    margins = plot.scale_container.contentsMargins()
    width = min(max(widget.minimumWidth(), widget.sizeHint().width()), widget.maximumWidth())
    return width + margins.left() + margins.right()


class PlotArea(QWidget):
    """Tabs and plot toolbar above the three plot views and the error bar."""

    def __init__(self, infobar: QWidget, parent=None):
        super().__init__(parent)
        self.tabs = _Tabs()
        for title in TAB_TITLES:
            self.tabs.addTab(title)
        self.description = QLabel()
        self.description.setProperty("kit", "muted")
        self.description.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.tool_bar = QHBoxLayout()
        self.tool_bar.setSpacing(2)
        self.tools_row = QHBoxLayout()
        self.tools_row.setSpacing(2)
        self.tools_row.addLayout(self.tool_bar)

        head = QWidget()
        head.setFixedHeight(40)
        row = QHBoxLayout(head)
        row.setContentsMargins(4, 0, 8, 0)
        row.setSpacing(10)
        row.addWidget(self.tabs)
        row.addWidget(self.description, stretch=1)
        row.addLayout(self.tools_row)

        self.map = ColorMapPlot()
        self.stacked = StackedPlot()
        self.reference = ColorMapPlot()
        self.map_splitter, self.map_scale = self._with_scale(self.map)
        self.reference_splitter, self.reference_scale = self._with_scale(self.reference)

        reference_page = QWidget()
        ref_layout = QVBoxLayout(reference_page)
        ref_layout.setContentsMargins(0, 0, 0, 0)
        ref_layout.setSpacing(0)
        options = QHBoxLayout()
        options.setContentsMargins(10, 4, 10, 4)
        self.ref_ratio = QRadioButton("Reference R(B)/R(0)")
        self.ref_data = QRadioButton("Reference data")
        self.ref_ratio.setChecked(True)
        self.ref_group = QButtonGroup(self)
        self.ref_group.addButton(self.ref_ratio)
        self.ref_group.addButton(self.ref_data)
        options.addWidget(self.ref_ratio)
        options.addWidget(self.ref_data)
        options.addStretch(1)
        ref_layout.addLayout(options)
        ref_layout.addWidget(self.reference_splitter, stretch=1)

        self.stack = QStackedWidget()
        self.stack.addWidget(self.map_splitter)
        self.stack.addWidget(self.stacked)
        self.stack.addWidget(reference_page)
        self.tabs.currentChanged.connect(self.stack.setCurrentIndex)

        self.plot_box = QWidget()
        box = QVBoxLayout(self.plot_box)
        box.setContentsMargins(0, 0, 0, 0)
        box.addWidget(self.stack)
        self.infobar = infobar
        self._overlay = _Overlay(infobar, self.plot_box)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(head)
        layout.addWidget(Separator())
        layout.addWidget(self.plot_box, stretch=1)

    @staticmethod
    def _with_scale(plot: ColorMapPlot) -> tuple[ScaleSplitter, SlidePanel]:
        splitter = ScaleSplitter()
        splitter.addWidget(plot)
        width = scale_width(plot)
        panel = SlidePanel(plot.scale_container, width)
        panel.setMaximumWidth(width)
        splitter.addWidget(panel)
        splitter.setStretchFactor(0, 1)
        splitter.setCollapsible(0, False)
        panel.openChanged.connect(plot.set_scale_visible)
        return splitter, panel

    def add_tool_button(self, name: str, tooltip: str, checkable: bool = False) -> QToolButton:
        button = QToolButton()
        button.setProperty("kit", "tool")
        button.setCheckable(checkable)
        button.setIconSize(QSize(16, 16))
        button.setToolTip(tooltip)
        button.setAccessibleName(tooltip.split(" (")[0])
        icons.set_icon(button, name, "muted", on_color="accent")
        self.tools_row.addWidget(button)
        return button

    def add_separator(self) -> None:
        line = Separator(Qt.Orientation.Vertical)
        line.setFixedHeight(18)
        self.tools_row.addSpacing(3)
        self.tools_row.addWidget(line)
        self.tools_row.addSpacing(3)

    def current_view(self) -> str:
        return VIEWS[self.tabs.currentIndex()]

    def set_current_view(self, view: str) -> None:
        self.tabs.setCurrentIndex(VIEWS.index(view))

    def scale_panels(self) -> dict[str, SlidePanel]:
        return {"map": self.map_scale, "reference": self.reference_scale}


# ---------------------------------------------------------------------- rendering
def render(window) -> None:
    """Draw the selected map and its stacked spectra in the display unit."""
    c, plots = window.controller, window.plots
    if c.result is None:
        plots.map.clear_map()
        plots.stacked.clear_map()
        return
    fmap = c.current_map()
    view = c.view
    plots.map.set_map(
        fmap,
        levels=c.current_levels(),
        cmap=view.colormap_for(c.selection.order),
        x_range=view.field_range,
        y_range=view.energy_range,
    )
    plots.stacked.set_map(
        fmap, view.stacked_offset, y_range=view.stacked_range, x_range=view.energy_range
    )


def render_reference(window) -> None:
    c, plot = window.controller, window.plots.reference
    fmap = c.reference_map()
    if fmap is None:
        plot.clear_map()
        return
    view = c.view
    plot.set_map(
        fmap,
        levels=c.reference_levels(),
        cmap=view.colormap_for(0),
        x_range=view.field_range,
        y_range=view.energy_range,
    )


# ---------------------------------------------------------------------- actions
@user_action("Plot")
def redraw(window) -> None:
    render(window)
    render_reference(window)
    update_description(window)


def update_description(window) -> None:
    area: PlotArea = window.plot_area
    c = window.controller
    if area.current_view() == "reference":
        kind = c.selection.reference_kind
        area.description.setText(f"Reference {KIND_LABELS[kind]}")
    else:
        area.description.setText(c.description() if c.result is not None else "")
    area.description.setToolTip(area.description.text())


def _store_levels(window, view: str, key: str, lo: float, hi: float) -> None:
    c = window.controller
    if not lo < hi:  # both ends dragged onto one level: draw the kept levels again
        fmap = c.current_map() if view == "map" else c.reference_map()
        levels = c.current_levels() if view == "map" else c.reference_levels()
        window.plots[view].set_levels(*(levels or robust_levels(fmap.values)))
        return
    c.set_levels(key, lo, hi)
    logger.info("Colour range of %s set to %.4g … %.4g", level_label(key), lo, hi)


@user_action("Export")
def export_table(window) -> None:
    c = window.controller
    if c.result is None:
        raise ValueError("nothing to export - process data first")
    fmap = c.current_map()
    name = c.selection.export_name()
    path = save_file(window, "Export current plot")
    if not path:
        return
    out = Path(path)
    if not out.suffix:
        out = out.with_suffix(".csv")
    if window.commands["export_suffix"].isChecked():
        out = out.with_name(f"{out.stem}_{name}{out.suffix}")
    save_tsv(fmap, out)
    logger.info("Exported %s to %s", name, out)


@user_action("Save image")
def save_image(window) -> None:
    if window.controller.result is None:
        raise ValueError("nothing to save - process data first")
    view = window.plots[window.plot_area.current_view()]
    path = save_file(window, "Save plot image", IMAGE_FILTER)
    if not path:
        return
    out = Path(path) if Path(path).suffix else Path(path).with_suffix(".png")
    view.export_image(out)
    logger.info("Saved image %s", out)


def set_scale_style(window, style: str) -> None:
    """Use *style* ("histogram" or "bar") for the colour scale of every map.

    Each scale has a fixed width, which becomes the panel's maximum (the content's is its
    minimum). Once the splitter is laid out the panel also gets that size, open or closed;
    before that (restoring settings) the limits alone size it when it is shown.
    """
    area: PlotArea = window.plot_area
    for name, panel in area.scale_panels().items():
        plot: ColorMapPlot = window.plots[name]
        plot.set_scale_style(style)
        width = scale_width(plot)
        panel.setMaximumWidth(width)
        splitter = panel.parentWidget()
        sizes = splitter.sizes()
        if not (splitter.isVisible() and sum(sizes) > 0):
            continue
        if panel.is_open():
            splitter.setSizes([sum(sizes) - width, width])
        else:
            panel.set_settings_value(json.dumps({"open": False, "size": width}))
    if area.scale_style_button.isChecked() != (style == "bar"):
        area.scale_style_button.setChecked(style == "bar")


def set_scales_open(window, open_: bool) -> None:
    for panel in window.plot_area.scale_panels().values():
        panel.set_open(open_)


def fit_to_data(window) -> None:
    window.controller.fit_ranges(window.plot_area.current_view())  # the View section draws it


def apply_theme(window) -> None:
    colors = PlotColors.from_mapping(window.theme.plot_colors())
    for _name, plot in window.plots.items():
        plot.apply_theme(colors)
    for splitter in (window.plot_area.map_splitter, window.plot_area.reference_splitter):
        for i in range(1, splitter.count()):
            splitter.handle(i).update()


def _cursor_text(view: str, x: float, y: float, value, unit: Unit) -> str:
    if view == "stacked":
        return f"E = {x:.4g} {unit}    I = {y:.4g}"
    text = f"B = {x:.3f} T    E = {y:.4g} {unit}"
    if value is not None and value == value:  # not NaN
        text += f"    {value:.5g}"
    return text


def _connect_clicks(window, view: str) -> None:
    plot = window.plots[view]

    def has_data() -> bool:
        if isinstance(plot, ColorMapPlot):
            return plot.image.image is not None
        return bool(plot.shown_fields().size)

    def on_click(event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        pos = event.scenePos()
        vb = plot.plot.vb
        if not vb.sceneBoundingRect().contains(pos) or not has_data():
            return
        point = vb.mapSceneToView(pos)
        window.tools.click(view, point.x(), point.y(), event.modifiers(), event.button())

    plot.plot.scene().sigMouseClicked.connect(on_click)


def on_escape(window) -> None:
    """Escape closes the error bar, else goes back to the default tool."""
    if not window.infobar.isHidden():
        window.infobar.dismiss()
    else:
        window.tools.set_active(window.tools.default())


def _set_mouse_mode(window, mode) -> None:
    for _name, plot in window.plots.items():
        plot.plot.vb.setMouseMode(mode)


# ---------------------------------------------------------------------- install
def install(window) -> None:
    """Build the plot area into the stage and wire it to the controller and the toolbar."""
    c = window.controller
    area = PlotArea(window.infobar)
    window.plot_area = area
    window.stage.layout().addWidget(area)
    window.plots = PlotViews(area.map, area.stacked, area.reference)

    # tool modes
    window.tools = ToolRegistry(area.tool_bar, window, window)
    window.tools.register(
        "navigate",
        "move",
        "Pan and zoom",
        "V",
        on_activate=lambda: _set_mouse_mode(window, pg.ViewBox.PanMode),
        views=VIEWS,
    )
    window.tools.register(
        "zoom",
        "zoom-in",
        "Box zoom",
        "Z",
        on_activate=lambda: _set_mouse_mode(window, pg.ViewBox.RectMode),
        on_deactivate=lambda: _set_mouse_mode(window, pg.ViewBox.PanMode),
        views=VIEWS,
    )
    for view in VIEWS:
        _connect_clicks(window, view)
    QShortcut(QKeySequence(Qt.Key.Key_Escape), window).activated.connect(lambda: on_escape(window))

    area.add_separator()
    fit = area.add_tool_button("scan", "Fit to data (A)")
    fit.clicked.connect(lambda: fit_to_data(window))
    QShortcut(QKeySequence("A"), window).activated.connect(lambda: fit_to_data(window))
    area.scale_style_button = area.add_tool_button(
        "palette", "Slim colour bars instead of histograms", checkable=True
    )
    area.scales_button = area.add_tool_button(
        "eye", "Show or hide all colour scales", checkable=True
    )
    area.scales_button.setChecked(True)
    area.inspector_button = area.add_tool_button(
        "panel-right", "Show or hide the inspector", checkable=True
    )
    area.inspector_button.setChecked(window.inspector_panel.is_open())
    image = area.add_tool_button("image", "Quick image (PNG/SVG)")
    image.clicked.connect(window.commands["export_image"].trigger)

    area.scale_style_button.toggled.connect(
        lambda bar: set_scale_style(window, "bar" if bar else DEFAULT_SCALE_STYLE)
    )
    area.scales_button.clicked.connect(lambda checked: set_scales_open(window, checked))

    def sync_scales_button(_open=None) -> None:
        all_open = all(p.is_open() for p in area.scale_panels().values())
        area.scales_button.setChecked(all_open)
        icons.set_icon(area.scales_button, "eye" if all_open else "eye-off", "muted", "accent")

    for panel in area.scale_panels().values():
        panel.openChanged.connect(sync_scales_button)
    area.inspector_button.toggled.connect(lambda on: window.inspector_panel.set_open(on))
    window.inspector_panel.openChanged.connect(area.inspector_button.setChecked)

    # tabs
    def on_tab(index: int) -> None:
        view = VIEWS[index]
        window.tools.set_view(view)
        window.set_inspector_title(TAB_TITLES[index])
        update_description(window)

    area.tabs.currentChanged.connect(on_tab)
    on_tab(area.tabs.currentIndex())

    # toolbar selection
    tb = window.toolbar
    tb.kind.valueChanged.connect(lambda v: c.set_selection(kind=PlotKind(v)))
    tb.order.valueChanged.connect(lambda v: c.set_selection(order=int(v)))
    tb.axis.valueChanged.connect(
        lambda v: c.set_selection(axis=Axis.FIELD if v == "B" else Axis.ENERGY)
    )
    tb.per_unit.toggled.connect(lambda on: c.set_selection(physical=on))
    area.ref_group.buttonToggled.connect(
        lambda _b, checked: (
            checked
            and c.set_selection(
                reference_kind=PlotKind.DATA if area.ref_data.isChecked() else PlotKind.RATIO
            )
        )
    )

    def follow_reference_kind() -> None:
        data = c.selection.reference_kind is PlotKind.DATA
        (area.ref_data if data else area.ref_ratio).setChecked(True)

    c.selectionChanged.connect(follow_reference_kind)

    # drawing (a new result also closes an old error bar; before drawing, which may report)
    c.resultChanged.connect(window.infobar.dismiss)
    c.resultChanged.connect(lambda: redraw(window))
    c.selectionChanged.connect(lambda: redraw(window))
    c.viewChanged.connect(lambda: redraw(window))
    c.unitChanged.connect(lambda _old, _new: redraw(window))
    window.plots.map.levelsEdited.connect(
        lambda lo, hi: _store_levels(window, "map", c.selection.level_key, lo, hi)
    )
    window.plots.reference.levelsEdited.connect(
        lambda lo, hi: _store_levels(
            window, "reference", level_key(c.selection.reference_kind), lo, hi
        )
    )

    # cursor read-out
    for view in VIEWS:
        window.plots[view].cursorMoved.connect(
            lambda x, y, value, v=view: window.set_cursor_text(_cursor_text(v, x, y, value, c.unit))
        )

    # exports
    window.commands["export_table"].triggered.connect(lambda: export_table(window))
    window.commands["export_image"].triggered.connect(lambda: save_image(window))

    # theme
    apply_theme(window)
    window.themeChanged.connect(lambda: apply_theme(window))

    # settings
    p = window.persistence
    if p is not None:
        p.bind("plot/scale_style_bar", area.scale_style_button)
        p.bind("plot/reference_ratio", area.ref_ratio)
        p.bind("plot/reference_data", area.ref_data)
        p.bind("export/type_suffix", CheckableSetting(window.commands["export_suffix"]))
    window.add_splitter("map_scale", area.map_splitter)
    window.add_splitter("reference_scale", area.reference_splitter)
