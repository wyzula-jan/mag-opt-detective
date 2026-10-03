"""The export window: a journal figure of the plot on screen, with a live preview.

Left, the figure at its true proportions, drawn by matplotlib at screen resolution again about
150 ms after any change, with its print size under it; right, the journal preset (or one of the
user's own presets, :mod:`~mag_opt_detective.gui.export_presets`), the size, text and lines, the
file format, what the figure includes, its colour range and colour bar, the ticks and the
labels. Problems show inline under the preview, never as dialogs. While it is open the window
follows the main window (unit, plot, ranges, colours, models, points), and it remembers its
choices (``export/figure``).
"""

from __future__ import annotations

import logging
import math
from contextlib import contextmanager
from pathlib import Path

from PySide6.QtCore import QEvent, QRectF, QSignalBlocker, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QImage, QKeySequence, QPainter, QPen, QShortcut
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from mag_opt_detective.core.units import Unit
from mag_opt_detective.export import PRESETS, FigureStyle, JournalPreset, get_preset
from mag_opt_detective.export.style import COLORBAR_LOCATIONS, MINOR_INTERVALS, TICK_DIRECTIONS
from mag_opt_detective.export.user_presets import COLOUR_RANGES, UserPreset
from mag_opt_detective.gui import icons
from mag_opt_detective.gui.controller import level_label, parse_level_key
from mag_opt_detective.gui.export_menu import ExportSettings, PresetStore
from mag_opt_detective.gui.export_presets import MyPresets
from mag_opt_detective.gui.export_render import FigureJob, FigureRenderer
from mag_opt_detective.gui.export_state import (
    FORMAT_LABELS,
    KINDS,
    POINTS_ALL,
    POINTS_CURRENT,
    POINTS_NONE,
    PRESET_NAMES,
    RANGE_AUTO,
    RANGE_FIXED,
    RANGE_NAMES,
    RANGE_WINDOW,
    RASTER_FORMATS,
    FigureContent,
    PrintSize,
    StyleCheck,
    auto_labels,
    auto_tick_sizes,
    canonical_levels,
    display_levels,
    figure_state,
    figure_style,
    file_name,
    levels_problem,
    map_levels,
    model_curves,
    point_sets,
    print_size,
    with_format,
)
from mag_opt_detective.gui.kit import SegmentedControl
from mag_opt_detective.gui.panels.common import (
    Note,
    SpinBox,
    SwitchRow,
    UnitField,
    block,
    hint,
    labelled,
    scaled_font,
    section_label,
)
from mag_opt_detective.gui.theme import current_tokens
from mag_opt_detective.gui.widgets import FloatEdit, Separator, last_dir, parse_float, set_last_dir

logger = logging.getLogger("mag_opt_detective")

DEBOUNCE_MS = 150
SETTINGS_WIDTH = 352
PANEL_LETTERS = ("a", "b", "c", "d", "e", "f")
NO_PANEL = "none"
VIEW_NAMES = {"map": "Map", "stacked": "Stacked"}
FILTERS = {
    "pdf": "PDF document (*.pdf)",
    "svg": "SVG image (*.svg)",
    "eps": "EPS file (*.eps)",
    "png": "PNG image (*.png)",
    "tif": "TIFF image (*.tif *.tiff)",
}
COLUMN_NAMES = {
    "single": "Single column",
    "double": "Double column",
    "1.5 narrow": "1.5 columns, narrow",
    "1.5 wide": "1.5 columns, wide",
}
MAX_PREVIEW_DPI = 600.0
RANGE_TIPS = {
    RANGE_WINDOW: "The levels of the main window, as they change",
    RANGE_AUTO: "1st–99th percentile of the whole map, as the window's Auto",
    RANGE_FIXED: "Levels typed here, remembered for each plot",
}
RANGE_NOTES = {
    RANGE_WINDOW: "The levels of the main window, as they change.",
    RANGE_AUTO: "1–99 % of the whole map, as the window's Auto.",
}
TICK_OPTIONS = {"in": ("In", "Ticks point into the plot"), "out": ("Out", "Ticks point outwards")}
LOCATION_OPTIONS = {
    "right": ("Right", "A vertical bar right of the plot"),
    "top": ("Top", "A horizontal bar above the plot, its label on top"),
}


def column_name(preset: JournalPreset, key: str) -> str:
    """What a preset's column width is called, e.g. "Single column · 8.6 cm" for APS."""
    text = COLUMN_NAMES.get(key, key.capitalize())
    return f"{text} · {preset.widths_mm[key] / 10:g} cm" if preset.key == "aps" else text


def number_field(unit: str, name: str) -> UnitField:
    edit = FloatEdit(name=name)
    edit.setAccessibleName(name)
    return UnitField(edit, unit)


def number(field: UnitField) -> float | None:
    return parse_float(field.edit.text())


def _number_text(value: float | None, digits: int = 6) -> str:
    return "" if value is None else f"{value:.{digits}g}"


def _row(
    *items: tuple[QWidget, int], spacing: int = 8, align=Qt.AlignmentFlag.AlignBottom
) -> QWidget:
    """Widgets side by side (with their stretch factors), their bottoms in line."""
    box = QWidget()
    layout = QHBoxLayout(box)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(spacing)
    for widget, stretch in items:
        layout.addWidget(widget, stretch, align)
    return box


def _stored_levels(value) -> tuple[float, float] | None:
    """A stored ``[lo, hi]`` pair, if it can be drawn."""
    if not isinstance(value, list | tuple) or len(value) != 2:
        return None
    if any(isinstance(v, bool) or not isinstance(v, int | float) for v in value):
        return None
    lo, hi = float(value[0]), float(value[1])
    return (lo, hi) if math.isfinite(lo) and math.isfinite(hi) and lo < hi else None


# ---------------------------------------------------------------------- preview
class FigurePreview(QWidget):
    """The figure at its true proportions on a sunken background (white paper, thin edge),
    with its print size under it.

    Until a new drawing arrives the last one is shown; "Drawing…" appears when one takes long.
    """

    MARGIN = 22
    BUSY_DELAY_MS = 200

    def __init__(self, parent=None):
        super().__init__(parent)
        self._image: QImage | None = None
        self._size_mm = (89.0, 70.0)
        self._caption = ""
        self._message = ""
        self._busy = False
        self._show_busy = False
        self._busy_timer = QTimer(self)
        self._busy_timer.setSingleShot(True)
        self._busy_timer.setInterval(self.BUSY_DELAY_MS)
        self._busy_timer.timeout.connect(self._busy_shown)
        self.setMinimumSize(300, 240)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setAccessibleName("Figure preview")

    def image(self) -> QImage | None:
        return self._image

    def message(self) -> str:
        return self._message

    def caption(self) -> str:
        return self._caption

    def set_size_mm(self, width: float, height: float, caption: str = "") -> None:
        self._size_mm = (width, height)
        self._caption = caption
        self.update()

    def set_image(self, image: QImage) -> None:
        self._image = image
        self._message = ""
        self.update()

    def set_message(self, text: str) -> None:
        """Show *text* (a problem) instead of the figure; "" shows the figure again."""
        self._message = text
        self.update()

    def set_busy(self, busy: bool) -> None:
        self._busy = busy
        if busy and not self._busy_timer.isActive() and not self._show_busy:
            self._busy_timer.start()
        if not busy:
            self._busy_timer.stop()
            self._show_busy = False
        self.update()

    def busy_shown(self) -> bool:
        return self._show_busy

    def _busy_shown(self) -> None:
        self._show_busy = self._busy
        self.update()

    def _area(self) -> QRectF:
        """Room for the page: the widget less the margins and the caption line."""
        m = self.MARGIN
        caption = self.fontMetrics().height() + 8
        return QRectF(self.rect()).adjusted(m, m, -m, -m - caption)

    def page_rect(self) -> QRectF:
        """Where the figure goes: its aspect fitted into the room, centred."""
        width, height = self._size_mm
        if self._image is not None and not self._image.isNull():
            width, height = self._image.width(), self._image.height()  # until redrawn
        area = self._area()
        if area.width() <= 0 or area.height() <= 0 or width <= 0 or height <= 0:
            return QRectF()
        scale = min(area.width() / width, area.height() / height)
        w, h = width * scale, height * scale
        return QRectF(area.center().x() - w / 2, area.center().y() - h / 2, w, h)

    def wanted_dpi(self) -> float:
        """The dpi (in logical pixels) at which a figure of the current size fills the room."""
        width_mm, height_mm = self._size_mm
        area = self._area()
        if area.width() <= 0 or area.height() <= 0:
            return 96.0
        scale = min(area.width() / width_mm, area.height() / height_mm)  # px per mm
        return max(20.0, min(scale * 25.4, MAX_PREVIEW_DPI))

    def sizeHint(self) -> QSize:
        return QSize(640, 520)

    def paintEvent(self, event) -> None:
        tokens = current_tokens()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.fillRect(self.rect(), tokens["sunken"])
        if self._message:
            painter.setPen(tokens["err"])
            flags = Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap
            painter.drawText(self.rect().adjusted(40, 40, -40, -40), int(flags), self._message)
            painter.end()
            return
        page = self.page_rect()
        if not page.isEmpty():
            halo = QColor(tokens["line"])
            for grow, alpha in ((4.0, 0.35), (2.0, 0.6)):
                halo.setAlphaF(alpha)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(halo)
                painter.drawRoundedRect(page.adjusted(-grow, -grow, grow, grow), grow, grow)
            painter.fillRect(page, QColor("white"))  # the paper: a data colour in every theme
            if self._image is not None:
                painter.drawImage(page, self._image)
            painter.setPen(QPen(tokens["line-strong"], 1))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(page.adjusted(-0.5, -0.5, 0.5, 0.5))
            if self._caption:
                below = QRectF(0, page.bottom() + 8, self.width(), self.fontMetrics().height())
                painter.setPen(tokens["muted"])
                painter.drawText(below, Qt.AlignmentFlag.AlignCenter, self._caption)
        if self._show_busy:
            self._paint_chip(painter, tokens, "Drawing…")
        painter.end()

    def _paint_chip(self, painter: QPainter, tokens, text: str) -> None:
        font = scaled_font(self, 0.92)
        painter.setFont(font)
        width = painter.fontMetrics().horizontalAdvance(text) + 20
        rect = QRectF(10, 10, width, painter.fontMetrics().height() + 10)
        painter.setPen(QPen(tokens["line-strong"], 1))
        painter.setBrush(tokens["win"])
        painter.drawRoundedRect(rect, 7, 7)
        painter.setPen(tokens["muted"])
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)


class Messages(QWidget):
    """Lines of text with a status icon (errors, warnings, the last save), newest last."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(3)
        self._shown: list[tuple[str, str]] = []

    def items(self) -> list[tuple[str, str]]:
        """``(level, text)`` of the lines shown."""
        return list(self._shown)

    def texts(self, level: str | None = None) -> list[str]:
        return [text for lvl, text in self._shown if level is None or lvl == level]

    def set_items(self, items: list[tuple[str, str]]) -> None:
        if items == self._shown:
            return
        self._shown = list(items)
        while self._layout.count():
            widget = self._layout.takeAt(0).widget()
            if widget is not None:
                widget.hide()
                widget.deleteLater()
        for level, text in items:
            self._layout.addWidget(Note(text, level))
        self.setVisible(bool(items))


# ---------------------------------------------------------------------- the window
class ExportDialog(QDialog):
    """Journal figure export of the main window's plot (non-modal; follows the window)."""

    previewUpdated = Signal()
    figureSaved = Signal(str)

    def __init__(self, window, settings: ExportSettings, presets: PresetStore | None = None):
        super().__init__(window)
        self.main_window = window
        self.controller = window.controller
        self.settings = settings
        self.preset_store = PresetStore() if presets is None else presets
        self._quiet = 0
        self._labels = {kind: {"x": "", "y": "", "colorbar": ""} for kind in KINDS}
        self._kind = "map"
        self._preview_error = ""
        self._note: tuple[str, str] | None = None  # the last save or preset action
        self._snapshot_counts = (0, 0, 0)  # model curves, point curves, points
        self._fixed: dict[str, tuple[float, float]] = {}  # level key -> levels (cm^-1 based)
        self._shown_levels: tuple | None = None  # (key, unit) of the fixed levels shown

        self.setWindowTitle("Journal figure")
        self.setWindowFlag(Qt.WindowType.WindowContextHelpButtonHint, False)
        self.setSizeGripEnabled(True)
        self.setModal(False)
        self.resize(1080, 720)
        self.setMinimumSize(860, 560)

        self.renderer = FigureRenderer(self)
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(DEBOUNCE_MS)
        self._debounce.timeout.connect(self._draw_preview)

        self._build()
        self._wire()
        self.my_presets.load()
        self._apply_stored()
        listener = self._apply_stored
        settings.listeners.append(listener)
        self.destroyed.connect(lambda: _forget(settings, listener))

    # ------------------------------------------------------------------ building
    def _build(self) -> None:
        # journal
        self.preset = SegmentedControl(size="sm", expand=True)
        for key, preset in PRESETS.items():
            self.preset.add_option(key, PRESET_NAMES.get(key, preset.name), preset.name)
        self.preset.setAccessibleName("Journal")
        self.my_presets = MyPresets(self, self.preset_store)
        self.notes = hint()
        self.notes.setFont(scaled_font(self.notes, 0.9))

        # size
        self.width_stack = QStackedWidget()
        self.widths: dict[str, SegmentedControl] = {}
        self.custom_width = number_field("mm", "Width")
        for key, preset in PRESETS.items():
            if preset.free_size:
                self.width_stack.addWidget(self.custom_width)
                continue
            control = SegmentedControl(size="sm", expand=True)
            for column, mm in preset.widths_mm.items():
                control.add_option(column, f"{mm:g}", f"{column_name(preset, column)}, {mm:g} mm")
            control.setAccessibleName(f"{preset.name} width")
            self.widths[key] = control
            self.width_stack.addWidget(control)
        self.width_stack.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.width_caption = hint()
        self.height_field = number_field("mm", "Height")
        self.font_field = number_field("pt", "Text size")
        self.line_field = number_field("pt", "Line width")

        # file
        self.format = SegmentedControl(size="sm", expand=True)
        for fmt, text in FORMAT_LABELS.items():
            kind = "raster image" if fmt in RASTER_FORMATS else "vector, editable text"
            self.format.add_option(fmt, text, f"{text} ({kind})")
        self.format.setAccessibleName("File format")
        self.dpi_field = number_field("dpi", "Resolution")
        self.dpi_hint = hint()

        # content
        self.view = SegmentedControl(size="sm", expand=True)
        for kind in KINDS:
            self.view.add_option(kind, VIEW_NAMES[kind], f"The {VIEW_NAMES[kind]} plot")
        self.view.setAccessibleName("View")
        self.models_row = SwitchRow("Model curves", "")
        self.models = self.models_row.switch
        self._models_wanted = True  # the choice, kept while the switch is off and disabled
        self.models.toggled.connect(self._on_models)
        self.points_row = SwitchRow("Picked points", "")
        self.points = self.points_row.switch
        self.points_scope = SegmentedControl(size="sm", expand=True)
        self.points_scope.add_option(POINTS_CURRENT, "Current curve", "Only the current curve")
        self.points_scope.add_option(POINTS_ALL, "All curves", "Every curve with points")
        self.points_scope.set_value(POINTS_ALL)
        self.points_scope.setAccessibleName("Picked points of")
        self.panel_label = SegmentedControl(size="xs", expand=True)
        self.panel_label.add_option(NO_PANEL, "None", "No panel label")
        for letter in PANEL_LETTERS:
            self.panel_label.add_option(letter, letter, f"Panel {letter}: bold, top left")
        self.panel_label.setAccessibleName("Panel label")

        # colour: the map's colour range and the colour bar
        self.levels_mode = SegmentedControl(size="sm", expand=True)
        for mode in COLOUR_RANGES:
            self.levels_mode.add_option(mode, RANGE_NAMES[mode], RANGE_TIPS[mode])
        self.levels_mode.setAccessibleName("Colour range")
        self.level_lo = number_field("", "Colour range minimum")
        self.level_hi = number_field("", "Colour range maximum")
        dash = QLabel("–")
        dash.setProperty("kit", "muted")
        self.levels_note = hint()
        self.levels_note.setFont(scaled_font(self.levels_note, 0.9))
        self.colorbar_row = SwitchRow("Colour bar", "")
        self.colorbar = self.colorbar_row.switch
        self.colorbar_position = SegmentedControl(size="sm")
        for location in COLORBAR_LOCATIONS:
            self.colorbar_position.add_option(location, *LOCATION_OPTIONS[location])
        self.colorbar_position.setAccessibleName("Colour bar position")
        self.colorbar_label = QLineEdit()
        self.colorbar_label.setAccessibleName("Colour bar label")

        # ticks
        self.tick_direction = SegmentedControl(size="sm", expand=True)
        for direction in TICK_DIRECTIONS:
            self.tick_direction.add_option(direction, *TICK_OPTIONS[direction])
        self.tick_direction.set_value("out")
        self.tick_direction.setAccessibleName("Tick direction")
        self.tick_length = number_field("pt", "Tick length")
        self.tick_width = number_field("pt", "Tick width")
        self.mirror_row = SwitchRow("All four sides", "Ticks on the top and right axes too")
        self.tick_mirror = self.mirror_row.switch
        self.minor_row = SwitchRow("Minor ticks", "Unlabelled ticks between the labelled ones")
        self.minor_ticks = self.minor_row.switch
        self.minor_intervals = SpinBox(*MINOR_INTERVALS, 2)
        self.minor_intervals.setAccessibleName("Minor intervals")
        self.minor_intervals.setToolTip("Into how many parts the minor ticks split each step")
        self.minor_length = number_field("pt", "Minor tick length")
        self.minor_intervals.setFixedHeight(self.minor_length.sizeHint().height())

        # labels
        self.x_label = QLineEdit()
        self.x_label.setAccessibleName("Horizontal axis label")
        self.y_label = QLineEdit()
        self.y_label.setAccessibleName("Vertical axis label")
        for edit in (self.x_label, self.y_label, self.colorbar_label):
            edit.setToolTip(
                "Empty: the automatic label. Only spaces: no label. Math in $…$, e.g. cm$^{-1}$."
            )

        # preview
        self.preview = FigurePreview()

        # settings column
        column = QWidget()
        layout = QVBoxLayout(column)
        layout.setContentsMargins(14, 12, 14, 16)
        layout.setSpacing(16)
        title = QLabel("Journal figure")
        title.setFont(scaled_font(title, 1.08, bold=True))
        subtitle = hint("The plot on screen at print size")
        layout.addWidget(block(title, subtitle, spacing=2))
        journal = _row((self.preset, 1), (self.my_presets.button, 0))
        layout.addWidget(
            block(section_label("Journal"), journal, self.my_presets.naming_box, self.notes)
        )
        sizes = _row(
            (labelled("Height", self.height_field), 1),
            (labelled("Text", self.font_field), 1),
            (labelled("Lines", self.line_field), 1),
        )
        width_box = block(labelled("Width (mm)", self.width_stack), self.width_caption, spacing=4)
        layout.addWidget(block(section_label("Size"), width_box, sizes))
        self.dpi_field.setFixedWidth(104)
        dpi_row = QWidget()
        dpi_line = QHBoxLayout(dpi_row)
        dpi_line.setContentsMargins(0, 0, 0, 0)
        dpi_line.setSpacing(10)
        dpi_line.addWidget(labelled("Resolution", self.dpi_field))
        dpi_line.addWidget(self.dpi_hint, 1, Qt.AlignmentFlag.AlignBottom)
        layout.addWidget(block(section_label("File"), self.format, dpi_row))
        panel_box = labelled("Panel label", self.panel_label)
        self.panel_caption = panel_box.findChild(QLabel)  # shows the preset's style
        self.points_box = block(self.points_row, self.points_scope, spacing=6)
        layout.addWidget(
            block(
                section_label("Content"),
                labelled("View", self.view),
                self.models_row,
                self.points_box,
                panel_box,
                spacing=12,
            )
        )
        levels_row = _row(
            (self.level_lo, 1),
            (dash, 0),
            (self.level_hi, 1),
            spacing=6,
            align=Qt.AlignmentFlag.AlignVCenter,
        )
        self.levels_box = block(
            labelled("Colour range", self.levels_mode), levels_row, self.levels_note, spacing=6
        )
        bar_row = _row(
            (labelled("Position", self.colorbar_position), 0),
            (labelled("Label", self.colorbar_label), 1),
            spacing=10,
        )
        self.colorbar_box = block(self.colorbar_row, bar_row, spacing=8)
        layout.addWidget(
            block(section_label("Colour"), self.levels_box, self.colorbar_box, spacing=12)
        )
        tick_sizes = _row(
            (labelled("Direction", self.tick_direction), 1),
            (labelled("Length", self.tick_length), 1),
            (labelled("Width", self.tick_width), 1),
        )
        minor_sizes = _row(
            (labelled("Intervals", self.minor_intervals), 1),
            (labelled("Length", self.minor_length), 1),
            (QWidget(), 1),
        )
        self.minor_box = block(self.minor_row, minor_sizes, spacing=8)
        layout.addWidget(
            block(
                section_label("Ticks"),
                tick_sizes,
                self.mirror_row,
                self.minor_box,
                hint("Empty lengths and widths follow the text size and the line width."),
                spacing=10,
            )
        )
        layout.addWidget(
            block(
                section_label("Axis labels"),
                labelled("Horizontal", self.x_label),
                labelled("Vertical", self.y_label),
                hint("Empty fields use the automatic labels."),
            )
        )
        layout.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(column)
        scroll.setFixedWidth(SETTINGS_WIDTH)
        self.settings_scroll = scroll

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        body.addWidget(self.preview, stretch=1)
        body.addWidget(Separator(Qt.Orientation.Vertical))
        body.addWidget(scroll)

        # footer
        self.messages = Messages()
        self.close_button = QPushButton("Close")
        self.close_button.setProperty("kit", "button")
        self.close_button.setAutoDefault(False)
        self.save_button = QPushButton("Save…")
        self.save_button.setProperty("kit", "primary")
        self.save_button.setAutoDefault(False)
        self.save_button.setToolTip("Choose a file and save the figure (Ctrl+S)")
        icons.set_icon(self.save_button, "save", "accent-fg")
        footer = QHBoxLayout()
        footer.setContentsMargins(14, 10, 14, 10)
        footer.setSpacing(8)
        footer.addWidget(self.messages, 1)
        footer.addWidget(self.close_button, 0, Qt.AlignmentFlag.AlignBottom)
        footer.addWidget(self.save_button, 0, Qt.AlignmentFlag.AlignBottom)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addLayout(body, stretch=1)
        root.addWidget(Separator())
        root.addLayout(footer)

    def _wire(self) -> None:
        self.preset.valueChanged.connect(self._on_preset)
        for control in self.widths.values():
            control.valueChanged.connect(self._changed)
        for field in self._number_fields().values():
            field.edit.textChanged.connect(self._changed)
        self.format.valueChanged.connect(self._changed)
        self.view.valueChanged.connect(self._on_view)
        for switch in (self.colorbar, self.models, self.points, self.tick_mirror, self.minor_ticks):
            switch.toggled.connect(self._changed)
        for control in (
            self.points_scope,
            self.panel_label,
            self.colorbar_position,
            self.tick_direction,
        ):
            control.valueChanged.connect(self._changed)
        self.levels_mode.valueChanged.connect(self._on_colour_range)
        for field in (self.level_lo, self.level_hi):
            field.edit.textChanged.connect(self._on_level_typed)
        for field in self._tick_fields().values():
            field.edit.textChanged.connect(self._changed)
        self.minor_intervals.valueChanged.connect(self._changed)
        for edit in (self.x_label, self.y_label, self.colorbar_label):
            edit.textChanged.connect(self._changed)
        self.close_button.clicked.connect(self.close)
        self.save_button.clicked.connect(self.save)
        QShortcut(QKeySequence(QKeySequence.StandardKey.Save), self).activated.connect(self.save)

        self.renderer.previewReady.connect(self._on_preview)
        self.renderer.saved.connect(self._on_saved)
        self.renderer.failed.connect(self._on_failed)
        self.renderer.busyChanged.connect(self._on_busy)

        c = self.controller
        for signal in (
            c.resultChanged,
            c.viewChanged,
            c.rangesChanged,
            c.selectionChanged,
            c.pointsChanged,
            c.overlaysChanged,
        ):
            signal.connect(self._follow_window)
        c.unitChanged.connect(self._follow_window)
        self.main_window.themeChanged.connect(self.update)

    def _number_fields(self) -> dict[str, UnitField]:
        return {
            "width": self.custom_width,
            "height": self.height_field,
            "font": self.font_field,
            "line": self.line_field,
            "dpi": self.dpi_field,
        }

    def _tick_fields(self) -> dict[str, UnitField]:
        return {
            "tick_length": self.tick_length,
            "tick_width": self.tick_width,
            "minor_length": self.minor_length,
        }

    @contextmanager
    def _quietly(self):
        """Change several controls as one change (one check, one drawing)."""
        self._quiet += 1
        try:
            yield
        finally:
            self._quiet -= 1

    # ------------------------------------------------------------------ state
    def current_preset(self) -> JournalPreset:
        return get_preset(self.preset.value())

    def width_key(self) -> str:
        """The preset column chosen ("" for the free width of Custom)."""
        control = self.widths.get(self.preset.value())
        return control.value() if control is not None else ""

    def print_size(self) -> PrintSize:
        preset = self.current_preset()
        key = self.width_key()
        width = preset.widths_mm[key] if key else number(self.custom_width)
        return print_size(
            preset,
            width,
            number(self.height_field),
            number(self.font_field),
            number(self.line_field),
            number(self.dpi_field),
            self.format.value(),
        )

    def style_check(self) -> StyleCheck:
        """The colour bar place and the ticks as typed, with their errors."""
        return figure_style(
            colorbar_location=self.colorbar_position.value(),
            direction=self.tick_direction.value(),
            mirror=self.tick_mirror.isChecked(),
            length=self.tick_length.edit.text(),
            width=self.tick_width.edit.text(),
            minor=self.minor_ticks.isChecked(),
            minor_intervals=self.minor_intervals.value(),
            minor_length=self.minor_length.edit.text(),
        )

    def current_style(self) -> FigureStyle:
        return self.style_check().style

    def level_key(self) -> str:
        """The level key of the map on screen (plot kind, derivative, axis, per unit)."""
        return self.controller.selection.level_key

    def fixed_levels(self) -> tuple[float, float] | None:
        """The fixed colour range kept for the map on screen, in the display unit."""
        key = self.level_key()
        kept = self._fixed.get(key)
        return None if kept is None else display_levels(key, kept, Unit(self.controller.unit))

    def colour_range_problem(self) -> str:
        """Why the typed colour range cannot be used ("" if it can, or does not apply)."""
        if self.view.value() != "map" or self.levels_mode.value() != RANGE_FIXED:
            return ""
        return levels_problem(number(self.level_lo), number(self.level_hi))

    def problems(self) -> list[str]:
        """What keeps the figure from being drawn and saved."""
        problems = self.print_size().errors + self.style_check().errors
        if self.colour_range_problem():
            problems.append(self.colour_range_problem())
        return problems

    def _on_models(self, checked: bool) -> None:
        if self.models.isEnabled():  # the user's choice (not the switch turned off with it)
            self._models_wanted = checked

    def content(self) -> FigureContent:
        points = self.points_scope.value() if self.points.isChecked() else POINTS_NONE
        return FigureContent(
            kind=self.view.value(),
            colorbar=self.colorbar.isChecked(),
            colorbar_label=self.colorbar_label.text(),
            models=self.models.isChecked(),
            points=points,
            x_label=self.x_label.text(),
            y_label=self.y_label.text(),
            colour_range=self.levels_mode.value(),
            fixed_levels=self.fixed_levels(),
        )

    def panel_letter(self) -> str | None:
        value = self.panel_label.value()
        return None if value == NO_PANEL else value

    def figure_state(self):
        """What the figure shows now (export FigureState), as it would be saved."""
        return figure_state(self.controller, self.content())

    def job(self, dpi: float, state=None) -> FigureJob:
        problems = self.problems()
        if problems:
            raise ValueError(problems[0])
        size = self.print_size()
        return FigureJob(
            state=self.figure_state() if state is None else state,
            preset=self.current_preset(),
            width_mm=size.width_mm,
            height_mm=size.height_mm,
            font_pt=size.font_pt,
            line_pt=size.line_pt,
            panel_label=self.panel_letter(),
            dpi=dpi,
            style=self.current_style(),
        )

    def default_file_name(self) -> str:
        return file_name(
            self.controller,
            self.view.value(),
            self.current_preset(),
            self.width_key(),
            self.format.value(),
        )

    # ------------------------------------------------------------------ colour range
    def _snapshot(self):
        """The window's figure snapshot, or None without data."""
        try:
            return self.controller.figure_state()
        except ValueError:
            return None

    def _start_fixed(self) -> None:
        """In Fixed, a plot without fixed levels gets the window's levels (and keeps them, as
        typed ones). Called where the mode, the settings or the window's plot change."""
        key = self.level_key()
        if self.levels_mode.value() != RANGE_FIXED or key in self._fixed:
            return
        snapshot = self._snapshot()
        if snapshot is None:
            return
        unit = Unit(self.controller.unit)
        self._fixed[key] = canonical_levels(key, tuple(snapshot.levels), unit)
        self.settings.update(fixed_levels=self._stored_fixed())

    def _stored_fixed(self) -> dict[str, list[float]]:
        return {key: [lo, hi] for key, (lo, hi) in self._fixed.items()}

    def _sync_level_fields(self, snapshot=None) -> None:
        """Show the colour range in the fields: the fixed levels of the plot on screen (again
        when the plot or the unit changed), else the levels Window or Auto draw with."""
        mode = self.levels_mode.value()
        key, unit = self.level_key(), Unit(self.controller.unit)
        if mode == RANGE_FIXED:
            if self._shown_levels == (key, unit):
                return
            levels = self.fixed_levels()
            if levels is None:
                return
            texts = [_number_text(v) for v in levels]
            self._shown_levels = (key, unit)
        else:
            self._shown_levels = None
            if snapshot is None:
                return
            levels = map_levels(snapshot, FigureContent(colour_range=mode))
            texts = [_number_text(v, 4) for v in levels]
        with self._quietly():
            self.level_lo.edit.setText(texts[0])
            self.level_hi.edit.setText(texts[1])

    def _on_colour_range(self, _mode: str) -> None:
        if self._quiet:
            return
        self._shown_levels = None
        self._changed()

    def _on_level_typed(self, _text: str = "") -> None:
        if self._quiet or self.levels_mode.value() != RANGE_FIXED:
            return
        lo, hi = number(self.level_lo), number(self.level_hi)
        if not levels_problem(lo, hi):
            key, unit = self.level_key(), Unit(self.controller.unit)
            self._fixed[key] = canonical_levels(key, (lo, hi), unit)
            self._shown_levels = (key, unit)
        self._changed()

    # ------------------------------------------------------------------ user presets
    def current_user_preset(self, name: str) -> UserPreset:
        """The window's style settings as a preset called *name*; ValueError if they cannot be
        kept (a field with an error, or no name)."""
        problems = self.print_size().errors + self.style_check().errors
        if problems:
            raise ValueError(f"fix the settings first: {problems[0]}")
        preset = self.current_preset()
        return UserPreset(
            name=name,
            journal=preset.key,
            width=self.width_key(),
            width_mm=number(self.custom_width) if preset.free_size else None,
            height_mm=number(self.height_field),
            font_pt=number(self.font_field),
            line_pt=number(self.line_field),
            format=self.format.value(),
            dpi=number(self.dpi_field),
            colorbar=self.colorbar.isChecked(),
            style=self.current_style(),
            colour_range=self.levels_mode.value(),
        )

    def show_user_preset(self, preset: UserPreset) -> None:
        """Show the settings of a user's preset (its content settings stay)."""
        with self._quietly():
            self.preset.set_value(preset.journal)
            self._set_preset_defaults(preset.journal)
            control = self.widths.get(preset.journal)
            if control is not None:
                control.set_value(preset.width)
            if preset.width_mm is not None:
                self.custom_width.edit.setText(f"{preset.width_mm:g}")
            for field, value in (
                (self.height_field, preset.height_mm),
                (self.font_field, preset.font_pt),
                (self.line_field, preset.line_pt),
                (self.dpi_field, preset.dpi),
            ):
                field.edit.setText(f"{value:g}")
            self.format.set_value(preset.format)
            self.colorbar.setChecked(preset.colorbar)
            self._show_style(preset.style)
            self.levels_mode.set_value(preset.colour_range)
        self._shown_levels = None
        self._changed()

    def show_note(self, level: str, text: str) -> None:
        """Show *text* (the last save or preset action) in the messages, until a change."""
        self._note = (level, text)
        self._update_form()

    # ------------------------------------------------------------------ opening
    def open_for_window(self) -> None:
        """Show the window for the plot on screen (raise it if it is open already)."""
        if not self.isVisible():
            w = self.main_window
            with self._quietly():
                tab = w.plot_area.current_view()
                self.view.set_value("stacked" if tab == "stacked" else "map")
                points = getattr(w.panels.get("points"), "markers", None)
                markers = points.value() if points is not None else POINTS_ALL
                self.points.setChecked(markers != "hidden")
                if markers in (POINTS_CURRENT, POINTS_ALL):
                    self.points_scope.set_value(markers)
                self.models.setChecked(True)
                self._models_wanted = True
            self._changed()
        self.show()
        self.raise_()
        self.activateWindow()

    def _apply_stored(self) -> None:
        """Show the remembered choices (checked against the presets)."""
        v = dict(self.settings.values)
        with self._quietly():
            key = v["preset"] if v["preset"] in PRESETS else "nature"
            self.preset.set_value(key)
            self._set_preset_defaults(key)
            control = self.widths.get(key)
            if control is not None and v["width"] in control.options():
                control.set_value(v["width"])
            for name, field in self._number_fields().items():
                stored = v["custom_width" if name == "width" else name]
                if stored is not None:
                    field.edit.setText(f"{stored:g}")
            if v["format"] in FORMAT_LABELS:
                self.format.set_value(v["format"])
            self.colorbar.setChecked(bool(v["colorbar"]))
            self._labels["map"]["colorbar"] = str(v["colorbar_label"])
            if self._kind == "map":
                self.colorbar_label.setText(str(v["colorbar_label"]))
            letter = v["panel_label"]
            self.panel_label.set_value(letter if letter in PANEL_LETTERS else NO_PANEL)
            location = v["colorbar_location"]
            self.colorbar_position.set_value(
                location if location in COLORBAR_LOCATIONS else "right"
            )
            mode = v["colour_range"]
            self.levels_mode.set_value(mode if mode in COLOUR_RANGES else RANGE_WINDOW)
            self._fixed = {}
            for level_key, pair in v["fixed_levels"].items():
                levels = _stored_levels(pair)
                if levels is not None and _is_level_key(level_key):
                    self._fixed[level_key] = levels
            direction = v["tick_direction"]
            self.tick_direction.set_value(direction if direction in TICK_DIRECTIONS else "out")
            self.tick_mirror.setChecked(bool(v["tick_mirror"]))
            self.minor_ticks.setChecked(bool(v["minor_ticks"]))
            lo, hi = MINOR_INTERVALS
            intervals = v["minor_intervals"]
            self.minor_intervals.setValue(intervals if lo <= intervals <= hi else 2)
            for name, field in self._tick_fields().items():
                field.edit.setText(_number_text(v[name]))
        self._shown_levels = None
        self._changed()

    def _show_style(self, style: FigureStyle) -> None:
        ticks = style.ticks
        self.colorbar_position.set_value(style.colorbar_location)
        self.tick_direction.set_value(ticks.direction)
        self.tick_mirror.setChecked(ticks.mirror)
        self.minor_ticks.setChecked(ticks.minor)
        self.minor_intervals.setValue(ticks.minor_intervals)
        self.tick_length.edit.setText(_number_text(ticks.length_pt))
        self.tick_width.edit.setText(_number_text(ticks.width_pt))
        self.minor_length.edit.setText(_number_text(ticks.minor_length_pt))

    def _set_preset_defaults(self, key: str) -> None:
        preset = get_preset(key)
        control = self.widths.get(key)
        if control is not None and preset.default_width is not None:
            control.set_value(preset.default_width)
        if preset.free_size:
            self.custom_width.edit.setText(f"{preset.default_width_mm:g}")
        self.height_field.edit.setText(f"{preset.default_height_mm:g}")
        self.font_field.edit.setText(f"{preset.font_size_pt:g}")
        self.line_field.edit.setText(f"{preset.line_width_pt:g}")
        self.dpi_field.edit.setText(f"{preset.raster_dpi:g}")

    # ------------------------------------------------------------------ reactions
    def _on_preset(self, key: str) -> None:
        if self._quiet:
            return
        with self._quietly():
            self._set_preset_defaults(key)
        self._changed()

    def _on_view(self, kind: str) -> None:
        old = self._labels[self._kind]
        old["x"], old["y"] = self.x_label.text(), self.y_label.text()
        old["colorbar"] = self.colorbar_label.text()
        self._kind = kind
        new = self._labels[kind]
        with self._quietly():
            self.x_label.setText(new["x"])
            self.y_label.setText(new["y"])
            self.colorbar_label.setText(new["colorbar"])
        self._changed()

    def _follow_window(self, *_args) -> None:
        """The main window changed what it shows: show it here too (no reprocessing)."""
        if self.isVisible():
            self._start_fixed()
            self._update_form()
            self._schedule()

    def _changed(self, *_args) -> None:
        if self._quiet:
            return
        self._note = None
        self._start_fixed()
        self._update_form()
        self._store()
        self._schedule()

    def _schedule(self) -> None:
        if self.isVisible():
            self._debounce.start()

    def flush(self) -> None:
        """Draw the preview now instead of after the pause (tests, first show)."""
        self._debounce.stop()
        self._draw_preview()

    def is_idle(self) -> bool:
        """Nothing waits to be drawn or saved."""
        return not self._debounce.isActive() and not self.renderer.busy()

    # ------------------------------------------------------------------ form
    def _update_form(self) -> None:
        preset = self.current_preset()
        size = self.print_size()
        style = self.style_check()
        self.width_stack.setCurrentIndex(list(PRESETS).index(preset.key))
        key = self.width_key()
        self.width_caption.setText(column_name(preset, key) if key else "Any width")
        self.notes.setText("\n".join(f"• {note}" for note in preset.notes))
        self.notes.setToolTip("\n".join(preset.sources))
        for letter in PANEL_LETTERS:
            styled = preset.panel_label(letter)
            self.panel_label.button(letter).setToolTip(f"Panel {styled}: bold, top left")
        name = PRESET_NAMES.get(preset.key, preset.name)
        self.panel_caption.setText(f"Panel label · {name} style: {preset.panel_label('a')}, bold")
        for name, field in self._number_fields().items():
            field.set_invalid(name in size.invalid)
        for name, field in self._tick_fields().items():
            field.set_invalid(name in style.invalid)
        if size.ok:
            for name, value in auto_tick_sizes(size.font_pt, size.line_pt).items():
                self._tick_fields()[name].edit.setPlaceholderText(f"{value:.3g}")

        fmt = self.format.value()
        raster = fmt in RASTER_FORMATS
        if size.ok:
            px_w, px_h = size.pixels()
            self.dpi_hint.setText(
                f"{px_w} × {px_h} px" if raster else "Of the map image in the file"
            )
            text = f"{size.width_mm:g} × {size.height_mm:g} mm · {FORMAT_LABELS[fmt]}"
            if raster:
                text += f" · {size.dpi:g} dpi"
            self.preview.set_size_mm(size.width_mm, size.height_mm, text)
        else:
            self.dpi_hint.setText("")

        kind = self.view.value()
        unit = Unit(self.controller.unit)
        x_auto, y_auto = auto_labels(kind, unit)
        self.x_label.setPlaceholderText(x_auto)
        self.y_label.setPlaceholderText(y_auto)
        self._update_content(kind)
        self._sync_level_fields()
        self._update_levels(kind)
        self.my_presets.update_button()

        items = [("err", text) for text in self.problems()]
        if self._preview_error:
            items.append(("err", self._preview_error))
        items += [("warn", text) for text in size.warnings]
        if self._note is not None:
            items.append(self._note)
        self.messages.set_items(items)
        self._update_save_button()

    def _update_content(self, kind: str) -> None:
        n_models, n_curves, n_points = self._snapshot_counts
        stacked = kind == "stacked"
        by_field = self.controller.view.stacked_by_field
        self.colorbar_label.setPlaceholderText(
            "Magnetic field (T)" if stacked else "No label (e.g. ΔT/T)"
        )
        can_bar = not stacked or by_field
        self.colorbar_row.switch.setEnabled(can_bar)
        self.colorbar_row.description_label.setText(
            ("Field of each trace" if can_bar else "Traces are not coloured by field")
            if stacked
            else "Values of the map"
        )
        self.colorbar_row.description_label.setVisible(True)
        bar_shown = can_bar and self.colorbar.isChecked()
        self.colorbar_label.setEnabled(bar_shown)
        self.colorbar_position.setEnabled(bar_shown)

        can_draw = not stacked and n_models > 0
        self.models.setEnabled(can_draw)
        with QSignalBlocker(self.models):  # off while nothing can be drawn, as its note says
            self.models.setChecked(can_draw and self._models_wanted)
        if stacked:
            models_text = "Not drawn on stacked spectra"
        elif n_models:
            models_text = f"{n_models} curve{'s' if n_models != 1 else ''} from the window"
        else:
            models_text = "None shown in the window"
        self.models_row.description_label.setText(models_text)
        self.models_row.description_label.setVisible(True)

        self.points.setEnabled(n_points > 0)
        if n_points:
            curves = f"{n_curves} curve{'s' if n_curves != 1 else ''}"
            points_text = f"{n_points} point{'s' if n_points != 1 else ''} on {curves}"
        else:
            points_text = "No picked points"
        self.points_row.description_label.setText(points_text)
        self.points_row.description_label.setVisible(True)
        self.points_scope.setEnabled(n_points > 0 and self.points.isChecked())

        minor = self.minor_ticks.isChecked()
        self.minor_intervals.setEnabled(minor)
        self.minor_length.setEnabled(minor)

    def _update_levels(self, kind: str) -> None:
        """The colour range: for maps only (stacked traces are coloured by field); its fields
        are typed in Fixed and show the levels drawn with otherwise."""
        is_map = kind == "map"
        mode = self.levels_mode.value()
        fixed = is_map and mode == RANGE_FIXED
        self.levels_mode.setEnabled(is_map)
        for field in (self.level_lo, self.level_hi):
            field.setEnabled(fixed)
        lo, hi = number(self.level_lo), number(self.level_hi)
        problem = self.colour_range_problem()
        reversed_ = lo is not None and hi is not None
        self.level_lo.set_invalid(bool(problem) and (lo is None or reversed_))
        self.level_hi.set_invalid(bool(problem) and (hi is None or reversed_))
        if not is_map:
            note = "Stacked traces are coloured by field: the range is for maps."
        elif mode == RANGE_FIXED:
            note = f"In the plot's values. Remembered for {level_label(self.level_key())}."
        else:
            note = RANGE_NOTES[mode]
        self.levels_note.setText(note)

    def _update_save_button(self) -> None:
        saving = self.renderer.saving()
        ok = not self.problems() and not self._preview_error
        self.save_button.setEnabled(ok and not saving)
        self.save_button.setText("Saving…" if saving else "Save…")

    def _store(self) -> None:
        if self._quiet:
            return
        self.settings.update(
            preset=self.preset.value(),
            width=self.width_key(),
            custom_width=number(self.custom_width),
            height=number(self.height_field),
            font=number(self.font_field),
            line=number(self.line_field),
            format=self.format.value(),
            dpi=number(self.dpi_field),
            colorbar=self.colorbar.isChecked(),
            colorbar_label=self._map_colorbar_label(),
            colorbar_location=self.colorbar_position.value(),
            panel_label="" if self.panel_letter() is None else self.panel_letter(),
            colour_range=self.levels_mode.value(),
            fixed_levels=self._stored_fixed(),
            tick_direction=self.tick_direction.value(),
            tick_mirror=self.tick_mirror.isChecked(),
            tick_length=number(self.tick_length),
            tick_width=number(self.tick_width),
            minor_ticks=self.minor_ticks.isChecked(),
            minor_intervals=self.minor_intervals.value(),
            minor_length=number(self.minor_length),
        )

    def _map_colorbar_label(self) -> str:
        """The colour bar label typed for the map (the one remembered)."""
        if self._kind == "map":
            return self.colorbar_label.text()
        return self._labels["map"]["colorbar"]

    # ------------------------------------------------------------------ drawing
    def _snapshot_state(self):
        """The figure state now; also counts the models and points the window has and shows
        the colour range drawn with."""
        snapshot = self.controller.figure_state()
        sets = point_sets(snapshot, POINTS_ALL)
        self._snapshot_counts = (
            len(model_curves(snapshot)),
            len(sets),
            sum(s.x.size for s in sets),
        )
        self._sync_level_fields(snapshot)
        return figure_state(self.controller, self.content(), snapshot=snapshot)

    def _draw_preview(self) -> None:
        if not self.isVisible():
            return
        try:
            state = self._snapshot_state()
        except ValueError as exc:
            self._preview_error = f"Nothing to draw: {exc}."
            self.preview.set_message(self._preview_error)
            self._update_form()
            return
        if self._preview_error:
            self._preview_error = ""
            self.preview.set_message("")
        self._update_form()
        if self.problems():
            return  # the errors are shown; the last drawing stays
        ratio = self.devicePixelRatioF()
        self.renderer.preview(self.job(self.preview.wanted_dpi() * ratio, state), ratio)

    def _on_preview(self, image: QImage) -> None:
        self.preview.set_image(image)
        self.previewUpdated.emit()

    def _on_busy(self, busy: bool) -> None:
        self.preview.set_busy(busy and not self.renderer.saving())
        self._update_save_button()
        if self.renderer.saving():
            self.setCursor(Qt.CursorShape.BusyCursor)
        else:
            self.unsetCursor()

    def _on_failed(self, kind: str, message: str) -> None:
        if kind == "preview":
            self.preview.set_message(f"The preview could not be drawn: {message}")
            return
        logger.error("Export figure: could not save the figure: %s", message)
        self.show_note("err", f"Could not save the figure: {message}")

    def _on_saved(self, path: str) -> None:
        size = self.print_size()
        name = Path(path).name
        what = f"{size.width_mm:g} × {size.height_mm:g} mm" if size.ok else ""
        logger.info("Saved figure %s (%s, %s)", path, self.current_preset().name, what)
        self.show_note("ok", f"Saved {name}")
        status = self.main_window.statusBar()
        status.showMessage(f"Saved figure {name}", 6000)
        self.figureSaved.emit(path)

    # ------------------------------------------------------------------ saving
    def save(self) -> None:
        """Ask for a file (suffix of the chosen format) and save the figure there."""
        size = self.print_size()
        if self.problems() or self.renderer.saving():
            return
        fmt = self.format.value()
        start = self.default_file_name()
        if last_dir():
            start = str(Path(last_dir()) / start)
        path, _ = QFileDialog.getSaveFileName(self, "Save figure", start, FILTERS[fmt])
        if not path:
            return
        out = with_format(path, fmt)
        set_last_dir(str(out.parent))
        try:
            job = self.job(size.dpi)
        except ValueError as exc:
            self.show_note("err", f"Could not save the figure: {exc}")
            return
        self._note = None
        self.renderer.save(job, out)
        self._update_form()

    # ------------------------------------------------------------------ events
    def showEvent(self, event) -> None:
        super().showEvent(event)
        QTimer.singleShot(0, self._draw_now)  # once laid out: the preview knows its size

    def _draw_now(self) -> None:
        if self.isVisible():
            self.flush()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._schedule()  # a sharper (or smaller) preview for the new size

    def changeEvent(self, event) -> None:
        super().changeEvent(event)
        if event.type() == QEvent.Type.ActivationChange and self.isActiveWindow():
            self._follow_window()  # e.g. a model changed in the window's inspector


def _is_level_key(key) -> bool:
    try:
        parse_level_key(key)
    except ValueError:
        return False
    return True


def _forget(holder, listener) -> None:
    if listener in holder.listeners:
        holder.listeners.remove(listener)
