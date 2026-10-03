"""The export window: a journal figure of the plot on screen, with a live preview.

Left, the figure at its true proportions, drawn by matplotlib at screen resolution again about
150 ms after any change, with its print size under it; right, the journal preset, the size,
text and lines, the file format, what the figure includes and its labels. Problems show inline
under the preview, never as dialogs. While it is open the window follows the main window (unit,
plot, ranges, colours, models, points), and it remembers its choices (``export/figure``).
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path

from PySide6.QtCore import QEvent, QRectF, QSize, Qt, QTimer, Signal
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
from mag_opt_detective.export import PRESETS, JournalPreset, get_preset
from mag_opt_detective.gui import icons
from mag_opt_detective.gui.export_menu import ExportSettings
from mag_opt_detective.gui.export_render import FigureJob, FigureRenderer
from mag_opt_detective.gui.export_state import (
    FORMAT_LABELS,
    KINDS,
    POINTS_ALL,
    POINTS_CURRENT,
    POINTS_NONE,
    RASTER_FORMATS,
    FigureContent,
    PrintSize,
    auto_labels,
    figure_state,
    file_name,
    model_curves,
    point_sets,
    print_size,
    with_format,
)
from mag_opt_detective.gui.kit import SegmentedControl
from mag_opt_detective.gui.panels.common import (
    Note,
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
PRESET_NAMES = {"nature": "Nature", "aps": "APS", "custom": "Custom"}
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

    def __init__(self, window, settings: ExportSettings):
        super().__init__(window)
        self.main_window = window
        self.controller = window.controller
        self.settings = settings
        self._quiet = 0
        self._labels = {kind: {"x": "", "y": "", "colorbar": ""} for kind in KINDS}
        self._kind = "map"
        self._preview_error = ""
        self._saved_note: tuple[str, str] | None = None
        self._snapshot_counts = (0, 0, 0)  # model curves, point curves, points

        self.setWindowTitle("Export figure")
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
        self.colorbar_row = SwitchRow("Colour bar", "")
        self.colorbar = self.colorbar_row.switch
        self.colorbar_label = QLineEdit()
        self.colorbar_label.setAccessibleName("Colour bar label")
        self.models_row = SwitchRow("Model curves", "")
        self.models = self.models_row.switch
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
        layout.addWidget(block(section_label("Journal"), self.preset, self.notes))
        sizes = QHBoxLayout()
        sizes.setSpacing(8)
        for text, field in (
            ("Height", self.height_field),
            ("Text", self.font_field),
            ("Lines", self.line_field),
        ):
            sizes.addWidget(labelled(text, field), 1)
        sizes_box = QWidget()
        sizes_box.setLayout(sizes)
        sizes.setContentsMargins(0, 0, 0, 0)
        width_box = block(labelled("Width (mm)", self.width_stack), self.width_caption, spacing=4)
        layout.addWidget(block(section_label("Size"), width_box, sizes_box))
        dpi_row = QWidget()
        dpi_line = QHBoxLayout(dpi_row)
        dpi_line.setContentsMargins(0, 0, 0, 0)
        dpi_line.setSpacing(10)
        self.dpi_field.setFixedWidth(104)
        dpi_line.addWidget(labelled("Resolution", self.dpi_field))
        dpi_line.addWidget(self.dpi_hint, 1, Qt.AlignmentFlag.AlignBottom)
        layout.addWidget(block(section_label("File"), self.format, dpi_row))
        self.colorbar_box = block(self.colorbar_row, self.colorbar_label, spacing=6)
        panel_box = labelled("Panel label", self.panel_label)
        self.panel_caption = panel_box.findChild(QLabel)  # shows the preset's style
        self.points_box = block(self.points_row, self.points_scope, spacing=6)
        layout.addWidget(
            block(
                section_label("Content"),
                labelled("View", self.view),
                self.colorbar_box,
                self.models_row,
                self.points_box,
                panel_box,
                spacing=12,
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
        for switch in (self.colorbar, self.models, self.points):
            switch.toggled.connect(self._changed)
        self.points_scope.valueChanged.connect(self._changed)
        self.panel_label.valueChanged.connect(self._changed)
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
        )

    def panel_letter(self) -> str | None:
        value = self.panel_label.value()
        return None if value == NO_PANEL else value

    def figure_state(self):
        """What the figure shows now (export FigureState), as it would be saved."""
        return figure_state(self.controller, self.content())

    def job(self, dpi: float, state=None) -> FigureJob:
        size = self.print_size()
        if not size.ok:
            raise ValueError(size.errors[0])
        return FigureJob(
            state=self.figure_state() if state is None else state,
            preset=self.current_preset(),
            width_mm=size.width_mm,
            height_mm=size.height_mm,
            font_pt=size.font_pt,
            line_pt=size.line_pt,
            panel_label=self.panel_letter(),
            dpi=dpi,
        )

    def default_file_name(self) -> str:
        return file_name(
            self.controller,
            self.view.value(),
            self.current_preset(),
            self.width_key(),
            self.format.value(),
        )

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
        self._changed()

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
            self._update_form()
            self._schedule()

    def _changed(self, *_args) -> None:
        if self._quiet:
            return
        self._saved_note = None
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

        items = [("err", text) for text in size.errors]
        if self._preview_error:
            items.append(("err", self._preview_error))
        items += [("warn", text) for text in size.warnings]
        if self._saved_note is not None:
            items.append(self._saved_note)
        self.messages.set_items(items)
        self._update_save_button(size)

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
        self.colorbar_label.setEnabled(can_bar and self.colorbar.isChecked())

        self.models.setEnabled(not stacked and n_models > 0)
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

    def _update_save_button(self, size: PrintSize | None = None) -> None:
        size = size or self.print_size()
        saving = self.renderer.saving()
        self.save_button.setEnabled(size.ok and not saving and not self._preview_error)
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
            panel_label="" if self.panel_letter() is None else self.panel_letter(),
        )

    def _map_colorbar_label(self) -> str:
        """The colour bar label typed for the map (the one remembered)."""
        if self._kind == "map":
            return self.colorbar_label.text()
        return self._labels["map"]["colorbar"]

    # ------------------------------------------------------------------ drawing
    def _snapshot_state(self):
        """The figure state now; also counts the models and points the window has."""
        snapshot = self.controller.figure_state()
        sets = point_sets(snapshot, POINTS_ALL)
        self._snapshot_counts = (
            len(model_curves(snapshot)),
            len(sets),
            sum(s.x.size for s in sets),
        )
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
        size = self.print_size()
        if not size.ok:
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
        self._saved_note = ("err", f"Could not save the figure: {message}")
        self._update_form()

    def _on_saved(self, path: str) -> None:
        size = self.print_size()
        name = Path(path).name
        what = f"{size.width_mm:g} × {size.height_mm:g} mm" if size.ok else ""
        logger.info("Saved figure %s (%s, %s)", path, self.current_preset().name, what)
        self._saved_note = ("ok", f"Saved {name}")
        self._update_form()
        status = self.main_window.statusBar()
        status.showMessage(f"Saved figure {name}", 6000)
        self.figureSaved.emit(path)

    # ------------------------------------------------------------------ saving
    def save(self) -> None:
        """Ask for a file (suffix of the chosen format) and save the figure there."""
        size = self.print_size()
        if not size.ok or self.renderer.saving():
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
            self._saved_note = ("err", f"Could not save the figure: {exc}")
            self._update_form()
            return
        self._saved_note = None
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


def _forget(settings: ExportSettings, listener) -> None:
    if listener in settings.listeners:
        settings.listeners.remove(listener)
