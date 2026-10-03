"""Library panel: processed maps to plot together, compare and combine into a product.

The maps live in ``controller.library`` (cm^-1). On top, the processed sweep goes into the
library in one click. Each map row has a tick (ticked maps are on the plot, drawn together, the
one ticked or clicked last on top), the name, a meta line (kind · fields · E range in the
display unit) and opens to its cut limits (E in the display unit, kept in cm^-1, and B in T; an
empty limit keeps everything) and a remove button. Rows on the plot are highlighted. The
footer combines the ticked maps by energy, by field or as an average, with a preview of the
result or the reason it cannot be made; the result is the product, with its name, what it was
made of and its export.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
from PySide6.QtCore import QEvent, QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QAction, QPainter, QPen
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from mag_opt_detective.core.pipeline import PlotKind
from mag_opt_detective.core.spectra import LIMIT_RTOL, FieldMap, save_tsv
from mag_opt_detective.core.units import Unit, from_cm1
from mag_opt_detective.gui import icons
from mag_opt_detective.gui.controller import (
    COMBINE_METHODS,
    FIELD_TOL,
    KIND_LABELS,
    SHOWS_MAPS,
    SHOWS_PROCESS,
    SHOWS_PRODUCT,
    CombinePreview,
    LibraryEntry,
    Product,
    cut_text,
    energy_range_text,
    fact_text,
    in_panel,
    user_action,
)
from mag_opt_detective.gui.display import format_range, unit_text
from mag_opt_detective.gui.kit import SegmentedControl
from mag_opt_detective.gui.panels.common import (
    Card,
    CheckBox,
    Divider,
    DropZone,
    ElidedLabel,
    FileDrops,
    LinkButton,
    Note,
    SwitchRow,
    UnitField,
    clear_layout,
    fit_segments,
    hint,
    labelled,
    scaled_font,
    section_label,
    small_button,
)
from mag_opt_detective.gui.panels.files import fmt_field
from mag_opt_detective.gui.theme import current_tokens
from mag_opt_detective.gui.widgets import (
    TABLE_FILTER,
    EnergyEdit,
    FloatEdit,
    FlowLayout,
    last_dir,
    open_files,
    parse_float,
    set_last_dir,
)

NEED_TWO = "Tick at least two maps"
ADD_TIP = "Add the processed sweep's R(B)/R(0) map to the library"
NOTHING_TO_ADD = "Process a sweep first: there is no map to add yet"
ROW_TIP = "Click to show {name} on top of the ticked maps"
METHODS = {  # value: (segment text, short segment text, explanation, details)
    "energy": (
        "By energy",
        "Energy",
        "Joins spectral ranges (FIR + MIR), split in the middle of an overlap.",
        "Each map is cut to its limits; where two overlap, the lower one is used up to the "
        "middle of the overlap (E limits move the join). Every measured energy is kept.",
    ),
    "field": (
        "By field",
        "Field",
        "Joins field ranges over the energies all maps share.",
        "Joins field ranges such as 0 – 8 T and 8 – 16 T; fields measured twice are averaged.",
    ),
    "average": (
        "Average",
        "Average",
        "Averages repeated sweeps over the energies they share.",
        "Averages repeated sweeps with the same fields, each cut to its limits.",
    ),
}
NEED_TWO_HINT = "Tick two or more maps to merge or average them."
PREVIEW_DELAY_MS = 120  # typing in a limit computes the preview once it pauses
LIMITS_DELAY_MS = 400  # ... and draws maps plotted cut to their limits once it pauses

logger = logging.getLogger("mag_opt_detective")


def meta_parts(entry: LibraryEntry, unit: Unit) -> list[str]:
    """``["R(B)/R(0)", "64 fields", "350 – 7800 cm⁻¹"]`` (energies in *unit*)."""
    return map_parts(entry.kind, entry.fmap, unit)


def map_parts(kind: str, fmap: FieldMap, unit: Unit) -> list[str]:
    """Kind, number of fields and energy range of *fmap* (cm^-1) in *unit*."""
    n = fmap.field.size
    e_lo, e_hi = from_cm1(np.array([np.nanmin(fmap.energy), np.nanmax(fmap.energy)]), unit)
    fields = f"{n} field{'s' if n != 1 else ''}"
    return [kind, fields, format_range(e_lo, e_hi, unit_text(unit))]


def size_text(fmap: FieldMap, unit: Unit) -> str:
    """``64 fields · 0.25 – 16 T · 30 – 3200 cm⁻¹ · 6341 energies`` of *fmap* (cm^-1)."""
    b = fmap.field
    e_lo, e_hi = from_cm1(np.array([fmap.energy.min(), fmap.energy.max()]), unit)
    fields = f"{b.size} field{'s' if b.size != 1 else ''}"
    return (
        f"{fields} · {format_range(b.min(), b.max(), 'T')} · "
        f"{format_range(e_lo, e_hi, unit_text(unit))} · {fmap.energy.size} energies"
    )


def name_parts(name: str) -> list[str]:
    """*name* cut after each ``_``, ``-`` or space, where a long name may wrap."""
    return [part for part in re.split(r"(?<=[_\- ])", name) if part]


def provenance_text(entry: LibraryEntry, unit: Unit) -> str:
    """The tooltip of a library map: how it was made."""
    if entry.provenance is None:
        return entry.source or f"{entry.name} (saved from the map shown)"
    return "\n".join(entry.provenance.lines(unit))


class PartsLabel(QLabel):
    """Text made of parts that wraps only between parts, so none is cut in two.

    The parts are joined by *separator*, which is dropped where a line breaks; a part wider
    than the label is elided (*mode*).
    """

    def __init__(self, separator: str = "", mode=Qt.TextElideMode.ElideRight, parent=None):
        super().__init__(parent)
        self.separator = separator
        self._mode = mode
        self._parts: list[str] = []
        policy = QSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        policy.setHeightForWidth(True)
        self.setSizePolicy(policy)

    def set_parts(self, parts: list[str]) -> None:
        self._parts = list(parts)
        self.setText(self.separator.join(self._parts))
        self.updateGeometry()

    def lines(self, width: int | None = None) -> list[str]:
        """The lines shown at *width* (the current width by default)."""
        width = self.contentsRect().width() if width is None else width
        metrics = self.fontMetrics()
        lines: list[str] = []
        for part in self._parts:
            joined = f"{lines[-1]}{self.separator}{part}" if lines else part
            if lines and metrics.horizontalAdvance(joined) <= width:
                lines[-1] = joined
            else:
                lines.append(part)
        return lines

    def _height(self, lines: int) -> int:
        margins = self.contentsMargins()
        return max(1, lines) * self.fontMetrics().lineSpacing() + margins.top() + margins.bottom()

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        margins = self.contentsMargins()
        return self._height(len(self.lines(width - margins.left() - margins.right())))

    def sizeHint(self) -> QSize:
        margins = self.contentsMargins()
        width = self.fontMetrics().horizontalAdvance(self.text()) + margins.left() + margins.right()
        return QSize(width, self._height(1))

    def minimumSizeHint(self) -> QSize:
        return QSize(20, self._height(1))

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setPen(self.palette().color(self.foregroundRole()))
        painter.setFont(self.font())
        rect = self.contentsRect()
        metrics = self.fontMetrics()
        step = metrics.lineSpacing()
        for i, line in enumerate(self.lines(rect.width())):
            painter.drawText(
                rect.adjusted(0, i * step, 0, 0),
                int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop),
                metrics.elidedText(line, self._mode, rect.width()),
            )
        painter.end()


def _holds_data(axis: np.ndarray, lo: float | None, hi: float | None, rtol: float) -> bool:
    """Some sample of *axis* lies in ``[lo, hi]`` (a None end is open; *rtol*: slack on an end,
    as :func:`~mag_opt_detective.core.spectra.energy_mask` gives)."""
    inside = np.ones(axis.size, dtype=bool)
    if lo is not None:
        inside &= axis >= lo - abs(lo) * rtol - FIELD_TOL
    if hi is not None:
        inside &= axis <= hi + abs(hi) * rtol + FIELD_TOL
    return bool(inside.any())


def file_name(name: str) -> str:
    """*name* usable as a file name on every system (other characters become ``_``)."""
    return re.sub(r"[^A-Za-z0-9._+-]+", "_", name).strip("._") or "product"


def _no_minimum_width(note: Note) -> None:
    """Let *note* be narrower than its longest word (a map name), which is then cut."""
    note.text_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)


class StateTag(QLabel):
    """A small accent pill saying a map is on the plot ("On plot", "Top")."""

    def __init__(self, text: str = "", parent=None):
        super().__init__(text, parent)
        self.setFont(scaled_font(self, 0.8, bold=True))
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setContentsMargins(6, 0, 6, 0)

    def paintEvent(self, event) -> None:
        tokens = current_tokens()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(tokens["accent-soft"])
        painter.drawRoundedRect(QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5), 4, 4)
        painter.setPen(tokens["accent"])
        painter.setFont(self.font())
        painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.text())
        painter.end()


class _Highlight(QWidget):
    """A widget that paints itself as being on the plot: an accent-soft fill, and an accent
    border for the map on top (:meth:`set_plot_state`)."""

    INSET = 3

    def __init__(self, parent=None):
        super().__init__(parent)
        self._state = ""  # "", "plotted" or "top"

    def plot_state(self) -> str:
        return self._state

    def set_plot_state(self, state: str) -> None:
        if state != self._state:
            self._state = state
            self.update()

    def paintEvent(self, event) -> None:
        if not self._state:
            return
        tokens = current_tokens()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        inset = self.INSET + 0.5
        rect = QRectF(self.rect()).adjusted(inset, inset, -inset, -inset)
        painter.setBrush(tokens["accent-soft"])
        if self._state == "top":
            painter.setPen(QPen(tokens["accent"], 1))
        else:
            painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(rect, 5, 5)
        painter.end()


class EntryRow(_Highlight):
    """One library map: tick, name, plot tag and meta line; it opens to the cut limits and
    the remove button. A click on the row brings the map on top of the plot.

    The name has the first line to itself (but for the tag and expand button) and wraps after a
    ``_``; the meta line wraps between its parts. Both stay whole in a narrow panel.
    """

    raiseRequested = Signal(int)
    removeRequested = Signal(int)
    expandedChanged = Signal(int, bool)

    def __init__(self, entry: LibraryEntry, parent=None):
        super().__init__(parent)
        self.key = entry.key
        self._axes = entry.fmap.energy, entry.fmap.field  # (cm^-1, T): limits must hold data
        self.use = CheckBox()
        self.use.setAccessibleName(f"Plot {entry.name}")
        self.use.setToolTip("Ticked maps are drawn on the plot and combined")
        self.name_label = PartsLabel(mode=Qt.TextElideMode.ElideMiddle)
        self.name_label.setFont(scaled_font(self.name_label, 0.98, bold=True))
        self.name_label.set_parts(name_parts(entry.name))
        self.meta_label = PartsLabel(" · ")
        self.meta_label.setProperty("kit", "muted")
        self.meta_label.setFont(scaled_font(self.meta_label, 0.9))
        self.tag = StateTag()
        self.tag.hide()
        self.problem = Note("", "warn")  # why a ticked map is not drawn
        self.problem.hide()
        _no_minimum_width(self.problem)
        self.expand_button = QToolButton()
        self.expand_button.setProperty("kit", "tool")
        self.expand_button.setIconSize(QSize(14, 14))
        self.expand_button.setToolTip(f"Limits for {entry.name}")
        self.expand_button.setAccessibleName(f"Limits for {entry.name}")
        self.expand_button.setCheckable(True)
        icons.set_icon(self.expand_button, "chevron-right", "muted")
        line = max(self.expand_button.sizeHint().height(), self.use.sizeHint().height())
        pad = max(0, (line - self.name_label.fontMetrics().lineSpacing()) // 2)
        self.name_label.setContentsMargins(0, pad, 0, pad)  # line 1 level with the buttons
        self.remove_button = small_button(
            "Remove", "trash-2", f"Remove {entry.name} from the library"
        )
        self.remove_button.setAccessibleName(f"Remove {entry.name}")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip(ROW_TIP.format(name=entry.name))

        self.e_min = EnergyEdit(name=f"{entry.name}: E min")
        self.e_max = EnergyEdit(name=f"{entry.name}: E max")
        self.b_min = FloatEdit(None, f"{entry.name}: B min")
        self.b_max = FloatEdit(None, f"{entry.name}: B max")
        self.fields = {
            edit: UnitField(edit, "T" if edit in (self.b_min, self.b_max) else "")
            for edit in (self.e_min, self.e_max, self.b_min, self.b_max)
        }
        for edit in self.fields:
            edit.setPlaceholderText("all")
            edit.setAcceptDrops(False)  # dropped tables go to the list
        self.cut_box = QWidget()
        self.cut_box.setCursor(Qt.CursorShape.ArrowCursor)
        grid = QGridLayout(self.cut_box)
        grid.setContentsMargins(31, 0, 9, 10)
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(6)
        for i, (text, edit) in enumerate(
            (("E min", self.e_min), ("E max", self.e_max), ("B min", self.b_min),
             ("B max", self.b_max))
        ):  # fmt: skip
            grid.addWidget(labelled(text, self.fields[edit]), i // 2, i % 2)
        grid.addWidget(self.remove_button, 2, 0, 1, 2, Qt.AlignmentFlag.AlignLeft)
        self.cut_box.setVisible(False)

        head = QGridLayout()
        head.setContentsMargins(9, 7, 5, 8)
        head.setHorizontalSpacing(6)
        head.setVerticalSpacing(1)
        tick = QWidget()  # keeps the tick level with the name's first line
        tick.setFixedHeight(line)
        tick_layout = QVBoxLayout(tick)
        tick_layout.setContentsMargins(0, 0, 0, 0)
        tick_layout.addWidget(self.use, 0, Qt.AlignmentFlag.AlignVCenter)
        side = QHBoxLayout()  # the tag (when shown) and the expand button
        side.setSpacing(4)
        side.addWidget(self.tag, 0, Qt.AlignmentFlag.AlignVCenter)
        side.addWidget(self.expand_button)
        top = Qt.AlignmentFlag.AlignTop
        head.addWidget(tick, 0, 0, top)
        head.addWidget(self.name_label, 0, 1)
        head.addLayout(side, 0, 2, top)
        head.addWidget(self.meta_label, 1, 1, 1, 2)
        head.addWidget(self.problem, 2, 1, 1, 2)
        head.setColumnStretch(1, 1)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addLayout(head)
        layout.addWidget(self.cut_box)

        self.remove_button.clicked.connect(lambda: self.removeRequested.emit(self.key))
        self.expand_button.toggled.connect(self.set_expanded)

    def _on_head(self, event) -> bool:
        return self.cut_box.isHidden() or event.position().y() < self.cut_box.y()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._on_head(event):
            event.accept()  # the release comes here
            return
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        inside = self.rect().contains(event.position().toPoint())
        if event.button() == Qt.MouseButton.LeftButton and inside and self._on_head(event):
            self.raiseRequested.emit(self.key)
        super().mouseReleaseEvent(event)

    def is_expanded(self) -> bool:
        return not self.cut_box.isHidden()

    def set_expanded(self, expanded: bool) -> None:
        if self.expand_button.isChecked() != expanded:
            self.expand_button.setChecked(expanded)  # comes back here
            return
        self.cut_box.setVisible(expanded)
        icons.set_icon(self.expand_button, "chevron-down" if expanded else "chevron-right", "muted")
        self.expandedChanged.emit(self.key, expanded)

    def set_plot_state(self, state: str, tag: str = "", problem: str = "") -> None:
        """Show the row as on the plot (*state* "plotted" or "top"; "" when it is not), or
        why the ticked map is not drawn (*problem*)."""
        super().set_plot_state(state)
        self.tag.setText(tag)
        self.tag.setVisible(bool(tag))
        text = f"Not drawn: {problem}" if problem else ""
        if text != self.problem.text():
            self.problem.set_text(text, "warn")
        self.problem.setVisible(bool(problem))

    def set_unit(self, unit: Unit) -> None:
        for edit in (self.e_min, self.e_max):
            edit.set_unit(unit)
            self.fields[edit].set_unit(unit_text(unit))

    def show_entry(self, entry: LibraryEntry, unit: Unit) -> None:
        """Show the entry's tick, meta line and limits (fields holding a value keep it)."""
        if self.use.isChecked() != entry.used:
            self.use.setChecked(entry.used)
        self.meta_label.set_parts(meta_parts(entry, unit))
        self.name_label.setToolTip(provenance_text(entry, unit))
        for edit, value in zip((self.e_min, self.e_max), entry.energy_cut, strict=True):
            if edit.cm1() != value:
                edit.set_cm1(value)
        for edit, value in zip((self.b_min, self.b_max), entry.field_cut, strict=True):
            if parse_float(edit.text()) != value:
                edit.setText("" if value is None else fmt_field(value))
        self.check()

    def energy_cut(self) -> tuple[float | None, float | None]:
        return self.e_min.cm1(), self.e_max.cm1()

    def field_cut(self) -> tuple[float | None, float | None]:
        return parse_float(self.b_min.text()), parse_float(self.b_max.text())

    def check(self) -> bool:
        """Flag limits that are not numbers, are reversed or hold none of the map's data (as
        while one is typed); returns whether all limits can be used."""
        valid = True
        energy, field = self._axes
        for lo, hi, values, axis, tol in (
            (self.e_min, self.e_max, self.energy_cut(), energy, LIMIT_RTOL),
            (self.b_min, self.b_max, self.field_cut(), field, 0.0),
        ):
            reversed_ = None not in values and values[0] >= values[1]
            empty = not reversed_ and not _holds_data(axis, *values, tol)
            for edit, value in zip((lo, hi), values, strict=True):
                not_a_number = value is None and bool(edit.text().strip())
                bad = reversed_ or not_a_number or (empty and value is not None)
                self.fields[edit].set_invalid(bad)
                valid &= not bad
        return valid


class SweepCard(Card):
    """The processed sweep: its name and size, Add to library, and whether it is on the plot
    (a click shows it again when a library map hides it)."""

    showRequested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.highlight = _Highlight()
        self.title = section_label("Processed sweep")
        self.tag = StateTag("On plot")
        self.name_label = PartsLabel(mode=Qt.TextElideMode.ElideMiddle)
        self.name_label.setFont(scaled_font(self.name_label, 0.98, bold=True))
        self.meta_label = PartsLabel(" · ")
        self.meta_label.setProperty("kit", "muted")
        self.meta_label.setFont(scaled_font(self.meta_label, 0.9))
        self.add_button = small_button("Add", "plus", ADD_TIP)
        self.add_button.setAccessibleName("Add the processed sweep to the library")
        self.added = Note("In library", "ok")
        self.added.setToolTip("The processed sweep's map is in the library")
        self.show_link = LinkButton("Show", "Show the processed sweep on the plot")
        layout = QVBoxLayout(self.highlight)
        layout.setContentsMargins(10, 7, 8, 8)
        layout.setSpacing(2)
        top = QHBoxLayout()
        top.setSpacing(6)
        top.addWidget(self.title, 1)
        top.addWidget(self.tag)
        top.addWidget(self.show_link)
        layout.addLayout(top)
        row = QHBoxLayout()  # the name and Add (or "In library")
        row.setSpacing(6)
        text = QVBoxLayout()
        text.setSpacing(1)
        text.addWidget(self.name_label)
        text.addWidget(self.meta_label)
        row.addLayout(text, 1)
        row.addWidget(self.add_button, 0, Qt.AlignmentFlag.AlignTop)
        row.addWidget(self.added, 0, Qt.AlignmentFlag.AlignTop)
        layout.addLayout(row)
        self.box.addWidget(self.highlight)
        self.show_link.clicked.connect(self.showRequested)

    def show_sweep(self, name: str, parts: list[str], on_plot: bool, in_library: bool) -> None:
        self.name_label.set_parts(name_parts(name))
        self.meta_label.set_parts(parts)
        self.highlight.set_plot_state("top" if on_plot else "")
        self.tag.setVisible(on_plot)
        self.show_link.setVisible(not on_plot)
        self.add_button.setVisible(not in_library)
        self.added.setVisible(in_library)


class OpacitySlider(QWidget):
    """Opacity of the maps drawn above the bottom one, in percent (settings protocol)."""

    valueChanged = Signal(float)  # 0 - 1

    def __init__(self, parent=None):
        super().__init__(parent)
        self.label = QLabel("Overlap")
        self.label.setFont(scaled_font(self.label, 0.94))
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(10, 100)
        self.slider.setSingleStep(5)
        self.slider.setPageStep(10)
        self.slider.setValue(50)
        self.slider.setAccessibleName("Overlap opacity")
        self.setToolTip("Opacity of the maps above the lowest one where maps overlap")
        self.value_label = QLabel()
        self.value_label.setProperty("kit", "muted")
        self.value_label.setFont(scaled_font(self.value_label, 0.94))
        self.value_label.setFixedWidth(self.value_label.fontMetrics().horizontalAdvance("100 %"))
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(self.label)
        layout.addWidget(self.slider, 1)
        layout.addWidget(self.value_label)
        self.slider.valueChanged.connect(self._on_value)
        self._on_value(self.slider.value())

    def _on_value(self, percent: int) -> None:
        self.value_label.setText(f"{percent} %")
        self.valueChanged.emit(percent / 100)

    def value(self) -> float:
        return self.slider.value() / 100

    def set_value(self, opacity: float) -> None:
        self.slider.setValue(round(opacity * 100))

    def settings_value(self) -> int:
        return self.slider.value()

    def set_settings_value(self, value) -> bool:
        try:
            percent = int(value)
        except (TypeError, ValueError):
            return False
        if not self.slider.minimum() <= percent <= self.slider.maximum():
            return False
        self.slider.setValue(percent)
        return True


class ProductBox(Card):
    """The product of combining maps: its name (editable), what it was made of, its size and
    what was done to make it, and Plot, Save to library, Table… and Figure…."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.highlight = _Highlight()
        self.title = section_label("Product")
        self.tag = StateTag("On plot")
        self.discard_button = QToolButton()
        self.discard_button.setProperty("kit", "tool")
        self.discard_button.setIconSize(QSize(14, 14))
        self.discard_button.setToolTip("Discard the product")
        self.discard_button.setAccessibleName("Discard the product")
        icons.set_icon(self.discard_button, "x", "muted")
        self.name_edit = QLineEdit()
        self.name_edit.setAccessibleName("Product name")
        self.name_edit.setToolTip("Name of the product (in the library and in exported files)")
        self.method_label = QLabel()
        self.method_label.setFont(scaled_font(self.method_label, 0.92))
        self.parts_box = QVBoxLayout()  # a line per map, wrapping only between name parts
        self.parts_box.setSpacing(1)
        self.size_label = hint()
        self.notes = Note("", "info")
        _no_minimum_width(self.notes)
        self.plot_button = small_button("Plot", "eye", "Show the product on the plot")
        self.save_button = small_button("Save to library", "save", "Add the product to the library")
        self.table_button = small_button(
            "Table…",
            "file-text",
            "Export the product as a data table, with a note of how it was made",
        )
        self.figure_button = small_button(
            "Figure…", "image", "Show the product and open the journal figure export"
        )
        layout = QVBoxLayout(self.highlight)
        layout.setContentsMargins(10, 8, 8, 10)
        layout.setSpacing(6)
        top = QHBoxLayout()
        top.setSpacing(6)
        top.addWidget(self.title, 1)
        top.addWidget(self.tag)
        top.addWidget(self.discard_button)
        layout.addLayout(top)
        layout.addWidget(self.name_edit)
        layout.addWidget(self.method_label)
        layout.addLayout(self.parts_box)
        layout.addWidget(self.size_label)
        layout.addWidget(self.notes)
        buttons = QWidget()
        buttons.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        flow = FlowLayout(buttons, spacing=6, row_spacing=6)
        for button in (self.plot_button, self.save_button, self.table_button, self.figure_button):
            flow.addWidget(button)
        layout.addWidget(buttons)
        self.box.addWidget(self.highlight)

    def show_product(self, product: Product, unit: Unit, on_plot: bool, saved: str) -> None:
        """Show *product* in *unit*; *saved* is the name of its library entry ("" if none)."""
        if self.name_edit.text() != product.name and not self.name_edit.hasFocus():
            self.name_edit.setText(product.name)
            self.name_edit.setCursorPosition(0)
        p = product.provenance
        self.method_label.setText(f"{COMBINE_METHODS[p.method]} of {len(p.parts)} maps:")
        clear_layout(self.parts_box)
        for part in p.parts:
            cuts = cut_text(part.energy_cut, part.field_cut, unit)
            label = PartsLabel()
            label.setFont(scaled_font(label, 0.92))
            label.set_parts(name_parts(f"• {part.name}") + ([f" ({cuts})"] if cuts else []))
            self.parts_box.addWidget(label)
        self.size_label.setText(size_text(product.fmap, unit))
        facts = [fact_text(fact, unit) for fact in p.facts]
        level = "warn" if any(lv == "warn" for _, lv in facts) else "info"
        self.notes.set_text("\n".join(text for text, _ in facts), level)
        self.notes.setVisible(bool(facts))
        self.highlight.set_plot_state("top" if on_plot else "")
        self.tag.setVisible(on_plot)
        self.plot_button.setEnabled(not on_plot)
        self.save_button.setEnabled(not saved)
        self.save_button.setText("Saved" if saved else "Save to library")
        self.save_button.setToolTip(
            f"The product is in the library as {saved}" if saved else "Add it to the library"
        )

    def part_lines(self) -> list[str]:
        """The maps the product was made of, as shown (one line each)."""
        items = (self.parts_box.itemAt(i).widget() for i in range(self.parts_box.count()))
        return [label.text() for label in items if label is not None]


class CombineBar(QWidget):
    """The footer: how to combine the ticked maps, a preview and Create product. Its header
    folds the options away; below two ticked maps it says only that two are needed.
    Settings protocol: whether the options are open."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._ticked = 0
        self._open = True
        self.toggle = QToolButton()
        self.toggle.setProperty("kit", "tool")
        self.toggle.setIconSize(QSize(14, 14))
        self.toggle.setToolTip("Show or hide the ways to combine the ticked maps")
        self.toggle.setAccessibleName("Combine options")
        self.title = section_label("Combine ticked maps")
        self.need = hint(NEED_TWO_HINT)
        self.method = SegmentedControl(size="sm", expand=True)
        for value, (text, _short, explanation, details) in METHODS.items():
            self.method.add_option(value, text, f"{explanation} {details}")
        self.method.setAccessibleName("How to combine the ticked maps")
        self.explanation = hint()
        self.preview = Note("", "muted")
        self.preview.setAccessibleName("Preview of the product")
        _no_minimum_width(self.preview)  # a long map name must not widen the panel
        self.create_button = QPushButton("Create product")
        self.create_button.setProperty("kit", "primary")
        icons.set_icon(self.create_button, "git-merge", "accent-fg")
        self.create_button.setIconSize(QSize(14, 14))
        self.create_button.setFixedHeight(26)
        self.body = QWidget()
        body = QVBoxLayout(self.body)
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(7)
        body.addWidget(self.method)
        body.addWidget(self.explanation)
        body.addWidget(self.preview)
        row = QHBoxLayout()
        row.addWidget(self.create_button)
        row.addStretch(1)
        body.addLayout(row)
        head = QHBoxLayout()
        head.setSpacing(4)
        head.addWidget(self.title, 1)
        head.addWidget(self.toggle)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 6, 10, 10)
        layout.setSpacing(5)
        layout.addLayout(head)
        layout.addWidget(self.need)
        layout.addWidget(self.body)
        self.method.valueChanged.connect(self._explain)
        self.toggle.clicked.connect(lambda: self.set_open(not self._open))
        self._explain(self.method.value())
        self._sync()

    def _explain(self, value: str) -> None:
        self.explanation.setText(METHODS[value][2])
        self.explanation.setToolTip(METHODS[value][3])

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        labels = {value: (texts[0], texts[1]) for value, texts in METHODS.items()}
        fit_segments(self.method, labels, self.body.contentsRect().width() or self.width() - 24)

    def _sync(self) -> None:
        enough = self._ticked >= 2
        self.need.setVisible(not enough)
        self.body.setVisible(enough and self._open)
        self.toggle.setVisible(enough)
        icons.set_icon(self.toggle, "chevron-down" if self._open else "chevron-right", "muted")

    def is_open(self) -> bool:
        """The options show (when two or more maps are ticked)."""
        return self._open

    def set_open(self, open_: bool) -> None:
        self._open = bool(open_)
        self._sync()

    def settings_value(self) -> bool:
        return self.is_open()

    def set_settings_value(self, value) -> bool:
        if isinstance(value, str):
            value = value.lower() == "true"
        if not isinstance(value, bool):
            return False
        self.set_open(value)
        return True

    def show_preview(self, preview: CombinePreview, unit: Unit) -> None:
        n = len(preview.entries)
        self._ticked = n
        self.title.setText(f"COMBINE {n} TICKED MAP{'S' if n != 1 else ''}")
        self._sync()
        if preview.fmap is None:
            problem = preview.problem
            text = problem[:1].upper() + problem[1:] if problem else NEED_TWO
            self.preview.set_text(text, "warn" if n >= 2 else "muted")
            self.create_button.setEnabled(False)
            self.create_button.setToolTip(text)
            return
        facts = [fact_text(fact, unit) for fact in preview.facts]
        level = "warn" if any(lv == "warn" for _, lv in facts) else "ok"
        lines = [f"Result: {size_text(preview.fmap, unit)}", *(text for text, _ in facts)]
        self.preview.set_text("\n".join(lines), level)
        self.create_button.setEnabled(True)
        self.create_button.setToolTip(f"{COMBINE_METHODS[preview.method]}: make the product")


class PlotNameChip(ElidedLabel):
    """The library map(s) or the product on the plot, for the status bar: it takes the room
    its text needs and elides it, down to nothing, when the bar is short of room."""

    def __init__(self, parent=None):
        super().__init__("", parent)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        self.setContentsMargins(4, 0, 4, 0)
        self.setAccessibleName("Library map on the plot")

    def sizeHint(self) -> QSize:
        margins = self.contentsMargins()
        text = self.fontMetrics().horizontalAdvance(self.text()) + 2  # (sub-pixel advances)
        return QSize(text + margins.left() + margins.right(), super().sizeHint().height())

    def minimumSizeHint(self) -> QSize:
        return QSize(0, super().minimumSizeHint().height())


class LibraryPanel(QWidget):
    """The processed sweep, the list of maps with the plot options, the product and the
    Combine footer."""

    shown = Signal()  # the panel came on screen

    def __init__(self, parent=None):
        super().__init__(parent)
        self.unit = Unit.CM1
        self.preview_due = False  # the Combine preview waits for the panel to be shown
        self.limits_timer = QTimer(self)  # draws maps plotted cut once typing in limits pauses
        self.limits_timer.setSingleShot(True)
        self.limits_timer.setInterval(LIMITS_DELAY_MS)
        self.rows: dict[int, EntryRow] = {}
        self._expanded: set[int] = set()

        self.sweep = SweepCard()
        self.sweep.hide()
        self.save_button = self.sweep.add_button  # adds the processed sweep
        self.maps_label = section_label("Maps")
        self.show_ticked_link = LinkButton("Show", "Show the ticked maps on the plot")
        self.show_ticked_link.hide()
        self.load_button = small_button(
            "Load table…", "upload", "Add exported R(B)/R(0) tables to the library"
        )
        self.list_card = Card()
        self.empty = DropZone(
            "No maps yet. Add the processed sweep, or drop exported tables here.",
            icon="library-big",
        )
        self.list_hint = hint(
            "Tick maps to plot them together; click a map to bring it on top. The top map is "
            "the one you pick on and export."
        )
        self.opacity = OpacitySlider()
        self.product_box = ProductBox()
        self.product_box.hide()
        self.full_energy_row = SwitchRow("Plot the full ranges")
        self.full_energy_row.setToolTip("Off: each map is plotted cut to its E and B limits")
        self.full_energy_row.title_label.setFont(scaled_font(self, 0.94))
        self.full_energy = self.full_energy_row.switch
        self.full_energy.setChecked(True)
        self.auto_field_row = SwitchRow(
            "Field from the table header", "Off: loaded tables use the Sample's custom range"
        )
        self.auto_field = self.auto_field_row.switch
        self.auto_field.setChecked(True)
        for row in (self.full_energy_row, self.auto_field_row):  # a narrow panel: no sideways
            row.title_label.setWordWrap(True)

        body = QWidget()
        self.body = body
        layout = QVBoxLayout(body)
        layout.setContentsMargins(14, 12, 14, 16)
        layout.setSpacing(12)
        layout.addWidget(self.sweep)
        head = QHBoxLayout()
        head.setSpacing(6)
        head.addWidget(self.maps_label, 1)
        head.addWidget(self.show_ticked_link)
        head.addWidget(self.load_button)
        maps = QVBoxLayout()
        maps.setSpacing(7)
        maps.addLayout(head)
        maps.addWidget(self.opacity)  # the plot options above the list, so they stay in view
        maps.addWidget(self.full_energy_row)
        maps.addWidget(self.list_card)
        maps.addWidget(self.empty)
        maps.addWidget(self.list_hint)
        layout.addLayout(maps)
        layout.addWidget(self.product_box)
        layout.addWidget(self.auto_field_row)
        layout.addStretch(1)
        self.scroll = QScrollArea()
        self.scroll.setFocusPolicy(Qt.FocusPolicy.NoFocus)  # Tab goes to the controls inside
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll.setWidget(body)

        self.plot_chip = PlotNameChip(self)  # for the status bar (see install)
        self.plot_chip.hide()
        self.combine = CombineBar()
        self.method = self.combine.method
        self.create_button = self.combine.create_button

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(self.scroll, 1)
        outer.addWidget(Divider())
        outer.addWidget(self.combine)

    def show_entries(self, entries: list[LibraryEntry]) -> None:
        """Rebuild the rows (entries added or removed); expanded rows stay open."""
        box = self.list_card.box
        clear_layout(box)
        self.rows = {}
        for i, entry in enumerate(entries):
            if i:
                box.addWidget(Divider())
            row = EntryRow(entry)
            row.set_unit(self.unit)
            row.show_entry(entry, self.unit)
            row.set_expanded(entry.key in self._expanded)
            row.expandedChanged.connect(self._remember_expanded)
            box.addWidget(row)
            self.rows[entry.key] = row
        self._expanded &= set(self.rows)
        self.list_card.setVisible(bool(entries))
        self.empty.setVisible(not entries)

    def _remember_expanded(self, key: int, expanded: bool) -> None:
        (self._expanded.add if expanded else self._expanded.discard)(key)

    def show_plotted(
        self, plotted: list[int], shown: bool, ticked: int, problems: dict[int, str]
    ) -> None:
        """Highlight the rows on the plot (*plotted* keys, bottom first) when the plot shows
        the ticked maps (*shown*), and say why a ticked map is not drawn (*problems*, by key);
        else offer to show the *ticked* ones again."""
        on_plot = [key for key in plotted if not problems.get(key)] if shown else []
        top = on_plot[-1] if on_plot else None
        for key, row in self.rows.items():
            if key == top:
                row.set_plot_state("top", "Top" if len(on_plot) > 1 else "On plot")
            else:
                row.set_plot_state("plotted" if key in on_plot else "", "", problems.get(key, ""))
        self.show_ticked_link.setVisible(bool(ticked) and not shown)
        self.opacity.setEnabled(len(on_plot) > 1)
        self.opacity.setToolTip(
            "Maps above the bottom one are drawn this opaque, so that overlaps show"
            if len(on_plot) > 1
            else "Tick two maps to see how they overlap"
        )

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self.shown.emit()

    def reveal_product(self) -> None:
        """Scroll the product box into view (once it is laid out)."""

        def reveal() -> None:
            QApplication.sendPostedEvents(None, QEvent.Type.LayoutRequest)  # its place
            self.scroll.ensureWidgetVisible(self.product_box, 0, 0)

        QTimer.singleShot(0, self, reveal)

    def set_unit(self, unit: Unit, entries: list[LibraryEntry]) -> None:
        self.unit = Unit(unit)
        for entry in entries:
            row = self.rows.get(entry.key)
            if row is not None:
                row.set_unit(self.unit)
                row.show_entry(entry, self.unit)


# ---------------------------------------------------------------------- actions
def _panel(window) -> LibraryPanel:
    return window.panels["library"]


def refresh_preview(window) -> CombinePreview:
    """Show the preview of combining the ticked maps now (it follows edits after a pause)."""
    panel, c = _panel(window), window.controller
    panel.preview_due = False
    preview = c.combine_preview(panel.method.value())
    panel.combine.show_preview(preview, c.unit)
    return preview


@user_action("Plot map")
def apply_limits(window) -> None:
    """Draw the ticked maps with the limits typed now, when they are plotted cut to them (the
    panel does it once typing pauses); not while the top map's limits cannot be used (they
    are marked)."""
    panel, c = _panel(window), window.controller
    panel.limits_timer.stop()
    plotted = c.plotted_entries()
    row = panel.rows.get(plotted[-1].key) if plotted else None
    if row is not None and not row.check():
        return
    c.redraw_ticked()


@user_action("Load table")
def load_tables(window, paths: list[str] | None = None) -> None:
    """Add tables to the library; without *paths* they are chosen in a dialog."""
    if paths is None:
        paths = open_files(window, "Load processed tables", TABLE_FILTER)
    if not paths:
        return
    c = window.controller
    field_values = None
    if not _panel(window).auto_field.isChecked():
        with in_panel("sample"):
            field_values = c.processing.sample_field.values()
    for path in paths:
        c.load_table(path, field_values)


@user_action("Add to library")
def add_processed(window) -> None:
    window.controller.add_processed()


@user_action("Plot map")
def plot_entry(window, key: int) -> None:
    """Show library map *key* on top of the ticked maps."""
    c = window.controller
    c.plot_entry(c.entry(key))


@user_action("Plot map")
def tick_entry(window, key: int, on: bool) -> None:
    """Tick (show on the plot) or untick library map *key*."""
    c = window.controller
    c.update_entry(c.entry(key), used=on)


@user_action("Plot map")
def show_ticked(window) -> None:
    window.controller.show_ticked()


@user_action("Show processed sweep")
def show_processed(window) -> None:
    window.controller.show_processed()


@user_action("Remove map")
def remove_entry(window, key: int) -> None:
    c = window.controller
    c.remove_entry(c.entry(key))


@user_action("Create product")
def make_product(window, method: str | None = None) -> None:
    """Combine the ticked maps by *method* (default: the one chosen in the panel)."""
    panel = _panel(window)
    window.controller.make_product(method or panel.method.value())
    panel.reveal_product()


@user_action("Plot product")
def plot_product(window) -> None:
    window.controller.plot_product()


@user_action("Save product")
def save_product(window) -> None:
    window.controller.save_product()


@user_action("Export product")
def export_product(window, path: str | None = None) -> Path | None:
    """Write the product as a data table as it is plotted (baseline region applied, display
    unit), to *path* or a file chosen in a dialog that suggests its name; next to it
    ``<file>_provenance.txt`` says how it was made (the table format has no room for
    comments)."""
    c = window.controller
    product = c.product
    if product is None:
        raise ValueError("there is no product to export: combine ticked maps first")
    if path is None:
        suggested = str(Path(last_dir()) / f"{file_name(product.name)}.csv")
        path, _ = QFileDialog.getSaveFileName(window, "Export product", suggested, TABLE_FILTER)
        if path:
            set_last_dir(str(Path(path).parent))
    if not path:
        return None
    out = Path(path)
    if not out.suffix:
        out = out.with_suffix(".csv")
    fmap, region = c.product_table()
    save_tsv(fmap, out)
    note = out.with_name(f"{out.stem}_provenance.txt")
    lines = [product.name, *product.provenance.lines(c.unit)]
    if region is not None:
        lines.append(f"Baseline region {energy_range_text(region, c.unit)}, as plotted")
    lines.append(size_text(product.fmap, c.unit))
    note.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Exported the product to %s (how it was made: %s)", out, note.name)
    return out


@user_action("Figure")
def product_figure(window) -> None:
    """Show the product and open the journal figure export."""
    window.controller.plot_product()
    window.commands["export_figure"].trigger()


@user_action("Discard product")
def discard_product(window) -> None:
    window.controller.discard_product()


def install(window) -> None:
    c = window.controller
    panel = LibraryPanel()
    page = window.add_panel(
        "library", "Library", "library-big", "Processed maps", panel,
        "Processed maps to plot together, compare and combine.",
    )  # fmt: skip
    syncing = False

    def connect_row(row: EntryRow) -> None:
        entry = c.entry(row.key)

        def push(**changes) -> None:
            if not syncing:
                c.update_entry(entry, **changes)

        def on_limits(**changes) -> None:
            row.check()
            push(**changes)  # kept as typed; maps plotted cut follow once typing pauses
            if not syncing and entry.used and not c.full_range:
                limits_timer.start()

        row.use.toggled.connect(
            lambda on, k=row.key: None if syncing else tick_entry(window, k, on)
        )
        for edit in (row.e_min, row.e_max):
            edit.valueChanged.connect(lambda: on_limits(energy_cut=row.energy_cut()))
        for edit in (row.b_min, row.b_max):
            edit.textChanged.connect(lambda _t: on_limits(field_cut=row.field_cut()))
        for edit in row.fields:
            edit.editingFinished.connect(lambda: limits_timer.isActive() and apply_limits(window))
        row.raiseRequested.connect(lambda key: plot_entry(window, key))
        row.removeRequested.connect(lambda key: remove_entry(window, key))

    def show_plotted() -> None:
        shown = c.showing() == SHOWS_MAPS
        problems = {layer.entry.key: layer.problem for layer in c.overlay_layers()}
        keys = [e.key for e in c.plotted_entries()]
        panel.show_plotted(keys, shown, len(c.ticked()), problems)

    def rebuild() -> None:
        nonlocal syncing
        syncing = True
        try:
            panel.show_entries(c.library)
        finally:
            syncing = False
        for row in panel.rows.values():
            connect_row(row)
        n = len(c.library)
        page.set_subtitle(
            f"{n} map{'s' if n != 1 else ''}: tick to plot, compare and combine."
            if n
            else "Processed maps to plot together, compare and combine."
        )
        show_plotted()
        show_sweep()
        schedule_preview()

    def pull(key: int) -> None:
        """Show an entry changed through the controller."""
        nonlocal syncing
        row = panel.rows.get(key)
        if row is None:
            return
        syncing = True
        try:
            row.show_entry(c.entry(key), panel.unit)
        finally:
            syncing = False
        schedule_preview()

    def show_sweep() -> None:
        result = c.processed_result()
        panel.sweep.setVisible(result is not None)
        if result is not None:
            name = c.sweep_name()
            parts = map_parts(KIND_LABELS[PlotKind.RATIO], result.ratio, panel.unit)
            in_library = c.processed_in_library()
            panel.sweep.show_sweep(name, parts, c.showing() == SHOWS_PROCESS, in_library)
        has_map = result is not None
        panel.save_button.setEnabled(has_map)
        panel.save_button.setToolTip(ADD_TIP if has_map else NOTHING_TO_ADD)
        add_action.setEnabled(has_map and not c.processed_in_library())

    def show_product() -> None:
        product = c.product
        panel.product_box.setVisible(product is not None)
        if product is None:
            return
        saved = ""
        if product.saved is not None:
            saved = next((e.name for e in c.library if e.key == product.saved), "")
        on_plot = c.showing() == SHOWS_PRODUCT
        panel.product_box.show_product(product, panel.unit, on_plot, saved)
        window.setWindowTitle(window.window_title())  # a renamed product shown

    def on_unit(_old=None, new=None) -> None:
        nonlocal syncing
        syncing = True
        try:
            panel.set_unit(c.unit, c.library)
        finally:
            syncing = False
        show_sweep()
        show_product()
        schedule_preview()

    # maps plotted cut to their limits are drawn again once typing pauses (or on Return)
    limits_timer = panel.limits_timer
    limits_timer.timeout.connect(lambda: apply_limits(window))

    # the preview of combining the ticked maps, computed once typing pauses
    preview_timer = QTimer(panel)
    preview_timer.setSingleShot(True)
    preview_timer.setInterval(PREVIEW_DELAY_MS)

    def schedule_preview() -> None:
        preview_timer.start()

    def on_preview_timer() -> None:
        if panel.isVisible():
            refresh_preview(window)
        else:
            panel.preview_due = True  # computed when the panel is shown

    preview_timer.timeout.connect(on_preview_timer)
    panel.shown.connect(lambda: refresh_preview(window) if panel.preview_due else None)
    panel.method.valueChanged.connect(lambda _v: refresh_preview(window))

    chip = panel.plot_chip
    window.add_status_chip(chip)  # after "Showing a library map" (and the baseline chip)

    def show_chip() -> None:
        title = c.plot_title() if c.result_source == "library" else ""
        chip.setText(title)
        chip.setToolTip(f"On the plot: {title}" if title else "")
        chip.setVisible(bool(title))

    def on_result() -> None:
        show_plotted()
        show_sweep()
        show_product()
        show_chip()

    add_action = QAction("Add processed map to library", window)
    add_action.setToolTip(ADD_TIP)
    icons.set_icon(add_action, "plus")
    add_action.triggered.connect(lambda: add_processed(window))
    window.add_file_action(add_action)

    c.libraryChanged.connect(rebuild)
    c.entryChanged.connect(pull)
    c.unitChanged.connect(on_unit)
    c.restored.connect(on_unit)
    c.resultChanged.connect(on_result)
    c.plottedChanged.connect(on_result)
    c.selectionChanged.connect(show_plotted)  # a map below may not have that plot kind
    c.productChanged.connect(show_product)
    c.productChanged.connect(show_chip)
    c.opacityChanged.connect(lambda value: panel.opacity.set_value(value))
    panel.unit = c.unit
    rebuild()
    on_result()
    refresh_preview(window)

    panel.save_button.clicked.connect(lambda: add_processed(window))
    panel.sweep.showRequested.connect(lambda: show_processed(window))
    panel.show_ticked_link.clicked.connect(lambda: show_ticked(window))
    panel.load_button.clicked.connect(lambda: load_tables(window))
    panel.empty.clicked.connect(lambda: load_tables(window))
    panel.empty.filesDropped.connect(lambda paths: load_tables(window, paths))
    FileDrops(panel.list_card, lambda paths: load_tables(window, paths))
    panel.opacity.valueChanged.connect(c.set_overlay_opacity)
    panel.full_energy.toggled.connect(lambda on: set_full_range(window, on))
    panel.create_button.clicked.connect(lambda: make_product(window))
    box = panel.product_box
    box.name_edit.editingFinished.connect(lambda: rename_product(window, box.name_edit.text()))
    box.plot_button.clicked.connect(lambda: plot_product(window))
    box.save_button.clicked.connect(lambda: save_product(window))
    box.table_button.clicked.connect(lambda: export_product(window))
    box.figure_button.clicked.connect(lambda: product_figure(window))
    box.discard_button.clicked.connect(lambda: discard_product(window))

    p = window.persistence
    if p is not None:
        p.bind("library/full_energy", panel.full_energy)
        p.bind("library/auto_field", panel.auto_field)
        p.bind("library/opacity", panel.opacity)
        p.bind("library/method", panel.method)
        p.bind("library/combine_open", panel.combine)


@user_action("Plot map")
def set_full_range(window, full: bool) -> None:
    window.controller.set_full_range(full)


@user_action("Rename product")
def rename_product(window, name: str) -> None:
    if window.controller.product is not None:
        window.controller.rename_product(name)
