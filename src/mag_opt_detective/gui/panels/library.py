"""Library panel: processed maps to plot again, merge or average.

The maps live in ``controller.library`` (cm^-1). Each row has a tick (used by Merge and
Average), the name, a meta line (kind · fields · E range in the display unit) and a Plot
button. It opens to the map's cut limits (E in the display unit, kept in cm^-1, and B in T; an
empty limit keeps everything) and a remove button. Merge and Average need at least two ticked
maps.
"""

from __future__ import annotations

import re

import numpy as np
from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QPainter
from PySide6.QtWidgets import (
    QCheckBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from mag_opt_detective.core.units import Unit, from_cm1
from mag_opt_detective.gui import icons
from mag_opt_detective.gui.controller import LibraryEntry, in_panel, user_action
from mag_opt_detective.gui.display import format_range, unit_text
from mag_opt_detective.gui.panels.common import (
    Card,
    Divider,
    DropZone,
    FileDrops,
    SwitchRow,
    UnitField,
    clear_layout,
    hint,
    labelled,
    scaled_font,
    small_button,
)
from mag_opt_detective.gui.panels.files import fmt_field
from mag_opt_detective.gui.widgets import (
    TABLE_FILTER,
    EnergyEdit,
    FloatEdit,
    FlowLayout,
    open_files,
    parse_float,
)

NEED_TWO = "Tick at least two maps"
SAVE_TIP = "Add the R(B)/R(0) map shown to the library"
NOTHING_TO_SAVE = "Process a sweep first: there is no map to save yet"
TOOLTIPS = {
    "energy": "Join maps measured in different spectral ranges, each cut to its E limits",
    "field": "Join maps measured over different field ranges, each cut to its B limits",
    "average": "Average repeated measurements (same fields), each cut to its E and B limits",
}


def meta_parts(entry: LibraryEntry, unit: Unit) -> list[str]:
    """``["R(B)/R(0)", "64 fields", "350 – 7800 cm⁻¹"]`` (energies in *unit*)."""
    fmap = entry.fmap
    n = fmap.field.size
    e_lo, e_hi = from_cm1(np.array([fmap.energy.min(), fmap.energy.max()]), unit)
    fields = f"{n} field{'s' if n != 1 else ''}"
    return [entry.kind, fields, format_range(e_lo, e_hi, unit_text(unit))]


def name_parts(name: str) -> list[str]:
    """*name* cut after each ``_``, ``-`` or space, where a long name may wrap."""
    return [part for part in re.split(r"(?<=[_\- ])", name) if part]


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


class EntryRow(QWidget):
    """One library map: tick, name, meta line and Plot; it opens to the cut limits and the
    remove button.

    The name has the first line to itself (but for the expand button) and wraps after a ``_``;
    the meta line wraps between its parts. Both stay whole in a narrow panel.
    """

    plotRequested = Signal(int)
    removeRequested = Signal(int)
    expandedChanged = Signal(int, bool)

    def __init__(self, entry: LibraryEntry, parent=None):
        super().__init__(parent)
        self.key = entry.key
        self.use = QCheckBox()
        self.use.setAccessibleName(f"Use {entry.name}")
        self.use.setToolTip("Ticked maps are merged or averaged")
        self.name_label = PartsLabel(mode=Qt.TextElideMode.ElideMiddle)
        self.name_label.setFont(scaled_font(self.name_label, 0.98, bold=True))
        self.name_label.set_parts(name_parts(entry.name))
        self.name_label.setToolTip(entry.source or f"{entry.name} (saved from a processed map)")
        self.meta_label = PartsLabel(" · ")
        self.meta_label.setProperty("kit", "muted")
        self.meta_label.setFont(scaled_font(self.meta_label, 0.9))
        self.plot_button = small_button("Plot", tooltip=f"Show {entry.name} on the plots")
        self.plot_button.setAccessibleName(f"Plot {entry.name}")
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
        grid = QGridLayout(self.cut_box)
        grid.setContentsMargins(31, 0, 8, 9)
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(6)
        for i, (text, edit) in enumerate(
            (("E min", self.e_min), ("E max", self.e_max), ("B min", self.b_min),
             ("B max", self.b_max))
        ):  # fmt: skip
            grid.addWidget(labelled(text, self.fields[edit]), i // 2, i % 2)
        grid.addWidget(self.remove_button, 2, 0, 1, 2, Qt.AlignmentFlag.AlignLeft)
        self.cut_box.setVisible(False)

        meta_line = QHBoxLayout()
        meta_line.setContentsMargins(0, 0, 0, 0)
        meta_line.setSpacing(6)
        meta_line.addWidget(self.meta_label, 1)
        meta_line.addWidget(self.plot_button, 0, Qt.AlignmentFlag.AlignVCenter)
        head = QGridLayout()
        head.setContentsMargins(8, 6, 4, 7)
        head.setHorizontalSpacing(6)
        head.setVerticalSpacing(1)
        tick = QWidget()  # keeps the tick level with the name's first line
        tick.setFixedHeight(line)
        tick_layout = QVBoxLayout(tick)
        tick_layout.setContentsMargins(0, 0, 0, 0)
        tick_layout.addWidget(self.use, 0, Qt.AlignmentFlag.AlignVCenter)
        top = Qt.AlignmentFlag.AlignTop
        head.addWidget(tick, 0, 0, top)
        head.addWidget(self.name_label, 0, 1)
        head.addWidget(self.expand_button, 0, 2, top)
        head.addLayout(meta_line, 1, 1, 1, 2)
        head.setColumnStretch(1, 1)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addLayout(head)
        layout.addWidget(self.cut_box)

        self.plot_button.clicked.connect(lambda: self.plotRequested.emit(self.key))
        self.remove_button.clicked.connect(lambda: self.removeRequested.emit(self.key))
        self.expand_button.toggled.connect(self.set_expanded)

    def is_expanded(self) -> bool:
        return not self.cut_box.isHidden()

    def set_expanded(self, expanded: bool) -> None:
        if self.expand_button.isChecked() != expanded:
            self.expand_button.setChecked(expanded)  # comes back here
            return
        self.cut_box.setVisible(expanded)
        icons.set_icon(self.expand_button, "chevron-down" if expanded else "chevron-right", "muted")
        self.expandedChanged.emit(self.key, expanded)

    def set_unit(self, unit: Unit) -> None:
        for edit in (self.e_min, self.e_max):
            edit.set_unit(unit)
            self.fields[edit].set_unit(unit_text(unit))

    def show_entry(self, entry: LibraryEntry, unit: Unit) -> None:
        """Show the entry's tick, meta line and limits (fields holding a value keep it)."""
        if self.use.isChecked() != entry.used:
            self.use.setChecked(entry.used)
        self.meta_label.set_parts(meta_parts(entry, unit))
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

    def check(self) -> None:
        """Flag limits that are not numbers or are reversed."""
        for lo, hi, values in (
            (self.e_min, self.e_max, self.energy_cut()),
            (self.b_min, self.b_max, self.field_cut()),
        ):
            reversed_ = None not in values and values[0] >= values[1]
            for edit, value in zip((lo, hi), values, strict=True):
                not_a_number = value is None and bool(edit.text().strip())
                self.fields[edit].set_invalid(reversed_ or not_a_number)


class LibraryPanel(QWidget):
    """Save / Load, the list of maps, the plot options and the batch actions."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.unit = Unit.CM1
        self.rows: dict[int, EntryRow] = {}
        self._expanded: set[int] = set()

        self.save_button = small_button("Save current map", "save", SAVE_TIP)
        self.load_button = small_button(
            "Load table…", "upload", "Add exported R(B)/R(0) tables to the library"
        )
        self.list_card = Card()
        self.empty = DropZone(
            "No maps yet. Save the current map after processing, or drop exported tables here.",
            icon="library-big",
        )
        self.full_energy_row = SwitchRow(
            "Plot the full energy range", "Off: Plot cuts each map to its E limits"
        )
        self.full_energy = self.full_energy_row.switch
        self.full_energy.setChecked(True)
        self.auto_field_row = SwitchRow(
            "Field from the table header", "Off: loaded tables use the Sample's custom range"
        )
        self.auto_field = self.auto_field_row.switch
        self.auto_field.setChecked(True)

        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setContentsMargins(14, 12, 14, 16)
        layout.setSpacing(12)
        top = QWidget()
        top.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        top_flow = FlowLayout(top, spacing=6, row_spacing=6)
        top_flow.addWidget(self.save_button)
        top_flow.addWidget(self.load_button)
        layout.addWidget(top)
        layout.addWidget(self.list_card)
        layout.addWidget(self.empty)
        layout.addWidget(
            hint(
                "Ticked maps are merged or averaged, each cut to its own E and B limits. "
                "Empty limits keep everything."
            )
        )
        layout.addWidget(self.full_energy_row)
        layout.addWidget(self.auto_field_row)
        layout.addStretch(1)
        scroll = QScrollArea()
        scroll.setFocusPolicy(Qt.FocusPolicy.NoFocus)  # Tab goes to the controls inside
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(body)

        self.count_label = hint()
        self.merge_energy_button = small_button("Merge by energy", "git-merge")
        self.merge_field_button = small_button("Merge by field")
        self.average_button = small_button("Average")
        foot = QWidget()
        foot_layout = QVBoxLayout(foot)
        foot_layout.setContentsMargins(14, 10, 14, 10)
        foot_layout.setSpacing(6)
        foot_layout.addWidget(self.count_label)
        buttons = QWidget()
        buttons.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        flow = FlowLayout(buttons, spacing=6, row_spacing=6)
        for button in self.batch_buttons():
            flow.addWidget(button)
        foot_layout.addWidget(buttons)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(scroll, 1)
        outer.addWidget(Divider())
        outer.addWidget(foot)

    def batch_buttons(self) -> tuple[QToolButton, QToolButton, QToolButton]:
        return self.merge_energy_button, self.merge_field_button, self.average_button

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
        self.show_counts(entries)

    def _remember_expanded(self, key: int, expanded: bool) -> None:
        (self._expanded.add if expanded else self._expanded.discard)(key)

    def show_counts(self, entries: list[LibraryEntry]) -> None:
        n, total = sum(e.used for e in entries), len(entries)
        text = f"{n} of {total} ticked"
        if total and n < 2:
            text += ". Tick at least two to merge or average."
        self.count_label.setText(text if total else "No maps in the library.")
        for button, what in zip(self.batch_buttons(), TOOLTIPS, strict=True):
            button.setEnabled(n >= 2)
            button.setToolTip(TOOLTIPS[what] if n >= 2 else f"{NEED_TWO} to use this")

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


@user_action("Save map")
def save_current(window) -> None:
    window.controller.save_current_map()


@user_action("Plot map")
def plot_entry(window, key: int) -> None:
    c = window.controller
    c.plot_entry(c.entry(key), cut_energy=not _panel(window).full_energy.isChecked())


@user_action("Remove map")
def remove_entry(window, key: int) -> None:
    c = window.controller
    c.remove_entry(c.entry(key))


@user_action("Merge by energy")
def merge_by_energy(window) -> None:
    window.controller.merge_by_energy()


@user_action("Merge by field")
def merge_by_field(window) -> None:
    window.controller.merge_by_field()


@user_action("Average")
def average(window) -> None:
    window.controller.average()


def install(window) -> None:
    c = window.controller
    panel = LibraryPanel()
    page = window.add_panel(
        "library", "Library", "library-big", "Processed maps", panel,
        "Processed maps to plot again, merge or average.",
    )  # fmt: skip
    syncing = False

    def connect_row(row: EntryRow) -> None:
        entry = c.entry(row.key)

        def push(**changes) -> None:
            if not syncing:
                c.update_entry(entry, **changes)

        row.use.toggled.connect(lambda on: push(used=on))
        for edit in (row.e_min, row.e_max):
            edit.valueChanged.connect(lambda r=row: (r.check(), push(energy_cut=r.energy_cut())))
        for edit in (row.b_min, row.b_max):
            edit.textChanged.connect(lambda _t, r=row: (r.check(), push(field_cut=r.field_cut())))
        row.plotRequested.connect(lambda key: plot_entry(window, key))
        row.removeRequested.connect(lambda key: remove_entry(window, key))

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
            f"{n} map{'s' if n != 1 else ''} to plot again, merge or average."
            if n
            else "Processed maps to plot again, merge or average."
        )

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
        panel.show_counts(c.library)

    def on_unit(_old=None, new=None) -> None:
        nonlocal syncing
        syncing = True
        try:
            panel.set_unit(c.unit, c.library)
        finally:
            syncing = False

    c.libraryChanged.connect(rebuild)
    c.entryChanged.connect(pull)
    c.unitChanged.connect(on_unit)
    c.restored.connect(on_unit)
    panel.unit = c.unit
    rebuild()

    panel.save_button.clicked.connect(lambda: save_current(window))

    def sync_save() -> None:
        has_map = c.result is not None
        panel.save_button.setEnabled(has_map)
        panel.save_button.setToolTip(SAVE_TIP if has_map else NOTHING_TO_SAVE)

    c.resultChanged.connect(sync_save)
    sync_save()
    panel.load_button.clicked.connect(lambda: load_tables(window))
    panel.empty.clicked.connect(lambda: load_tables(window))
    panel.empty.filesDropped.connect(lambda paths: load_tables(window, paths))
    FileDrops(panel.list_card, lambda paths: load_tables(window, paths))
    panel.merge_energy_button.clicked.connect(lambda: merge_by_energy(window))
    panel.merge_field_button.clicked.connect(lambda: merge_by_field(window))
    panel.average_button.clicked.connect(lambda: average(window))

    p = window.persistence
    if p is not None:
        p.bind("library/full_energy", panel.full_energy)
        p.bind("library/auto_field", panel.auto_field)
