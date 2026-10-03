"""The fit area of a model card: how picked curves map to branches, Fit, and the results.

The picked curves come from the Points panel; each is assigned to a branch (or skipped), by
branch, by sorted energy or to the nearest branch. A fit shows every free parameter ± sigma
(energies in the display unit) and chi², previewed on the map until it is applied or
discarded; the results can be copied or exported as a tab-separated table.
"""

from __future__ import annotations

from typing import Protocol

from PySide6.QtCore import QSignalBlocker, Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from mag_opt_detective.core.fitting import Assignment, FitResult
from mag_opt_detective.core.units import Unit
from mag_opt_detective.gui import icons
from mag_opt_detective.gui.controller import EXPECTED_ERRORS
from mag_opt_detective.gui.display import UNIT_TEXT
from mag_opt_detective.gui.inspector import model_state as ms
from mag_opt_detective.gui.inspector.model_widgets import (
    FIELD_HEIGHT,
    VELOCITY_UNIT,
    CurveDot,
    TableBox,
    mono_label,
    muted_label,
    tool_button,
)
from mag_opt_detective.gui.kit import SegmentedControl
from mag_opt_detective.gui.panels.common import (
    ElidedLabel,
    LinkButton,
    Note,
    WidthWatcher,
    fit_segments,
    scaled_font,
    section_label,
    small_button,
)

SKIP = "Skip"
MODE_LABELS = {  # long and short labels of the assignment modes (narrow inspector)
    value.value: (text, "Branch" if value is Assignment.BRANCH else text)
    for value, text, _tip in ms.ASSIGNMENTS
}
NO_POINTS = "No picked points yet. Pick points on the map, then fit them here."
ORDINALS = ("Lowest", "2nd lowest", "3rd lowest")


class FitOwner(Protocol):
    """What the fit area needs from the section (``models.Models``)."""

    @property
    def unit(self) -> Unit: ...

    def picked(self) -> list[tuple[str, int, QColor]]: ...

    def fit(self, entry: ms.ModelEntry) -> None: ...

    def is_fitting(self, entry: ms.ModelEntry) -> bool: ...

    def cancel_fit(self, entry: ms.ModelEntry) -> None: ...

    def result(self, entry: ms.ModelEntry) -> FitResult | None: ...

    def apply(self, entry: ms.ModelEntry) -> None: ...

    def discard(self, entry: ms.ModelEntry) -> None: ...

    def copy_results(self, entry: ms.ModelEntry) -> None: ...

    def export_results(self, entry: ms.ModelEntry) -> None: ...

    def open_points(self) -> None: ...


def rank_name(k: int) -> str:
    return ORDINALS[k] if k < len(ORDINALS) else f"{k + 1}th lowest"


def choices(entry: ms.ModelEntry) -> list[tuple[str, int | None]]:
    """The (text, branch) items of a curve's branch menu for the entry's assignment mode."""
    names = entry.branch_names()
    mode = entry.fit.assignment
    if mode is Assignment.NEAREST:
        return [(SKIP, None), ("Use", 0)] if names else [(SKIP, None)]
    if mode is Assignment.SORTED:
        return [(SKIP, None), *((rank_name(k), k) for k in range(len(names)))]
    return [(SKIP, None), *zip(names, range(len(names)), strict=True)]


def display_unit(text: str, unit: Unit) -> str:
    if text == str(unit):
        return UNIT_TEXT.get(unit, text)
    return VELOCITY_UNIT if text == "1e5 m/s" else text


class FitArea(QWidget):
    """Assignment mode, the curve -> branch table, Fit, the error and the results."""

    def __init__(self, entry: ms.ModelEntry, owner: FitOwner, parent=None):
        super().__init__(parent)
        self.entry, self.owner = entry, owner
        self.close_button = tool_button("x", "Close the fit")
        self.close_button.setFixedSize(20, 20)
        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.addWidget(section_label("Fit to picked points"))
        head.addStretch(1)
        head.addWidget(self.close_button)

        self.mode = SegmentedControl(size="xs", expand=True)
        for value, text, tip in ms.ASSIGNMENTS:
            button = self.mode.add_option(value.value, text, tip)
            button.setFont(scaled_font(button, 0.92))
        self.mode.setAccessibleName("Assignment of the picked points")
        self.mode.setMinimumWidth(10)  # the labels shorten instead (WidthWatcher)
        WidthWatcher(self, lambda width: fit_segments(self.mode, MODE_LABELS, width))
        self.table = TableBox()
        self.table.grid.addWidget(muted_label("Curve", 0.85), 0, 0, 1, 3)
        self.table.grid.addWidget(muted_label("Branch", 0.85), 0, 3)
        self.table.grid.setColumnStretch(1, 1)
        self.table.grid.setColumnStretch(3, 2)
        self.combos: dict[str, QComboBox] = {}
        self._rows: list[list[QWidget]] = []
        self._shown: list[tuple[str, int, str]] = []
        self._error_panel: str | None = None  # the panel that fixes the error shown
        self.empty = Note(NO_POINTS, "info")
        self.points_link = LinkButton("Open Points", "Open the Points panel")
        self.fit_button = QPushButton("Fit")
        self.fit_button.setProperty("kit", "primary")
        self.fit_button.setToolTip("Fit the free parameters to the assigned curves")
        icons.set_icon(self.fit_button, "play", "accent-fg")
        self.cancel_button = LinkButton("Cancel", "Stop the fit")
        self.cancel_button.hide()
        self.error = Note("", "err")
        self.error.hide()
        fit_row = QHBoxLayout()
        fit_row.setContentsMargins(0, 0, 0, 0)
        fit_row.addWidget(self.fit_button)
        fit_row.addWidget(self.cancel_button)
        fit_row.addStretch(1)
        fit_row.addWidget(self.points_link)

        self.results = QWidget()
        self.result_table = TableBox()  # name | value ± sigma | unit, numbers on the right
        self.result_table.grid.setColumnStretch(0, 1)
        self.result_table.grid.setHorizontalSpacing(3)
        self.stats = QLabel()
        self.stats.setProperty("kit", "muted")
        self.stats.setWordWrap(True)
        self.apply_button = small_button("Apply", "check", "Write the fitted values into the model")
        self.discard_button = small_button("Discard", None, "Drop the fit and its preview")
        self.copy_button = LinkButton("Copy", "Copy the results as a table")
        self.export_button = LinkButton("Export…", "Save the results as a table (TSV)")
        result_head = QHBoxLayout()
        result_head.setContentsMargins(0, 0, 0, 0)
        result_head.setSpacing(2)
        result_head.addWidget(section_label("Result"))
        result_head.addStretch(1)
        result_head.addWidget(self.copy_button)
        result_head.addWidget(self.export_button)
        buttons = QHBoxLayout()
        buttons.setContentsMargins(0, 0, 0, 0)
        buttons.setSpacing(6)
        buttons.addWidget(self.apply_button)
        buttons.addWidget(self.discard_button)
        buttons.addStretch(1)
        results = QVBoxLayout(self.results)
        results.setContentsMargins(0, 0, 0, 0)
        results.setSpacing(6)
        results.addLayout(result_head)
        results.addWidget(self.result_table)
        results.addWidget(self.stats)
        results.addLayout(buttons)
        self.results.hide()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 0)
        layout.setSpacing(8)
        layout.addLayout(head)
        layout.addWidget(self.mode)
        layout.addWidget(self.table)
        layout.addWidget(self.empty)
        layout.addLayout(fit_row)
        layout.addWidget(self.error)
        layout.addWidget(self.results)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        self.hide()

        self.mode.valueChanged.connect(self._on_mode)
        self.fit_button.clicked.connect(self._on_fit)
        self.cancel_button.clicked.connect(lambda: owner.cancel_fit(entry))
        self.points_link.clicked.connect(owner.open_points)
        self.apply_button.clicked.connect(lambda: owner.apply(entry))
        self.discard_button.clicked.connect(lambda: owner.discard(entry))
        self.copy_button.clicked.connect(lambda: owner.copy_results(entry))
        self.export_button.clicked.connect(lambda: owner.export_results(entry))

    # --- showing --------------------------------------------------------------------------
    def refresh(self) -> None:
        """Show the picked curves, their branches and the result (if the area is open)."""
        if self.isHidden():
            return
        with QSignalBlocker(self.mode):
            self.mode.set_value(self.entry.fit.assignment.value)
        picked = self.owner.picked()
        shown = [(name, count, color.name()) for name, count, color in picked]
        if shown != self._shown:
            self._build_rows(picked)
            self._shown = shown
        self._fill_combos()
        has_points = bool(picked)
        busy = self.owner.is_fitting(self.entry)
        self.table.setVisible(has_points)
        self.table.setEnabled(not busy)
        self.mode.setEnabled(not busy)
        self.empty.setVisible(not has_points)
        self.points_link.setVisible(not has_points or self._error_panel == "points")
        self.fit_button.setText("Fitting…" if busy else "Fit")
        self.fit_button.setEnabled(has_points and self.entry.is_drawable() and not busy)
        self.cancel_button.setVisible(busy)
        if not self.entry.is_drawable():
            self.fit_button.setToolTip("Enter a valid expression first")
        else:
            self.fit_button.setToolTip("Fit the free parameters to the assigned curves")
        self.refresh_results()

    def _build_rows(self, picked: list[tuple[str, int, QColor]]) -> None:
        grid = self.table.grid
        for widgets in self._rows:
            for widget in widgets:
                grid.removeWidget(widget)
                widget.hide()
                widget.deleteLater()
        self._rows, self.combos = [], {}
        for i, (name, count, color) in enumerate(picked):
            dot = CurveDot(color)
            label = QLabel(name)
            label.setToolTip(f"{name}: {count} point{'s' * (count != 1)}")
            number = muted_label(str(count), 0.88)
            combo = QComboBox()
            combo.setAccessibleName(f"Branch of {name}")
            combo.setSizeAdjustPolicy(
                QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
            )
            combo.setMinimumContentsLength(6)
            combo.setFixedHeight(FIELD_HEIGHT)
            combo.currentIndexChanged.connect(lambda _i, n=name: self._on_combo(n))
            row = i + 1
            grid.addWidget(dot, row, 0)
            grid.addWidget(label, row, 1)
            grid.addWidget(number, row, 2)
            grid.addWidget(combo, row, 3)
            self._rows.append([dot, label, number, combo])
            self.combos[name] = combo

    def _fill_combos(self) -> None:
        items = choices(self.entry)
        mapping = ms.resolve_mapping(self.entry, list(self.combos))
        for name, combo in self.combos.items():
            with QSignalBlocker(combo):
                if [combo.itemText(k) for k in range(combo.count())] != [t for t, _ in items]:
                    combo.clear()
                    for text, branch in items:
                        combo.addItem(text, branch)
                value = mapping.get(name)
                index = next((k for k, (_t, b) in enumerate(items) if b == value), 0)
                combo.setCurrentIndex(index)

    def refresh_results(self) -> None:
        result = self.owner.result(self.entry)
        self.results.setVisible(result is not None)
        if result is None:
            return
        unit = self.owner.unit
        report = ms.fit_report(self.entry, result, unit)
        grid = self.result_table.grid
        while grid.count():
            widget = grid.takeAt(0).widget()
            if widget is not None:
                widget.hide()
                widget.deleteLater()
        right = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        for row, (label, value, sigma, text) in enumerate(report.rows):
            value_text, sigma_text = ms.format_with_sigma(value, sigma)
            name = ElidedLabel(label)
            name.setProperty("kit", "muted")
            name.setFont(scaled_font(name, 0.94))
            name.setToolTip(label)
            number = mono_label(value_text)
            number.setAlignment(right)
            number.setAccessibleName(f"{label} = {value_text} ± {sigma_text}")
            plus_minus = mono_label("±")
            plus_minus.setProperty("kit", "muted")
            error = mono_label(sigma_text)
            error.setAccessibleName(f"Uncertainty of {label}")
            unit_label = muted_label(display_unit(text, unit), 0.88)
            unit_label.setContentsMargins(3, 0, 0, 0)
            for column, widget in enumerate((name, number, plus_minus, error, unit_label)):
                grid.addWidget(widget, row, column)
        squared = "(cm⁻¹)²" if unit is Unit.CM1 else f"{UNIT_TEXT.get(unit, str(unit))}²"
        self.stats.setText(  # a line each: a wrapped line would split an item
            f"χ² {report.chi2:.3g} {squared}\n"
            f"reduced χ² {report.reduced_chi2:.3g} {squared}\n"
            f"dof {report.dof} · {report.n_points} points"
        )

    def show_error(self, message: str | None, panel: str | None = None) -> None:
        self._error_panel = panel if message else None
        self.error.setVisible(bool(message))
        if message:
            self.error.set_text(message[:1].upper() + message[1:], "err")
        has_points = bool(self.combos)
        self.points_link.setVisible(not has_points or self._error_panel == "points")

    # --- editing --------------------------------------------------------------------------
    def _on_mode(self, value: str) -> None:
        for name, combo in self.combos.items():  # what is shown carries over (a skip stays)
            if name not in self.entry.fit.mapping:
                self._record(name, combo.currentData())
        self.entry.fit.assignment = Assignment(value)
        self._fill_combos()

    def _on_combo(self, name: str) -> None:
        self._record(name, self.combos[name].currentData())
        # the other curves keep what they show (their defaults become explicit)
        for other, box in self.combos.items():
            if other not in self.entry.fit.mapping:
                self._record(other, box.currentData())

    def _record(self, name: str, branch: int | None) -> None:
        """Remember the choice for curve *name*. "Use" (nearest) keeps the branch the curve
        had (or gets by default), so By branch and Sorted show it again."""
        mapping = self.entry.fit.mapping
        if self.entry.fit.assignment is Assignment.NEAREST and branch is not None:
            old = mapping.get(name)
            if old is None:
                old = ms.default_branch(self.entry, list(self.combos), name)
            branch = 0 if old is None else old
        mapping[name] = branch

    def _on_fit(self) -> None:
        self.show_error(None)
        try:
            self.owner.fit(self.entry)  # the result (or its error) is shown when it is done
        except EXPECTED_ERRORS as exc:
            self.show_error(str(exc), getattr(exc, "panel", None))
        self.refresh()
