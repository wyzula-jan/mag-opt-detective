"""The model cards of the Models section and their editors.

A card has a header (collapse chevron, colour swatch, name, show/hide switch, remove) and a
body with the model's editor and its fit area. Editors write straight into the core model of
the :class:`~mag_opt_detective.gui.inspector.model_state.ModelEntry` and tell the owner (the
section, see ``models.py``) what changed; energies are typed and shown in the display unit.
"""

from __future__ import annotations

from typing import Protocol

from PySide6.QtCore import QRectF, QSignalBlocker, QSize, Qt
from PySide6.QtGui import QFont, QPainter, QPen
from PySide6.QtWidgets import (
    QCheckBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from mag_opt_detective.core.units import Unit, convert
from mag_opt_detective.core.zeeman import Form
from mag_opt_detective.gui import icons
from mag_opt_detective.gui.inspector import model_state as ms
from mag_opt_detective.gui.inspector.model_widgets import (
    CodeBox,
    ColorSwatch,
    NumberField,
    SliderField,
    TableBox,
    compact,
    mono_label,
    muted_label,
    tool_button,
)
from mag_opt_detective.gui.inspector.view import UNIT_TEXT
from mag_opt_detective.gui.kit import SegmentedControl, Switch
from mag_opt_detective.gui.panels.common import (
    Divider,
    ElidedLabel,
    LinkButton,
    Note,
    SwitchRow,
    UnitField,
    clear_layout,
    hint,
    mono_font,
    scaled_font,
    section_label,
    small_button,
)
from mag_opt_detective.gui.theme import current_tokens

DELTA_SPAN = 200.0  # meV: the half-gap slider's end
DIRAC_FORMULA = "E = √(2eħv²Bn + Δ²) + √(2eħv²B(n+1) + Δ²), n = 0 … N-1"
EXPRESSION_HINT = (
    "One branch per line in B (T); # starts its label. Constants muB, hbar, kB, e, c, pi; "
    "sqrt, exp, log, tanh, hypot, …"
)
NO_PARAMETERS = "Every other name becomes a parameter, shared between the lines."
FORMS = {
    Form.LINEAR: ("lin", "Linear: E = E₀ + m g μB B"),
    Form.HYPERBOLIC: ("hyp", "Hyperbolic: E = √(E₀² + (m g μB B)²)"),
}
OUTPUT_UNITS = ((Unit.MEV, "meV"), (Unit.CM1, "cm⁻¹"), (Unit.THZ, "THz"))


class Owner(Protocol):
    """What the cards need from the section (``models.Models``)."""

    @property
    def unit(self) -> Unit: ...

    def edited(self, entry: ms.ModelEntry, structure: bool = False) -> None: ...

    def set_visible(self, entry: ms.ModelEntry, visible: bool) -> None: ...

    def set_color(self, entry: ms.ModelEntry, color: str) -> None: ...

    def set_expanded(self, entry: ms.ModelEntry, expanded: bool) -> None: ...

    def remove(self, entry: ms.ModelEntry) -> None: ...


def unit_text(unit: Unit | str) -> str:
    return UNIT_TEXT.get(Unit(unit), str(unit))


def column_label(text: str) -> QLabel:
    label = muted_label(text, 0.85)
    label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
    return label


# ---------------------------------------------------------------------- Dirac
class DiracEditor(QWidget):
    """Fermi velocity, half-gap (display unit) and number of transitions: sliders and fields."""

    def __init__(self, entry: ms.ModelEntry, owner: Owner, parent=None):
        super().__init__(parent)
        self.entry, self.owner = entry, owner
        self.velocity = SliderField("Fermi velocity v", 0.1, 30.0, "×10⁵ m/s", minimum=0.0)
        self.delta = SliderField("Half-gap Δ", 0.0, DELTA_SPAN, "meV", minimum=0.0)
        self.n_lines = SliderField("Transitions shown", 1, 40, integer=True, minimum=1)
        formula = QLabel(DIRAC_FORMULA)
        formula.setProperty("kit", "muted")
        formula.setWordWrap(True)
        formula.setFont(mono_font(0.85))
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        for widget in (self.velocity, self.delta, self.n_lines, formula):
            layout.addWidget(widget)
        self.velocity.valueChanged.connect(self._on_velocity)
        self.delta.valueChanged.connect(self._on_delta)
        self.n_lines.valueChanged.connect(self._on_n_lines)

    def refresh(self) -> None:
        unit = self.owner.unit
        p = ms.params(self.entry)
        self.velocity.set_value(p["velocity"].value)
        self.delta.set_unit(unit_text(unit))
        self.delta.set_range(0.0, float(convert(DELTA_SPAN, Unit.MEV, unit)))
        self.delta.set_value(ms.shown(self.entry, p["delta"], p["delta"].value, unit))
        self.n_lines.set_value(self.entry.model.n_lines)

    def _on_velocity(self, value: float) -> None:
        ms.params(self.entry)["velocity"].value = value
        self.owner.edited(self.entry)

    def _on_delta(self, value: float) -> None:
        p = ms.params(self.entry)["delta"]
        p.value = ms.stored(self.entry, p, value, self.owner.unit)
        self.owner.edited(self.entry)

    def _on_n_lines(self, value: float) -> None:
        self.entry.model.n_lines = max(1, round(value))
        self.owner.edited(self.entry, structure=True)


# ---------------------------------------------------------------------- Zeeman
class FormButton(QToolButton):
    """Linear or hyperbolic branch; a click switches."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.form = Form.LINEAR
        self.setFont(scaled_font(self, 0.9))
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedHeight(24)
        self.setMinimumWidth(34)

    def set_form(self, form: Form | str) -> None:
        self.form = Form(form)
        text, tip = FORMS[self.form]
        self.setText(text)
        self.setToolTip(f"{tip} (click to switch)")
        self.setAccessibleName(f"Form: {self.form.value}")

    def paintEvent(self, event) -> None:
        tokens = current_tokens()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(self.rect()).adjusted(1.5, 1.5, -1.5, -1.5)
        hot = self.underMouse() or self.hasFocus()
        painter.setPen(QPen(tokens["accent" if self.hasFocus() else "line-strong"], 1))
        painter.setBrush(tokens["hover" if hot else "sunken"])
        painter.drawRoundedRect(rect, 5, 5)
        painter.setPen(tokens["fg"])
        painter.setFont(self.font())
        painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.text())
        painter.end()


def captioned(field: UnitField, caption: str) -> UnitField:
    """*field* with a muted *caption* inside its box, before the number (E₀, g, m)."""
    label = muted_label(caption, 0.9)
    field.layout().insertWidget(0, label)
    field.caption = label
    return field


class BranchRow(QWidget):
    """One branch in two lines: label and E₀ (display unit); then g, m, form and remove."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.label = compact(UnitField(QLineEdit()))
        self.label.edit.setMaxLength(24)
        self.label.edit.setAccessibleName("Branch label")
        self.e0 = captioned(NumberField(name="E₀"), "E₀")
        self.g = captioned(NumberField(name="g factor"), "g")
        self.m = captioned(NumberField(name="m, the multiplier of g μB B"), "m")
        self.e0.setToolTip("Energy at zero field")
        self.g.setToolTip("g factor")
        self.m.setToolTip("Multiplier of g μB B (e.g. ±1, or 0 for a field-independent line)")
        self.form = FormButton()
        self.remove = tool_button("x", "Remove branch", "faint")
        self.remove.setFixedSize(20, 20)
        self.remove.setIconSize(QSize(12, 12))
        first = QHBoxLayout()
        first.setContentsMargins(0, 0, 0, 0)
        first.setSpacing(4)
        first.addWidget(self.label, stretch=4)
        first.addWidget(self.e0, stretch=5)
        second = QHBoxLayout()
        second.setContentsMargins(0, 0, 0, 0)
        second.setSpacing(4)
        second.addWidget(self.g, stretch=3)
        second.addWidget(self.m, stretch=2)
        second.addWidget(self.form)
        second.addWidget(self.remove)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addLayout(first)
        layout.addLayout(second)


class ZeemanEditor(QWidget):
    """Branches (label, E0, g, m, form), the coupling switch and the couplings Δij."""

    def __init__(self, entry: ms.ModelEntry, owner: Owner, parent=None):
        super().__init__(parent)
        self.entry, self.owner = entry, owner
        self.rows: list[BranchRow] = []
        self.add_button = LinkButton("Add branch", "Add a branch")
        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.addWidget(section_label("Branches"))
        head.addStretch(1)
        head.addWidget(self.add_button)
        self.table = TableBox(rows=True)
        self.coupled_row = SwitchRow(
            "Coupled", "Avoided crossings: the energies are the eigenvalues of diag(Eᵢ) + Δ."
        )
        self.coupled = self.coupled_row.switch
        self.coupling_title = section_label("Couplings")
        self.coupling_box = QWidget()
        self.coupling_grid = QGridLayout(self.coupling_box)
        self.coupling_grid.setContentsMargins(0, 0, 0, 0)
        self.coupling_grid.setHorizontalSpacing(8)
        self.coupling_grid.setVerticalSpacing(6)
        self.coupling_grid.setColumnStretch(0, 1)
        self.coupling_grid.setColumnStretch(1, 1)
        self.couplings: dict[tuple[int, int], NumberField] = {}
        self.coupling_note = hint("Add a second branch to couple branches.")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(7)
        layout.addLayout(head)
        layout.addWidget(self.table)
        layout.addSpacing(4)
        layout.addWidget(self.coupled_row)
        layout.addWidget(self.coupling_note)
        layout.addWidget(self.coupling_title)
        layout.addWidget(self.coupling_box)
        self.add_button.clicked.connect(self._on_add)
        self.coupled.toggled.connect(self._on_coupled)

    # --- showing --------------------------------------------------------------------------
    def refresh(self) -> None:
        unit = self.owner.unit
        model = self.entry.model
        branches = model.branches
        if len(self.rows) != len(branches):
            self._build_rows(len(branches))
        p = ms.params(self.entry)
        for i, (row, branch) in enumerate(zip(self.rows, branches, strict=True)):
            if row.label.edit.text() != branch.label:
                with QSignalBlocker(row.label.edit):
                    row.label.edit.setText(branch.label)
            e0 = p[f"e0_{i}"]
            row.e0.set_unit(unit_text(unit))
            row.e0.set_value(ms.shown(self.entry, e0, e0.value, unit))
            row.g.set_value(p[f"g_{i}"].value)
            row.m.set_value(branch.m)
            row.form.set_form(branch.form)
            row.remove.setEnabled(len(branches) > 1)
            row.e0.edit.setAccessibleName(f"E₀ of {branch.label}")
        self._refresh_couplings()

    def _build_rows(self, n: int) -> None:
        clear_layout(self.table.rows)
        self.rows = []
        for i in range(n):
            row = BranchRow()
            if i:
                self.table.rows.addWidget(Divider())
            self.table.rows.addWidget(row)
            row.label.edit.textChanged.connect(lambda text, k=i: self._on_label(k, text))
            row.e0.valueEdited.connect(lambda v, k=i: self._on_e0(k, v))
            row.g.valueEdited.connect(lambda v, k=i: self._on_param(f"g_{k}", v))
            row.m.valueEdited.connect(lambda v, k=i: self._on_m(k, v))
            row.form.clicked.connect(lambda _checked=False, k=i: self._on_form(k))
            row.remove.clicked.connect(lambda _checked=False, k=i: self._on_remove(k))
            self.rows.append(row)

    def _refresh_couplings(self) -> None:
        model, unit = self.entry.model, self.owner.unit
        n = len(model.branches)
        with QSignalBlocker(self.coupled):
            self.coupled.setChecked(model.coupled)
        self.coupled.setEnabled(n > 1)
        self.coupling_note.setVisible(n < 2)
        show = model.coupled and n > 1
        pairs = sorted(model.couplings)
        if sorted(self.couplings) != pairs:
            clear_layout(self.coupling_grid)
            self.couplings = {}
            for k, (i, j) in enumerate(pairs):
                field = NumberField(unit_text(unit), name=f"Coupling Δ {i + 1}–{j + 1}")
                field.valueEdited.connect(lambda v, pair=(i, j): self._on_coupling(pair, v))
                box = QWidget()
                box_layout = QVBoxLayout(box)
                box_layout.setContentsMargins(0, 0, 0, 0)
                box_layout.setSpacing(2)
                box_layout.addWidget(muted_label(f"Δ {i + 1}–{j + 1}", 0.85))
                box_layout.addWidget(field)
                self.coupling_grid.addWidget(box, k // 2, k % 2)
                self.couplings[(i, j)] = field
        p = ms.params(self.entry)
        for (i, j), field in self.couplings.items():
            field.set_unit(unit_text(unit))
            delta = p[f"delta_{i}_{j}"]
            value = model.couplings[(i, j)]
            field.set_value(ms.shown(self.entry, delta, value, unit))
        self.coupling_title.setVisible(show)
        self.coupling_box.setVisible(show)

    # --- editing --------------------------------------------------------------------------
    def _on_add(self) -> None:
        ms.add_branch(self.entry)
        self.owner.edited(self.entry, structure=True)

    def _on_remove(self, index: int) -> None:
        ms.remove_branch(self.entry, index)
        self.owner.edited(self.entry, structure=True)

    def _on_label(self, index: int, text: str) -> None:
        ms.set_branch(self.entry, index, label=text.strip() or f"Branch {index + 1}")
        self.owner.edited(self.entry)

    def _on_e0(self, index: int, value) -> None:
        p = ms.params(self.entry)[f"e0_{index}"]
        p.value = ms.stored(self.entry, p, value, self.owner.unit)
        self.owner.edited(self.entry)

    def _on_param(self, name: str, value) -> None:
        ms.params(self.entry)[name].value = float(value)
        self.owner.edited(self.entry)

    def _on_m(self, index: int, value) -> None:
        ms.set_branch(self.entry, index, m=value)
        self.owner.edited(self.entry)

    def _on_form(self, index: int) -> None:
        form = self.entry.model.branches[index].form
        new = Form.HYPERBOLIC if form is Form.LINEAR else Form.LINEAR
        ms.set_branch(self.entry, index, form=new)
        self.rows[index].form.set_form(new)
        self.owner.edited(self.entry)

    def _on_coupled(self, on: bool) -> None:
        self.entry.model.set_coupled(on)
        self.owner.edited(self.entry, structure=True)

    def _on_coupling(self, pair: tuple[int, int], value) -> None:
        i, j = pair
        p = ms.params(self.entry)[f"delta_{i}_{j}"]
        p.value = ms.stored(self.entry, p, value, self.owner.unit)
        self.owner.edited(self.entry)


# ---------------------------------------------------------------------- custom expression
class ParamRow:
    """The cells of one expression parameter: name, value, fixed, min and max."""

    def __init__(self, name: str):
        self.name = name
        self.label = mono_label(name)
        self.label.setToolTip(name)
        self.value = NumberField(name=f"{name} value")
        self.fixed = QCheckBox()
        self.fixed.setToolTip(f"Hold {name} fixed in fits")
        self.fixed.setAccessibleName(f"{name} fixed")
        self.lo = NumberField(name=f"{name} minimum", optional=True, placeholder="-∞")
        self.hi = NumberField(name=f"{name} maximum", optional=True, placeholder="∞")

    def widgets(self) -> list[QWidget]:
        return [self.label, self.value, self.fixed, self.lo, self.hi]


class ExpressionEditor(QWidget):
    """The expression (validated as you type), its output unit and its parameters."""

    def __init__(self, entry: ms.ModelEntry, owner: Owner, parent=None):
        super().__init__(parent)
        self.entry, self.owner = entry, owner
        self.code = CodeBox(ms.EXAMPLE)
        self.message = Note(EXPRESSION_HINT)
        self.unit = SegmentedControl(size="xs")
        for value, text in OUTPUT_UNITS:
            button = self.unit.add_option(
                value.value, text, f"The expression gives energies in {text}"
            )
            button.setFont(scaled_font(button, 0.92))
        self.unit.setAccessibleName("Output unit of the expression")
        unit_label = muted_label("Unit", 0.94)
        unit_label.setToolTip("The unit of the energies the expression gives (and of its terms)")
        unit_row = QHBoxLayout()
        unit_row.setContentsMargins(0, 0, 0, 0)
        unit_row.addWidget(unit_label)
        unit_row.addStretch(1)
        unit_row.addWidget(self.unit)
        self.table = TableBox()
        for column, text in enumerate(("Name", "Value", "Fix", "Min", "Max")):
            self.table.grid.addWidget(column_label(text), 0, column)
        for column, stretch in enumerate((3, 5, 0, 4, 4)):
            self.table.grid.setColumnStretch(column, stretch)
        self.rows: dict[str, ParamRow] = {}
        self.empty = hint(NO_PARAMETERS)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(7)
        layout.addWidget(section_label("Branches in B"))
        layout.addWidget(self.code)
        layout.addWidget(self.message)
        layout.addLayout(unit_row)
        layout.addSpacing(2)
        layout.addWidget(section_label("Parameters"))
        layout.addWidget(self.table)
        layout.addWidget(self.empty)
        self.code.textChanged.connect(self._on_text)
        self.unit.valueChanged.connect(self._on_unit)

    def refresh(self) -> None:
        entry = self.entry
        self.code.set_text(entry.text)
        with QSignalBlocker(self.unit):
            self.unit.set_value(entry.unit.value)
        self._show_message()
        names = [p.name for p in entry.model.params] if entry.model is not None else []
        if not entry.text.strip():
            names = []
        if list(self.rows) != names:
            self._build_rows(names)
        p = ms.params(entry)
        for name, row in self.rows.items():
            param = p[name]
            row.value.set_value(param.value)
            with QSignalBlocker(row.fixed):
                row.fixed.setChecked(param.fixed)
            row.lo.set_value(param.lo)
            row.hi.set_value(param.hi)
        self.table.setVisible(bool(self.rows))
        self.empty.setVisible(not self.rows)

    def _show_message(self) -> None:
        error = self.entry.error
        if error is None:
            self.message.set_text(EXPRESSION_HINT, "muted")
            self.code.mark(None)
        else:
            text = f"Line {error.line}, column {error.column}: {error.message}"
            self.message.set_text(text, "err")
            self.code.mark(error.line, error.column)

    def _build_rows(self, names: list[str]) -> None:
        grid = self.table.grid
        for row in self.rows.values():
            for widget in row.widgets():
                grid.removeWidget(widget)
                widget.hide()
                widget.deleteLater()
        self.rows = {}
        for i, name in enumerate(names):
            row = ParamRow(name)
            for column, widget in enumerate(row.widgets()):
                align = Qt.AlignmentFlag.AlignCenter if column == 2 else Qt.AlignmentFlag(0)
                grid.addWidget(widget, i + 1, column, alignment=align)
            row.value.valueEdited.connect(lambda v, n=name: self._on_value(n, v))
            row.fixed.toggled.connect(lambda on, n=name: self._on_fixed(n, on))
            row.lo.valueEdited.connect(lambda v, n=name: self._on_bound(n, "lo", v))
            row.hi.valueEdited.connect(lambda v, n=name: self._on_bound(n, "hi", v))
            self.rows[name] = row

    def _on_text(self, text: str) -> None:
        old = [p.name for p in self.entry.model.params] if self.entry.model is not None else []
        ms.set_expression(self.entry, text)
        new = [p.name for p in self.entry.model.params] if self.entry.model is not None else []
        self._show_message()
        self.owner.edited(self.entry, structure=new != old)

    def _on_unit(self, value: str) -> None:
        ms.set_output_unit(self.entry, value)
        self.owner.edited(self.entry)

    def _on_value(self, name: str, value) -> None:
        ms.params(self.entry)[name].value = float(value)
        self.owner.edited(self.entry)

    def _on_fixed(self, name: str, fixed: bool) -> None:
        ms.params(self.entry)[name].fixed = fixed

    def _on_bound(self, name: str, which: str, value) -> None:
        row = self.rows[name]
        lo = row.lo.value() if which != "lo" else value
        hi = row.hi.value() if which != "hi" else value
        bad = lo is not None and hi is not None and lo >= hi
        row.lo.set_invalid(bad)
        row.hi.set_invalid(bad)
        if bad:
            return
        p = ms.params(self.entry)[name]
        p.lo = -float("inf") if lo is None else float(lo)
        p.hi = float("inf") if hi is None else float(hi)


EDITORS = {ms.DIRAC: DiracEditor, ms.ZEEMAN: ZeemanEditor, ms.CUSTOM: ExpressionEditor}


# ---------------------------------------------------------------------- the card
class _Name(ElidedLabel):
    """The model name in the card header; a click collapses or expands the card."""

    def __init__(self, card: ModelCard):
        super().__init__()
        self._card = card
        font = QFont(self.font())
        font.setWeight(QFont.Weight.DemiBold)
        self.setFont(font)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._card.toggle()
        super().mouseReleaseEvent(event)


class ModelCard(QWidget):
    """One model: header (chevron, colour, name, show switch, remove), editor and fit area."""

    def __init__(self, entry: ms.ModelEntry, owner: Owner, fit_area: QWidget, parent=None):
        super().__init__(parent)
        self.entry, self.owner = entry, owner
        self.chevron = tool_button("chevron-down", "Collapse")
        self.chevron.setFixedSize(20, 20)
        self.chevron.setIconSize(QSize(13, 13))
        self.swatch = ColorSwatch()
        self.name = _Name(self)
        self.visible_switch = Switch()
        self.visible_switch.setToolTip("Show the curves on the map")
        self.remove_button = tool_button("trash-2", "Remove model")
        head = QHBoxLayout()
        head.setContentsMargins(6, 5, 6, 5)
        head.setSpacing(4)
        head.addWidget(self.chevron)
        head.addWidget(self.swatch)
        head.addWidget(self.name, stretch=1)
        head.addWidget(self.visible_switch)
        head.addWidget(self.remove_button)

        self.editor = EDITORS[entry.kind](entry, owner)
        self.fit_button = small_button(
            "Fit to points…", "chart-scatter", "Fit to the picked points"
        )
        self.fit_button.setCheckable(True)
        self.fit_area = fit_area
        self.body = QWidget()
        body = QVBoxLayout(self.body)
        body.setContentsMargins(12, 2, 12, 12)
        body.setSpacing(10)
        body.addWidget(self.editor)
        body.addWidget(self.fit_button, alignment=Qt.AlignmentFlag.AlignLeft)
        body.addWidget(self.fit_area)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(1, 1, 1, 1)
        layout.setSpacing(0)
        layout.addLayout(head)
        layout.addWidget(self.body)

        self.chevron.clicked.connect(self.toggle)
        self.swatch.colorChosen.connect(lambda color: owner.set_color(entry, color))
        self.visible_switch.toggled.connect(lambda on: owner.set_visible(entry, on))
        self.remove_button.clicked.connect(lambda: owner.remove(entry))
        self.fit_button.toggled.connect(self._on_fit_button)

    def toggle(self) -> None:
        self.owner.set_expanded(self.entry, not self.entry.expanded)

    def refresh(self) -> None:
        """Show the entry (values in the display unit)."""
        entry = self.entry
        self.name.setText(entry.name)
        self.name.setToolTip(f"{entry.name}: click to {'collapse' if entry.expanded else 'expand'}")
        self.swatch.set_color(entry.color)
        with QSignalBlocker(self.visible_switch):
            self.visible_switch.setChecked(entry.visible)
        self.visible_switch.setAccessibleName(f"Show {entry.name}")
        self.remove_button.setAccessibleName(f"Remove {entry.name}")
        self.body.setVisible(entry.expanded)
        icons.set_icon(self.chevron, "chevron-down" if entry.expanded else "chevron-right", "muted")
        self.chevron.setToolTip("Collapse" if entry.expanded else "Expand")
        self.editor.refresh()
        with QSignalBlocker(self.fit_button):
            self.fit_button.setChecked(not self.fit_area.isHidden())

    def _on_fit_button(self, on: bool) -> None:
        self.fit_area.setVisible(on)
        if on:
            self.fit_area.refresh()

    def paintEvent(self, event) -> None:
        tokens = current_tokens()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(tokens["line"], 1))
        painter.setBrush(tokens["surface"])
        painter.drawRoundedRect(QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5), 7, 7)
        painter.end()
