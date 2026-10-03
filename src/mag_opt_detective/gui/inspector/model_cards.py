"""The model cards of the Models section and their editors.

A card has a header (collapse chevron, colour swatch, name, show/hide switch, remove) and a
body with the model's editor and its fit area. Editors write straight into the core model of
the :class:`~mag_opt_detective.gui.inspector.model_state.ModelEntry` and tell the owner (the
section, see ``models.py``) what changed; energies are typed and shown in the display unit.
"""

from __future__ import annotations

from typing import Protocol

from PySide6.QtCore import QRectF, QSignalBlocker, Qt
from PySide6.QtGui import QFont, QPainter, QPen
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from mag_opt_detective.core.units import Unit, convert
from mag_opt_detective.core.zeeman import Form
from mag_opt_detective.gui import icons
from mag_opt_detective.gui.display import unit_text
from mag_opt_detective.gui.inspector import model_state as ms
from mag_opt_detective.gui.inspector.model_widgets import (
    FIELD_HEIGHT,
    VELOCITY_UNIT,
    CodeBox,
    ColorSwatch,
    IconButton,
    NumberField,
    ParamRow,
    SliderMode,
    TableBox,
    columns,
    compact,
    default_range,
    fixed_range,
    muted_label,
    subscript,
    symmetric_range,
    tool_button,
)
from mag_opt_detective.gui.kit import SegmentedControl, Switch
from mag_opt_detective.gui.panels.common import (
    CheckBox,
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

DELTA_SPAN = 200.0  # meV: the half-gap slider's end (range mode)
COUPLING_SPAN = 20.0  # meV: the couplings' slider end (range mode)
VELOCITY_SPAN = (0.0, 30.0)  # 10⁵ m/s
G_SPAN = 10.0  # g spans -10 … 10 (g factors can be negative)
EXPRESSION_FLOOR = 0.1  # an expression parameter's relative span: at least 10 % of its scale
ENERGY_FLOOR = 1.0  # meV: the relative span of a smaller energy is taken of this
# no-break spaces: the formula wraps only between its terms
DIRAC_FORMULA = (
    "E\u00a0=\u00a0√(2eħv²Bn\u00a0+\u00a0Δ²)\u00a0+ √(2eħv²B(n+1)\u00a0+\u00a0Δ²), "
    "n\u00a0=\u00a00\u00a0…\u00a0N-1"
)
EXPRESSION_HINT = (
    "One branch per line in B (T); # starts its label. muB, hbar and kB are in the output "
    "unit (muB per T, kB per K); also e, c, pi and sqrt, exp, log, tanh, hypot, …"
)
NO_PARAMETERS = "Every other name becomes a parameter, shared between the lines."
FORMS = {
    Form.LINEAR: ("lin", "Linear: E = E₀ + m g μB B"),
    Form.HYPERBOLIC: ("hyp", "Hyperbolic: E = √(E₀² + (m g μB B)²)"),
}
OUTPUT_UNITS = ((Unit.MEV, "meV"), (Unit.CM1, "cm⁻¹"), (Unit.THZ, "THz"))
M_WIDEST = "-0.50"  # the multiplier m as typed
LIMIT_MIN_WIDTH = 48  # a fit limit's field in a narrow inspector
LIMIT_NAME_WIDTH = 40  # a fit limit's name keeps this much (or less if shorter), else elided


class Owner(Protocol):
    """What the cards need from the section (``models.Models``)."""

    @property
    def unit(self) -> Unit: ...

    @property
    def slider_mode(self) -> SliderMode: ...

    def energy_span(self) -> tuple[float, float] | None: ...

    def edited(self, entry: ms.ModelEntry, structure: bool = False, live: bool = False) -> None: ...

    def set_visible(self, entry: ms.ModelEntry, visible: bool) -> None: ...

    def set_color(self, entry: ms.ModelEntry, color: str) -> None: ...

    def set_expanded(self, entry: ms.ModelEntry, expanded: bool) -> None: ...

    def remove(self, entry: ms.ModelEntry) -> None: ...


def column_label(text: str) -> QLabel:
    label = muted_label(text, 0.85)
    label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
    return label


def in_unit(mev: float, unit: Unit) -> float:
    return float(convert(mev, Unit.MEV, unit))


def energy_range(owner: Owner, span_mev: float | None = None):
    """The range-mode span of an energy in the display unit: 0 to *span_mev*, else to the top
    of the processed map, else around the value."""
    if span_mev is not None:
        return fixed_range(0.0, in_unit(span_mev, owner.unit))
    span = owner.energy_span()
    if span is None or span[1] <= 0:
        return default_range
    return fixed_range(0.0, in_unit(span[1], owner.unit))


def stack(*widgets: QWidget, spacing: int = 8) -> QVBoxLayout:
    layout = QVBoxLayout()
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(spacing)
    for widget in widgets:
        layout.addWidget(widget)
    return layout


# ---------------------------------------------------------------------- Dirac
class DiracEditor(QWidget):
    """Fermi velocity v, half-gap Δ (display unit) and number of transitions N as parameter
    rows (the formula under them names the symbols)."""

    def __init__(self, entry: ms.ModelEntry, owner: Owner, parent=None):
        super().__init__(parent)
        self.entry, self.owner = entry, owner
        mode = owner.slider_mode
        self.velocity = ParamRow(
            "v",
            VELOCITY_UNIT,
            name="Fermi velocity v",
            digits=5,
            minimum=0.0,
            mode=mode,
            range_for=fixed_range(*VELOCITY_SPAN),
            floor=1.0,
        )
        self.delta = ParamRow("Δ", "meV", name="Half-gap Δ", minimum=0.0, mode=mode)
        most = ms.DIRAC_MAX_LINES
        self.n_lines = ParamRow(
            "N",
            name="Transitions shown",
            integer=True,
            minimum=1,
            maximum=most,
            range_for=fixed_range(1.0, float(most)),
        )
        self.n_lines.field.edit.setToolTip(f"Transitions shown, N (1 to {most})")
        formula = QLabel(DIRAC_FORMULA)
        formula.setProperty("kit", "muted")
        formula.setWordWrap(True)
        formula.setFont(mono_font(0.85))
        layout = stack(self.velocity, self.delta, self.n_lines, formula)
        self.setLayout(layout)
        self.velocity.valueEdited.connect(self._on_velocity)
        self.delta.valueEdited.connect(self._on_delta)
        self.n_lines.valueEdited.connect(self._on_n_lines)

    def rows(self) -> list[ParamRow]:
        return [self.velocity, self.delta, self.n_lines]

    def refresh(self) -> None:
        unit = self.owner.unit
        p = ms.params(self.entry)
        self.velocity.set_value(p["velocity"].value)
        self.delta.set_unit(unit_text(unit))
        self.delta.set_range_for(energy_range(self.owner, DELTA_SPAN))
        self.delta.set_floor(in_unit(ENERGY_FLOOR, unit))
        self.delta.set_value(ms.shown(self.entry, p["delta"], p["delta"].value, unit))
        self.n_lines.set_value(self.entry.model.n_lines)

    def _on_velocity(self, value: float, live: bool) -> None:
        ms.params(self.entry)["velocity"].value = value
        self.owner.edited(self.entry, live=live)

    def _on_delta(self, value: float, live: bool) -> None:
        p = ms.params(self.entry)["delta"]
        p.value = ms.stored(self.entry, p, value, self.owner.unit)
        self.owner.edited(self.entry, live=live)

    def _on_n_lines(self, value: float, live: bool) -> None:
        n = ms.dirac_lines(value)
        if n != self.entry.model.n_lines or not live:
            self.entry.model.n_lines = n
            self.owner.edited(self.entry, structure=True, live=live)


# ---------------------------------------------------------------------- Zeeman
class FormButton(QToolButton):
    """Linear or hyperbolic branch; a click switches."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.form = Form.LINEAR
        self.setFont(scaled_font(self, 0.9))
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedSize(36, FIELD_HEIGHT)

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
    """*field* with a muted *caption* inside its box, before the number (m)."""
    label = muted_label(caption, 0.9)
    field.layout().insertWidget(0, label)
    field.caption = label
    return field


class BranchRow(QWidget):
    """One branch: its label, m, form and remove on the first line, then E₀ (display unit) and
    g as parameter rows."""

    def __init__(self, mode: SliderMode, parent=None):
        super().__init__(parent)
        self.label = compact(UnitField(QLineEdit()))
        self.label.setFixedHeight(FIELD_HEIGHT)
        self.label.edit.setMaxLength(24)
        self.label.edit.setAccessibleName("Branch label")
        self.label.setToolTip("Label of the branch")
        self.m = captioned(NumberField(name="m, the multiplier of g μB B"), "m")
        self.m.setToolTip("Multiplier of g μB B (e.g. ±1, or 0 for a field-independent line)")
        room = self.m.edit.fontMetrics().horizontalAdvance(M_WIDEST) + 6
        self.m.setFixedWidth(self.m.caption.sizeHint().width() + room + 14)
        self.form = FormButton()
        self.remove = tool_button("x", "Remove branch", "faint")
        # E₀ may be typed below 0 (a linear branch), but the slider stops at 0
        self.e0 = ParamRow(
            "E₀", name="E₀, the energy at zero field", digits=5, mode=mode, nonnegative=True
        )
        self.g = ParamRow(
            "g", name="g factor", mode=mode, range_for=symmetric_range(G_SPAN), floor=1.0
        )
        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.setSpacing(4)
        head.addWidget(self.label, stretch=1)
        head.addWidget(self.m)
        head.addWidget(self.form)
        head.addWidget(self.remove)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addLayout(head)
        layout.addWidget(self.e0)
        layout.addWidget(self.g)


class ZeemanEditor(QWidget):
    """Branches (label, m, form; E₀ and g with sliders), the coupling switch and the couplings
    Δij (sliders)."""

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
        self.branches = QVBoxLayout()
        self.branches.setContentsMargins(0, 0, 0, 0)
        self.branches.setSpacing(10)
        self.coupled_row = SwitchRow(
            "Coupled", "Avoided crossings: the energies are the eigenvalues of diag(Eᵢ) + Δ."
        )
        self.coupled = self.coupled_row.switch
        self.coupling_title = section_label("Couplings")
        self.coupling_box = QWidget()
        self.coupling_rows = QVBoxLayout(self.coupling_box)
        self.coupling_rows.setContentsMargins(0, 0, 0, 0)
        self.coupling_rows.setSpacing(8)
        self.couplings: dict[tuple[int, int], ParamRow] = {}
        self.coupling_note = hint("Add a second branch to couple branches.")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addLayout(head)
        layout.addLayout(self.branches)
        layout.addSpacing(4)
        layout.addWidget(self.coupled_row)
        layout.addWidget(self.coupling_note)
        layout.addWidget(self.coupling_title)
        layout.addWidget(self.coupling_box)
        self.add_button.clicked.connect(self._on_add)
        self.coupled.toggled.connect(self._on_coupled)

    def param_rows(self) -> list[ParamRow]:
        rows = [row for branch in self.rows for row in (branch.e0, branch.g)]
        return rows + list(self.couplings.values())

    # --- showing --------------------------------------------------------------------------
    def refresh(self) -> None:
        unit = self.owner.unit
        model = self.entry.model
        branches = model.branches
        if len(self.rows) != len(branches):
            self._build_rows(len(branches))
        p = ms.params(self.entry)
        e0_range = energy_range(self.owner)
        for i, (row, branch) in enumerate(zip(self.rows, branches, strict=True)):
            if row.label.edit.text() != branch.label:
                with QSignalBlocker(row.label.edit):
                    row.label.edit.setText(branch.label)
            e0 = p[f"e0_{i}"]
            row.e0.set_unit(unit_text(unit))
            row.e0.set_range_for(e0_range)
            row.e0.set_floor(in_unit(ENERGY_FLOOR, unit))
            row.e0.set_value(ms.shown(self.entry, e0, e0.value, unit))
            row.g.set_value(p[f"g_{i}"].value)
            row.m.set_value(branch.m)
            row.form.set_form(branch.form)
            row.remove.setEnabled(len(branches) > 1)
            for widget, text in (
                (row.label.edit, f"Branch {i + 1} label"),
                (row.m.edit, f"Branch {i + 1} m, the multiplier of g μB B"),
                (row.e0.field.edit, f"E₀ of {branch.label}"),
                (row.e0.slider, f"E₀ of {branch.label}"),
                (row.g.field.edit, f"g factor of {branch.label}"),
                (row.g.slider, f"g factor of {branch.label}"),
            ):
                widget.setAccessibleName(text)
        self._refresh_couplings()

    def _build_rows(self, n: int) -> None:
        clear_layout(self.branches)
        self.rows = []
        for i in range(n):
            row = BranchRow(self.owner.slider_mode)
            if i:
                self.branches.addWidget(Divider())
            self.branches.addWidget(row)
            row.label.edit.textChanged.connect(lambda text, k=i: self._on_label(k, text))
            row.label.edit.editingFinished.connect(lambda r=row, k=i: self._tidy_label(r, k))
            row.e0.valueEdited.connect(lambda v, live, k=i: self._on_e0(k, v, live))
            row.g.valueEdited.connect(lambda v, live, k=i: self._on_param(f"g_{k}", v, live))
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
            clear_layout(self.coupling_rows)
            self.couplings = {}
            for i, j in pairs:
                row = ParamRow(
                    f"Δ{subscript(i + 1, j + 1)}",
                    unit_text(unit),
                    name=f"Coupling Δ {i + 1}–{j + 1}",
                    digits=5,
                    mode=self.owner.slider_mode,
                    nonnegative=True,
                )
                row.valueEdited.connect(
                    lambda v, live, pair=(i, j): self._on_coupling(pair, v, live)
                )
                self.coupling_rows.addWidget(row)
                self.couplings[(i, j)] = row
        p = ms.params(self.entry)
        coupling_range = energy_range(self.owner, COUPLING_SPAN)
        for (i, j), row in self.couplings.items():
            row.set_unit(unit_text(unit))
            row.set_range_for(coupling_range)
            row.set_floor(in_unit(ENERGY_FLOOR, unit))
            delta = p[f"delta_{i}_{j}"]
            row.set_value(ms.shown(self.entry, delta, model.couplings[(i, j)], unit))
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

    def _tidy_label(self, row: BranchRow, index: int) -> None:
        """A label left empty shows the name it fell back to ("Branch n")."""
        branches = self.entry.model.branches
        if index < min(len(branches), len(self.rows)) and self.rows[index] is row:
            label = branches[index].label
            if row.label.edit.text() != label:
                with QSignalBlocker(row.label.edit):
                    row.label.edit.setText(label)

    def _on_e0(self, index: int, value: float, live: bool) -> None:
        p = ms.params(self.entry)[f"e0_{index}"]
        p.value = ms.stored(self.entry, p, value, self.owner.unit)
        self.owner.edited(self.entry, live=live)

    def _on_param(self, name: str, value: float, live: bool) -> None:
        ms.params(self.entry)[name].value = float(value)
        self.owner.edited(self.entry, live=live)

    def _on_m(self, index: int, value) -> None:
        if float(value) == self.entry.model.branches[index].m:  # e.g. "1." typed after "1"
            return
        ms.set_branch(self.entry, index, m=value)
        self.owner.edited(self.entry, structure=True)  # a fit with the old m is void

    def _on_form(self, index: int) -> None:
        form = self.entry.model.branches[index].form
        new = Form.HYPERBOLIC if form is Form.LINEAR else Form.LINEAR
        ms.set_branch(self.entry, index, form=new)
        self.rows[index].form.set_form(new)
        self.owner.edited(self.entry, structure=True)

    def _on_coupled(self, on: bool) -> None:
        self.entry.model.set_coupled(on)
        self.owner.edited(self.entry, structure=True)

    def _on_coupling(self, pair: tuple[int, int], value: float, live: bool) -> None:
        i, j = pair
        p = ms.params(self.entry)[f"delta_{i}_{j}"]
        p.value = ms.stored(self.entry, p, value, self.owner.unit)
        self.owner.edited(self.entry, live=live)


# ---------------------------------------------------------------------- custom expression
class ExpressionParam:
    """One expression parameter: its row (name, slider, value) and its fit limits (fixed, min
    and max)."""

    def __init__(self, name: str, mode: SliderMode):
        self.name = name
        self.value = ParamRow(name, name=name, mode=mode, mono=True, floor_share=EXPRESSION_FLOOR)
        self.label = ElidedLabel(name)  # whole if the table has room, else elided
        self.label.setFont(mono_font(0.88))
        self.label.setToolTip(name)
        self.label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        whole = self.label.fontMetrics().horizontalAdvance(name) + 2
        self.label.setMinimumWidth(min(whole, LIMIT_NAME_WIDTH))
        self.fixed = CheckBox()
        self.fixed.setToolTip(f"Hold {name} fixed in fits")
        self.fixed.setAccessibleName(f"{name} fixed")
        self.lo = NumberField(name=f"{name} minimum", optional=True, placeholder="-∞")
        self.hi = NumberField(name=f"{name} maximum", optional=True, placeholder="∞")
        width = columns(self.lo).value
        for field in (self.lo, self.hi):  # the names first; up to the value fields' width
            field.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
            field.setMaximumWidth(width)
            field.setMinimumWidth(LIMIT_MIN_WIDTH)

    def limits(self) -> list[QWidget]:
        return [self.label, self.fixed, self.lo, self.hi]


class ExpressionEditor(QWidget):
    """The expression (validated as you type), its output unit, its parameters (sliders) and
    their fit limits."""

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
        self.params_title = section_label("Parameters")
        self.param_rows = QVBoxLayout()
        self.param_rows.setContentsMargins(0, 0, 0, 0)
        self.param_rows.setSpacing(8)
        self.limits_title = section_label("Fit limits")
        self.table = TableBox()
        for column, text in enumerate(("Name", "Fixed", "Min", "Max")):
            label = column_label(text)
            if column:  # over a tick box or right-aligned numbers
                label.setAlignment(
                    (Qt.AlignmentFlag.AlignHCenter if column == 1 else Qt.AlignmentFlag.AlignRight)
                    | Qt.AlignmentFlag.AlignVCenter
                )
            self.table.grid.addWidget(label, 0, column)
        for column in (0, 2, 3):  # names and limits share the room (limits up to a field)
            self.table.grid.setColumnStretch(column, 1)
        self.table.grid.setHorizontalSpacing(8)
        self.rows: dict[str, ExpressionParam] = {}
        self.empty = hint(NO_PARAMETERS)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(section_label("Branches in B"))
        layout.addWidget(self.code)
        layout.addWidget(self.message)
        layout.addLayout(unit_row)
        layout.addSpacing(2)
        layout.addWidget(self.params_title)
        layout.addLayout(self.param_rows)
        layout.addWidget(self.empty)
        layout.addSpacing(2)
        layout.addWidget(self.limits_title)
        layout.addWidget(self.table)
        self.code.textChanged.connect(self._on_text)
        self.unit.valueChanged.connect(self._on_unit)

    def param_rows_shown(self) -> list[ParamRow]:
        return [row.value for row in self.rows.values()]

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
        for widget in (self.table, self.limits_title):
            widget.setVisible(bool(self.rows))
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
        clear_layout(self.param_rows)
        for row in self.rows.values():
            for widget in row.limits():
                grid.removeWidget(widget)
                widget.hide()
                widget.deleteLater()
        self.rows = {}
        for i, name in enumerate(names):
            row = ExpressionParam(name, self.owner.slider_mode)
            self.param_rows.addWidget(row.value)
            for column, widget in enumerate(row.limits()):
                align = Qt.AlignmentFlag.AlignCenter if column == 1 else Qt.AlignmentFlag(0)
                grid.addWidget(widget, i + 1, column, alignment=align)
            row.value.valueEdited.connect(lambda v, live, n=name: self._on_value(n, v, live))
            row.fixed.toggled.connect(lambda on, n=name: self._on_fixed(n, on))
            row.lo.valueEdited.connect(lambda v, n=name: self._on_bound(n, "lo", v))
            row.hi.valueEdited.connect(lambda v, n=name: self._on_bound(n, "hi", v))
            self.rows[name] = row
        # the names get a caption column as wide as the widest (the symbols' is too narrow)
        rows = [row.value for row in self.rows.values()]
        if rows:
            metrics = rows[0].caption.fontMetrics()
            widest = max(metrics.horizontalAdvance(name) for name in names) + 4
            for row in rows:
                row.set_label_width(widest)

    def _on_text(self, text: str) -> None:
        ms.set_expression(self.entry, text)
        self._show_message()
        self.owner.edited(self.entry, structure=True)  # a fit of another expression is void

    def _on_unit(self, value: str) -> None:
        ms.set_output_unit(self.entry, value)
        self.owner.edited(self.entry, structure=True)  # the fitted values were in the old unit

    def _on_value(self, name: str, value: float, live: bool) -> None:
        ms.params(self.entry)[name].value = float(value)
        self.entry.edited.add(name)
        self.owner.edited(self.entry, live=live)

    def _on_fixed(self, name: str, fixed: bool) -> None:
        ms.params(self.entry)[name].fixed = fixed
        self.entry.edited.add(name)

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
        self.entry.edited.add(name)


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
    """One model: header (chevron, colour, name, show switch, remove), editor and fit area
    ("Fit to points…" opens it in its place)."""

    def __init__(self, entry: ms.ModelEntry, owner: Owner, fit_area: QWidget, parent=None):
        super().__init__(parent)
        self.entry, self.owner = entry, owner
        self.chevron = IconButton("chevron-down", "Collapse", 14, 12)
        self.swatch = ColorSwatch()
        self.name = _Name(self)
        self.visible_switch = Switch()
        self.visible_switch.setToolTip("Show the curves on the map")
        self.remove_button = IconButton("trash-2", "Remove model", 18, 12)
        head = QHBoxLayout()  # tight: the name keeps its room in a narrow inspector
        head.setContentsMargins(4, 5, 4, 5)
        head.setSpacing(2)
        head.addWidget(self.chevron)
        head.addWidget(self.swatch)
        head.addSpacing(2)
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
        self.fit_button.setVisible(self.fit_area.isHidden())

    def _on_fit_button(self, on: bool) -> None:
        """The fit area takes the button's place (its header closes it again)."""
        self.fit_area.setVisible(on)
        self.fit_button.setVisible(not on)
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
