"""Overlays section: model transition energies drawn over the map (massive Dirac model).

The model works in meV; its curves are converted to the display unit when drawn.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QCheckBox, QGridLayout, QLabel, QSpinBox, QWidget

from mag_opt_detective.core.models import dirac_interband
from mag_opt_detective.core.units import Unit, convert
from mag_opt_detective.gui.controller import Curve, user_action
from mag_opt_detective.gui.widgets import SliderSpin

LAYER = "models"


@dataclass(frozen=True)
class DiracSettings:
    velocity: float  # 10^5 m/s
    delta: float  # half-gap, meV
    n_lines: int


class ModelsPage(QWidget):
    """Massive Dirac Landau-level transitions drawn over the colour map."""

    changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        grid = QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)
        self.show_dirac = QCheckBox("Massive Dirac model")
        self.show_dirac.setToolTip("Interband transitions L(-n) → L(n+1)")
        self.velocity = SliderSpin(0.1, 30.0, 0.05, 5.0, " ×10⁵ m/s")
        self.delta = SliderSpin(0.0, 500.0, 0.5, 0.0, " meV")
        self.n_lines = QSpinBox()
        self.n_lines.setRange(1, 40)
        self.n_lines.setValue(5)
        formula = QLabel("E = √(2eħv²Bn + Δ²) + √(2eħv²B(n+1) + Δ²),  n = 0 … N-1")
        formula.setWordWrap(True)
        formula.setProperty("kit", "muted")
        grid.addWidget(self.show_dirac, 0, 0, 1, 2)
        grid.addWidget(QLabel("Fermi velocity v"), 1, 0, 1, 2)
        grid.addWidget(self.velocity, 2, 0, 1, 2)
        grid.addWidget(QLabel("Half-gap Δ"), 3, 0, 1, 2)
        grid.addWidget(self.delta, 4, 0, 1, 2)
        grid.addWidget(QLabel("Lines N"), 5, 0)
        grid.addWidget(self.n_lines, 5, 1)
        grid.addWidget(formula, 6, 0, 1, 2)
        grid.setColumnStretch(1, 1)

        self.show_dirac.toggled.connect(self.changed)
        self.velocity.valueChanged.connect(self.changed)
        self.delta.valueChanged.connect(self.changed)
        self.n_lines.valueChanged.connect(self.changed)

    def dirac(self) -> DiracSettings | None:
        if not self.show_dirac.isChecked():
            return None
        return DiracSettings(self.velocity.value(), self.delta.value(), self.n_lines.value())


def model_curves(page: ModelsPage, result, unit: Unit) -> list[Curve]:
    """The Dirac transitions over the field range of *result*, in *unit*."""
    model = page.dirac()
    if model is None or result is None:
        return []
    shown = result.ratio
    field = np.linspace(max(0.0, float(shown.field.min())), float(shown.field.max()), 300)
    lines = dirac_interband(field, model.velocity, model.delta, model.n_lines)
    return [(field, line) for line in convert(lines, Unit.MEV, unit)]


@user_action("Model")
def draw(window, page: ModelsPage) -> None:
    curves = model_curves(page, window.controller.result, window.controller.unit)
    plot = window.plots.map
    if not curves:
        plot.set_model_curves(np.array([]), None)
        return
    plot.set_model_curves(curves[0][0], np.array([energy for _b, energy in curves]))


def install(window) -> None:
    c = window.controller
    page = ModelsPage()
    window.add_inspector_section("overlays", "Overlays", page)
    page.changed.connect(lambda: draw(window, page))
    c.resultChanged.connect(lambda: draw(window, page))
    c.unitChanged.connect(lambda _old, _new: draw(window, page))
    c.set_overlay("dirac", lambda unit: model_curves(page, c.result, unit))

    p = window.persistence
    if p is not None:
        p.bind("models/show_dirac", page.show_dirac)
        p.bind("models/velocity", page.velocity.spin)
        p.bind("models/delta", page.delta.spin)
        p.bind("models/n_lines", page.n_lines)
