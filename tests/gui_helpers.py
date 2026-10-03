"""Fixtures and helpers of the GUI tests.

Test modules take the fixtures over with
``window, errors = gui_helpers.window, gui_helpers.errors``.
"""

from __future__ import annotations

import numpy as np
import pytest
from PySide6.QtCore import QPointF, Qt
from PySide6.QtWidgets import QFileDialog, QMessageBox

from mag_opt_detective.gui.main_window import MainWindow


@pytest.fixture
def errors(monkeypatch):
    """Messages of the errors the window reports; dialogs never block."""
    messages: list[str] = []
    original = MainWindow.report_error

    def spy(self, title, message, panel=None, expected=True, hint=None):
        messages.append(message)
        original(self, title, message, panel=panel, expected=expected, hint=hint)

    monkeypatch.setattr(MainWindow, "report_error", spy)
    monkeypatch.setattr(QMessageBox, "warning", lambda *_args, **_kwargs: None)
    return messages


@pytest.fixture
def window(qtbot, errors):
    """A main window without settings; closed and deleted after the test (see conftest)."""
    w = MainWindow()
    qtbot.addWidget(w)
    yield w
    w.close()
    w.deleteLater()


def load_sweep(window, sweep) -> None:
    """Put the sweep into the Sample panel's file lists."""
    tab = window.panels["sample"].measurement
    tab.zero_list.set_paths(sweep["zero"])
    tab.field_list.set_paths(sweep["field"])


def process(window) -> None:
    window.commands["process"].trigger()


def select(window, kind=None, order=None, axis=None, per_unit=None) -> None:
    """Choose the plot in the toolbar (kind "Ratio", "Data", ...; order 0-2; axis "E"/"B")."""
    tb = window.toolbar
    if kind is not None:
        tb.kind.set_value(str(kind))
    if order is not None:
        tb.order.set_value(str(order))
    if axis is not None:
        tb.axis.set_value(axis)
    if per_unit is not None:
        tb.per_unit.setChecked(per_unit)


def set_unit(window, unit) -> None:
    window.toolbar.unit.set_value(str(unit))


def shown_image(window) -> np.ndarray:
    return window.plots.map.image.image


def energy_label(window) -> str:
    return window.plots.map.plot.getAxis("left").labelText


def current_marker_energies(window) -> np.ndarray:
    """Energies of the current curve's markers (drawn last on the "points" layer)."""
    return window.plots.map.layer("points").point_data()[-1][1]


def click_map(window, b: float, energy: float, modifiers=Qt.KeyboardModifier.NoModifier):
    """A click on the map at (*b*, *energy* in the display unit)."""
    return window.tools.click("map", b, energy, modifiers)


def click_stacked(window, b: float, energy: float, modifiers=Qt.KeyboardModifier.NoModifier):
    """A click on the stacked plot on the trace of field *b* at *energy* (display unit)."""
    stacked = window.plots.stacked
    j = int(np.abs(window.controller.current_map().field - b).argmin())  # the map drawn
    return window.tools.click("stacked", energy, stacked.trace_y(j, energy), modifiers)


def hover(qtbot, plot, x: float, y: float) -> tuple:
    """Move the mouse to (*x*, *y*) on *plot* (a map or stacked view); the cursorMoved args."""
    pos = plot.plot.vb.mapViewToScene(QPointF(x, y))
    with qtbot.waitSignal(plot.cursorMoved) as blocker:
        plot.plot.scene().sigMouseMoved.emit(pos)
    return tuple(blocker.args)


def save_to(monkeypatch, path) -> None:
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(path), ""))


def open_from(monkeypatch, path) -> None:
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: (str(path), ""))


def infobar_text(window) -> str:
    bar = window.infobar
    return "" if bar.isHidden() else f"{bar.title_label.text()}: {bar.text_label.text()}"


def inspector_page(window, name: str):
    """The content widget of inspector section *name* ("view", "colour", "traces", ...)."""
    return window.inspector[name].body_layout().itemAt(0).widget()
