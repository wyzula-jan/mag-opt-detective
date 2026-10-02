"""Points panel: the point extractor, its markers on the map and the Pick tool.

Picked energies are kept in cm^-1 by the controller; the table, the markers and the files use
the display unit.
"""

from __future__ import annotations

from pathlib import Path

import pyqtgraph as pg
from PySide6.QtCore import QRegularExpression, QSignalBlocker, Qt
from PySide6.QtGui import QRegularExpressionValidator
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from mag_opt_detective.core.units import from_cm1
from mag_opt_detective.gui.controller import user_action
from mag_opt_detective.gui.plot_panel import PlotClick
from mag_opt_detective.gui.widgets import PointTableModel, open_file, save_file

PICK = "pick"


class PointsPanel(QWidget):
    """Curve name, click mode, the curve actions and the table of picked points."""

    def __init__(self, parent=None):
        super().__init__(parent)
        grid = QGridLayout()
        self.column_name = QLineEdit("LL 1")
        self.column_name.setValidator(
            QRegularExpressionValidator(QRegularExpression(r"[A-Za-z0-9_ .+-]{1,32}"), self)
        )
        self.column_name.setToolTip("Name of the curve the clicked points are stored in")
        self.init_table = QCheckBox("New table on next Process")
        self.init_table.setChecked(True)
        grid.addWidget(QLabel("Curve"), 0, 0)
        grid.addWidget(self.column_name, 0, 1, 1, 2)
        grid.addWidget(self.init_table, 1, 0, 1, 3)

        self.point_group = QButtonGroup(self)
        self.point_off = QRadioButton("Off")
        self.point_record = QRadioButton("Record")
        self.point_remove = QRadioButton("Remove")
        for i, button in enumerate((self.point_off, self.point_record, self.point_remove)):
            self.point_group.addButton(button, i)
        self.point_off.setChecked(True)
        self.point_record.setToolTip("A click on the map records a point (Pick tool, P)")
        self.point_remove.setToolTip("A click on the map removes a point (or Alt-click)")
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Click mode:"))
        for button in (self.point_off, self.point_record, self.point_remove):
            mode_row.addWidget(button)
        mode_row.addStretch(1)
        grid.addLayout(mode_row, 2, 0, 1, 3)

        self.show_all_button = QPushButton("Show All Points")
        self.show_all_button.setCheckable(True)
        self.drop_button = QPushButton("Drop Curve")
        self.load_button = QPushButton("Load Points…")
        self.export_button = QPushButton("Export Points…")
        buttons = QGridLayout()
        for i, button in enumerate(
            (self.show_all_button, self.drop_button, self.load_button, self.export_button)
        ):
            buttons.addWidget(button, i // 2, i % 2)
        grid.addLayout(buttons, 3, 0, 1, 3)
        grid.setColumnStretch(1, 1)

        self.model = PointTableModel(self)
        self.table = QTableView()
        self.table.setModel(self.model)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout = QVBoxLayout(self)
        layout.addLayout(grid)
        layout.addWidget(self.table, stretch=1)


def draw_markers(window) -> None:
    """Mark the current curve (crosses) and, with Show All, every curve (circles)."""
    c, panel = window.controller, window.panels["points"]
    plot = window.plots.map
    if c.points is None:
        plot.set_points(None)
        return

    def shown(curve: str):
        b, e = c.points.points(curve)
        return b, from_cm1(e, c.unit)

    current = shown(c.curve) if c.curve in c.points.names else None
    others = []
    if panel.show_all_button.isChecked():
        names = c.points.names
        for i, name in enumerate(names):
            others.append((*shown(name), pg.intColor(i, hues=max(len(names), 1))))
    plot.set_points(current, others)


@user_action("Pick point")
def on_pick(window, click: PlotClick) -> None:
    """Record the clicked point; Alt-click (or the Remove mode) removes it."""
    panel = window.panels["points"]
    alt = bool(click.modifiers & Qt.KeyboardModifier.AltModifier)
    if alt or panel.point_remove.isChecked():
        window.controller.remove_point(click.x)
    else:
        window.controller.record_point(click.x, click.y)


@user_action("Drop curve")
def drop_curve(window) -> None:
    window.controller.drop_curve()


@user_action("Load points")
def load_points(window) -> None:
    path = open_file(window, "Load points")
    if path:
        window.controller.load_points(path)


@user_action("Export points")
def export_points(window) -> None:
    if window.controller.points is None:
        raise ValueError("no points to export")
    path = save_file(window, "Export points")
    if not path:
        return
    out = Path(path) if Path(path).suffix else Path(path).with_suffix(".csv")
    window.controller.save_points(out)


def install(window) -> None:
    c = window.controller
    panel = PointsPanel()
    window.add_panel(
        "points", "Points", "chart-scatter", "Picked points", panel,
        "Transition energies picked on the map, one per field and curve.",
    )  # fmt: skip
    tools = window.tools
    tools.register(
        PICK,
        "crosshair",
        "Pick points: click records, Alt-click removes",
        "P",
        views=("map",),
        on_click=lambda click: on_pick(window, click),
    )

    # click mode <-> pick tool (Record or Remove shows the map, where the tool works)
    def on_mode(_button, checked: bool) -> None:
        if not checked:
            return
        if panel.point_off.isChecked():
            if tools.active() == PICK:
                tools.set_active(tools.default())
            return
        if tools.view() not in tools.tool(PICK).views:
            window.plot_area.set_current_view("map")
        if not tools.set_active(PICK):
            on_tool(tools.active())

    def on_tool(name: str) -> None:
        with QSignalBlocker(panel.point_group):
            if name != PICK:
                panel.point_off.setChecked(True)
            elif panel.point_off.isChecked():
                panel.point_record.setChecked(True)

    panel.point_group.buttonToggled.connect(on_mode)
    tools.toolChanged.connect(on_tool)

    # state <-> widgets
    def sync() -> None:
        model = panel.model
        if model.table() is not c.points:
            model.set_table(c.points)
        else:
            model.refresh()
        with QSignalBlocker(panel.init_table):
            panel.init_table.setChecked(c.new_table)
        if panel.column_name.text().strip() != c.curve:
            with QSignalBlocker(panel.column_name):
                panel.column_name.setText(c.curve)
        draw_markers(window)

    panel.column_name.textChanged.connect(lambda text: c.set_curve(text))
    panel.init_table.toggled.connect(c.set_new_table)
    panel.show_all_button.toggled.connect(lambda: draw_markers(window))
    panel.drop_button.clicked.connect(lambda: drop_curve(window))
    panel.load_button.clicked.connect(lambda: load_points(window))
    panel.export_button.clicked.connect(lambda: export_points(window))
    c.pointsChanged.connect(sync)
    c.resultChanged.connect(lambda: draw_markers(window))

    def on_unit(_old, new) -> None:
        panel.model.set_unit(new)
        draw_markers(window)

    c.unitChanged.connect(on_unit)
    panel.model.set_unit(c.unit)
    c.set_curve(panel.column_name.text())
    c.set_new_table(panel.init_table.isChecked())

    p = window.persistence
    if p is not None:
        p.bind("points/column_name", panel.column_name)
