"""Small widgets of gui/widgets.py: the energy field, the flow layout, the action setting."""

import pytest
from PySide6.QtCore import QRect, QSize
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QWidget

from mag_opt_detective.gui.widgets import CheckableSetting, EnergyEdit, FlowLayout, parse_float


def test_energy_edit_keeps_cm1(qtbot):
    edit = EnergyEdit(None, "E")
    qtbot.addWidget(edit)
    changes = []
    edit.valueChanged.connect(lambda: changes.append(edit.cm1()))
    edit.set_unit("meV")
    edit.setText("55")  # typed in meV
    assert edit.cm1() == pytest.approx(55 * 8.0656)
    assert changes[-1] == edit.cm1()
    edit.set_unit("cm-1")
    assert edit.text() == "443.608" and edit.value() == pytest.approx(443.608)
    edit.set_unit("THz")
    edit.set_unit("meV")
    assert edit.text() == "55"
    assert edit.cm1() == pytest.approx(55 * 8.0656)
    n = len(changes)
    edit.set_unit("THz")  # showing in another unit is not an edit
    assert len(changes) == n
    edit.setText("abc")
    assert edit.cm1() is None


def test_energy_edit_settings_protocol(qtbot):
    edit = EnergyEdit(100.0)
    qtbot.addWidget(edit)
    edit.set_unit("meV")
    assert edit.settings_value() == 100.0  # stored in cm-1 whatever is shown
    assert edit.set_settings_value("806.56")
    assert edit.text() == "100"
    assert edit.set_settings_value("")
    assert edit.cm1() is None and edit.settings_value() == ""
    assert not edit.set_settings_value("nan")
    assert not edit.set_settings_value("x")
    edit.set_value(2.0)  # in the display unit
    assert edit.cm1() == pytest.approx(2 * 8.0656)


def test_parse_float():
    assert parse_float(" 1e3 ") == 1000.0
    assert parse_float("") is None and parse_float("inf") is None and parse_float("-") is None


def test_checkable_setting(qapp):
    action = QAction("Thing")
    action.setCheckable(True)
    setting = CheckableSetting(action)
    assert setting.settings_value() is False
    assert setting.set_settings_value("true") and action.isChecked()
    assert not setting.set_settings_value("maybe")


class _Box(QWidget):
    def __init__(self, width: int, height: int):
        super().__init__()
        self._hint = QSize(width, height)

    def sizeHint(self) -> QSize:
        return self._hint


def test_flow_layout_wraps(qtbot):
    host = QWidget()
    qtbot.addWidget(host)
    flow = FlowLayout(host, spacing=10, row_spacing=4)
    boxes = [_Box(100, 20), _Box(100, 30), _Box(100, 20)]
    for box in boxes:
        flow.addWidget(box)
    assert flow.heightForWidth(400) == 30  # one row
    assert flow.heightForWidth(250) == 30 + 4 + 20  # the third box wraps
    flow.setGeometry(QRect(0, 0, 250, 54))
    assert [b.geometry().topLeft().toTuple() for b in boxes] == [(0, 5), (110, 0), (0, 34)]
    assert flow.minimumSize() == QSize(100, 30)
