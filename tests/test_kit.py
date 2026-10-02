import pytest
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QWidget

from mag_opt_detective.gui.settings import Persistence


def ini_settings(tmp_path) -> QSettings:
    return QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)


class Duck(QWidget):
    """A minimal widget with the settings protocol."""

    def __init__(self, value=3):
        super().__init__()
        self.value = value
        self.raise_on_set = False

    def settings_value(self):
        return self.value

    def set_settings_value(self, value):
        if self.raise_on_set:
            raise ValueError("bad")
        if str(value) == "invalid":
            return False
        self.value = int(value)
        return None  # anything but False is success


def test_settings_protocol_duck_typing(qtbot, tmp_path):
    duck = Duck()
    qtbot.addWidget(duck)
    persistence = Persistence(ini_settings(tmp_path))
    persistence.bind("duck", duck)
    duck.value = 7
    persistence.save()
    duck.value = 0
    persistence.restore()
    assert duck.value == 7  # stored as text in the ini file, parsed by the widget

    persistence.set_value("duck", "invalid")
    persistence.restore()
    assert duck.value == 7
    duck.raise_on_set = True
    persistence.set_value("duck", "8")
    persistence.restore()  # errors in the widget count as invalid values
    assert duck.value == 7

    duck.value = [1, 2]
    with pytest.raises(TypeError):
        persistence.save()
