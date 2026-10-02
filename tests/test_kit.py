import json

import pytest
from PySide6.QtCore import QPoint, QSettings, Qt
from PySide6.QtWidgets import QLabel, QSplitter, QVBoxLayout, QWidget

from mag_opt_detective.gui.kit import (
    CollapsibleSection,
    RangeControl,
    RangeSlider,
    SegmentedControl,
    SlidePanel,
)
from mag_opt_detective.gui.settings import Persistence


def focused(qtbot, widget: QWidget) -> QWidget:
    """Show *widget*'s window, activate it and give *widget* the keyboard focus."""
    window = widget.window()
    window.show()
    window.activateWindow()
    qtbot.waitExposed(window)
    widget.setFocus(Qt.FocusReason.OtherFocusReason)
    qtbot.waitUntil(widget.hasFocus)
    return widget


# --- RangeSlider ---------------------------------------------------------------------------
@pytest.fixture
def slider(qtbot):
    s = RangeSlider()
    qtbot.addWidget(s)
    s.setAccessibleName("Field range")
    s.resize(220, s.sizeHint().height())
    s.set_extent(0.0, 100.0)
    s.set_values(20.0, 80.0)
    return s


def test_range_slider_programmatic_values_are_silent(qtbot, slider):
    with qtbot.assertNotEmitted(slider.valuesChanged):
        slider.set_values(90.0, 10.0)  # sorted
        slider.set_extent(50.0, -50.0)  # sorted
    assert slider.values() == (10.0, 90.0)
    assert slider.extent() == (-50.0, 50.0)
    assert slider.min_span() == pytest.approx(1.0)
    slider.set_values(-200.0, 200.0)  # outside the extent: kept, drawn at the ends
    assert slider.values() == (-200.0, 200.0)


def test_range_slider_keyboard(qtbot, slider):
    focused(qtbot, slider)
    assert slider.active_handle() == 0
    with qtbot.waitSignal(slider.valuesChanged) as moved:
        qtbot.keyClick(slider, Qt.Key.Key_Right)
    assert moved.args == pytest.approx([21.0, 80.0])
    qtbot.keyClick(slider, Qt.Key.Key_Right, Qt.KeyboardModifier.ShiftModifier)
    assert slider.values() == pytest.approx((31.0, 80.0))
    with qtbot.waitSignal(slider.editingFinished):
        qtbot.keyClick(slider, Qt.Key.Key_Home)
    assert slider.values() == pytest.approx((0.0, 80.0))

    qtbot.keyClick(slider, Qt.Key.Key_Tab)  # second handle; focus stays
    assert slider.hasFocus()
    assert slider.active_handle() == 1
    qtbot.keyClick(slider, Qt.Key.Key_Down)
    assert slider.values() == pytest.approx((0.0, 79.0))
    qtbot.keyClick(slider, Qt.Key.Key_End)
    assert slider.values() == pytest.approx((0.0, 100.0))
    qtbot.keyClick(slider, Qt.Key.Key_Backtab)
    assert slider.active_handle() == 0

    slider.set_values(50.0, 50.5)  # the minimum span keeps the handles apart
    slider.set_active_handle(1)
    qtbot.keyClick(slider, Qt.Key.Key_Left)
    lo, hi = slider.values()
    assert hi - lo == pytest.approx(slider.min_span())
    assert "from" in slider.accessibleDescription()


def test_range_slider_mouse(qtbot, slider):
    slider.show()
    qtbot.waitExposed(slider)
    y = slider.height() // 2
    x_hi = round(slider._x_for(80.0))
    with qtbot.waitSignal(slider.editingFinished):
        qtbot.mousePress(slider, Qt.MouseButton.LeftButton, pos=QPoint(x_hi, y))
        qtbot.mouseMove(slider, QPoint(round(slider._x_for(60.0)), y))
        qtbot.mouseRelease(slider, Qt.MouseButton.LeftButton, pos=QPoint(x_hi, y))
    assert slider.values()[1] == pytest.approx(60.0, abs=1.0)
    assert slider.active_handle() == 1

    # a press on the groove moves the nearest handle there
    with qtbot.waitSignal(slider.valuesChanged):
        qtbot.mouseClick(slider, Qt.MouseButton.LeftButton, pos=QPoint(round(slider._x_for(5)), y))
    assert slider.values()[0] == pytest.approx(5.0, abs=1.0)
    assert slider.active_handle() == 0


# --- RangeControl --------------------------------------------------------------------------
@pytest.fixture
def control(qtbot):
    rc = RangeControl("Field <i>B</i>", "T")
    qtbot.addWidget(rc)
    rc.set_extent(0.25, 16.0)
    rc.set_range(0.25, 16.0)
    return rc


def type_into(qtbot, spin, text: str) -> None:
    spin.lineEdit().selectAll()
    qtbot.keyClicks(spin.lineEdit(), text)
    qtbot.keyClick(spin.lineEdit(), Qt.Key.Key_Return)


def test_range_control_shows_state(control):
    assert control.is_auto()
    assert control.range() == (0.25, 16.0)
    assert control.note.text() == "Data 0.25 – 16 T"
    assert control.lo_spin.suffix() == " T"
    assert control.lo_spin.locale().decimalPoint() == "."
    assert control.lo_spin.decimals() == 2
    assert control.slider.is_muted()
    assert control.mode.value() == "auto"
    assert control.accessibleName() == "Field B"  # the title without markup
    assert control.lo_spin.accessibleName() == "Field B minimum"
    assert control.slider.accessibleName() == "Field B range"

    control.set_unit("mT")
    assert control.hi_spin.suffix() == " mT"
    assert control.note.text().endswith("16 mT")
    control.set_extent(0.0, 4000.0, note="Data shared with Stacked")
    assert control.note.text() == "Data shared with Stacked"
    assert control.lo_spin.decimals() == 0
    control.set_extent(0.8, 1.3)
    assert control.lo_spin.decimals() == 4


def test_range_control_programmatic_setters_are_silent(qtbot, control):
    with (
        qtbot.assertNotEmitted(control.rangeEdited),
        qtbot.assertNotEmitted(control.autoRequested),
    ):
        control.set_auto(False)
        control.set_range(3.0, 1.0)
        control.set_auto(True)
    assert control.range() == (1.0, 3.0)
    with pytest.raises(ValueError):
        control.set_range(float("nan"), 1.0)


def test_range_control_spin_edit_switches_to_fixed(qtbot, control):
    focused(qtbot, control.hi_spin)
    with qtbot.waitSignal(control.rangeEdited) as edited:
        type_into(qtbot, control.hi_spin, "10")
    assert edited.args == pytest.approx([0.25, 10.0])
    assert control.range() == pytest.approx((0.25, 10.0))
    assert not control.is_auto()
    assert control.mode.value() == "fixed"
    assert not control.slider.is_muted()
    assert control.slider.values() == pytest.approx((0.25, 10.0))


def test_range_control_refuses_an_empty_range_inline(qtbot, control):
    focused(qtbot, control.lo_spin)
    with qtbot.assertNotEmitted(control.rangeEdited):
        type_into(qtbot, control.lo_spin, "20")
    assert control.error() == "The lower limit must be below the upper limit."
    assert control.note.text() == control.error()
    assert control.note.property("error") is True
    assert control.lo_spin.property("invalid") is True
    assert control.range() == (0.25, 16.0)
    assert control.is_auto()

    focused(qtbot, control.hi_spin)  # fixing the other field applies both
    with qtbot.waitSignal(control.rangeEdited) as edited:
        type_into(qtbot, control.hi_spin, "30")
    assert edited.args == pytest.approx([20.0, 30.0])
    assert control.error() == ""
    assert control.note.property("error") is False
    assert control.lo_spin.property("invalid") is False


def test_range_control_slider_and_auto_fixed(qtbot, control):
    focused(qtbot, control.slider)
    with qtbot.waitSignal(control.rangeEdited) as edited:
        qtbot.keyClick(control.slider, Qt.Key.Key_Right)
    lo, hi = edited.args
    assert lo == pytest.approx(0.25 + 0.1575)
    assert hi == 16.0
    assert not control.is_auto()
    assert control.lo_spin.value() == pytest.approx(lo, abs=0.01)

    with qtbot.waitSignal(control.autoRequested):
        control.mode.button("auto").click()
    assert control.is_auto()
    assert control.slider.is_muted()
    control.set_range(0.25, 16.0)  # the owner answers with the data range

    with qtbot.waitSignal(control.rangeEdited) as fixed:
        control.mode.button("fixed").click()
    assert fixed.args == [0.25, 16.0]
    assert not control.is_auto()


def test_range_control_settings_protocol(control):
    control.set_auto(False)
    control.set_range(1.0, 2.5)
    assert json.loads(control.settings_value()) == {"auto": False, "lo": 1.0, "hi": 2.5}
    assert control.set_settings_value('{"auto": true, "lo": 0.5, "hi": 4}')
    assert control.is_auto()
    assert control.range() == (0.5, 4.0)
    for bad in ('{"auto": true, "lo": 5, "hi": 4}', "nonsense", '{"lo": 1, "hi": 2}', 3):
        assert control.set_settings_value(bad) is False
    assert control.range() == (0.5, 4.0)

    control.set_settings_scale(8.0)  # e.g. cm⁻¹ stored while meV is shown
    assert json.loads(control.settings_value())["hi"] == 32.0
    assert control.set_settings_value('{"auto": false, "lo": 8, "hi": 16}')
    assert control.range() == (1.0, 2.0)


# --- SegmentedControl ----------------------------------------------------------------------
@pytest.fixture
def segmented(qtbot):
    control = SegmentedControl(size="sm")
    qtbot.addWidget(control)
    control.setAccessibleName("Derivative")
    for value, text in (("0", "Off"), ("1", "1st"), ("2", "2nd")):
        control.add_option(value, text, tooltip=f"Order {value}")
    return control


def test_segmented_values_and_signals(qtbot, segmented):
    assert segmented.options() == ["0", "1", "2"]
    assert segmented.value() == "0"  # the first option starts selected
    with qtbot.waitSignal(segmented.valueChanged) as changed:
        segmented.set_value("2")
    assert changed.args == ["2"]
    with qtbot.waitSignal(segmented.valueChanged) as clicked:
        segmented.button("1").click()
    assert clicked.args == ["1"]
    assert segmented.button("1").isChecked() and not segmented.button("2").isChecked()
    assert segmented.button("1").accessibleName() == "1st"
    assert segmented.button("1").toolTip() == "Order 1"
    with pytest.raises(ValueError):
        segmented.set_value("3")
    with pytest.raises(ValueError):
        segmented.add_option("1", "again")
    icon_only = segmented.add_option("i", "", tooltip="With icon", icon="sigma")
    assert not icon_only.icon().isNull()
    assert icon_only.accessibleName() == "With icon"


def test_segmented_keyboard_and_roving_focus(qtbot, segmented):
    focused(qtbot, segmented.button("0"))
    in_tab_order = [v for v in segmented.options() if segmented.button(v).focusPolicy() != 0]
    assert in_tab_order == ["0"]
    qtbot.keyClick(segmented.button("0"), Qt.Key.Key_Right)
    assert segmented.value() == "1"
    assert segmented.button("1").hasFocus()
    segmented.set_option_enabled("2", False)
    qtbot.keyClick(segmented.button("1"), Qt.Key.Key_Right)  # disabled options are skipped
    assert segmented.value() == "1"
    qtbot.keyClick(segmented.button("1"), Qt.Key.Key_Left)
    assert segmented.value() == "0"
    qtbot.keyClick(segmented.button("0"), Qt.Key.Key_End)
    assert segmented.value() == "1"


def test_segmented_settings_protocol(segmented):
    segmented.set_value("2")
    assert segmented.settings_value() == "2"
    assert segmented.set_settings_value("1")
    assert segmented.value() == "1"
    assert segmented.set_settings_value("7") is False
    assert segmented.set_settings_value(1) is False
    assert segmented.value() == "1"


# --- SlidePanel ----------------------------------------------------------------------------
class Content(QWidget):
    def __init__(self, text: str):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(text))
        self.setMinimumSize(120, 60)


def make_splitter(qtbot, orientation=Qt.Orientation.Horizontal, size=(1000, 600)):
    splitter = QSplitter(orientation)
    qtbot.addWidget(splitter)
    left = SlidePanel(Content("left"), 250)
    centre = QLabel("plot")
    right = SlidePanel(Content("right"), 200)
    for widget in (left, centre, right):
        splitter.addWidget(widget)
    splitter.setStretchFactor(1, 1)
    splitter.resize(*size)
    return splitter, left, right


def shown(qtbot, splitter):
    splitter.show()
    qtbot.waitExposed(splitter)
    return splitter


def test_slide_panel_open_close_without_animation(qtbot):
    splitter, left, _right = make_splitter(qtbot)
    shown(qtbot, splitter)
    total = sum(splitter.sizes())
    assert splitter.sizes()[0] == 250
    assert splitter.sizes()[2] == 200

    with qtbot.waitSignal(left.openChanged) as closed:
        left.set_open(False, animate=False)
    assert closed.args == [False]
    assert not left.is_open()
    assert splitter.sizes()[0] == 0
    assert sum(splitter.sizes()) == total  # the plot got the space
    assert not left.content().isVisible()  # hidden content leaves the Tab order

    left.toggle(animate=False)
    assert left.is_open()
    assert splitter.sizes() == [250, total - 450, 200]
    assert left.content().isVisible()
    with qtbot.assertNotEmitted(left.openChanged):
        left.set_open(True, animate=False)


def test_slide_panel_user_drag_and_restore(qtbot):
    splitter, left, _right = make_splitter(qtbot)
    shown(qtbot, splitter)
    sizes = splitter.sizes()
    splitter.setSizes([180, sizes[1] + 70, sizes[2]])  # the user drags the handle
    splitter.splitterMoved.emit(180, 1)
    assert left.open_size() == 180

    with qtbot.waitSignal(left.openChanged) as closed:  # dragging to zero closes
        splitter.setSizes([0, sizes[1] + 250, sizes[2]])
        splitter.splitterMoved.emit(0, 1)
    assert closed.args == [False]
    assert not left.is_open()

    left.set_open(True, animate=False)  # reopening restores the last open size
    assert splitter.sizes()[0] == 180

    left.set_open(False, animate=False)
    with qtbot.waitSignal(left.openChanged) as reopened:  # dragging out of zero opens
        splitter.setSizes([150, sizes[1] + 100, sizes[2]])
        splitter.splitterMoved.emit(150, 1)
    assert reopened.args == [True]
    assert left.open_size() == 150


def test_slide_panel_vertical_drawer(qtbot):
    splitter, _top, drawer = make_splitter(qtbot, Qt.Orientation.Vertical, size=(500, 800))
    shown(qtbot, splitter)
    assert splitter.sizes()[2] == 200
    drawer.set_open(False, animate=False)
    assert splitter.sizes()[2] == 0
    drawer.set_open(True, animate=False)
    assert splitter.sizes()[2] == 200


def test_slide_panel_animation_keeps_the_content_size(qtbot):
    splitter, left, right = make_splitter(qtbot)
    shown(qtbot, splitter)
    right.duration_ms = 1500  # slow enough to look at a frame halfway
    right.set_open(False)
    assert right.is_animating()
    assert not right.is_open()
    qtbot.waitUntil(lambda: 20 < splitter.sizes()[2] < 180)
    content = right.content().geometry()
    assert content.width() == 200  # not squeezed, clipped by the pane
    assert content.x() == 0  # the trailing pane slides towards the plot
    assert right.minimumSizeHint().width() == 0
    right.set_open(True)  # reverses from where it is
    left.set_open(False)
    qtbot.waitUntil(lambda: not left.is_animating() and not right.is_animating(), timeout=4000)
    assert splitter.sizes()[0] == 0
    assert splitter.sizes()[2] == 200
    assert right.minimumSizeHint().width() >= 120


def test_slide_panel_before_show_and_settings(qtbot):
    splitter, left, _right = make_splitter(qtbot)
    assert left.set_settings_value('{"open": false, "size": 240}')
    assert not left.is_open()
    assert json.loads(left.settings_value()) == {"open": False, "size": 240}
    shown(qtbot, splitter)
    qtbot.waitUntil(lambda: splitter.sizes()[0] == 0)
    left.set_open(True, animate=False)
    assert splitter.sizes()[0] == 240
    for bad in ("{}", '{"open": true, "size": 0}', '{"open": "maybe", "size": 10}', "x"):
        assert left.set_settings_value(bad) is False
    assert left.is_open()


def test_slide_panel_restores_an_exact_open_size_before_show(qtbot):
    splitter, left, _right = make_splitter(qtbot)
    assert left.set_settings_value('{"open": true, "size": 320}')
    shown(qtbot, splitter)  # setSizes before show is only relative; the panel settles after
    qtbot.waitUntil(lambda: splitter.sizes()[0] == 320)


def test_slide_panel_outside_a_splitter(qtbot):
    panel = SlidePanel(Content("alone"), 100)
    qtbot.addWidget(panel)
    panel.show()
    panel.set_open(False)
    assert not panel.content().isVisible()
    panel.set_open(True)
    assert panel.content().isVisible()


# --- CollapsibleSection --------------------------------------------------------------------
def test_collapsible_section(qtbot):
    sub = QLabel("Ratio · 0th")
    section = CollapsibleSection("Colour", trailing=sub)
    qtbot.addWidget(section)
    section.body_layout().addWidget(QLabel("body"))
    section.show()
    qtbot.waitExposed(section)
    assert section.is_expanded()
    assert section.trailing() is sub
    assert section.button.accessibleName() == "Colour"

    with qtbot.waitSignal(section.toggled) as toggled:
        section.set_expanded(False, animate=False)
    assert toggled.args == [False]
    assert not section.body.isVisible()
    assert not section.button.isChecked()

    section.duration_ms = 300
    with qtbot.waitSignal(section.toggled) as clicked:
        section.button.click()  # animated
    assert clicked.args == [True]
    assert section.is_animating()
    qtbot.waitUntil(lambda: not section.is_animating())
    assert section.body.isVisible()
    assert section.body.height() > 0

    with qtbot.waitSignal(section.toggled):  # a click anywhere on the header toggles
        pos = QPoint(section.header.width() - 3, section.header.height() // 2)
        qtbot.mouseClick(section.header, Qt.MouseButton.LeftButton, pos=pos)
    assert not section.is_expanded()


def test_collapsible_section_settings_protocol(qtbot):
    section = CollapsibleSection("View", expanded=False)
    qtbot.addWidget(section)
    assert section.settings_value() is False
    assert section.set_settings_value("true")
    assert section.is_expanded()
    assert section.set_settings_value(False)
    assert not section.is_expanded()
    assert section.set_settings_value("maybe") is False


# --- settings protocol through Persistence -------------------------------------------------
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
