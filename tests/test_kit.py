import json

import pytest
from PySide6.QtCore import QPoint, QSettings, QSize, Qt
from PySide6.QtWidgets import QCheckBox, QLabel, QLineEdit, QSplitter, QVBoxLayout, QWidget

from mag_opt_detective.gui import icons, theme
from mag_opt_detective.gui.kit import (
    CollapsibleSection,
    InfoBar,
    NudgeSlider,
    RangeControl,
    RangeSlider,
    SegmentedControl,
    SlidePanel,
    SmallButton,
    Switch,
)
from mag_opt_detective.gui.kit.nudge_slider import jog_fraction
from mag_opt_detective.gui.kit.range_slider import BOTH, HI, LO
from mag_opt_detective.gui.settings import PREFIX, Persistence


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
    qtbot.keyClick(slider, Qt.Key.Key_PageUp)
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
    x_hi = round(slider.x_for(80.0))
    with qtbot.waitSignal(slider.editingFinished):
        qtbot.mousePress(slider, Qt.MouseButton.LeftButton, pos=QPoint(x_hi, y))
        qtbot.mouseMove(slider, QPoint(round(slider.x_for(60.0)), y))
        qtbot.mouseRelease(slider, Qt.MouseButton.LeftButton, pos=QPoint(x_hi, y))
    assert slider.values()[1] == pytest.approx(60.0, abs=1.0)
    assert slider.active_handle() == 1

    # a press on the groove moves the nearest handle there
    with qtbot.waitSignal(slider.valuesChanged):
        qtbot.mouseClick(slider, Qt.MouseButton.LeftButton, pos=QPoint(round(slider.x_for(5)), y))
    assert slider.values()[0] == pytest.approx(5.0, abs=1.0)
    assert slider.active_handle() == 0


@pytest.mark.parametrize(
    ("start", "press_at", "drag_to"),
    [((100.0, 100.0), 101.0, 60.0), ((0.0, 0.0), -1.0, 40.0), ((99.5, 100.0), 101.0, 60.0)],
)
def test_range_slider_handles_stay_in_the_extent(qtbot, slider, start, press_at, drag_to):
    slider.show()
    qtbot.waitExposed(slider)
    y = slider.height() // 2
    slider.set_values(*start)  # handles drawn on top of each other at an end
    press = QPoint(round(slider.x_for(start[0]) + press_at - start[0]), y)  # a pixel beside
    qtbot.mousePress(slider, Qt.MouseButton.LeftButton, pos=press)
    qtbot.mouseMove(slider, QPoint(round(slider.x_for(drag_to)), y))
    qtbot.mouseRelease(slider, Qt.MouseButton.LeftButton, pos=press)
    lo, hi = slider.values()
    assert 0.0 <= lo < hi <= 100.0
    assert drag_to == pytest.approx(lo if drag_to < start[0] else hi, abs=1.0)  # separated

    slider.set_values(99.5, 100.0)  # a key on the high handle cannot push it past the end
    slider.set_active_handle(1)
    focused(qtbot, slider)
    qtbot.keyClick(slider, Qt.Key.Key_Right)
    assert slider.values() == pytest.approx((99.0, 100.0))


def test_range_slider_groove_press_beside_handles_on_top_of_each_other(qtbot, slider):
    slider.show()
    qtbot.waitExposed(slider)
    y = slider.height() // 2
    for press, expected in ((70.0, (50.0, 70.0)), (30.0, (30.0, 50.0))):
        slider.set_values(50.0, 50.0)  # the handle on the side of the press moves there
        at = QPoint(round(slider.x_for(press)), y)
        qtbot.mouseClick(slider, Qt.MouseButton.LeftButton, pos=at)
        assert slider.values() == pytest.approx(expected, abs=1.0)


def record(slider) -> tuple[list, list]:
    """The values of every valuesChanged and a list that grows on every editingFinished."""
    moves, finished = [], []
    slider.valuesChanged.connect(lambda lo, hi: moves.append((lo, hi)))
    slider.editingFinished.connect(lambda: finished.append(True))
    return moves, finished


def drag(qtbot, slider, *xs: float) -> None:
    """Press at the first x, move through the others, release at the last."""
    y = slider.height() // 2
    qtbot.mousePress(slider, Qt.MouseButton.LeftButton, pos=QPoint(round(xs[0]), y))
    for x in xs[1:]:
        qtbot.mouseMove(slider, QPoint(round(x), y))
    qtbot.mouseRelease(slider, Qt.MouseButton.LeftButton, pos=QPoint(round(xs[-1]), y))


def test_range_slider_drag_between_the_handles_moves_the_range(qtbot, slider):
    slider.show()
    qtbot.waitExposed(slider)
    moves, finished = record(slider)
    x = slider.x_for
    assert slider.part_at(x(50.0)) == BOTH
    drag(qtbot, slider, x(50.0), x(55.0), x(60.0), x(60.0))
    assert len(moves) == 2  # one per move that changed the values
    assert len(finished) == 1
    assert all(hi - lo == pytest.approx(60.0) for lo, hi in moves)
    assert slider.values() == pytest.approx((30.0, 90.0), abs=1.0)
    assert slider.active_handle() == 0  # the keyboard keeps its handle


def test_range_slider_range_drag_stops_at_the_ends(qtbot, slider):
    slider.show()
    qtbot.waitExposed(slider)
    moves, finished = record(slider)
    y = slider.height() // 2
    qtbot.mousePress(slider, Qt.MouseButton.LeftButton, pos=QPoint(round(slider.x_for(50)), y))
    qtbot.mouseMove(slider, QPoint(slider.width() + 40, y))
    assert slider.values() == pytest.approx((40.0, 100.0))
    assert slider.values()[1] == 100.0  # exactly at the end, the width kept
    qtbot.mouseMove(slider, QPoint(-40, y))
    assert slider.values() == pytest.approx((0.0, 60.0))
    assert slider.values()[0] == 0.0
    qtbot.mouseMove(slider, QPoint(-80, y))  # further out: nothing changes, nothing emitted
    qtbot.mouseRelease(slider, Qt.MouseButton.LeftButton, pos=QPoint(-80, y))
    assert (len(moves), len(finished)) == (2, 1)


def test_range_slider_handles_win_on_a_narrow_range(qtbot, slider):
    slider.show()
    qtbot.waitExposed(slider)
    x = slider.x_for
    assert slider.part_at(x(80.0) - 4) == HI  # where a handle overlaps the bar, it wins
    assert slider.part_at(x(10.0)) is None  # the groove
    slider.set_values(40.0, 51.0)  # 22 px apart: 3 px of bar between the handles' reach
    assert slider.part_at(x(45.0)) == LO  # too little to grab: the nearer handle
    assert slider.part_at(x(46.0)) == HI

    slider.set_values(50.0, 52.0)  # nearly on top of each other
    _, finished = record(slider)
    drag(qtbot, slider, x(52.0), x(70.0))
    assert slider.values() == pytest.approx((50.0, 70.0), abs=1.0)  # the high handle alone
    assert slider.active_handle() == 1
    assert len(finished) == 1


def test_range_slider_shift_keys_move_the_range(qtbot, slider):
    focused(qtbot, slider)
    moves, finished = record(slider)
    shift = Qt.KeyboardModifier.ShiftModifier
    qtbot.keyClick(slider, Qt.Key.Key_Right, shift)
    assert slider.values() == pytest.approx((21.0, 81.0))
    assert (len(moves), len(finished)) == (1, 1)
    qtbot.keyClick(slider, Qt.Key.Key_Left, shift)
    assert slider.values() == pytest.approx((20.0, 80.0))
    qtbot.keyClick(slider, Qt.Key.Key_PageUp, shift)
    qtbot.keyClick(slider, Qt.Key.Key_PageUp, shift)  # stops at the end, the width kept
    assert slider.values() == pytest.approx((40.0, 100.0))
    assert (len(moves), len(finished)) == (4, 4)
    qtbot.keyClick(slider, Qt.Key.Key_Up, shift)  # no room: no signal
    assert (len(moves), len(finished)) == (4, 4)
    qtbot.keyClick(slider, Qt.Key.Key_Home, shift)
    assert slider.values() == pytest.approx((0.0, 60.0))
    assert all(hi - lo == pytest.approx(60.0) for lo, hi in moves)
    assert slider.active_handle() == 0
    qtbot.keyClick(slider, Qt.Key.Key_Right)  # without Shift: the active handle alone
    assert slider.values() == pytest.approx((1.0, 60.0))
    assert "Shift+arrow" in slider.toolTip()
    assert "Shift+arrow" in slider.accessibleDescription()

    slider.set_values(-20.0, 50.0)  # an end outside the extent does not go further out
    qtbot.keyClick(slider, Qt.Key.Key_Left, shift)
    assert slider.values() == (-20.0, 50.0)
    qtbot.keyClick(slider, Qt.Key.Key_Right, shift)
    assert slider.values() == pytest.approx((-19.0, 51.0))

    slider.set_min_span(10.0)  # pinned at the end, the high handle moves with the range
    slider.set_values(90.0, 100.0)
    qtbot.keyClick(slider, Qt.Key.Key_End, shift)
    assert slider.values() == (90.0, 100.0)
    qtbot.keyClick(slider, Qt.Key.Key_Left, shift)
    assert slider.values() == pytest.approx((89.0, 99.0))

    slider.set_values(0.0, 100.0)  # a range that fills the extent cannot move
    with qtbot.assertNotEmitted(slider.valuesChanged):
        qtbot.keyClick(slider, Qt.Key.Key_Right, shift)


@pytest.mark.parametrize("values", [(0.0, 100.0), (-5.0, 105.0)])
def test_range_slider_bar_of_a_range_that_cannot_move(qtbot, slider, values):
    slider.show()
    qtbot.waitExposed(slider)
    slider.set_values(*values)  # fills the extent (as on Auto) or goes beyond it
    moves, finished = record(slider)
    x, y = slider.x_for(50.0), slider.height() // 2
    assert slider.part_at(x) == BOTH  # grabbed as a bar, as a narrower range would be
    qtbot.mousePress(slider, Qt.MouseButton.LeftButton, pos=QPoint(round(x), y))
    assert slider.cursor().shape() == Qt.CursorShape.ArrowCursor  # no closed hand
    for to in (x + 40, x - 40, slider.width() + 40, -40):
        qtbot.mouseMove(slider, QPoint(round(to), y))
    slider.grab()
    qtbot.mouseRelease(slider, Qt.MouseButton.LeftButton, pos=QPoint(-40, y))
    assert slider.values() == values
    assert (moves, finished) == ([], [])  # nothing changed: no signal at all


def test_range_slider_cursor_and_hover_over_the_range(qtbot, slider):
    slider.show()
    qtbot.waitExposed(slider)
    y = slider.height() // 2
    middle = QPoint(round(slider.x_for(50.0)), y)
    # a move without buttons sets the cursor position, which is ignored when unchanged:
    # coming from the groove makes sure that the move to the middle is seen
    qtbot.mouseMove(slider, QPoint(round(slider.x_for(5.0)), y))
    assert slider.cursor().shape() == Qt.CursorShape.ArrowCursor
    qtbot.mouseMove(slider, middle)
    assert slider.cursor().shape() == Qt.CursorShape.OpenHandCursor
    slider.grab()  # paints the hover band
    qtbot.mousePress(slider, Qt.MouseButton.LeftButton, pos=middle)
    assert slider.cursor().shape() == Qt.CursorShape.ClosedHandCursor
    slider.grab()
    qtbot.mouseRelease(slider, Qt.MouseButton.LeftButton, pos=middle)
    assert slider.cursor().shape() == Qt.CursorShape.OpenHandCursor
    qtbot.mouseMove(slider, QPoint(round(slider.x_for(80.0)), y))  # a handle
    assert slider.cursor().shape() == Qt.CursorShape.ArrowCursor
    qtbot.mouseMove(slider, middle)
    slider.set_values(0.0, 100.0)  # the range can no longer move
    assert slider.cursor().shape() == Qt.CursorShape.ArrowCursor


# --- NudgeSlider ---------------------------------------------------------------------------
@pytest.fixture
def nudge(qtbot):
    s = NudgeSlider()
    qtbot.addWidget(s)
    s.setAccessibleName("g factor")
    s.resize(221, s.sizeHint().height())  # a track of 200 px: the half track is 100 px
    s.set_range(0.0, 100.0)
    s.set_value(20.0)
    s.duration_ms = 0
    return s


def nudge_record(slider) -> tuple[list, list]:
    """The values of every valueChanged and a list that grows on every editingFinished."""
    moves, finished = [], []
    slider.valueChanged.connect(moves.append)
    slider.editingFinished.connect(lambda: finished.append(True))
    return moves, finished


def jog(qtbot, slider, *offsets: float, release: bool = True) -> None:
    """Press at the centre, move the handle *offsets* (of the half track) away, release."""
    x0, x1 = slider.track()
    y = slider.height() // 2
    centre = round(slider.centre_x() - 0.5)
    qtbot.mousePress(slider, Qt.MouseButton.LeftButton, pos=QPoint(centre, y))
    for offset in offsets:
        qtbot.mouseMove(slider, QPoint(round(centre + offset * (x1 - x0) / 2), y))
    if release:
        end = round(centre + offsets[-1] * (x1 - x0) / 2) if offsets else centre
        qtbot.mouseRelease(slider, Qt.MouseButton.LeftButton, pos=QPoint(end, y))


def test_nudge_slider_range_mode_maps_its_range(qtbot, nudge):
    assert nudge.mode() == "range" and nudge.handle_x() == pytest.approx(nudge.x_for(20.0))
    moves, finished = nudge_record(nudge)
    y = nudge.height() // 2
    qtbot.mousePress(nudge, Qt.MouseButton.LeftButton, pos=QPoint(round(nudge.x_for(60)), y))
    assert nudge.value() == pytest.approx(60.0, abs=0.5)  # the groove: the handle jumps
    qtbot.mouseMove(nudge, QPoint(round(nudge.x_for(80)), y))
    assert nudge.value() == pytest.approx(80.0, abs=0.5) and len(moves) == 2 and not finished
    qtbot.mouseRelease(nudge, Qt.MouseButton.LeftButton, pos=QPoint(round(nudge.x_for(80)), y))
    assert finished == [True]
    with qtbot.assertNotEmitted(nudge.valueChanged):
        nudge.set_value(150.0)  # kept, drawn at the end
    assert nudge.value() == 150.0 and nudge.handle_x() == nudge.track()[1]


def test_nudge_slider_relative_drag_springs_back(qtbot, nudge):
    nudge.set_mode("relative", 10.0)
    assert nudge.handle_x() == nudge.centre_x()  # at rest in the centre
    moves, finished = nudge_record(nudge)
    jog(qtbot, nudge, 0.25, 0.5, release=False)
    assert nudge.is_dragging() and nudge.handle_offset() == pytest.approx(0.5)
    expected = 20.0 * (1 + 0.10 * jog_fraction(0.5))  # +3.125 % on the gentle curve
    assert nudge.value() == pytest.approx(expected, abs=1e-3) and len(moves) == 2
    assert nudge.change_text() == "+3.1 %"
    assert not finished
    y = nudge.height() // 2
    qtbot.mouseRelease(nudge, Qt.MouseButton.LeftButton, pos=QPoint(160, y))
    assert finished == [True] and nudge.value() == moves[-1]  # the release changes nothing
    assert nudge.handle_offset() == 0.0 and not nudge.is_dragging()

    jog(qtbot, nudge, 1.0)  # the next drag works around the new value, the end gives the span
    assert nudge.value() == pytest.approx(expected * 1.1, abs=1e-3)
    jog(qtbot, nudge, -1.0, -2.0)  # beyond the end: still the whole span
    assert nudge.value() == pytest.approx(expected * 1.1 * 0.9, abs=1e-3)
    count = len(moves)
    jog(qtbot, nudge)  # a click alone changes nothing
    assert len(moves) == count and len(finished) == 3


def test_nudge_slider_springs_back_with_an_animation(qtbot, nudge):
    nudge.set_mode("relative", 50.0)
    nudge.duration_ms = 60
    nudge.show()
    qtbot.waitExposed(nudge)
    jog(qtbot, nudge, 0.8)
    value = nudge.value()
    assert value > 20.0 and nudge.is_springing() and nudge.handle_offset() > 0
    qtbot.waitUntil(lambda: not nudge.is_springing(), timeout=2000)
    assert nudge.handle_offset() == 0.0 and nudge.value() == value


def test_nudge_slider_relative_zero_floor_and_bounds(qtbot, nudge):
    nudge.set_mode("relative", 10.0)
    nudge.set_value(0.0)
    jog(qtbot, nudge, 1.0)
    assert nudge.value() == pytest.approx(0.1)  # 10 % of 1 for a zero value without a floor
    nudge.set_value(0.0)
    nudge.set_floor(5.0)  # a typical scale
    jog(qtbot, nudge, 1.0)
    assert nudge.value() == pytest.approx(0.5)
    jog(qtbot, nudge, 1.0)  # still small: the floor keeps the span
    assert nudge.value() == pytest.approx(1.0)
    nudge.set_bounds(minimum=0.0)
    jog(qtbot, nudge, -1.0)
    jog(qtbot, nudge, -1.0)
    jog(qtbot, nudge, -1.0)
    assert nudge.value() == 0.0  # held at the bound
    nudge.set_value(2.0)
    nudge.set_floor(0.0)
    nudge.set_mode("relative", 1.0)
    jog(qtbot, nudge, 1.0)
    assert nudge.value() == pytest.approx(2.02)


def test_nudge_slider_keyboard(qtbot, nudge):
    focused(qtbot, nudge)
    moves, finished = nudge_record(nudge)
    qtbot.keyClick(nudge, Qt.Key.Key_Right)  # range: 1 % of the range
    assert nudge.value() == pytest.approx(21.0) and finished == [True]
    qtbot.keyClick(nudge, Qt.Key.Key_Right, Qt.KeyboardModifier.ShiftModifier)  # 10 %
    assert nudge.value() == pytest.approx(31.0)
    qtbot.keyClick(nudge, Qt.Key.Key_Home)
    assert nudge.value() == 0.0
    qtbot.keyClick(nudge, Qt.Key.Key_Left)  # nothing below the range: nothing emitted
    assert len(moves) == 3 and len(finished) == 3

    nudge.set_mode("relative", 10.0)
    nudge.set_value(20.0)
    qtbot.keyClick(nudge, Qt.Key.Key_Right)  # a tenth of the span: 1 %
    assert nudge.value() == pytest.approx(20.2)
    qtbot.keyClick(nudge, Qt.Key.Key_Left, Qt.KeyboardModifier.ShiftModifier)  # the span
    assert nudge.value() == pytest.approx(20.2 * 0.9)
    qtbot.keyClick(nudge, Qt.Key.Key_PageUp)
    assert nudge.value() == pytest.approx(20.2 * 0.9 * 1.1)
    assert len(finished) == 6 and nudge.handle_offset() == 0.0
    qtbot.keyClick(nudge, Qt.Key.Key_Escape)  # QTest leaves Shift held until the next key


def test_nudge_slider_whole_numbers(qtbot, nudge):
    nudge.set_step(1.0)
    nudge.set_range(1.0, 40.0)
    nudge.set_bounds(1.0, 40.0)
    nudge.set_modes(("range",))
    y = nudge.height() // 2
    qtbot.mouseClick(nudge, Qt.MouseButton.LeftButton, pos=QPoint(round(nudge.x_for(7.3)), y))
    assert nudge.value() == 7.0
    nudge.set_mode("relative")  # not offered: ignored
    assert nudge.mode() == "range"
    menu = nudge.mode_menu()
    assert [a.isEnabled() for a in menu.actions()] == [True, False, False, False]


def test_nudge_slider_set_value_during_a_relative_drag(qtbot, nudge):
    """A value shown anew mid-drag (e.g. in another unit) goes on from there, without a jump."""
    nudge.set_mode("relative", 10.0)
    jog(qtbot, nudge, 0.5, release=False)
    shown = nudge.value() * 8.0656  # the same value in cm-1
    nudge.set_value(shown)
    y = nudge.height() // 2
    x = round(nudge.handle_x())
    qtbot.mouseMove(nudge, QPoint(x + 1, y))
    assert nudge.value() == pytest.approx(shown, rel=0.01)  # a pixel further, not 3 % away
    qtbot.mouseRelease(nudge, Qt.MouseButton.LeftButton, pos=QPoint(x + 1, y))


def test_nudge_slider_keeps_a_value_set_beyond_its_bounds(qtbot, nudge):
    nudge.set_mode("relative", 10.0)
    nudge.set_bounds(minimum=0.0)
    nudge.set_value(-5.0)  # e.g. typed: kept, and not pulled up to 0 by a drag
    jog(qtbot, nudge, 0.5)
    assert -5.0 < nudge.value() < -4.5
    start = nudge.value()
    jog(qtbot, nudge, -1.0)  # further out: held where it was
    assert nudge.value() == start


def test_nudge_slider_context_menus_are_deleted(qtbot, nudge):
    from PySide6.QtCore import QEvent, QTimer
    from PySide6.QtGui import QContextMenuEvent
    from PySide6.QtWidgets import QApplication, QMenu

    def close_menus():
        for widget in QApplication.topLevelWidgets():
            if isinstance(widget, QMenu) and widget.isVisible():
                widget.close()

    nudge.show()
    qtbot.waitExposed(nudge)
    at = QPoint(10, 10)
    for _ in range(3):
        QTimer.singleShot(30, close_menus)
        event = QContextMenuEvent(QContextMenuEvent.Reason.Mouse, at, nudge.mapToGlobal(at))
        QApplication.sendEvent(nudge, event)
    QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    assert nudge.findChildren(QMenu) == []


def test_nudge_slider_mode_menu_and_settings(qtbot, nudge):
    menu = nudge.mode_menu()
    texts = [a.text() for a in menu.actions()]
    assert texts == ["Range", "Relative ±1 %", "Relative ±10 %", "Relative ±50 %"]
    assert [a.isChecked() for a in menu.actions()] == [True, False, False, False]
    with qtbot.waitSignal(nudge.modeChanged) as changed:
        menu.actions()[3].trigger()
    assert changed.args == ["relative", 50.0]
    assert (nudge.mode(), nudge.span()) == ("relative", 50.0)
    assert "±50 %" in nudge.accessibleDescription()
    assert [a.isChecked() for a in nudge.mode_menu().actions()] == [False, False, False, True]
    with qtbot.assertNotEmitted(nudge.modeChanged):
        nudge.set_mode("range")
        nudge.mode_menu().actions()[0].trigger()  # already the mode: nothing changes
    assert json.loads(nudge.settings_value()) == {"mode": "range", "span": 50.0}
    assert nudge.set_settings_value('{"mode": "relative", "span": 1}')
    assert (nudge.mode(), nudge.span()) == ("relative", 1.0)
    for bad in ('{"mode": "jog"}', '{"mode": "relative", "span": -5}', "nonsense", 3):
        assert not nudge.set_settings_value(bad)
    assert (nudge.mode(), nudge.span()) == ("relative", 1.0)
    with pytest.raises(ValueError):
        nudge.set_mode("relative", 0.0)


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
    assert (control.lo_spin.text(), control.hi_spin.text()) == ("0.25 T", "16 T")  # as the note
    assert control.lo_spin.decimals() == 6  # 0.25 to four significant digits, and two more
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
    assert control.lo_spin.resolution == 0
    control.set_extent(0.8, 1.3)
    assert control.lo_spin.resolution == 4


def test_range_control_fields_and_note_show_the_same_digits(control):
    control.set_unit("meV")
    control.set_extent(3500 / 80.656, 3200 / 8.0656)  # 43.394 – 396.746
    control.set_range(*control.extent())
    assert control.note.text() == "Data 43.39 – 396.7 meV"
    assert (control.lo_spin.text(), control.hi_spin.text()) == ("43.39 meV", "396.7 meV")
    control.set_extent(0.995, 1.002)  # a narrow range keeps the digits that resolve it
    control.set_range(0.995123, 1.0019)
    assert control.lo_spin.text() == "0.995123 meV"


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

    focused(qtbot, control.lo_spin)  # resetting the fields drops a pending error
    type_into(qtbot, control.lo_spin, "40")
    assert control.error()
    control.set_extent(0.0, 50.0)
    assert control.error() == ""
    assert control.note.text() == "Data 0 – 50 T"
    assert control.lo_spin.property("invalid") is False
    assert control.lo_spin.value() == 20.0
    type_into(qtbot, control.lo_spin, "40")
    control.set_unit("mT")
    assert control.error() == ""
    assert control.note.property("error") is False


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


def test_range_control_range_drag_switches_to_fixed(qtbot, control):
    control.resize(300, control.sizeHint().height())
    control.show()
    qtbot.waitExposed(control)
    control.set_range(4.0, 8.0)  # still Auto
    slider = control.slider
    edits = []
    control.rangeEdited.connect(lambda lo, hi: edits.append((lo, hi)))
    drag(qtbot, slider, slider.x_for(6.0), slider.x_for(8.0), slider.x_for(10.0))
    lo, hi = control.range()
    assert hi - lo == pytest.approx(4.0)
    assert lo == pytest.approx(8.0, abs=0.2)
    assert edits[-1] == (lo, hi)
    assert len(edits) == 2
    assert not control.is_auto()
    assert control.mode.value() == "fixed"
    assert not slider.is_muted()
    assert (control.lo_spin.value(), control.hi_spin.value()) == pytest.approx((lo, hi), abs=1e-4)

    control.set_auto(True)  # Shift+arrows count as an edit too
    focused(qtbot, slider)
    with qtbot.waitSignal(control.rangeEdited) as edited:
        qtbot.keyClick(slider, Qt.Key.Key_Left, Qt.KeyboardModifier.ShiftModifier)
    assert edited.args == pytest.approx([lo - 0.1575, hi - 0.1575])
    assert not control.is_auto()
    assert control.hi_spin.value() == pytest.approx(hi - 0.1575, abs=1e-4)


@pytest.mark.parametrize("data_range", [(0.25, 16.0), (0.0, 20.0)])
def test_range_control_bar_drag_on_auto_range_changes_nothing(qtbot, control, data_range):
    control.resize(300, control.sizeHint().height())
    control.show()
    qtbot.waitExposed(control)
    control.set_range(*data_range)  # Auto: the data range fills the slider (or goes beyond)
    slider = control.slider
    with qtbot.assertNotEmitted(control.rangeEdited):
        drag(qtbot, slider, slider.x_for(8.0), slider.x_for(8.0) + 40, slider.x_for(8.0) - 40)
    assert control.range() == data_range
    assert control.is_auto()
    assert control.mode.value() == "auto"
    assert slider.is_muted()


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
    given = icons.icon("pin", "#ff0000")
    with_qicon = segmented.add_option("q", "Pinned", icon=given)
    assert with_qicon.icon().cacheKey() == given.cacheKey()
    assert with_qicon.toolButtonStyle() == Qt.ToolButtonStyle.ToolButtonTextBesideIcon


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


def test_segments_are_as_wide_as_their_text_and_padding(segmented):
    """As the mockup's ``.seg.sm`` button: the text in an 8 px padding and a 1 px border (Qt's
    tool button would add the width of two spaces)."""
    segmented.setStyleSheet(theme.build_stylesheet(theme.tokens_for(False)))
    for value in segmented.options():
        button = segmented.button(value)
        text = button.fontMetrics().horizontalAdvance(button.text())
        assert text + 2 * 9 <= button.sizeHint().width() <= text + 2 * 9 + 5
        assert button.minimumSizeHint() == button.sizeHint()


def test_small_button_puts_the_mockup_gap_between_icon_and_text(qtbot):
    """As the mockup's ``.btn.sm``: a 14 px icon and 6 px before the text (Qt leaves 2 px)."""
    sheet = theme.build_stylesheet(theme.tokens_for(False))
    plain, with_icon = SmallButton("Save current map"), SmallButton("Save current map", "save")
    for button in (plain, with_icon):
        qtbot.addWidget(button)
        button.setStyleSheet(sheet)
    assert with_icon.property("kit") == "button" and with_icon.toolTip() == ""
    assert with_icon.sizeHint().width() - plain.sizeHint().width() == 14 + 6
    with_icon.setCheckable(True)
    with_icon.setChecked(True)  # painted in the accent colour, as the stylesheet's checked one
    assert not with_icon.grab().isNull()


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
        self.edit = QLineEdit()
        layout.addWidget(self.edit)
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


def drag_handle(qtbot, splitter: QSplitter, index: int, to: int) -> None:
    """Drag handle *index* of a horizontal splitter with the mouse until it is at x = *to*."""
    handle = splitter.handle(index)
    grab = QPoint(handle.width() // 2, handle.height() // 2)
    qtbot.mousePress(handle, Qt.MouseButton.LeftButton, pos=grab)
    target = QPoint(to - handle.x() + grab.x(), grab.y())
    qtbot.mouseMove(handle, target)
    qtbot.mouseRelease(handle, Qt.MouseButton.LeftButton, pos=target)


def test_slide_panel_user_drag_and_restore(qtbot):
    splitter, left, _right = make_splitter(qtbot)
    shown(qtbot, splitter)
    drag_handle(qtbot, splitter, 1, 180)
    assert splitter.sizes()[0] == 180
    assert left.open_size() == 180

    with qtbot.waitSignal(left.openChanged) as closed:  # dragging to zero closes
        drag_handle(qtbot, splitter, 1, 0)
    assert closed.args == [False]
    assert splitter.sizes()[0] == 0
    assert not left.content().isVisible()

    left.set_open(True, animate=False)  # reopening restores the last open size
    assert splitter.sizes()[0] == 180

    left.set_open(False, animate=False)
    with qtbot.waitSignal(left.openChanged) as reopened:  # dragging out of zero opens
        drag_handle(qtbot, splitter, 1, 150)
    assert reopened.args == [True]
    assert splitter.sizes()[0] == 150
    assert left.open_size() == 150
    assert left.content().isVisible()


def test_slide_panel_follows_set_sizes(qtbot):
    splitter, left, right = make_splitter(qtbot)
    shown(qtbot, splitter)
    total = sum(splitter.sizes())
    with qtbot.waitSignal(left.openChanged) as closed:
        splitter.setSizes([0, total - 200, 200])
    assert closed.args == [False]
    assert not left.is_open()
    assert not left.content().isVisible()
    left.toggle(animate=False)  # the first toggle opens it again
    assert left.is_open()
    assert splitter.sizes()[0] == 250

    with qtbot.waitSignal(right.openChanged):
        splitter.setSizes([250, total - 250, 0])
    assert not right.is_open()
    with qtbot.waitSignal(right.openChanged) as opened:
        splitter.setSizes([250, total - 420, 170])
    assert opened.args == [True]
    assert right.open_size() == 170
    assert right.content().isVisible()

    right.hide()  # a hidden pane gets no space, but it is not closed
    right.show()
    qtbot.waitUntil(lambda: splitter.sizes()[2] == 170)
    assert right.is_open()


def saved_states(qtbot) -> tuple:
    """``saveState()`` with the left panel closed, and with it open at 180 px."""
    splitter, left, _right = make_splitter(qtbot)
    shown(qtbot, splitter)
    left.set_open(False, animate=False)
    closed = splitter.saveState()
    sizes = splitter.sizes()
    splitter.setSizes([180, sizes[1] - 180, sizes[2]])
    assert left.is_open()
    opened = splitter.saveState()
    splitter.hide()
    return closed, opened


def test_slide_panel_follows_restore_state(qtbot):
    closed, opened = saved_states(qtbot)
    splitter, left, _right = make_splitter(qtbot)
    shown(qtbot, splitter)
    with qtbot.waitSignal(left.openChanged) as changed:
        assert splitter.restoreState(closed)
    assert changed.args == [False]
    assert splitter.sizes()[0] == 0
    assert not left.content().isVisible()
    with qtbot.waitSignal(left.openChanged) as changed:
        assert splitter.restoreState(opened)
    assert changed.args == [True]
    assert splitter.sizes()[0] == 180
    assert left.open_size() == 180
    assert left.content().isVisible()


def test_slide_panel_restore_state_at_launch(qtbot):
    closed, opened = saved_states(qtbot)
    splitter, left, _right = make_splitter(qtbot)
    assert splitter.restoreState(closed)  # before show, as at launch
    shown(qtbot, splitter)
    assert splitter.sizes()[0] == 0
    assert not left.is_open()
    assert not left.content().edit.isVisible()  # out of the Tab order
    left.toggle(animate=False)  # the first toggle opens it
    assert left.is_open()
    assert splitter.sizes()[0] == 250

    # restoreState after a set_open made before show wins: it came later
    splitter, left, _right = make_splitter(qtbot)
    left.set_open(False, animate=False)
    assert splitter.restoreState(opened)
    shown(qtbot, splitter)
    qtbot.waitUntil(left.is_open)
    assert splitter.sizes()[0] == 180
    assert left.content().isVisible()


def test_slide_panel_closed_before_it_is_added(qtbot):
    splitter = QSplitter()  # no stretch factor: the closed panel still gets no space
    qtbot.addWidget(splitter)
    panel = SlidePanel(Content("early"), 220)
    panel.set_open(False, animate=False)
    splitter.addWidget(panel)
    splitter.addWidget(QLabel("plot"))
    splitter.resize(1000, 600)
    shown(qtbot, splitter)
    assert splitter.sizes()[0] == 0
    assert not panel.is_open()
    assert not panel.content().isVisible()

    late = SlidePanel(Content("late"), 160)
    late.set_open(False, animate=False)
    splitter.addWidget(late)  # into a splitter that is already shown
    assert splitter.sizes()[2] == 0
    assert not late.is_open()
    panel.toggle(animate=False)
    assert splitter.sizes()[0] == 220
    assert panel.content().isVisible()


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


def test_slide_panel_at_start_behind_a_rail(qtbot):
    splitter = QSplitter()
    qtbot.addWidget(splitter)
    panel = SlidePanel(Content("left"), 200, at_start=True)
    for widget in (QLabel("rail"), panel, QLabel("plot")):
        splitter.addWidget(widget)
    splitter.setStretchFactor(2, 1)
    splitter.resize(1000, 600)
    shown(qtbot, splitter)
    panel.duration_ms = 1500
    panel.set_open(False)
    qtbot.waitUntil(lambda: 20 < splitter.sizes()[1] < 180)
    content = panel.content().geometry()
    assert content.width() == 200
    assert content.right() == panel.width() - 1  # slides out towards the window's start


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


def test_collapsible_section_fits_wrapped_text(qtbot):
    section = CollapsibleSection("Colour")
    label = QLabel("Remembered for R(B)/R(0). Drag here or on the colour scale. " * 3)
    label.setWordWrap(True)
    section.body_layout().addWidget(label)
    host = QWidget()
    layout = QVBoxLayout(host)
    layout.addWidget(section)
    layout.addStretch(1)
    qtbot.addWidget(host)
    host.resize(220, 600)
    host.show()
    qtbot.waitExposed(host)
    needed = label.heightForWidth(label.width())
    assert needed > label.sizeHint().height()  # narrower than the label would like
    assert label.visibleRegion().boundingRect().height() == needed  # nothing is clipped


def test_collapsible_section_settings_protocol(qtbot):
    section = CollapsibleSection("View", expanded=False)
    qtbot.addWidget(section)
    assert section.settings_value() is False
    assert section.set_settings_value("true")
    assert section.is_expanded()
    assert section.set_settings_value(False)
    assert not section.is_expanded()
    assert section.set_settings_value("maybe") is False


# --- Switch, InfoBar -----------------------------------------------------------------------
def test_switch_behaves_like_a_checkbox(qtbot):
    switch = Switch("Smooth the reference")
    qtbot.addWidget(switch)
    assert isinstance(switch, QCheckBox)
    assert switch.sizeHint().width() > 30 + switch.fontMetrics().horizontalAdvance("Smooth")
    focused(qtbot, switch)
    with qtbot.waitSignal(switch.toggled) as toggled:
        qtbot.keyClick(switch, Qt.Key.Key_Space)
    assert toggled.args == [True]
    text_pos = QPoint(switch.width() - 5, switch.height() // 2)  # the label toggles too
    qtbot.mouseClick(switch, Qt.MouseButton.LeftButton, pos=text_pos)
    assert not switch.isChecked()
    switch.grab()  # paints without errors


def test_infobar_action_and_close(qtbot):
    bar = InfoBar()
    qtbot.addWidget(bar)
    assert bar.isHidden()
    calls = []
    bar.show_message(
        "error", "Can't process", "No reference files.", "Open Reference", lambda: calls.append(1)
    )
    assert bar.isVisible()
    assert bar.level() == "error"
    assert bar.property("level") == "error"
    assert bar.title_label.text() == "Can't process"
    assert bar.action_button.isVisible()
    assert bar.accessibleName() == "Can't process"
    with qtbot.waitSignal(bar.closed):
        bar.action_button.click()
    assert calls == [1]
    assert bar.isHidden()

    bar.show_message("info", "Saved.")
    assert bar.property("level") == "info"
    assert not bar.action_button.isVisible()
    assert not bar.text_label.isVisible()
    with qtbot.waitSignal(bar.closed):
        bar.close_button.click()
    assert bar.isHidden()
    with qtbot.assertNotEmitted(bar.closed):
        bar.dismiss()  # already hidden
    bar.show_message("warning", "Uneven steps")
    with qtbot.waitSignal(bar.closed):
        qtbot.keyClick(bar, Qt.Key.Key_Escape)
    with pytest.raises(ValueError):
        bar.show_message("fatal", "?")


def test_infobar_tells_closing_by_the_user_from_closing_by_the_program(qtbot):
    bar = InfoBar()
    qtbot.addWidget(bar)
    bar.show_message("info", "Version 9.0.0 is available", actions=[("Skip", lambda: None)])
    with qtbot.assertNotEmitted(bar.closedByUser), qtbot.waitSignal(bar.closed):
        bar.dismiss()  # a new result, Process
    for close in (
        bar.close_button.click,
        lambda: qtbot.keyClick(bar, Qt.Key.Key_Escape),
        lambda: bar.row_buttons()[0].click(),
        bar.dismiss_by_user,  # a window-wide Escape
    ):
        bar.show_message("info", "Version 9.0.0 is available", actions=[("Skip", lambda: None)])
        with qtbot.waitSignals([bar.closed, bar.closedByUser], order="strict"):
            close()
    with qtbot.assertNotEmitted(bar.closedByUser):
        bar.dismiss_by_user()  # already hidden


def test_infobar_shows_several_actions_in_a_row(qtbot):
    bar = InfoBar()
    qtbot.addWidget(bar)
    calls = []
    actions = [(text, lambda t=text: calls.append(t)) for text in ("Download", "Skip")]
    bar.show_message("info", "Version 9.0.0 is available", "Notes.", actions=actions)
    assert not bar.action_button.isVisible() and bar.action_row.isVisible()
    assert [b.text() for b in bar.row_buttons()] == ["Download", "Skip"]
    with qtbot.waitSignal(bar.closed):
        bar.row_buttons()[1].click()
    assert calls == ["Skip"] and bar.isHidden()

    def again() -> None:  # an action that shows the next message (and its buttons)
        calls.append("again")
        bar.show_message("info", "Next", actions=[("Download", lambda: None)])

    bar.show_message("info", "Version 9.0.0 is available", actions=[("Retry", again)])
    bar.row_buttons()[0].click()
    assert calls == ["Skip", "again"] and bar.isVisible()
    assert [b.text() for b in bar.row_buttons()] == ["Download"]
    bar.show_message("error", "Can't process", "No files.", "Open Sample", lambda: None)
    assert bar.row_buttons() == [] and not bar.action_row.isVisible()
    assert bar.action_button.isVisible()


def test_infobar_icon_is_crisp_at_any_pixel_ratio(qtbot):
    bar = InfoBar()
    qtbot.addWidget(bar)
    bar.show_message("warning", "Uneven steps")
    view = bar.icon_view  # paints a QIcon, which picks the pixmap for the screen's ratio
    assert view.icon().cacheKey() == icons.icon("triangle-alert", "warn").cacheKey()
    hidpi = view.icon().pixmap(QSize(16, 16), 2.0)
    assert hidpi.devicePixelRatio() == 2.0
    assert hidpi.toImage() == icons.pixmap("triangle-alert", 16, "warn", 2.0).toImage()
    shot = view.grab().toImage()
    assert len({shot.pixel(x, y) for x in range(16) for y in range(16)}) > 1  # painted


# --- settings protocol through Persistence -------------------------------------------------
class Kit(QWidget):
    """One of each stateful kit widget, as a window would bind them."""

    def __init__(self):
        super().__init__()
        self.segmented = SegmentedControl()
        for value in ("cm-1", "meV", "THz"):
            self.segmented.add_option(value, value)
        self.range = RangeControl("Energy", "cm⁻¹")
        self.range.set_extent(100, 4000)
        self.range.set_range(100, 4000)
        self.section = CollapsibleSection("View")
        self.switch = Switch("Energy window")
        self.splitter = QSplitter()
        self.panel = SlidePanel(Content("panel"), 250)
        self.splitter.addWidget(self.panel)
        self.splitter.addWidget(QLabel("plot"))
        layout = QVBoxLayout(self)
        for widget in (self.segmented, self.range, self.section, self.switch, self.splitter):
            layout.addWidget(widget)

    def bind(self, persistence: Persistence) -> None:
        persistence.bind("unit", self.segmented)
        persistence.bind("view/energy", self.range)
        persistence.bind("inspector/view", self.section)
        persistence.bind("processing/window", self.switch)
        persistence.bind("layout/left", self.panel)


def ini_settings(tmp_path) -> QSettings:
    return QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)


def test_settings_protocol_round_trip(qtbot, tmp_path):
    first = Kit()
    qtbot.addWidget(first)
    persistence = Persistence(ini_settings(tmp_path))
    first.bind(persistence)
    first.segmented.set_value("meV")
    first.range.set_auto(False)
    first.range.set_range(400.5, 2200.25)
    first.section.set_expanded(False, animate=False)
    first.switch.setChecked(True)
    first.panel.set_open(False, animate=False)
    persistence.save()

    second = Kit()
    qtbot.addWidget(second)
    restored = Persistence(ini_settings(tmp_path))
    second.bind(restored)
    restored.restore()
    assert second.segmented.value() == "meV"
    assert not second.range.is_auto()
    assert second.range.range() == (400.5, 2200.25)
    assert not second.section.is_expanded()
    assert second.switch.isChecked()
    assert not second.panel.is_open()
    assert second.panel.open_size() == 250

    restored.reset()  # back to the defaults read at bind time
    assert second.segmented.value() == "cm-1"
    assert second.range.is_auto()
    assert second.section.is_expanded()
    assert second.panel.is_open()


def test_settings_protocol_ignores_invalid_values(qtbot, tmp_path):
    raw = ini_settings(tmp_path)
    raw.setValue(f"{PREFIX}/unit", "eV")
    raw.setValue(f"{PREFIX}/view/energy", '{"auto": false, "lo": 9, "hi": 1}')
    raw.setValue(f"{PREFIX}/inspector/view", "sometimes")
    raw.setValue(f"{PREFIX}/layout/left", "[1, 2]")
    raw.sync()
    kit = Kit()
    qtbot.addWidget(kit)
    persistence = Persistence(ini_settings(tmp_path))
    kit.bind(persistence)
    persistence.restore()
    assert kit.segmented.value() == "cm-1"
    assert kit.range.is_auto()
    assert kit.range.range() == (100.0, 4000.0)
    assert kit.section.is_expanded()
    assert kit.panel.is_open()


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


def test_gallery_shows_every_widget(qtbot, tmp_path):
    from mag_opt_detective.gui.kit.gallery import Gallery
    from mag_opt_detective.gui.theme import Theme

    gallery = Gallery(Theme("light"))
    qtbot.addWidget(gallery)
    gallery.resize(1100, 800)
    gallery.show()
    qtbot.waitExposed(gallery)
    kinds = (CollapsibleSection, InfoBar, RangeControl, RangeSlider, SegmentedControl)
    for kind in (*kinds, SlidePanel, Switch):
        assert gallery.findChildren(kind), kind.__name__
    gallery.log.set_open(False, animate=False)
    assert gallery.stage.sizes()[1] == 0
    assert gallery.grab().save(str(tmp_path / "gallery.png"))


def test_gallery_shows_the_nudge_slider(qtbot):
    from mag_opt_detective.gui.kit.gallery import Gallery
    from mag_opt_detective.gui.theme import Theme

    gallery = Gallery(Theme("light"))
    qtbot.addWidget(gallery)
    modes = sorted(slider.mode() for slider in gallery.findChildren(NudgeSlider))
    assert modes == ["range", "relative"]
