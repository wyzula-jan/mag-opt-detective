"""The baseline chip in the status bar: the baseline of the map on screen (P4-15)."""

import math

import pyqtgraph as pg
import pytest
from PySide6.QtCore import QEvent, QObject, QPoint, QPointF, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QLabel

import golden
import gui_helpers
from gui_helpers import process, set_unit
from mag_opt_detective.core.units import Unit, from_cm1
from mag_opt_detective.gui.baseline_chip import BaselineChip, BaselineMark, baseline_mark
from mag_opt_detective.gui.controller import SweepFiles
from mag_opt_detective.gui.display import process_key, unit_text
from mag_opt_detective.gui.panels import library
from mag_opt_detective.gui.panels.processing import LiveApply, show_baseline

window, errors = gui_helpers.window, gui_helpers.errors  # shared fixtures

LEFT, PLAIN = Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier


@pytest.fixture
def lines(tmp_path) -> SweepFiles:
    """The golden sweep (energies 400 - 1200 cm-1)."""
    zero, field = golden.write_sweep(tmp_path)
    return SweepFiles(tuple(map(str, zero)), tuple(map(str, field)))


@pytest.fixture
def processed(window, lines):
    """The sweep processed with the baseline region 450 - 550 cm-1 (Live off)."""
    c = window.controller
    c.set_processing(sample_files=lines, baseline=(450.0, 550.0))
    process(window)
    assert c.result.baseline_region == (450.0, 550.0)
    return window


def chip(window) -> BaselineChip:
    return window.baseline_chip


def mark(window) -> BaselineMark | None:
    """What the chip shows, checked against the controller (up to date) and against what it
    says to people."""
    shown = chip(window).mark()
    assert shown == baseline_mark(window.controller)
    assert chip(window).isVisibleTo(window) == (shown is not None)
    if shown is not None:
        assert chip(window).text() == shown.text()
        assert chip(window).accessibleName() == shown.accessible_name()
        assert chip(window).toolTip() == shown.tooltip()
    return shown


def region_text(lo: float, hi: float, unit: Unit) -> str:
    """The region *lo* - *hi* (cm^-1) as the Processing panel's fields write it in *unit*."""
    a, b = (float(from_cm1(v, unit)) for v in (lo, hi))
    return f"{a:.6g} – {b:.6g} {unit_text(unit)}"


def type_region(window, lo: float | None, hi: float | None) -> None:
    """Type the baseline region (display unit) into the Processing panel's fields."""
    panel = window.panels["processing"]
    for edit, value in ((panel.baseline_lo, lo), (panel.baseline_hi, hi)):
        edit.setText("" if value is None else f"{value:g}")


# ---------------------------------------------------------------------- what it shows
def test_the_chip_is_in_the_status_bar_and_hidden_before_a_map(window):
    assert chip(window).parentWidget() is window.statusBar()
    assert mark(window) is None and chip(window).isHidden()  # nothing processed yet


def test_no_baseline_while_the_panel_has_one_to_apply(window, lines):
    w, c = window, window.controller
    panel = w.panels["processing"]
    c.set_processing(sample_files=lines)
    process(w)
    assert c.result is not None and mark(w) is None  # no baseline, none set
    panel.baseline_on.setChecked(True)  # starts a region (empty fields), not applied yet
    lo, hi = c.processing.baseline
    shown = mark(w)
    assert shown.text() == "No baseline" and shown.region is None and shown.pending
    assert shown.tooltip().splitlines()[:2] == [
        "The map shown has no baseline correction.",
        f"The Processing panel has {region_text(lo, hi, Unit.CM1)}. "
        f"Process ({process_key()}) to apply it.",
    ]
    assert shown.accessible_name() == "No baseline, changed, process again to apply"
    assert chip(w).form() is not None or not chip(w).isVisible()
    type_region(w, 600, None)  # incomplete: nothing the next Process would apply
    assert mark(w) is None
    type_region(w, 700, 600)  # reversed: the same
    assert mark(w) is None
    type_region(w, 1300, 1400)  # outside the data: the same
    assert mark(w) is None
    type_region(w, 600, 700)
    assert mark(w).text() == "No baseline"
    set_unit(w, Unit.MEV)
    assert region_text(600, 700, Unit.MEV) in mark(w).tooltip()
    process(w)
    assert mark(w).text() == f"Baseline {region_text(600, 700, Unit.MEV)}"
    assert not mark(w).pending


def test_no_baseline_never_shows_with_live_or_a_library_map(window, lines):
    w, c = window, window.controller
    panel = w.panels["processing"]
    c.set_processing(sample_files=lines)
    process(w)
    library.add_processed(w)
    panel.baseline_live.setChecked(True)
    type_region(w, 600, 700)
    panel.baseline_on.setChecked(True)  # applied at once
    assert mark(w).text() == "Baseline 600 – 700 cm⁻¹" and mark(w).live
    panel.baseline_on.setChecked(False)
    panel.baseline_live.setChecked(False)
    library.plot_entry(w, c.library[0].key)  # plotted without a baseline
    panel.baseline_on.setChecked(True)
    assert c.result_source == "library" and mark(w) is None


def test_the_applied_region_in_the_display_unit(processed):
    w = processed
    shown = mark(w)
    assert shown.text() == "Baseline 450 – 550 cm⁻¹"
    assert not (shown.live or shown.pending or shown.library)
    assert "baseline-corrected over 450 – 550 cm⁻¹" in shown.tooltip()
    assert "Processing panel" in shown.tooltip()
    for unit in (Unit.MEV, Unit.THZ, Unit.CM1):
        set_unit(w, unit)
        assert mark(w).text() == f"Baseline {region_text(450, 550, unit)}"
        assert not mark(w).pending  # a unit switch changes no baseline
    w.controller.set_unit(Unit.MEV)  # through the controller: the chip follows too
    assert mark(w).text() == f"Baseline {region_text(450, 550, Unit.MEV)}"


def test_the_numbers_are_the_fields_numbers(window, lines):
    w, c = window, window.controller
    panel = w.panels["processing"]
    c.set_processing(sample_files=lines, baseline=(450.123456, 550.98765))
    process(w)
    for unit in (Unit.CM1, Unit.MEV, Unit.THZ):
        set_unit(w, unit)
        fields = f"{panel.baseline_lo.text()} – {panel.baseline_hi.text()} {unit_text(unit)}"
        assert mark(w).text() == f"Baseline {fields}"
    wide = BaselineMark((10000.5, 12000.25), Unit.CM1)
    assert wide.text() == f"Baseline {10000.5:.6g} – {12000.25:.6g} cm⁻¹"
    assert wide.text().startswith("Baseline 10000.5 – 12000.")


def test_a_pending_difference_always_shows(processed):
    w, c = processed, processed.controller
    c.set_processing(baseline=(450.0000001, 550.0))
    shown = mark(w)
    assert shown.pending
    wanted = shown.tooltip().split("The Processing panel has ")[1].split(". Process")[0]
    assert wanted != shown.range_text()
    assert shown.range_text() == "450 – 550 cm⁻¹" and wanted == "450.0000001 – 550 cm⁻¹"
    c.set_processing(baseline=(600.0, 700.0))
    assert mark(w).digits == 6


def test_a_changed_region_is_pending_until_processed(processed):
    w, c = processed, processed.controller
    type_region(w, 600, 700)
    shown = mark(w)
    assert shown.pending and shown.text() == "Baseline 450 – 550 cm⁻¹"  # still what is applied
    assert shown.tooltip().splitlines()[1] == (
        f"The Processing panel has 600 – 700 cm⁻¹. Process ({process_key()}) to apply it."
    )
    assert "changed" in shown.accessible_name()
    set_unit(w, Unit.MEV)
    assert region_text(600, 700, Unit.MEV) in mark(w).tooltip()
    set_unit(w, Unit.CM1)
    type_region(w, 450, 550)  # back to the applied one
    assert not mark(w).pending
    type_region(w, 600, 700)
    process(w)
    assert mark(w).text() == "Baseline 600 – 700 cm⁻¹" and not mark(w).pending
    assert c.result.baseline_region == (600.0, 700.0)


@pytest.mark.parametrize(
    ("change", "says"),
    [
        ("off", f"has the baseline off. Process ({process_key()}) to remove it."),
        ("reversed", "cannot be applied: 700 – 600 cm⁻¹ is reversed"),
        ("incomplete", "cannot be applied: it is incomplete; enter both limits."),
        ("outside", "cannot be applied: 1300 – 1400 cm⁻¹ holds no data (the data span 400 –"),
    ],
)
def test_pending_says_what_the_panel_has(processed, change, says):
    w = processed
    panel = w.panels["processing"]
    if change == "off":
        panel.baseline_on.setChecked(False)
    elif change == "reversed":
        type_region(w, 700, 600)
    elif change == "incomplete":
        type_region(w, 600, None)
    else:
        type_region(w, 1300, 1400)
    shown = mark(w)
    assert shown.pending and shown.text() == "Baseline 450 – 550 cm⁻¹" and says in shown.tooltip()
    assert (change == "off") == ("Process (" in shown.tooltip())  # only what Process can apply
    assert shown.accessible_name().endswith("process again to apply") == (change == "off")
    if change == "off":
        process(w)
        assert mark(w) is None  # the new map has no baseline


APPLIES = "applies"


@pytest.mark.parametrize(
    ("last", "new", "region", "says"),
    [
        ((600.0, 1100.0), None, (450.0, 550.0), APPLIES),  # the whole sweep again
        ((600.0, 1100.0), (900.0, None), (850.0, 950.0), APPLIES),  # reaches into the window
        ((600.0, 1100.0), (900.0, 1200.0), (450.0, 550.0), "lies outside the energy window"),
        (None, (600.0, 1100.0), (450.0, 550.0), "holds no data (the data span 600 – 1100"),
        (None, (400.0, 1100.0), (450.0, 550.0), APPLIES),
        ((600.0, 1100.0), (600.0, 1100.0), (450.0, 550.0), "holds no data (the data span 600 –"),
    ],
)
@pytest.mark.parametrize("applied", [None, (700.0, 800.0)])
def test_pending_judges_the_data_the_next_process_loads(
    window, lines, last, new, region, says, applied
):
    w, c = window, window.controller
    c.set_processing(sample_files=lines, energy_cut=last, baseline=applied)
    process(w)
    c.set_processing(energy_cut=new, baseline=region)
    shown = mark(w)
    if says == APPLIES:  # also "No baseline" on a map without one
        assert shown.pending and not shown.problem
        assert shown.text() == ("No baseline" if applied is None else "Baseline 700 – 800 cm⁻¹")
        assert f"The Processing panel has {region_text(*region, Unit.CM1)}. Process" in (
            shown.tooltip()
        )
        process(w)
        assert c.result.baseline_region == region and not mark(w).pending
    elif applied is None:
        assert shown is None  # nothing the next Process would apply
    else:
        assert shown.pending and says in shown.problem and "Process (" not in shown.tooltip()


def test_another_sweep_is_not_judged_by_the_map_shown(window, lines, tmp_path):
    w, c = window, window.controller
    c.set_processing(sample_files=lines, baseline=(700.0, 800.0))
    process(w)
    other = SweepFiles(lines.zero, lines.field[:-1])
    c.set_processing(sample_files=other, baseline=(1300.0, 1400.0))
    assert mark(w).pending and not mark(w).problem  # (Process says it, if it must)


def test_live_applies_at_once_and_is_never_pending(processed):
    w, c = processed, processed.controller
    panel = w.panels["processing"]
    panel.baseline_live.setChecked(True)
    shown = mark(w)
    assert shown.live and not shown.pending and "live" in shown.accessible_name()
    assert "Live" in shown.tooltip()
    states = []  # the chip after every change (connected after the chip's own update)
    for signal in (c.processingChanged, c.resultChanged):
        signal.connect(lambda: states.append(chip(w).mark()))
    c.set_processing(baseline=(600.0, 700.0))  # (both ends at once, as a drag does)
    panel.live.flush()
    assert states and not any(s.pending for s in states if s is not None)
    assert mark(w).text() == "Baseline 600 – 700 cm⁻¹" and mark(w).live
    type_region(w, 700, 600)  # cannot be applied: it waits, and says so
    assert mark(w).pending and mark(w).live and mark(w).text() == "Baseline 600 – 700 cm⁻¹"
    panel.baseline_on.setChecked(False)  # applied at once: no baseline on the map
    assert mark(w) is None
    panel.baseline_live.setChecked(False)
    panel.baseline_on.setChecked(True)
    assert mark(w) is None  # (700 - 600 is reversed: nothing applied)
    type_region(w, 450, 550)
    process(w)
    assert not mark(w).live and not mark(w).pending


def test_library_maps_show_the_region_they_were_plotted_with(processed, errors):
    w, c = processed, processed.controller
    panel = w.panels["processing"]
    library.add_processed(w)
    type_region(w, 600, 700)
    library.plot_entry(w, c.library[0].key)  # plotted with the panel's region
    assert c.result_source == "library" and not errors
    shown = mark(w)
    assert shown.library and shown.text() == "Baseline 600 – 700 cm⁻¹"
    assert not (shown.live or shown.pending) and "library map" in shown.tooltip()
    panel.baseline_live.setChecked(True)  # Live leaves library maps alone
    type_region(w, 800, 900)
    panel.live.flush()
    assert mark(w) == shown
    set_unit(w, Unit.MEV)
    assert mark(w).text() == f"Baseline {region_text(600, 700, Unit.MEV)}"
    panel.baseline_on.setChecked(False)
    library.plot_entry(w, c.library[0].key)  # plotted without a baseline
    assert mark(w) is None and not errors


def test_restored_settings_show_after_the_restore(processed, qtbot):
    w, c = processed, processed.controller
    with c.restoring():
        c.set_processing(baseline=(600.0, 700.0))
        assert mark(w).pending  # (Live off: waits for Process)
    c.restored.emit()
    assert mark(w).pending and mark(w).text() == "Baseline 450 – 550 cm⁻¹"


# ---------------------------------------------------------------------- the widget
def test_the_chip_gets_narrower_and_never_widens_the_window(processed, qtbot):
    w = processed
    w.panels["processing"].baseline_live.setChecked(True)  # the widest chip
    type_region(w, 450, 550)
    w.resize(1100, 800)
    w.side_panel.set_open(True, animate=False)
    w.inspector_panel.set_open(True, animate=False)
    w.show()
    qtbot.waitExposed(w)
    assert chip(w).isVisible() and chip(w).form() == 0  # all of it fits at 1100 px
    labels = w.statusBar().findChildren(QLabel)
    state = next(label for label in labels if label.text() == w.state_text())
    summary = next(label for label in labels if label.text() == w.summary_text())
    left, right = chip(w).geometry().left(), chip(w).geometry().right()
    assert state.geometry().right() < left and right < summary.geometry().left()
    assert w.minimumSizeHint().width() <= 1100
    shown = w.minimumSizeHint().width()
    chip(w).hide()
    assert w.minimumSizeHint().width() == shown  # it adds nothing to the minimum
    chip(w).show()
    widths = chip(w).widths(mark(w))  # (the three forms)
    forms = []
    for width in (widths[0], widths[0] - 1, widths[1], widths[2], widths[2] - 1, 0):
        chip(w).resize(width + chip(w).MARGIN, chip(w).height())
        forms.append(chip(w).form())
    assert forms == [0, 1, 1, 2, None, None]
    assert chip(w).minimumSizeHint().width() == 0


def test_the_chip_keeps_its_width_while_held(qtbot):
    widget = BaselineChip()
    qtbot.addWidget(widget)
    widget.set_mark(BaselineMark((612.0, 992.0), Unit.CM1))
    narrow = widget.sizeHint().width()
    widget.set_held(True)
    wide = widget.sizeHint().width()
    assert wide > narrow  # room for one more digit at each end
    for region in ((707.0, 1087.0), (1000.0, 9999.0), (1.0, 2.0)):
        widget.set_mark(BaselineMark(region, Unit.CM1))
        assert widget.sizeHint().width() == wide
    widget.set_mark(BaselineMark((999.5, 999.5), Unit.CM1))  # (a decimal not reserved)
    assert widget.sizeHint().width() > wide
    widget.set_held(False)
    widget.set_mark(BaselineMark((612.0, 992.0), Unit.CM1))
    widget.set_held(True, decimals=2)  # (a drag's values have two decimals)
    assert widget.sizeHint().width() > wide
    two = widget.sizeHint().width()
    widget.set_mark(BaselineMark((9999.75, 9999.25), Unit.CM1))
    assert widget.sizeHint().width() == two
    widget.set_held(False)
    widget.set_mark(BaselineMark((612.0, 992.0), Unit.CM1))
    widget.set_held(True)
    assert widget.sizeHint().width() == wide
    widget.set_mark(BaselineMark((99999.5, 99999.75), Unit.CM1))  # (more than one digit)
    wider = widget.sizeHint().width()
    assert wider > wide
    widget.set_mark(BaselineMark((612.0, 992.0), Unit.CM1))
    assert widget.sizeHint().width() == wider
    widget.set_held(False)
    assert widget.sizeHint().width() == narrow
    widget.set_mark(None)
    assert widget.isHidden()


class Paints(QObject):
    """The form a chip has whenever it is painted."""

    def __init__(self, chip_widget: BaselineChip):
        super().__init__(chip_widget)
        self.forms: list[int | None] = []
        self._chip = chip_widget
        chip_widget.installEventFilter(self)

    def eventFilter(self, watched, event) -> bool:
        if event.type() == QEvent.Type.Paint:
            self.forms.append(self._chip.form())
        return False


def test_a_wider_region_is_painted_at_its_new_width(processed, qtbot):
    w, c = processed, processed.controller
    w.resize(1400, 900)
    w.show()
    qtbot.waitExposed(w)
    w.panels["processing"].baseline_live.setChecked(True)
    paints = Paints(chip(w))
    for region in ((612.0, 992.0), (707.0, 1087.0), (707.25, 1087.75), (450.0, 550.0)):
        c.set_processing(baseline=region)  # applied at once (Live)
        w.panels["processing"].live.flush()
        assert c.result.baseline_region == region
        qtbot.waitUntil(lambda: bool(paints.forms), timeout=2000)
        qtbot.wait(20)
    assert paints.forms and set(paints.forms) == {0}


def move_pause_ms() -> int:
    """pyqtgraph drops a mouse move within 1/mouseRateLimit s of the last one it took."""
    rate = pg.getConfigOption("mouseRateLimit")
    return 2 * math.ceil(1000 / rate) if rate > 0 else 0


def viewport_pos(plot, x: float, y: float) -> QPoint:
    return plot.view.mapFromScene(plot.plot.vb.mapViewToScene(QPointF(x, y)))


def live_drag(w, qtbot, edges: list[float], pause: bool = False) -> list:
    """Drag the upper edge of the map's band through *edges* (cm^-1) with Live on; the chip's
    (mark, asked width) after every change while dragged. *pause*: wait until Live applied the
    last position before letting go (then the release changes nothing)."""
    c = w.controller
    seen = []

    def record() -> None:
        seen.append((chip(w).mark(), chip(w).sizeHint().width()))

    c.processingChanged.connect(record)
    c.resultChanged.connect(record)
    plot = w.plots.map
    viewport = plot.view.viewport()
    steps = [viewport_pos(plot, 1.2, edge) for edge in edges]
    QTest.mouseMove(viewport, steps[0])
    QTest.mousePress(viewport, LEFT, PLAIN, steps[0])
    for step in steps[1:]:
        qtbot.wait(move_pause_ms())
        QTest.mouseMove(viewport, step)
    live = w.panels["processing"].live
    if pause:
        qtbot.waitUntil(lambda: not live.is_pending() and not live.is_busy(), timeout=3000)
        assert c.result.baseline_region == c.processing.baseline
    during = list(seen)
    QTest.mouseRelease(viewport, LEFT, PLAIN, steps[-1])
    c.processingChanged.disconnect(record)
    c.resultChanged.disconnect(record)
    return during


@pytest.fixture
def dragging(processed, qtbot):
    """The processed window at 1400 x 900 with the Processing panel open and Live on."""
    w = processed
    w.resize(1400, 900)
    w.show_panel("processing")
    w.show()
    qtbot.waitExposed(w)
    w.panels["processing"].baseline_live.setChecked(True)
    return w


def test_a_live_drag_neither_flickers_nor_moves_the_status_bar(dragging, qtbot):
    w, c = dragging, dragging.controller
    paints = Paints(chip(w))
    edges = [550.0 + 20 * k for k in range(29)]  # 550 - 1110: across 999 -> 1000
    during = live_drag(w, qtbot, edges)
    texts = [m.text() for m, _ in during]
    assert len(set(texts)) > 1  # the applied region changed on the way
    assert any("– 9" in t for t in texts) and any("– 10" in t for t in texts)
    assert not any(m.pending for m, _ in during) and all(m.live for m, _ in during)
    widths = [width for _, width in during]
    assert len(set(widths)) == 1  # room for one more digit from the first change on
    assert paints.forms and set(paints.forms) == {0}  # never painted without "Baseline"
    assert mark(w).text() == f"Baseline {region_text(*c.result.baseline_region, Unit.CM1)}"
    assert c.result.baseline_region == c.processing.baseline
    assert chip(w).sizeHint().width() == chip(w).widths(mark(w))[0] + chip(w).MARGIN


def test_a_zoomed_drag_reserves_room_for_decimals(dragging, qtbot):
    w, c = dragging, dragging.controller
    c.set_view(energy_range=(530.0, 1080.0))  # the band's values get a decimal
    qtbot.wait(20)
    band = w.panels["processing"].regions["map"]
    assert band.step() == pytest.approx(0.1)
    plot = w.plots.map
    viewport = plot.view.viewport()

    def energy(pixel: QPoint) -> float:
        return plot.plot.vb.mapSceneToView(plot.view.mapToScene(pixel)).y()

    press = viewport_pos(plot, 1.2, 550.0)  # the upper edge
    # the first step lands on a whole number, so the drag starts without decimals
    first = next(
        press - QPoint(0, dy)
        for dy in range(3, 80)
        if (value := round(550.0 + energy(press - QPoint(0, dy)) - energy(press), 1))
        == round(value)
    )
    steps = [first - QPoint(0, 50 * k) for k in range(12)]  # up to 1000 and more
    paints = Paints(chip(w))
    seen = []

    def record() -> None:
        seen.append((chip(w).mark(), chip(w).sizeHint().width()))

    c.resultChanged.connect(record)
    QTest.mouseMove(viewport, press)
    QTest.mousePress(viewport, LEFT, PLAIN, press)
    for step in steps:
        qtbot.wait(move_pause_ms())
        QTest.mouseMove(viewport, step)
        if len(seen) == 1:
            assert "." not in seen[0][0].range_text()  # the first change: no decimals
    qtbot.wait(2 * LiveApply.INTERVAL)  # what is waiting applies while still dragged
    during = list(seen)
    QTest.mouseRelease(viewport, LEFT, PLAIN, steps[-1])
    c.resultChanged.disconnect(record)
    assert any("." in m.range_text() and m.region[1] > 1000 for m, _ in during)
    assert len({width for _, width in during}) == 1  # never wider while dragged
    # room for any value with a digit more than at the start and the drag's decimal (the
    # widest digit is font-dependent: the drag above need not reach it)
    widest = BaselineMark((8888.8, 8888.8), Unit.CM1, live=True)
    assert during[0][1] >= chip(w).widths(widest)[0] + chip(w).MARGIN
    assert paints.forms and set(paints.forms) == {0}
    assert chip(w).sizeHint().width() == chip(w).widths(mark(w))[0] + chip(w).MARGIN


def test_the_width_is_released_when_a_drag_ends_without_a_change(dragging, qtbot):
    w = dragging
    edges = [550.0 + 20 * k for k in range(8)]
    during = live_drag(w, qtbot, edges, pause=True)
    assert during and during[-1][1] > chip(w).widths(during[-1][0])[0] + chip(w).MARGIN
    assert chip(w).sizeHint().width() == chip(w).widths(mark(w))[0] + chip(w).MARGIN


# ---------------------------------------------------------------------- the click
@pytest.mark.parametrize("on", [True, False])
def test_a_click_opens_the_panel_at_the_baseline(processed, qtbot, on):
    w = processed
    panel = w.panels["processing"]
    if not on:
        panel.baseline_on.setChecked(False)  # pending: the map still has 450 - 550
    w.resize(1100, 600)
    w.show_panel("sample")
    w.side_panel.set_open(False, animate=False)
    w.show()
    qtbot.waitExposed(w)
    w.activateWindow()
    qtbot.waitUntil(w.isActiveWindow, timeout=2000)
    QTest.mouseClick(chip(w), LEFT)
    assert w.side_panel.is_open() and w.current_panel() == "processing"
    target = panel.baseline_lo if on else panel.baseline_on
    qtbot.waitUntil(target.hasFocus, timeout=2000)
    page = w.panel_pages["processing"]
    qtbot.waitUntil(lambda: page.scroll.viewport().width() > 0, timeout=2000)
    block = panel.baseline_block
    top = block.mapTo(page.scroll.viewport(), QPoint(0, 0)).y()
    assert 0 <= top < page.scroll.viewport().height()  # scrolled to the baseline


def test_show_baseline_from_another_panel(processed):
    w = processed
    w.show_panel("library")
    show_baseline(w)
    assert w.current_panel() == "processing" and w.side_panel.is_open()
