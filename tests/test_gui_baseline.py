"""The baseline region on the plots and the Live baseline (P4-08)."""

import pyqtgraph as pg
import pytest
from PySide6.QtGui import QColor

import golden
from mag_opt_detective.gui.controller import AppController, SweepFiles
from mag_opt_detective.gui.plots.baseline_region import (
    VERTICAL,
    BaselineRegion,
    data_style,
    theme_style,
)


@pytest.fixture
def lines(tmp_path) -> SweepFiles:
    """A sweep whose ratios change along energy (a line moving on a sloped background), so
    every baseline region gives other maps; energies 400 - 1200 cm-1."""
    zero, field = golden.write_sweep(tmp_path)
    return SweepFiles(tuple(map(str, zero)), tuple(map(str, field)))


# ---------------------------------------------------------------------- the region
def test_the_region_item_keeps_its_limits_and_reports_only_drags(qtbot):
    plot = pg.PlotWidget()
    qtbot.addWidget(plot)
    region = BaselineRegion(VERTICAL, data_style())
    plot.addItem(region, ignoreBounds=True)
    edits, ends = [], []
    region.edited.connect(lambda lo, hi: edits.append((lo, hi)))
    region.editFinished.connect(lambda lo, hi: ends.append((lo, hi)))
    region.set_limits(0.0, 100.0, 5.0)
    region.set_region(10, 20)
    assert region.region() == (10, 20) and not edits  # typed: nothing reported
    region.set_region(-50, 300)
    assert region.region() == (0, 100) and not edits
    region.setRegion((30, 40))  # as a drag reports it
    assert edits == [(30, 40)] and ends == [(30, 40)]
    region.lines[0].setValue(80)  # an edge stops short of the other ...
    assert region.region() == (35, 40) and edits[-1] == (35, 40)
    region.lines[1].setValue(200)  # ... and at the limit
    assert region.region() == (35, 100)
    region.set_limits(None, 60.0, 1.0)
    assert region.region() == (35, 60)
    region.lines[0].setValue(-1e6)
    assert region.region() == (-1e6, 60)
    assert not region.is_dragging() and region.step() == 0  # not on screen: no rounding
    region.set_style(theme_style(QColor("#7a2a8c"), QColor("#ffffff")))
    assert region.lines[0].pen.color().name() == "#7a2a8c"


# ---------------------------------------------------------------------- controller
def test_apply_baseline_needs_a_processed_map(qapp, lines):
    c = AppController()
    c.set_processing(baseline=(450.0, 550.0))
    with pytest.raises(ValueError, match="no map is processed"):
        c.apply_baseline()
    c.set_processing(sample_files=lines)
    c.process()
    assert not c.apply_baseline()  # nothing changed
    c.set_processing(baseline=(600.0, 700.0))
    assert c.changed_since_process()
    c.set_live_baseline(True)
    assert not c.changed_since_process()  # counts as applied ...
    assert c.result.baseline_region == (450.0, 550.0)  # ... once apply_baseline runs
    assert c.apply_baseline() and c.result.baseline_region == (600.0, 700.0)
    c.set_live_baseline(False)
    assert not c.changed_since_process()
    c.set_processing(baseline=(600.0, None))
    with pytest.raises(ValueError, match="enter both limits"):
        c.apply_baseline()
    c.set_unit("meV")
    c.set_processing(baseline=(5000.0, 6000.0))
    with pytest.raises(ValueError, match=r"619\.917 – 743\.9 meV contains no data"):
        c.apply_baseline()
