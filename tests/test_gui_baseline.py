"""The baseline region on the plots and the Live baseline (P4-08)."""

import pytest

import golden
from mag_opt_detective.gui.controller import AppController, SweepFiles


@pytest.fixture
def lines(tmp_path) -> SweepFiles:
    """A sweep whose ratios change along energy (a line moving on a sloped background), so
    every baseline region gives other maps; energies 400 - 1200 cm-1."""
    zero, field = golden.write_sweep(tmp_path)
    return SweepFiles(tuple(map(str, zero)), tuple(map(str, field)))


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
