"""The folder watcher's parts that need no window (gui/watch.py)."""

import numpy as np

from helpers import sweep_name, write_text
from mag_opt_detective.gui import watch
from mag_opt_detective.gui.controller import SweepFiles
from mag_opt_detective.gui.watch import Arrivals, folder_listing, path_key, sweep_folder

X = np.linspace(100.0, 1000.0, 11)


def test_a_file_is_complete_when_it_stopped_changing():
    arrivals = Arrivals(settle=2.0)
    arrivals.observe({"a": (10, 1)}, now=0.0)
    assert arrivals.complete() == [] and arrivals.waiting()
    arrivals.observe({"a": (10, 1)}, now=1.5)
    assert arrivals.complete() == []
    arrivals.observe({"a": (20, 2)}, now=2.5)  # still written: starts again
    arrivals.observe({"a": (20, 2)}, now=4.0)
    assert arrivals.complete() == []
    arrivals.observe({"a": (20, 2), "empty": (0, 3)}, now=4.5)
    assert arrivals.complete() == ["a"]
    assert not arrivals.waiting()  # an empty file is not waited for
    arrivals.observe({"empty": (0, 3)}, now=9.0)  # "a" is gone
    assert arrivals.complete() == []


def test_files_changed_long_ago_are_complete_at_once():
    arrivals = Arrivals(settle=2.0)
    wall = 1_000.0
    arrivals.observe({"old": (5, int(990e9)), "new": (5, int(999.5e9))}, now=0.0, wall=wall)
    assert arrivals.complete() == ["old"]


def test_a_file_that_keeps_failing_is_left_out_until_it_changes():
    arrivals = Arrivals(settle=0.0)
    arrivals.observe({"a": (5, 1)}, now=0.0)
    arrivals.observe({"a": (5, 1)}, now=0.0)
    assert arrivals.complete() == ["a"]
    assert [arrivals.failed("a") for _ in range(watch.MAX_FAILURES)] == [False, False, True]
    assert arrivals.complete() == [] and not arrivals.waiting()
    arrivals.observe({"a": (6, 2)}, now=1.0)  # changed: a new chance
    arrivals.observe({"a": (6, 2)}, now=1.0)
    assert arrivals.complete() == ["a"]


def test_folder_listing_and_the_folder_of_a_sweep(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / ".hidden").write_text("x")
    a = str(write_text(tmp_path / "s_a00p000T_a00p000T.txt", X, X))
    b = str(write_text(tmp_path / sweep_name(1.0), X, X))
    assert set(folder_listing(tmp_path)) == {a, b}
    other = tmp_path / "sub" / "z_a00p000T.txt"
    assert sweep_folder(SweepFiles((a,), (b,))) == str(tmp_path)
    assert sweep_folder(SweepFiles((str(other),), (b,))) == str(tmp_path)  # in-field decide
    assert sweep_folder(SweepFiles()) is None
    assert path_key(f"{tmp_path}/./x.txt") == path_key(tmp_path / "x.txt")
