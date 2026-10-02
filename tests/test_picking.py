import subprocess
import sys

import numpy as np
import pytest

from mag_opt_detective.core import picking as pk
from mag_opt_detective.core.picking import Feature, Peak

STEP = 0.5  # energy step of the synthetic maps
ENERGY = np.arange(100.0, 200.0 + STEP / 2, STEP)


def lorentz(x, x0, g):
    return 1.0 / (1.0 + ((x - x0) / g) ** 2)


def dispersive(x, x0, g):
    """Minus the derivative of a Lorentzian (scaled): rising inflection at x0, falling at x0±g."""
    u = (x - x0) / g
    return 2.0 * u / (1.0 + u**2) ** 2


# ---- find_features ----------------------------------------------------------------------


@pytest.mark.parametrize("x0", [140.0, 140.13, 140.25, 140.4])
def test_max_is_refined_below_one_sample(x0):
    peaks = pk.find_features(ENERGY, lorentz(ENERGY, x0, 1.5), Feature.MAX)
    assert len(peaks) == 1
    peak = peaks[0]
    assert abs(peak.energy - x0) < STEP / 4
    assert peak.index == int(np.abs(ENERGY - peak.energy).argmin())
    assert peak.strength == pytest.approx(1.0, abs=0.05)


def test_min_finds_inverted_peaks():
    values = 1.0 - 0.4 * lorentz(ENERGY, 150.2, 2.0) - 0.3 * lorentz(ENERGY, 170.6, 2.0)
    peaks = pk.find_features(ENERGY, values, "min")
    assert [p.energy for p in peaks] == pytest.approx([150.2, 170.6], abs=STEP / 4)
    assert [p.strength for p in peaks] == pytest.approx([0.4, 0.3], abs=0.04)


@pytest.mark.parametrize("x0", [150.0, 150.2, 150.37])
def test_dispersive_inflections_at_analytic_positions(x0):
    g = 4.0
    values = dispersive(ENERGY, x0, g)
    rising = pk.find_features(ENERGY, values, Feature.RISING)
    falling = pk.find_features(ENERGY, values, Feature.FALLING)
    assert [p.energy for p in rising] == pytest.approx([x0], abs=STEP / 4)
    assert [p.energy for p in falling] == pytest.approx([x0 - g, x0 + g], abs=STEP / 4)
    # |slope| at the inflection points: 2/g at the centre, 1/(2g) at x0 ± g
    assert rising[0].strength == pytest.approx(2.0 / g, rel=0.1)
    assert [p.strength for p in falling] == pytest.approx([0.5 / g] * 2, rel=0.1)


def test_inflections_of_a_step_edge():
    values = np.tanh((ENERGY - 160.3) / 3.0)
    rising = pk.find_features(ENERGY, values, "rising", prominence=0.01)
    assert [p.energy for p in rising] == pytest.approx([160.3], abs=STEP / 4)
    assert pk.find_features(ENERGY, values, "falling", prominence=0.01) == []
    falling = pk.find_features(ENERGY, -values, "falling", prominence=0.01)
    assert [p.energy for p in falling] == pytest.approx([160.3], abs=STEP / 4)


def test_non_uniform_energy_grid():
    steps = np.linspace(0.2, 1.0, 160)
    energy = 100.0 + np.concatenate([[0.0], np.cumsum(steps)])
    for x0 in (130.37, 150.81, 170.05):
        local = np.interp(x0, energy[1:], steps)
        peaks = pk.find_features(energy, lorentz(energy, x0, 3.0), "max")
        assert [p.energy for p in peaks] == pytest.approx([x0], abs=local / 4)
        rising = pk.find_features(energy, dispersive(energy, x0, 6.0), "rising")
        assert [p.energy for p in rising] == pytest.approx([x0], abs=local / 4)


def test_unsorted_energy_and_nan_samples():
    order = np.random.default_rng(3).permutation(ENERGY.size)
    energy = ENERGY[order]
    values = lorentz(energy, 160.2, 1.5)
    values[np.abs(energy - 130.0) < 3] = np.nan
    values[np.abs(energy - 161.0) < 0.1] = np.nan  # one sample on the flank
    peaks = pk.find_features(energy, values, "max")
    assert len(peaks) == 1
    assert peaks[0].energy == pytest.approx(160.2, abs=STEP / 4)
    assert energy[peaks[0].index] == 160.0
    assert pk.find_features(energy, np.full(energy.size, np.nan), "max") == []


def test_e_range_restricts_the_search():
    values = lorentz(ENERGY, 130.0, 1.5) + lorentz(ENERGY, 170.0, 1.5)
    peaks = pk.find_features(ENERGY, values, "max", e_range=(150.0, None))
    assert [p.energy for p in peaks] == pytest.approx([170.0], abs=STEP / 4)
    assert len(pk.find_features(ENERGY, values, "max", e_range=(180.0, 120.0))) == 2
    assert pk.find_features(ENERGY, values, "max", e_range=(140.0, 141.0)) == []


def test_prominence_removes_noise_peaks():
    rng = np.random.default_rng(1)
    noise = 0.01 * rng.standard_normal(ENERGY.size)
    assert len(pk.find_features(ENERGY, noise, "max")) > 20
    assert pk.find_features(ENERGY, noise, "max", prominence=0.2) == []
    assert pk.find_features(ENERGY, noise, "rising", prominence=0.2) == []
    peaks = pk.find_features(ENERGY, noise + lorentz(ENERGY, 150.0, 1.5), "max", prominence=0.2)
    assert [p.energy for p in peaks] == pytest.approx([150.0], abs=STEP / 4)


def test_smoothing_improves_a_noisy_peak():
    rng = np.random.default_rng(2)
    truth = rng.uniform(140.0, 160.0, 200)
    raw, smooth = [], []
    for x0 in truth:
        values = lorentz(ENERGY, x0, 2.0) + 0.03 * rng.standard_normal(ENERGY.size)
        for out, sg in ((raw, None), (smooth, (9, 3))):
            peaks = pk.find_features(ENERGY, values, "max", smooth=sg, prominence=0.3)
            out.append(min(abs(p.energy - x0) for p in peaks))
    raw, smooth = np.array(raw), np.array(smooth)
    assert np.sqrt(np.mean(smooth**2)) < 0.5 * np.sqrt(np.mean(raw**2))
    assert np.mean(raw < STEP / 4) < 0.8
    assert np.mean(smooth < STEP / 4) > 0.95


def test_smoothing_window_is_checked_and_clipped():
    with pytest.raises(ValueError, match="odd"):
        pk.find_features(ENERGY, ENERGY, "max", smooth=(8, 2))
    with pytest.raises(ValueError, match="odd"):
        pk.find_features(ENERGY, ENERGY, "max", smooth=(3, 3))
    energy = np.arange(10.0)
    values = lorentz(energy, 4.6, 1.5)
    peaks = pk.find_features(energy, values, "max", smooth=(51, 2))
    assert len(peaks) == 1
    assert pk.find_features(energy[:3], values[:3], "max", smooth=(5, 3)) == []


def test_find_features_rejects_bad_input():
    with pytest.raises(ValueError, match="equal length"):
        pk.find_features(ENERGY, ENERGY[:-1], "max")
    with pytest.raises(ValueError, match="repeated"):
        pk.find_features(np.array([1.0, 2.0, 2.0, 3.0]), np.zeros(4), "max")
    with pytest.raises(ValueError):
        pk.find_features(ENERGY, ENERGY, "peak")


# ---- types ------------------------------------------------------------------------------


def test_feature_and_peak_values():
    assert Feature("rising") is Feature.RISING
    assert [str(f) for f in Feature] == ["max", "min", "rising", "falling"]
    assert Peak(1.0, 0.5, 3) == Peak(1.0, 0.5, 3)


def test_picking_does_not_import_qt():
    code = (
        "import sys, mag_opt_detective.core.picking; "
        "sys.exit(any(m.startswith(('PySide6', 'pyqtgraph')) for m in sys.modules))"
    )
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0
