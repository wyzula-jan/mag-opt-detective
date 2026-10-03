import subprocess
import sys

import numpy as np
import pytest

from mag_opt_detective.core import picking as pk
from mag_opt_detective.core.picking import Feature, Peak, Track
from mag_opt_detective.core.spectra import FieldMap

STEP = 0.5  # energy step of the synthetic maps
ENERGY = np.arange(100.0, 200.0 + STEP / 2, STEP)
FIELD = np.linspace(0.25, 9.0, 36)


def lorentz(x, x0, g):
    return 1.0 / (1.0 + ((x - x0) / g) ** 2)


def dispersive(x, x0, g):
    """Minus the derivative of a Lorentzian (scaled): rising inflection at x0, falling at x0±g."""
    u = (x - x0) / g
    return 2.0 * u / (1.0 + u**2) ** 2


def line_map(lines, *, energy=ENERGY, field=FIELD, width=1.5, noise=0.0, seed=0, shape=lorentz):
    """Map with one line per callable ``E(B)``; ``nan`` energies leave a column empty."""
    rng = np.random.default_rng(seed)
    values = noise * rng.standard_normal((energy.size, field.size))
    for line in lines:
        for j, b in enumerate(field):
            e0 = line(b)
            if np.isfinite(e0):
                values[:, j] += shape(energy, e0, width)
    return FieldMap(energy=energy, field=field, values=values, unit="meV")


def linear(b):
    return 130.0 + 4.0 * b


def sqrt_line(b):
    return 120.0 + 15.0 * np.sqrt(b)


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


# ---- track ------------------------------------------------------------------------------


@pytest.mark.parametrize("line", [linear, sqrt_line])
def test_track_follows_a_noisy_line(line):
    fmap = line_map([line], noise=0.01, seed=4)
    seed = (FIELD[18], line(FIELD[18]) + 0.8)
    result = pk.track(fmap, seed, Feature.MAX, window=3.0, prominence=0.2)
    np.testing.assert_array_equal(result.field, FIELD)
    assert np.max(np.abs(result.energy - line(FIELD))) < STEP / 4
    assert np.all(result.strength > 0.8)
    assert len(result) == FIELD.size


def test_track_minimum_of_inverted_lines():
    fmap = line_map([linear], noise=0.01, seed=5)
    fmap = fmap.with_values(1.0 - 0.5 * fmap.values)
    result = pk.track(fmap, (FIELD[0], linear(FIELD[0])), "min", window=3.0, prominence=0.1)
    assert np.max(np.abs(result.energy - linear(FIELD))) < STEP / 4


def test_track_rising_inflection():
    fmap = line_map([sqrt_line], width=4.0, shape=dispersive)
    result = pk.track(fmap, (FIELD[-1], sqrt_line(FIELD[-1])), "rising", window=3.0)
    assert result.field.size == FIELD.size
    assert np.max(np.abs(result.energy - sqrt_line(FIELD))) < STEP / 4


def test_track_direction():
    fmap = line_map([linear])
    seed = (FIELD[10], linear(FIELD[10]))
    up = pk.track(fmap, seed, "max", window=3.0, direction="up")
    down = pk.track(fmap, seed, "max", window=3.0, direction="down")
    np.testing.assert_array_equal(up.field, FIELD[10:])
    np.testing.assert_array_equal(down.field, FIELD[:11])
    with pytest.raises(ValueError, match="direction"):
        pk.track(fmap, seed, "max", window=3.0, direction="sideways")


def test_track_stops_after_max_misses_when_the_line_ends():
    end = FIELD[25]
    fmap = line_map([lambda b: linear(b) if b <= end else np.nan], noise=0.01, seed=6)
    result = pk.track(fmap, (FIELD[5], linear(FIELD[5])), "max", window=3.0, prominence=0.2)
    np.testing.assert_array_equal(result.field, FIELD[:26])


def test_track_bridges_at_most_max_misses_columns():
    gap = FIELD[[12, 13]]
    fmap = line_map([lambda b: np.nan if b in gap else linear(b)], noise=0.01, seed=7)
    seed = (FIELD[5], linear(FIELD[5]))
    bridged = pk.track(fmap, seed, "max", window=3.0, prominence=0.2, max_misses=2)
    np.testing.assert_array_equal(bridged.field, np.delete(FIELD, [12, 13]))
    stopped = pk.track(fmap, seed, "max", window=3.0, prominence=0.2, max_misses=1)
    np.testing.assert_array_equal(stopped.field, FIELD[:12])


def test_track_skips_nan_columns():
    fmap = line_map([sqrt_line])
    values = fmap.values.copy()
    values[:, [3, 4, 5, 20]] = np.nan
    values[:40, 10] = np.nan  # partly empty column, away from the line
    fmap = fmap.with_values(values)
    result = pk.track(fmap, (FIELD[15], sqrt_line(FIELD[15])), "max", window=3.0, max_misses=0)
    np.testing.assert_array_equal(result.field, np.delete(FIELD, [3, 4, 5, 20]))
    assert np.max(np.abs(result.energy - sqrt_line(result.field))) < STEP / 4


def test_track_on_a_non_uniform_energy_grid():
    steps = np.linspace(0.3, 0.8, 180)
    energy = 100.0 + np.concatenate([[0.0], np.cumsum(steps)])
    fmap = line_map([linear], energy=energy, width=2.0)
    result = pk.track(fmap, (FIELD[0], linear(FIELD[0])), "max", window=3.0)
    local = np.interp(result.energy, energy[1:], steps)
    assert result.field.size == FIELD.size
    assert np.all(np.abs(result.energy - linear(FIELD)) < local / 4)


def test_track_seed_errors():
    fmap = line_map([linear])
    with pytest.raises(ValueError, match="outside"):
        pk.track(fmap, (FIELD[-1] + 1.0, 150.0), "max", window=3.0)
    with pytest.raises(ValueError, match="outside"):
        pk.track(fmap, (FIELD[0], ENERGY[0] - 1.0), "max", window=3.0)
    with pytest.raises(ValueError, match="no max feature"):
        pk.track(fmap, (FIELD[0], linear(FIELD[0]) + 10.0), "max", window=3.0)
    with pytest.raises(ValueError, match="no min feature"):
        pk.track(fmap, (FIELD[0], linear(FIELD[0])), "min", window=3.0, prominence=0.1)
    with pytest.raises(ValueError, match="window"):
        pk.track(fmap, (FIELD[0], linear(FIELD[0])), "max", window=0.0)


# ---- detect -----------------------------------------------------------------------------


def falling_line(b):
    return 190.0 - 1.5 * b


def test_detect_finds_two_separate_lines():
    fmap = line_map([linear, falling_line], noise=0.01, seed=8)
    tracks = pk.detect(fmap, feature="max", prominence=0.2)
    assert len(tracks) == 2
    for result, line in zip(tracks, [linear, falling_line], strict=True):
        np.testing.assert_array_equal(result.field, FIELD)
        assert np.max(np.abs(result.energy - line(FIELD))) < STEP / 4


def test_detect_box_limits():
    fmap = line_map([linear, falling_line])
    tracks = pk.detect(fmap, feature="max", b_range=(2.0, 6.0), e_range=(155.0, None))
    assert len(tracks) == 1
    inside = (FIELD >= 2.0) & (FIELD <= 6.0)
    np.testing.assert_array_equal(tracks[0].field, FIELD[inside])
    assert np.allclose(tracks[0].energy, falling_line(FIELD[inside]), atol=STEP / 4)


def test_detect_prominence_excludes_noise_only_columns():
    fmap = line_map([lambda b: linear(b) if b < 4.0 else np.nan], noise=0.01, seed=9)
    tracks = pk.detect(fmap, feature="max", prominence=0.2)
    assert len(tracks) == 1
    np.testing.assert_array_equal(tracks[0].field, FIELD[FIELD < 4.0])
    assert len(pk.detect(fmap, feature="max")) > 1


def test_detect_skips_nan_columns_and_drops_short_tracks():
    fmap = line_map([linear, lambda b: 185.0 if b < 0.6 else np.nan])
    values = fmap.values.copy()
    values[:, [6, 7, 30]] = np.nan
    fmap = fmap.with_values(values)
    tracks = pk.detect(fmap, feature="max")
    assert len(tracks) == 1
    np.testing.assert_array_equal(tracks[0].field, np.delete(FIELD, [6, 7, 30]))
    assert len(pk.detect(fmap, feature="max", min_length=3)) == 1
    assert len(pk.detect(fmap, feature="max", min_length=2)) == 2


def test_detect_max_jump_limits_linking():
    fmap = line_map([lambda b: 110.0 + 8.0 * b])  # 2 meV (4 samples) per column
    assert len(pk.detect(fmap, feature="max", min_length=1)) == FIELD.size
    tracks = pk.detect(fmap, feature="max", max_jump=3.0)
    assert len(tracks) == 1
    assert len(tracks[0]) == FIELD.size


def test_detect_rejects_bad_options():
    fmap = line_map([linear])
    with pytest.raises(ValueError, match="max_jump"):
        pk.detect(fmap, feature="max", max_jump=0.0)
    with pytest.raises(ValueError, match="min_length"):
        pk.detect(fmap, feature="max", min_length=0)


# ---- types ------------------------------------------------------------------------------


def test_track_is_sorted_by_field():
    result = Track(field=[2.0, 1.0, 3.0], energy=[20.0, 10.0, 30.0], strength=[0.2, 0.1, 0.3])
    np.testing.assert_array_equal(result.field, [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(result.energy, [10.0, 20.0, 30.0])
    np.testing.assert_array_equal(result.strength, [0.1, 0.2, 0.3])
    with pytest.raises(ValueError, match="equal length"):
        Track(field=[1.0], energy=[1.0, 2.0], strength=[1.0])


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


# ---- detect: gaps and prediction, auto prominence ----------------------------------------


def test_detect_bridges_up_to_max_misses_columns_with_data():
    gap = FIELD[[12]]
    fmap = line_map([lambda b: np.nan if b in gap else linear(b)], noise=0.01, seed=10)
    split = pk.detect(fmap, feature="max", prominence=0.2)
    assert [len(t) for t in split] == [12, 23]
    bridged = pk.detect(fmap, feature="max", prominence=0.2, max_misses=1)
    assert len(bridged) == 1
    np.testing.assert_array_equal(bridged[0].field, np.delete(FIELD, 12))
    two = FIELD[[12, 13]]
    fmap = line_map([lambda b: np.nan if b in two else linear(b)], noise=0.01, seed=10)
    assert len(pk.detect(fmap, feature="max", prominence=0.2, max_misses=1)) == 2
    assert len(pk.detect(fmap, feature="max", prominence=0.2, max_misses=2)) == 1


def test_detect_predicts_from_history():
    def curved(b):
        return 120.0 + 0.9 * b**2  # the step per column grows to 4 meV

    fmap = line_map([curved], width=2.0)
    assert len(pk.detect(fmap, feature="max", max_jump=1.0, min_length=1)) > 1
    tracks = pk.detect(fmap, feature="max", max_jump=1.0, history=3)
    assert len(tracks) == 1
    np.testing.assert_array_equal(tracks[0].field, FIELD)
    assert np.max(np.abs(tracks[0].energy - curved(FIELD))) < STEP / 4
    with pytest.raises(ValueError, match="history"):
        pk.detect(fmap, feature="max", history=0)
    with pytest.raises(ValueError, match="max_misses"):
        pk.detect(fmap, feature="max", max_misses=-1)


def test_auto_prominence_sits_between_noise_and_lines():
    fmap = line_map([linear, falling_line], noise=0.01, seed=11)
    auto = pk.auto_prominence(fmap, "max")
    assert 0.03 < auto < 0.3
    tracks = pk.detect(fmap, feature="max", prominence=auto)
    assert len(tracks) == 2
    assert [len(t) for t in tracks] == [FIELD.size] * 2
    slope = pk.auto_prominence(fmap, "rising")
    rising = pk.detect(fmap, feature="rising", prominence=slope, max_jump=1.5)
    assert [len(t) for t in rising] == [FIELD.size] * 2


def test_auto_prominence_follows_the_smoothing():
    fmap = line_map([sqrt_line], width=3.0, noise=0.03, seed=13)
    raw = pk.auto_prominence(fmap, "rising")
    smooth = pk.auto_prominence(fmap, "rising", smooth=(15, 2))
    assert smooth < raw / 3  # the slope of smoothed spectra is far less noisy
    options = {"max_jump": 1.5, "max_misses": 1, "history": 3}
    found = pk.detect(fmap, feature="rising", smooth=(15, 2), prominence=smooth, **options)
    assert len(found) == 1 and len(found[0]) >= 30
    rising = sqrt_line(found[0].field) - 3.0 / np.sqrt(3)  # x0 - g/sqrt(3) of a Lorentzian
    assert np.max(np.abs(found[0].energy - rising)) < 3 * STEP  # smoothing over 15 > the line


def test_auto_prominence_of_a_map_without_noise_keeps_weak_lines():
    fmap = line_map([linear])
    weak = line_map([falling_line])
    fmap = fmap.with_values(fmap.values + 0.3 * weak.values)
    auto = pk.auto_prominence(fmap, "max")
    assert 0.0 <= auto < 0.01
    assert len(pk.detect(fmap, feature="max", prominence=auto)) == 2


def test_auto_prominence_box_and_empty_cases():
    fmap = line_map([linear], noise=0.01, seed=12)
    quiet = fmap.with_values(np.where(fmap.energy[:, None] > 160.0, 0.0, fmap.values))
    assert pk.auto_prominence(quiet, "max", e_range=(165.0, None)) == 0.0
    assert pk.auto_prominence(quiet, "max", b_range=(20.0, 30.0)) == 0.0
    boxed = pk.auto_prominence(fmap, "max", b_range=(2.0, 4.0), e_range=(100.0, 125.0))
    assert boxed == pytest.approx(6 * 0.01, rel=0.2)  # six times the noise
    assert pk.auto_prominence(fmap, "min", columns=4) > 0.0
    with pytest.raises(ValueError, match="odd"):
        pk.auto_prominence(fmap, "max", smooth=(8, 2))


# ---- detect in a region (mask) ------------------------------------------------------------


def ellipse(center, axes, n=180):
    """Vertices of an axis-aligned ellipse (field, energy) around *center*."""
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.column_stack([center[0] + axes[0] * np.cos(t), center[1] + axes[1] * np.sin(t)])


def rotated_rectangle(center, length, width, slope, scale=10.0):
    """Corners of a rectangle along a line of *slope* (meV/T); lengths in meV, with 1 T drawn
    as long as *scale* meV (as on a plot whose axes have different units)."""
    u = np.array([1.0, slope / scale]) / np.hypot(1.0, slope / scale)
    v = np.array([-u[1], u[0]])
    corners = [
        s * length / 2 * u + t * width / 2 * v for s, t in ((-1, -1), (1, -1), (1, 1), (-1, 1))
    ]
    return np.array(center) + np.array(corners) / [scale, 1.0]


def test_polygon_mask_matches_a_point_in_polygon_test():
    from matplotlib.path import Path

    rng = np.random.default_rng(14)
    field = np.sort(rng.uniform(0.0, 9.0, 30))[::-1]  # uneven and falling
    energy = rng.permutation(np.concatenate([np.linspace(100.0, 200.0, 150), [np.nan]]))
    fmap = FieldMap(energy=energy, field=field, values=np.zeros((energy.size, field.size)))
    radius = np.tile([4.0, 1.5], 5)
    angle = np.linspace(0.0, 2 * np.pi, 10, endpoint=False)
    star = np.column_stack([4.5 + radius * np.cos(angle), 150.0 + 10 * radius * np.sin(angle)])
    mask = pk.polygon_mask(fmap, star)
    b, e = np.meshgrid(field, energy)
    inside = Path(star).contains_points(np.column_stack([b.ravel(), e.ravel()]))
    np.testing.assert_array_equal(mask, inside.reshape(mask.shape) & np.isfinite(e))
    assert 0 < mask.sum() < mask.size
    box = [(2.0, 120.0), (6.0, 120.0), (6.0, 160.0), (2.0, 160.0)]
    in_b, in_e = (field > 2.0) & (field < 6.0), (energy > 120.0) & (energy < 160.0)
    np.testing.assert_array_equal(pk.polygon_mask(fmap, box), in_e[:, None] & in_b[None, :])
    assert not pk.polygon_mask(fmap, box[:2]).any()  # not an area


def test_detect_in_an_ellipse_skips_lines_outside_it():
    centre = lambda b: 150.0  # noqa: E731 - through the middle of the ellipse
    corner = lambda b: 175.0 if b <= 2.6 else np.nan  # noqa: E731 - in the box, not the ellipse
    fmap = line_map([centre, corner], noise=0.01, seed=15)
    region = ellipse((4.5, 150.0), (3.0, 30.0))
    box = {"b_range": (1.5, 7.5), "e_range": (120.0, 180.0)}
    in_box = pk.detect(fmap, feature="max", prominence=0.2, **box)
    assert [round(float(t.energy.mean())) for t in in_box] == [150, 175]
    mask = pk.polygon_mask(fmap, region)
    tracks = pk.detect(fmap, feature="max", prominence=0.2, mask=mask, **box)
    assert len(tracks) == 1
    assert np.allclose(tracks[0].energy, 150.0, atol=STEP / 4)
    inner = FIELD[(FIELD > 1.75) & (FIELD < 7.25)]  # the tips are too thin for a peak
    assert set(inner) <= set(tracks[0].field)
    assert np.all(((tracks[0].field - 4.5) / 3.0) ** 2 < 1.0)
    assert len(pk.detect(fmap, feature="max", prominence=0.2, mask=mask)) == 1  # box optional


def test_detect_in_a_rotated_rectangle_follows_one_of_two_parallel_lines():
    low, high = (lambda b: 110.0 + 8.0 * b), (lambda b: 126.0 + 8.0 * b)
    fmap = line_map([low, high], noise=0.01, seed=16)
    region = rotated_rectangle((4.5, low(4.5)), length=80.0, width=10.0, slope=8.0)
    b_range = (region[:, 0].min(), region[:, 0].max())
    e_range = (region[:, 1].min(), region[:, 1].max())
    assert e_range[0] < high(b_range[0]) < e_range[1]  # the other line is in the bounds
    options = {"prominence": 0.2, "max_jump": 3.0, "b_range": b_range, "e_range": e_range}
    assert len(pk.detect(fmap, feature="max", **options)) == 2
    mask = pk.polygon_mask(fmap, region)
    tracks = pk.detect(fmap, feature="max", mask=mask, **options)
    assert len(tracks) == 1
    assert np.max(np.abs(tracks[0].energy - low(tracks[0].field))) < STEP / 4
    assert len(tracks[0]) >= 20


def test_detect_in_a_concave_polygon():
    flat_low, flat_high = (lambda b: 120.0), (lambda b: 180.0)
    fmap = line_map([flat_low, flat_high], noise=0.01, seed=17)
    ell = [(1.1, 110.0), (7.9, 110.0), (7.9, 130.0), (3.1, 130.0), (3.1, 190.0), (1.1, 190.0)]
    mask = pk.polygon_mask(fmap, ell)
    tracks = pk.detect(fmap, feature="max", prominence=0.2, mask=mask)
    assert len(tracks) == 2
    low, high = tracks
    np.testing.assert_array_equal(low.field, FIELD[(FIELD > 1.1) & (FIELD < 7.9)])
    np.testing.assert_array_equal(high.field, FIELD[(FIELD > 1.1) & (FIELD < 3.1)])
    assert np.allclose(high.energy, 180.0, atol=STEP / 4)
    rect = pk.polygon_mask(fmap, [(1.9, 99.0), (6.1, 99.0), (6.1, 201.0), (1.9, 201.0)])
    assert pk.auto_prominence(fmap, "max", mask=rect) == pytest.approx(
        pk.auto_prominence(fmap, "max", b_range=(2.0, 6.0))
    )
    with pytest.raises(ValueError, match="mask shape"):
        pk.detect(fmap, feature="max", mask=mask[:, :3])
