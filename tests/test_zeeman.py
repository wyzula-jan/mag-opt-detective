import math

import numpy as np
import pytest

from mag_opt_detective.core.fitting import Assignment, Model, Observation, fit
from mag_opt_detective.core.zeeman import MU_B, Branch, Form, ZeemanModel, branch_energy


def within(result, name, true, sigmas=2.0):
    return abs(result.values[name] - true) <= sigmas * result.stderr[name]


def noisy(model, field, noise, seed):
    rng = np.random.default_rng(seed)
    lines = model.evaluate(field)
    return [
        Observation(field, line + rng.normal(0.0, noise, field.size), k)
        for k, line in enumerate(lines)
    ]


def test_bohr_magneton_in_mev_per_tesla():
    assert pytest.approx(0.0578838, rel=1e-6) == MU_B


def test_branch_forms():
    field = np.array([0.0, 1.0, 10.0])
    np.testing.assert_allclose(branch_energy(field, 5.0, 2.0, -0.5), 5.0 - MU_B * field)
    hyperbolic = branch_energy(field, 3.0, 2.0, 1.0, "hyperbolic")
    np.testing.assert_allclose(hyperbolic, np.sqrt(9.0 + (2.0 * MU_B * field) ** 2))
    assert Branch(1.0, form="hyperbolic").form is Form.HYPERBOLIC


def test_linear_fit_recovers_e0_and_g():
    truth = ZeemanModel([Branch(5.0, 2.0, 1.0), Branch(5.5, 2.0, -1.0)])
    observations = noisy(truth, np.linspace(0.0, 16.0, 25), 0.05, seed=2)
    model = ZeemanModel([Branch(4.5, 1.5, 1.0), Branch(6.0, 2.5, -1.0)])
    result = fit(model, observations)
    assert result.success
    for name, value in {"e0_0": 5.0, "g_0": 2.0, "e0_1": 5.5, "g_1": 2.0}.items():
        assert within(result, name, value), name
    assert math.isnan(result.stderr["delta_0_1"])  # uncoupled: held at 0
    assert result.values["delta_0_1"] == 0.0


def test_hyperbolic_fit_recovers_e0_and_g():
    truth = ZeemanModel([Branch(3.0, 2.1, 1.0, "hyperbolic")])
    observations = noisy(truth, np.linspace(0.0, 20.0, 30), 0.03, seed=2)
    model = ZeemanModel([Branch(2.0, 1.5, 1.0, "hyperbolic")])
    assert model.params[0].lo == 0.0  # the zero-field gap of a hyperbolic branch
    result = fit(model, observations)
    assert within(result, "e0_0", 3.0)
    assert within(result, "g_0", 2.1)


def crossing(delta=0.3, coupled=True):
    # 4 + 2 muB B meets 6 - 2 muB B at B = 2 / (4 muB) = 8.64 T
    branches = [Branch(4.0, 2.0, 1.0, label="up"), Branch(6.0, 2.0, -1.0, label="down")]
    return ZeemanModel(branches, {(0, 1): delta}, coupled=coupled)


def test_coupled_crossing_opens_a_gap_of_two_delta():
    model = crossing(delta=0.3)
    field = np.linspace(0.0, 20.0, 20001)
    modes = model.evaluate(field)
    assert modes.shape == (2, field.size)
    assert (np.diff(modes, axis=0) > 0).all()  # ascending per field
    gap = modes[1] - modes[0]
    b_cross = 2.0 / (4.0 * MU_B)
    assert gap.min() == pytest.approx(0.6, abs=1e-6)
    assert field[gap.argmin()] == pytest.approx(b_cross, abs=2e-3)
    # far from the crossing the modes follow the bare branches
    uncoupled = crossing(coupled=False).evaluate(field)
    np.testing.assert_allclose(modes[:, 0], np.sort(uncoupled[:, 0]), atol=0.05)
    assert model.branch_names() == ["mode 1", "mode 2"]


def test_branch_weights_label_the_modes():
    model = crossing(delta=0.3)
    field = np.array([0.0, 2.0 / (4.0 * MU_B), 20.0])
    weights = model.branch_weights(field)
    assert weights.shape == (2, 2, 3)
    np.testing.assert_allclose(weights.sum(axis=1), 1.0)
    assert weights[0, 0, 0] > 0.95  # below the crossing the lower mode is "up"
    np.testing.assert_allclose(weights[:, :, 1], 0.5, atol=1e-9)
    assert weights[0, 1, 2] > 0.95  # above it the lower mode is "down"
    model.set_coupled(False)
    np.testing.assert_array_equal(model.branch_weights(field)[:, :, 0], np.eye(2))


def test_coupled_fit_recovers_delta_with_sorted_assignment():
    observations = noisy(crossing(delta=0.3), np.linspace(3.0, 14.0, 34), 0.02, seed=2)
    model = ZeemanModel(
        [Branch(3.8, 1.8, 1.0), Branch(6.2, 2.2, -1.0)], {(0, 1): 0.15}, coupled=True
    )
    result = fit(model, observations, Assignment.SORTED)
    assert result.success
    assert within(result, "delta_0_1", 0.3)
    for name, value in {"e0_0": 4.0, "g_0": 2.0, "e0_1": 6.0, "g_1": 2.0}.items():
        assert within(result, name, value), name


def test_uncoupled_holds_deltas_at_zero_and_restores_them():
    model = crossing(delta=0.4, coupled=False)
    assert isinstance(model, Model)
    delta = model.params[-1]
    assert (delta.name, delta.value, delta.fixed) == ("delta_0_1", 0.0, True)
    assert model.couplings == {(0, 1): 0.4}
    field = np.array([0.0, 20.0])
    # plain branches in branch order, not sorted: "up" is above "down" at 20 T
    np.testing.assert_allclose(
        model.evaluate(field),
        [branch_energy(field, 4.0, 2.0, 1.0), branch_energy(field, 6.0, 2.0, -1.0)],
    )
    assert model.branch_names() == ["up", "down"]
    model.set_coupled(True)
    assert (delta.value, delta.fixed) == (0.4, False)
    delta.value, delta.fixed = 0.25, True
    model.set_coupled(False)
    assert (delta.value, delta.fixed) == (0.0, True)
    model.set_coupled(True)
    assert (delta.value, delta.fixed) == (0.25, True)


def test_add_and_remove_branches_keep_values():
    model = crossing(delta=0.3)
    model.params[0].fixed = True
    model.add_branch(Branch(10.0, 0.0, 0.0, label="phonon"))
    assert [p.name for p in model.params] == [
        "e0_0",
        "g_0",
        "e0_1",
        "g_1",
        "e0_2",
        "g_2",
        "delta_0_1",
        "delta_0_2",
        "delta_1_2",
    ]
    assert model.couplings == {(0, 1): 0.3, (0, 2): 0.0, (1, 2): 0.0}
    model.params[-1].value = 0.1
    model.remove_branch(0)
    assert [p.name for p in model.params] == ["e0_0", "g_0", "e0_1", "g_1", "delta_0_1"]
    assert model.couplings == {(0, 1): 0.1}
    assert [b.label for b in model.branches] == ["down", "phonon"]
    assert model.branches[0].e0 == 6.0
    with pytest.raises(IndexError):
        model.remove_branch(5)


def test_couplings_are_validated():
    branches = [Branch(1.0), Branch(2.0)]
    assert ZeemanModel(branches, {(1, 0): 0.2}, coupled=True).couplings == {(0, 1): 0.2}
    with pytest.raises(ValueError):
        ZeemanModel(branches, {(0, 0): 0.2})
    with pytest.raises(ValueError):
        ZeemanModel(branches, {(0, 2): 0.2})


def test_nearest_assignment_matches_points_to_the_closest_line():
    true = ZeemanModel([Branch(5.0, 2.0, 1.0), Branch(8.0, 2.0, -1.0)])
    field = np.linspace(0.0, 8.0, 20)
    lines = true.evaluate(field)
    rng = np.random.default_rng(5)
    # one unlabelled curve holding points of both lines
    mixed = Observation(
        np.concatenate([field, field]),
        np.concatenate(lines) + rng.normal(0.0, 0.02, 2 * field.size),
        label="all points",
    )
    model = ZeemanModel([Branch(5.3, 1.6, 1.0), Branch(7.6, 2.5, -1.0)])
    result = fit(model, [mixed], Assignment.NEAREST)
    for name, value in {"e0_0": 5.0, "g_0": 2.0, "e0_1": 8.0, "g_1": 2.0}.items():
        assert within(result, name, value)
    assert result.residuals[0].size == 40
