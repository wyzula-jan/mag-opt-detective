import math

import numpy as np
import pytest

from mag_opt_detective.core.fitting import (
    Model,
    Observation,
    Param,
    apply,
    fit,
    param_values,
)
from mag_opt_detective.core.models import DiracModel, dirac_interband
from mag_opt_detective.core.units import Unit


def within(result, name, true, sigmas=2.0):
    """True when the fitted *name* is within *sigmas* standard errors of *true*."""
    return abs(result.values[name] - true) <= sigmas * result.stderr[name]


TRUE_DIRAC = {"velocity": 5.4, "delta": 12.0}


def dirac_points(seed=3, noise=0.3, n_lines=3):
    rng = np.random.default_rng(seed)
    field = np.linspace(0.5, 16.0, 25)
    lines = dirac_interband(field, 5.4, 12.0, n_lines)
    return [
        Observation(field, lines[k] + rng.normal(0.0, noise, field.size), k, f"n={k}")
        for k in range(n_lines)
    ]


def test_dirac_model_wraps_interband():
    model = DiracModel(velocity=5.0, delta=10.0, n_lines=3)
    assert isinstance(model, Model)
    field = np.array([0.0, 1.0, 4.0])
    np.testing.assert_allclose(model.evaluate(field), dirac_interband(field, 5.0, 10.0, 3))
    np.testing.assert_allclose(
        model.evaluate(field, {"delta": 0.0}), dirac_interband(field, 5.0, 0.0, 3)
    )
    assert model.unit is Unit.MEV
    assert [p.kind for p in model.params] == ["velocity", "energy"]
    assert model.branch_names()[0] == "n=0 (L-0 -> L1)"


def test_dirac_fit_recovers_velocity_and_delta():
    observations = dirac_points()
    model = DiracModel(velocity=4.0, delta=5.0, n_lines=3)
    result = fit(model, observations)
    assert result.success
    assert within(result, "velocity", 5.4)
    assert within(result, "delta", 12.0)
    assert result.dof == 75 - 2
    assert result.free == ["velocity", "delta"]
    assert result.covariance.shape == (2, 2)
    assert 0.3**2 / 2 < result.reduced_chi2 < 2 * 0.3**2
    assert [r.size for r in result.residuals] == [25, 25, 25]
    total = sum(float(np.sum(r**2)) for r in result.residuals)
    assert result.chi2 == pytest.approx(total)


def test_standard_errors_match_the_scatter_of_repeated_fits():
    pulls = []
    for seed in range(40):
        result = fit(DiracModel(4.0, 5.0, n_lines=3), dirac_points(seed))
        pulls.append([(result.values[k] - v) / result.stderr[k] for k, v in TRUE_DIRAC.items()])
    assert 0.7 < np.std(pulls) < 1.3


def test_fit_does_not_mutate_the_model_until_applied():
    model = DiracModel(velocity=4.0, delta=5.0, n_lines=3)
    before = [(p.name, p.value, p.fixed) for p in model.params]
    result = fit(model, dirac_points())
    assert [(p.name, p.value, p.fixed) for p in model.params] == before
    apply(model, result)
    assert param_values(model.params) == result.values


def test_fixed_parameter_keeps_its_value_and_has_no_error():
    model = DiracModel(velocity=4.0, delta=12.0, n_lines=3)
    model.params[1].fixed = True
    result = fit(model, dirac_points())
    assert result.values["delta"] == 12.0
    assert math.isnan(result.stderr["delta"])
    assert result.free == ["velocity"]
    assert result.covariance.shape == (1, 1)
    assert within(result, "velocity", 5.4)


def test_bounds_are_respected():
    model = DiracModel(velocity=4.0, delta=5.0, n_lines=3)
    model.params[0].hi = 5.0
    result = fit(model, dirac_points())
    assert result.values["velocity"] <= 5.0
    assert result.values["velocity"] == pytest.approx(5.0, abs=1e-3)


def test_stderr_is_nan_without_degrees_of_freedom():
    field = np.array([2.0, 8.0])
    energy = dirac_interband(field, 5.4, 12.0, 1)[0]
    result = fit(DiracModel(4.0, 5.0, n_lines=1), [Observation(field, energy, 0)])
    assert result.dof == 0
    assert math.isnan(result.stderr["velocity"])
    assert math.isnan(result.stderr["delta"])
    assert math.isnan(result.reduced_chi2)
    assert np.isnan(result.covariance).all()


def test_no_free_parameters_only_evaluates():
    model = DiracModel(5.4, 12.0, n_lines=1)
    for p in model.params:
        p.fixed = True
    field = np.array([1.0, 2.0])
    energy = dirac_interband(field, 5.4, 12.0, 1)[0] + 0.5
    result = fit(model, [Observation(field, energy, 0)])
    assert result.success
    assert result.free == []
    assert result.covariance.shape == (0, 0)
    assert result.chi2 == pytest.approx(0.5)
    np.testing.assert_allclose(result.residuals[0], [-0.5, -0.5])


def test_invalid_observations_are_rejected():
    model = DiracModel(n_lines=2)
    field = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="branch"):
        fit(model, [Observation(field, field)])
    with pytest.raises(ValueError, match="branch 2"):
        fit(model, [Observation(field, field, 2)])
    with pytest.raises(ValueError, match="no points"):
        fit(model, [])
    with pytest.raises(ValueError, match="finite"):
        fit(model, [Observation(field, [1.0, np.nan], 0)])
    with pytest.raises(ValueError, match="field values"):
        Observation(field, [1.0])
    model.params[0].lo = model.params[0].hi = 5.0
    with pytest.raises(ValueError, match="bounds"):
        fit(model, [Observation(field, field, 0)])


def test_param_defaults():
    p = Param("x", 1.0)
    assert (p.lo, p.hi, p.fixed, p.kind) == (-math.inf, math.inf, False, "other")
    assert param_values([p], {"x": 2}) == {"x": 2.0}
