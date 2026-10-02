import math
import time

import numpy as np
import pytest
from scipy.constants import c, e, physical_constants

from mag_opt_detective.core.expressions import (
    MAX_LENGTH,
    ExpressionError,
    ExpressionModel,
    parse,
)
from mag_opt_detective.core.fitting import Assignment, Model, Observation, fit
from mag_opt_detective.core.units import Unit
from mag_opt_detective.core.zeeman import MU_B, branch_energy

FIELD = np.linspace(0.0, 16.0, 9)


def test_zeeman_formulas_written_by_hand():
    linear = parse("E0 + m*g*muB*B")
    values = {"E0": 5.0, "m": -0.5, "g": 2.1}
    np.testing.assert_allclose(
        linear.evaluate(FIELD, values)[0], branch_energy(FIELD, 5.0, 2.1, -0.5)
    )
    hyperbolic = parse("sqrt(E0**2 + (g*muB*B)**2)")
    np.testing.assert_allclose(
        hyperbolic.evaluate(FIELD, {"E0": 3.0, "g": 2.0})[0],
        branch_energy(FIELD, 3.0, 2.0, 1.0, "hyperbolic"),
    )


def test_lines_share_parameters_in_order_of_first_appearance():
    compiled = parse("# spin-split pair\n\nE0 + g*muB*B/2  # up\nE1*0 + E0 - g*muB*B/2\n")
    assert compiled.parameters == ["E0", "g", "E1"]
    assert [b.label for b in compiled.branches] == ["up", "E1*0 + E0 - g*muB*B/2"]
    assert [b.line for b in compiled.branches] == [3, 4]
    assert compiled.branches[0].text == "E0 + g*muB*B/2"
    out = compiled.evaluate(FIELD, {"E0": 5.0, "g": 2.0, "E1": 7.0})
    assert out.shape == (2, FIELD.size)
    np.testing.assert_allclose(out[0] - out[1], 2.0 * MU_B * FIELD)
    assert parse("b + a*B + b").parameters == ["b", "a"]


def test_constants_functions_and_float_literals():
    def value(text, **params):
        return parse(text).evaluate(np.array([2.0]), params)[0, 0]

    assert value("muB") == MU_B
    assert value("hbar") == pytest.approx(
        physical_constants["reduced Planck constant in eV s"][0] * 1e3
    )
    assert value("kB") == pytest.approx(physical_constants["Boltzmann constant in eV/K"][0] * 1e3)
    assert (value("e"), value("c"), value("pi")) == (e, c, math.pi)
    assert value("1/2") == 0.5
    assert value("2**-1") == 0.5
    assert value("-B + +B") == 0.0
    assert value("arctan2(1, B)") == pytest.approx(math.atan2(1, 2))
    assert value("hypot(3, B)") == pytest.approx(math.hypot(3, 2))
    assert value("minimum(B, 1) + maximum(B, 5)") == 6.0
    assert value("abs(-B) * sign(-B)") == -2.0
    assert value("log10(100) + log(exp(B)) + cosh(0) + tanh(0)") == pytest.approx(5.0)
    assert np.isnan(value("sqrt(-B)"))
    assert value("1/(B - 2)") == math.inf


def test_constant_expression_is_broadcast_to_the_field():
    out = parse("E0").evaluate(FIELD, {"E0": 4.0})
    assert out.shape == (1, FIELD.size)
    np.testing.assert_array_equal(out, 4.0)


def test_power_tower_overflows_to_inf_quickly():
    start = time.perf_counter()
    out = parse("10**10**10").evaluate(FIELD, {})
    assert time.perf_counter() - start < 1.0
    assert np.isinf(out).all()


@pytest.mark.parametrize(
    ("text", "message", "line", "column"),
    [
        ("1 + np.sqrt(B)", "attribute access", 1, 5),
        ("B\n  a[0]", "subscripts", 2, 3),
        ("(lambda x: x)(B)", "lambda", 1, 2),
        ("sqrt([x for x in B])", "comprehensions", 1, 6),
        ("2 * {x: x for x in B}", "comprehensions", 1, 5),
        ("sqrt(x=B)", "keyword arguments", 1, 6),
        ("hypot(*B)", "starred arguments", 1, 7),
        ("B + _hidden", "underscore", 1, 5),
        ("__import__('os')", "underscore", 1, 1),
        ("import os", "imports", 1, 1),
        ("B + 'text'", "text is not allowed", 1, 5),
        ("B + b'raw'", "text is not allowed", 1, 5),
        ("B * True", "True is not allowed", 1, 5),
        ("B + 2j", "complex numbers", 1, 5),
        ("open(B)", "unknown function open", 1, 1),
        ("B(2)", "B is not a function", 1, 1),
        ("2 + sqrt(B, B)", "sqrt takes 1 argument, got 2", 1, 5),
        ("hypot(B)", "hypot takes 2 arguments, got 1", 1, 1),
        ("B + sqrt", "sqrt is a function", 1, 5),
        ("B // 2", "operator //", 1, 3),
        ("μ + B % 2", "operator %", 1, 7),
        ("B > 1", "comparisons", 1, 1),
        ("B and 1", "'and' and 'or'", 1, 1),
        ("1 if B else 2", "conditional", 1, 1),
        ("a, b", "commas", 1, 1),
        ("(x := B)", "assignments", 1, 2),
        ("B +", "invalid syntax", 1, 4),
        ("\n\n   sqrt(B", "never closed", 3, 8),
    ],
)
def test_rejections_report_the_position(text, message, line, column):
    with pytest.raises(ExpressionError, match=message) as info:
        parse(text)
    assert (info.value.line, info.value.column) == (line, column)
    assert str(info.value).startswith(f"line {line}, column {column}: ")


def test_size_limits():
    with pytest.raises(ExpressionError, match="longer than") as info:
        parse("B\n" + "1" * MAX_LENGTH)
    assert (info.value.line, info.value.column) == (2, MAX_LENGTH - 1)
    with pytest.raises(ExpressionError, match="more than 500") as info:
        parse("B\n" + " + ".join(["B"] * 251))  # 251 names and 250 operations
    assert (info.value.line, info.value.column) == (2, 1)
    parse(" + ".join(["B"] * 250))


def test_empty_text_and_missing_values():
    with pytest.raises(ExpressionError, match="at least one"):
        parse("# only a comment\n\n")
    with pytest.raises(ValueError, match="no value for parameter g"):
        parse("E0 + g*B").evaluate(FIELD, {"E0": 1.0})


def test_expression_model_parameters_and_branches():
    model = ExpressionModel("E0 + g*muB*B  # up\nE0 - g*muB*B", Unit.THZ, initial={"g": 2.0})
    assert isinstance(model, Model)
    assert model.unit is Unit.THZ
    assert [(p.name, p.value, p.kind) for p in model.params] == [
        ("E0", 1.0, "other"),
        ("g", 2.0, "other"),
    ]
    assert model.branch_names() == ["up", "E0 - g*muB*B"]
    np.testing.assert_allclose(model.evaluate(FIELD)[0], 1.0 + 2.0 * MU_B * FIELD)


def test_set_text_keeps_parameters_by_name():
    model = ExpressionModel("E0 + g*muB*B", initial={"E0": 5.0})
    params = model.params
    model.params[0].fixed = True
    model.set_text("E0 + a*B", initial={"a": 0.1, "unused": 3.0})
    assert model.params is params
    assert [(p.name, p.value, p.fixed) for p in model.params] == [
        ("E0", 5.0, True),
        ("a", 0.1, False),
    ]
    with pytest.raises(ExpressionError):
        model.set_text("E0 + ")
    assert model.text == "E0 + a*B"
    assert [p.name for p in model.params] == ["E0", "a"]


def test_expression_model_fit_recovers_shared_parameters():
    truth = ExpressionModel("E0 + g*muB*B/2\nE0 - g*muB*B/2", initial={"E0": 5.0, "g": 2.0})
    field = np.linspace(0.0, 16.0, 25)
    rng = np.random.default_rng(4)
    lines = truth.evaluate(field)
    observations = [
        Observation(field, line + rng.normal(0.0, 0.02, field.size), k)
        for k, line in enumerate(lines)
    ]
    model = ExpressionModel(truth.text, initial={"E0": 4.0, "g": 1.0})
    result = fit(model, observations)
    assert result.success
    assert abs(result.values["E0"] - 5.0) <= 2 * result.stderr["E0"]
    assert abs(result.values["g"] - 2.0) <= 2 * result.stderr["g"]
    assert model.params[0].value == 4.0  # not applied


def test_sorted_assignment_uses_the_rank_of_the_energy():
    # two uncoupled crossing lines; observation k is the k-th lowest energy at each field
    model = ExpressionModel("E0 + a*B\nE1 - a*B", initial={"E0": 1.0, "E1": 3.0, "a": 0.8})
    field = np.linspace(0.0, 4.0, 21)
    truth = np.sort(model.evaluate(field, {"a": 1.0}), axis=0)
    observations = [Observation(field, truth[0]), Observation(field, truth[1])]
    result = fit(model, observations, Assignment.SORTED)
    assert result.values["a"] == pytest.approx(1.0, abs=1e-6)
    assert result.chi2 == pytest.approx(0.0, abs=1e-12)
