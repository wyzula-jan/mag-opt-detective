"""Fit transition-energy models E(B) to picked points."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal, Protocol, runtime_checkable

import numpy as np
from scipy.optimize import least_squares

from mag_opt_detective.core.units import Unit

ParamKind = Literal["energy", "velocity", "dimensionless", "other"]

# share of a parameter in a direction the data do not constrain above which its error is NaN
NULL_SHARE = 1e-6


@dataclass
class Param:
    """A model parameter; ``energy`` parameters are in the model unit."""

    name: str
    value: float
    lo: float = -math.inf
    hi: float = math.inf
    fixed: bool = False
    kind: ParamKind = "other"


@runtime_checkable
class Model(Protocol):
    """Transition energies of one or more branches as a function of the field B (T)."""

    unit: Unit
    params: list[Param]

    def branch_names(self) -> list[str]: ...

    def evaluate(self, field: np.ndarray, values: Mapping[str, float] | None = None) -> np.ndarray:
        """Energies, shape ``(n_branch, n_field)``; *values* None means the current ones."""
        ...


def param_values(
    params: Sequence[Param], overrides: Mapping[str, float] | None = None
) -> dict[str, float]:
    """Current values of *params* by name, replaced by *overrides* where given."""
    values = {p.name: float(p.value) for p in params}
    if overrides is not None:
        values.update({k: float(v) for k, v in overrides.items()})
    return values


class Assignment(StrEnum):
    """How picked points are matched with model branches."""

    BRANCH = "branch"  # each observation names its branch index
    SORTED = "sorted"  # branch k is the k-th lowest model energy at each field
    NEAREST = "nearest"  # each point is compared with the closest branch at its field


@dataclass
class Observation:
    """Picked points of one curve, with energies in the model unit."""

    field: np.ndarray
    energy: np.ndarray
    branch: int | None = None
    label: str = ""

    def __post_init__(self) -> None:
        self.field = np.asarray(self.field, dtype=float).ravel()
        self.energy = np.asarray(self.energy, dtype=float).ravel()
        if self.field.shape != self.energy.shape:
            raise ValueError(
                f"{_name(self)}: {self.field.size} field values but {self.energy.size} energies"
            )


@dataclass
class FitResult:
    """Outcome of :func:`fit`; covariance rows follow the order of *free*."""

    values: dict[str, float]
    stderr: dict[str, float]
    free: list[str]
    covariance: np.ndarray
    chi2: float
    reduced_chi2: float
    dof: int
    residuals: list[np.ndarray] = field(default_factory=list)
    success: bool = True
    message: str = ""
    nfev: int = 0


def fit(
    model: Model,
    observations: Sequence[Observation],
    assignment: Assignment | str = Assignment.BRANCH,
) -> FitResult:
    """Least-squares fit of the free parameters of *model* to *observations*.

    Residuals are model minus observed energy. Standard errors come from the Jacobian,
    ``cov = inv(J^T J) * chi2 / dof``, and are NaN for fixed parameters, for parameters the
    points do not constrain, or when ``dof <= 0``. The model itself is not changed; see
    :func:`apply`.
    """
    assignment = Assignment(assignment)
    names = [p.name for p in model.params]
    if len(set(names)) != len(names):
        raise ValueError("parameter names must be unique")
    observations = list(observations)
    fields, energies, rows = _stack(observations, assignment, len(model.branch_names()))
    for obs in observations:
        if not (np.isfinite(obs.field).all() and np.isfinite(obs.energy).all()):
            raise ValueError(f"{_name(obs)}: points must be finite")

    free = [p for p in model.params if not p.fixed]
    for p in free:
        if not p.lo < p.hi:
            raise ValueError(f"the bounds of {p.name} leave no room to fit")
    base = param_values(model.params)

    def residuals(x: np.ndarray) -> np.ndarray:
        values = base | {p.name: float(v) for p, v in zip(free, x, strict=True)}
        return _residuals(model, values, fields, energies, rows, assignment)

    if free:
        lo = np.array([p.lo for p in free], dtype=float)
        hi = np.array([p.hi for p in free], dtype=float)
        x0 = np.clip(np.array([p.value for p in free], dtype=float), lo, hi)
        r0 = residuals(x0)
        if not np.isfinite(r0).all():
            raise ValueError("the model is not finite at the starting values")
        sol = least_squares(residuals, x0, bounds=(lo, hi), x_scale="jac")
        x, jac = sol.x, np.atleast_2d(sol.jac)
        success, message, nfev = bool(sol.success), str(sol.message), int(sol.nfev)
    else:
        x, jac = np.empty(0), np.empty((fields.size, 0))
        success, message, nfev = True, "no free parameters", 1

    values = base | {p.name: float(v) for p, v in zip(free, x, strict=True)}
    res = _residuals(model, values, fields, energies, rows, assignment)
    chi2 = float(np.sum(res**2))
    dof = int(fields.size - len(free))
    covariance = _covariance(jac, chi2, dof)
    errors = np.sqrt(np.abs(np.diag(covariance)))
    stderr = dict.fromkeys(names, math.nan)
    stderr.update({p.name: float(s) for p, s in zip(free, errors, strict=True)})
    split = np.cumsum([obs.field.size for obs in observations])[:-1]
    return FitResult(
        values=values,
        stderr=stderr,
        free=[p.name for p in free],
        covariance=covariance,
        chi2=chi2,
        reduced_chi2=chi2 / dof if dof > 0 else math.nan,
        dof=dof,
        residuals=np.split(res, split),
        success=success,
        message=message,
        nfev=nfev,
    )


def apply(model: Model, result: FitResult) -> None:
    """Copy the fitted values of *result* into the parameters of *model*."""
    for p in model.params:
        if p.name in result.values:
            p.value = result.values[p.name]


def _name(obs: Observation) -> str:
    return obs.label or "observation"


def _stack(
    observations: list[Observation], assignment: Assignment, n_branch: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Concatenated fields and energies, and the branch row of every point."""
    if not observations or sum(obs.field.size for obs in observations) == 0:
        raise ValueError("there are no points to fit")
    rows = []
    for k, obs in enumerate(observations):
        branch = obs.branch
        if assignment is Assignment.SORTED and branch is None:
            branch = k
        if assignment is Assignment.NEAREST:
            branch = 0
        elif branch is None:
            raise ValueError(f"{_name(obs)}: choose a branch")
        elif not 0 <= branch < n_branch:
            raise ValueError(f"{_name(obs)}: branch {branch} not in the model (0..{n_branch - 1})")
        rows.append(np.full(obs.field.size, branch, dtype=int))
    fields = np.concatenate([obs.field for obs in observations])
    energies = np.concatenate([obs.energy for obs in observations])
    return fields, energies, np.concatenate(rows)


def _residuals(
    model: Model,
    values: Mapping[str, float],
    fields: np.ndarray,
    energies: np.ndarray,
    rows: np.ndarray,
    assignment: Assignment,
) -> np.ndarray:
    curves = np.asarray(model.evaluate(fields, values), dtype=float).reshape(-1, fields.size)
    cols = np.arange(fields.size)
    if assignment is Assignment.SORTED:
        curves = np.sort(curves, axis=0)
    diff = curves - energies
    if assignment is Assignment.NEAREST:
        rows = np.where(np.isnan(diff), np.inf, np.abs(diff)).argmin(axis=0)
    return diff[rows, cols]


def _covariance(jac: np.ndarray, chi2: float, dof: int) -> np.ndarray:
    """``inv(J^T J) * chi2 / dof``, from the SVD of J with its columns scaled to unit length.

    A parameter the data do not constrain (its column is zero, or it has a share in a
    direction that does not change the model) gets NaN rows and columns: its error is not
    known, rather than zero.
    """
    n = jac.shape[1]
    if n == 0:
        return np.empty((0, 0))
    if dof <= 0 or not np.isfinite(jac).all():
        return np.full((n, n), math.nan)
    norms = np.linalg.norm(jac, axis=0)
    dead = norms == 0
    scale = np.where(dead, 1.0, norms)
    _, s, vt = np.linalg.svd(jac / scale, full_matrices=False)
    tol = s.max(initial=0.0) * max(jac.shape) * np.finfo(float).eps
    keep = s > tol
    v = vt.T
    inverse = (v[:, keep] / s[keep] ** 2) @ v[:, keep].T / np.outer(scale, scale)
    loose = dead | (np.abs(v[:, ~keep]) > NULL_SHARE).any(axis=1)
    covariance = inverse * (chi2 / dof)
    covariance[loose, :] = math.nan
    covariance[:, loose] = math.nan
    return covariance
