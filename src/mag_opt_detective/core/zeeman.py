"""Zeeman / magnon branches E(B), optionally coupled into avoided crossings."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from scipy.constants import physical_constants

from mag_opt_detective.core.fitting import Param, param_values
from mag_opt_detective.core.units import Unit

MU_B = physical_constants["Bohr magneton in eV/T"][0] * 1e3  # meV/T


class Form(StrEnum):
    LINEAR = "linear"  # E = e0 + m g muB B
    HYPERBOLIC = "hyperbolic"  # E = sqrt(e0^2 + (m g muB B)^2), e.g. AFMR with B perp. easy axis


def branch_energy(
    field: np.ndarray, e0: float, g: float, m: float, form: Form | str = Form.LINEAR
) -> np.ndarray:
    """Energy (meV) of one uncoupled branch at *field* (T)."""
    zeeman = m * g * MU_B * np.asarray(field, dtype=float)
    if Form(form) is Form.HYPERBOLIC:
        return np.sqrt(e0**2 + zeeman**2)
    return e0 + zeeman


@dataclass
class Branch:
    """One branch: zero-field energy *e0* (meV), g-factor *g* and structural multiplier *m*."""

    e0: float
    g: float = 2.0
    m: float = 1.0
    form: Form | str = Form.LINEAR
    label: str = ""

    def __post_init__(self) -> None:
        self.form = Form(self.form)


class ZeemanModel:
    """Branches ``E_i(B)``; when coupled, the eigenvalues of ``diag(E_i(B)) + Delta``.

    Parameters are ``e0_<i>`` and ``g_<i>`` per branch and ``delta_<i>_<j>`` for every pair
    ``i < j`` (0-based branch indices, energies in meV). Uncoupled, the deltas are held fixed
    at 0 and the user's values come back when coupling is switched on again. Coupled, branch
    *k* of :meth:`evaluate` is the *k*-th lowest eigenvalue at each field.
    """

    unit = Unit.MEV

    def __init__(
        self,
        branches: Sequence[Branch],
        couplings: Mapping[tuple[int, int], float] | None = None,
        coupled: bool = False,
    ):
        self._structure: list[Branch] = []
        self._pairs: dict[tuple[int, int], Param] = {}
        self._saved: dict[tuple[int, int], tuple[float, bool]] = {}
        self._coupled = True
        self.params: list[Param] = []
        for branch in branches:
            self.add_branch(branch)
        for pair, value in (couplings or {}).items():
            self._pairs[self._pair(*pair)].value = float(value)
        self.set_coupled(coupled)

    @property
    def coupled(self) -> bool:
        return self._coupled

    def set_coupled(self, coupled: bool) -> None:
        """Switch coupling; uncoupled keeps the user's deltas aside and holds them at 0."""
        coupled = bool(coupled)
        if coupled == self._coupled:
            return
        self._coupled = coupled
        for pair, p in self._pairs.items():
            if coupled:
                p.value, p.fixed = self._saved.pop(pair, (0.0, False))
            else:
                self._saved[pair] = (p.value, p.fixed)
                p.value, p.fixed = 0.0, True

    @property
    def couplings(self) -> dict[tuple[int, int], float]:
        """The user's delta of every pair (meV), also while uncoupled."""
        if self._coupled:
            return {pair: p.value for pair, p in self._pairs.items()}
        return {pair: self._saved[pair][0] for pair in self._pairs}

    @property
    def branches(self) -> list[Branch]:
        """The branches with their current e0 and g."""
        values = param_values(self.params)
        return [
            Branch(values[f"e0_{i}"], values[f"g_{i}"], b.m, b.form, b.label)
            for i, b in enumerate(self._structure)
        ]

    def add_branch(self, branch: Branch) -> None:
        """Append *branch*, with zero couplings to the existing ones."""
        i = len(self._structure)
        self._structure.append(Branch(branch.e0, branch.g, branch.m, branch.form, branch.label))
        lo = 0.0 if Form(branch.form) is Form.HYPERBOLIC else -np.inf
        e0 = Param(f"e0_{i}", float(branch.e0), lo=lo, kind="energy")
        g = Param(f"g_{i}", float(branch.g), kind="dimensionless")
        self.params[2 * i : 2 * i] = [e0, g]
        for j in range(i):
            p = Param(f"delta_{j}_{i}", 0.0, kind="energy")
            self._pairs[(j, i)] = p
            if not self._coupled:
                self._saved[(j, i)] = (0.0, False)
                p.fixed = True
        self._order_params()

    def remove_branch(self, index: int) -> None:
        """Remove branch *index* and its couplings; later branches move down by one."""
        n = len(self._structure)
        if not 0 <= index < n:
            raise IndexError(f"no branch {index}")
        del self._structure[index]
        del self.params[2 * index : 2 * index + 2]

        def shift(k: int) -> int:
            return k - 1 if k > index else k

        pairs, saved = {}, {}
        for (i, j), p in self._pairs.items():
            if index in (i, j):
                continue
            pair = (shift(i), shift(j))
            pairs[pair] = p
            if (i, j) in self._saved:
                saved[pair] = self._saved[(i, j)]
        self._pairs, self._saved = pairs, saved
        for k in range(n - 1):
            self.params[2 * k].name, self.params[2 * k + 1].name = f"e0_{k}", f"g_{k}"
        for (i, j), p in self._pairs.items():
            p.name = f"delta_{i}_{j}"
        self._order_params()

    def branch_names(self) -> list[str]:
        if self._coupled and len(self._structure) > 1:
            return [f"mode {k + 1}" for k in range(len(self._structure))]
        return [b.label or f"branch {i + 1}" for i, b in enumerate(self._structure)]

    def evaluate(self, field: np.ndarray, values: Mapping[str, float] | None = None) -> np.ndarray:
        """Energies (meV), shape ``(n_branch, n_field)``; sorted ascending when coupled."""
        field = np.asarray(field, dtype=float)
        vals = param_values(self.params, values)
        energies = self._uncoupled(field, vals)
        if not self._coupled or len(self._structure) < 2:
            return energies
        return np.moveaxis(np.linalg.eigvalsh(self._hamiltonian(energies, vals)), -1, 0)

    def branch_weights(
        self, field: np.ndarray, values: Mapping[str, float] | None = None
    ) -> np.ndarray:
        """Weight of uncoupled branch *i* in mode *k*, shape ``(n_mode, n_branch, n_field)``.

        Weights of a mode sum to 1; uncoupled, mode *k* is branch *k*.
        """
        field = np.asarray(field, dtype=float)
        n = len(self._structure)
        if not self._coupled or n < 2:
            eye = np.eye(n).reshape((n, n) + (1,) * field.ndim)
            return np.broadcast_to(eye, (n, n, *field.shape)).copy()
        vals = param_values(self.params, values)
        _, vectors = np.linalg.eigh(self._hamiltonian(self._uncoupled(field, vals), vals))
        return np.moveaxis(vectors**2, (-1, -2), (0, 1))

    def _uncoupled(self, field: np.ndarray, vals: Mapping[str, float]) -> np.ndarray:
        rows = [
            branch_energy(field, vals[f"e0_{i}"], vals[f"g_{i}"], b.m, b.form)
            for i, b in enumerate(self._structure)
        ]
        return np.stack(rows) if rows else np.empty((0, *field.shape))

    def _hamiltonian(self, energies: np.ndarray, vals: Mapping[str, float]) -> np.ndarray:
        """``diag(E_i(B)) + Delta`` for every field, shape ``(*field.shape, n, n)``."""
        n = energies.shape[0]
        h = np.zeros((*energies.shape[1:], n, n))
        idx = np.arange(n)
        h[..., idx, idx] = np.moveaxis(energies, 0, -1)
        for (i, j), p in self._pairs.items():
            h[..., i, j] = h[..., j, i] = vals[p.name]
        return h

    def _pair(self, i: int, j: int) -> tuple[int, int]:
        pair = (min(i, j), max(i, j))
        if pair not in self._pairs:
            raise ValueError(f"no coupling between branches {i} and {j}")
        return pair

    def _order_params(self) -> None:
        n = len(self._structure)
        deltas = [self._pairs[pair] for pair in sorted(self._pairs)]
        self.params[:] = self.params[: 2 * n] + deltas
