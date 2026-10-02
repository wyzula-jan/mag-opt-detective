import numpy as np
import pytest
from scipy.constants import e, hbar

from mag_opt_detective.core.models import dirac_interband, dirac_level


def test_graphene_like_cyclotron_energy():
    # v = 10^6 m/s, massless: E_1(1 T) = sqrt(2 e hbar v^2 B) = 36.3 meV
    expected = np.sqrt(2 * e * hbar * (1e6) ** 2 * 1.0) / e * 1000
    assert dirac_level(1, np.array([1.0]), velocity=10.0, delta=0.0)[0] == pytest.approx(expected)
    assert expected == pytest.approx(36.3, abs=0.1)


def test_interband_limits():
    field = np.array([0.0, 1.0, 4.0])
    lines = dirac_interband(field, velocity=5.0, delta=10.0, n_lines=3)
    assert lines.shape == (3, 3)
    np.testing.assert_allclose(lines[:, 0], 20.0)  # B = 0 gives 2 delta
    massless = dirac_interband(field, velocity=5.0, delta=0.0, n_lines=3)
    np.testing.assert_allclose(massless[:, 2], 2 * massless[:, 1])  # sqrt(B) scaling
    # n = 0 is L0 -> L1: delta + E_1
    np.testing.assert_allclose(lines[0], 10.0 + dirac_level(1, field, 5.0, 10.0))
    with pytest.raises(ValueError):
        dirac_interband(field, velocity=-1.0, delta=0.0, n_lines=1)
