"""nilss_jax.fd: the finite-difference reference."""
import numpy as np
import pytest
import jax.numpy as jnp

from nilss_jax import fd
from nilss_jax.systems import lorenz63


def _oscillator(u, p):
    return jnp.array([p['w'] * u[1], -p['w'] * u[0]])


def _energy(u):
    return u[0] ** 2 + u[1] ** 2


def test_time_average_of_an_oscillator():
    parts = fd.time_average(_oscillator, lambda u: u[0] ** 2, {'w': 1.0}, [[1.0, 0.0], [0.0, 2.0]], T=20 * np.pi, dt=0.01, parts=10)
    assert parts.shape == (2, 10)
    np.testing.assert_allclose(parts.mean(axis=1), [0.5, 2.0], rtol=2e-2)


def test_fit_slope_recovers_a_line():
    v = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    slope, err, chi2, dof = fd.fit_slope(v, 2.0 * v + 1.0, np.full(5, 0.1))
    assert slope == pytest.approx(2.0) and chi2 < 1e-12 and dof == 3 and err == pytest.approx(0.1 / np.sqrt(10.0))


def test_finite_difference_of_a_smooth_response():
    # <u0^2 + u1^2> / 2 of the oscillator started at fixed radius r is r^2 -- a parameter that enters the initial condition
    # through `project` has the slope 2 r. Here the parameter is the radius itself.
    project = lambda u, p: p['r'] * np.array(u) / np.linalg.norm(u)
    res = fd.finite_difference(_oscillator, _energy, {'w': 1.0, 'r': 2.0}, 'r', 0.1, [[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]],
                               T=10.0, dt=0.01, npoint=3, project=project, parts=5)
    assert res.slope == pytest.approx(4.0, rel=1e-3) and len(res.values) == 3 and 'finite differences in r' in res.summary()
    with pytest.raises(KeyError):
        fd.finite_difference(_oscillator, _energy, {'w': 1.0}, 'nope', 0.1, [[1.0, 0.0]], 1.0, 0.01)


def test_select_drops_copies():
    sel = lambda parts: np.arange(len(parts)) < 2
    res = fd.finite_difference(_oscillator, _energy, {'w': 1.0, 'r': 2.0}, 'r', 0.1, [[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]], T=2.0,
                               dt=0.01, npoint=3, project=lambda u, p: p['r'] * np.array(u), parts=2, select=sel)
    assert list(res.n_copies) == [2, 2, 2]


def test_lorenz_finite_difference_agrees_with_the_published_sensitivity():
    u0s = np.array([lorenz63.initial_condition(i) for i in range(24)])
    res = fd.finite_difference(lorenz63.rhs, lorenz63.J, lorenz63.DEFAULTS, 'rho', 1.0, u0s, T=400.0, dt=0.01, npoint=3, T_spinup=20.0)
    assert res.slope == pytest.approx(1.0, abs=0.15)
