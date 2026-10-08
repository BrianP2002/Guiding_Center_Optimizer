"""Guiding-center model: parameter table, derivatives, integrator, tangent equations."""
import numpy as np
import pytest
import jax
import jax.numpy as jnp

import app_gc
from app_gc import default_params, PARAM_BOUNDS, canon_par

STATES = [np.array([0.05, 1.0, 2.0]), np.array([0.1, 4.0, 0.5]), np.array([0.02, 3.0, 5.5])]


def test_param_table_is_consistent():
    # the 'lam' / 'lambda' mismatch between the scripts made one parameter silently useless
    assert set(PARAM_BOUNDS) == set(default_params)
    for name, (lo, hi) in PARAM_BOUNDS.items():
        assert lo < default_params[name] < hi
    assert canon_par('lambda') == 'lam'
    with pytest.raises(KeyError):
        canon_par('bogus')


def test_ddt_rejects_unknown_parameter():
    u = jnp.array(STATES[0])
    with pytest.raises(KeyError):
        app_gc.ddt((u, jnp.zeros((1, 3)), jnp.zeros(3)), default_params, 'bogus')


@pytest.mark.parametrize('u', STATES)
def test_jacobian_matches_finite_difference(u):
    Df = np.asarray(jax.jacobian(app_gc.f_ode_wrapper, argnums=0)(u, default_params))
    eps = 1e-6
    fd = np.stack([(np.asarray(app_gc.f_ode_wrapper(u + eps * e, default_params))
                    - np.asarray(app_gc.f_ode_wrapper(u - eps * e, default_params))) / (2 * eps)
                   for e in np.eye(3)], axis=1)
    assert np.max(np.abs(Df - fd)) < 1e-6 * np.max(np.abs(Df))


@pytest.mark.parametrize('par', sorted(default_params))
@pytest.mark.parametrize('u', STATES)
def test_dfdpar_matches_finite_difference(u, par):
    u = jnp.array(u)
    # with w = vstar = 0 the inhomogeneous tangent rate is exactly df/dpar
    _, _, dfdpar = app_gc.ddt((u, jnp.zeros((1, 3)), jnp.zeros(3)), default_params, par)
    eps = 1e-6
    def f_at(p):
        return np.asarray(app_gc.f_ode_wrapper(u, {**default_params, par: p}))
    fd = (f_at(default_params[par] + eps) - f_at(default_params[par] - eps)) / (2 * eps)
    assert np.max(np.abs(np.asarray(dfdpar) - fd)) < 1e-6 * np.max(np.abs(fd))


def _integrate(u0, w0, v0, par, s, dt, nstep):
    uwv = (jnp.array(u0), jnp.array(w0), jnp.array(v0))
    params = {**default_params, par: s}
    for _ in range(nstep):
        uwv = app_gc.RK4(uwv, params, par, dt)
    return [np.asarray(a) for a in uwv]


def test_rk4_is_fourth_order():
    u0, w0, v0, T = STATES[0], np.eye(3)[:1], np.zeros(3), 0.05
    ref = _integrate(u0, w0, v0, 'a0', 0.1, T / 320, 320)
    errs = []
    for n in (10, 20, 40):
        out = _integrate(u0, w0, v0, 'a0', 0.1, T / n, n)
        errs.append(max(np.max(np.abs(a - r)) for a, r in zip(out, ref)))
    for coarse, fine in zip(errs[:-1], errs[1:]):
        assert 12 < coarse / fine < 20, errs


@pytest.mark.parametrize('par', sorted(default_params))
def test_inhomogeneous_tangent_matches_trajectory_difference(par):
    u0, T, dt, eps = STATES[0], 0.2, 1e-3, 1e-6
    n = int(round(T / dt))
    s = default_params[par]
    _, _, vstar = _integrate(u0, np.zeros((1, 3)), np.zeros(3), par, s, dt, n)
    up = _integrate(u0, np.zeros((1, 3)), np.zeros(3), par, s + eps, dt, n)[0]
    um = _integrate(u0, np.zeros((1, 3)), np.zeros(3), par, s - eps, dt, n)[0]
    assert np.max(np.abs(vstar - (up - um) / (2 * eps))) < 1e-6 * np.max(np.abs(vstar))


def test_homogeneous_tangent_matches_trajectory_difference():
    u0, T, dt, eps = STATES[1], 0.2, 1e-3, 1e-6
    n = int(round(T / dt))
    q = np.array([[0.3, -0.5, 0.8]])
    _, w, _ = _integrate(u0, q, np.zeros(3), 'a0', 0.1, dt, n)
    up = _integrate(u0 + eps * q[0], q, np.zeros(3), 'a0', 0.1, dt, n)[0]
    um = _integrate(u0 - eps * q[0], q, np.zeros(3), 'a0', 0.1, dt, n)[0]
    assert np.max(np.abs(w[0] - (up - um) / (2 * eps))) < 1e-6 * np.max(np.abs(w))


def test_float64_is_enabled():
    # float32 made the xi = 0 checks and the QR of the tangents unreliable
    assert app_gc.f_ode_wrapper(jnp.array(STATES[0]), default_params).dtype == jnp.float64
