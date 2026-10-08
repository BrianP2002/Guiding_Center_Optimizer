"""The public API of nilss_jax.core: results, saving, input validation, higher-dimensional systems."""
import numpy as np
import pytest
import jax
import jax.numpy as jnp

from nilss_jax import NILSS, NILSSResult
from nilss_jax.systems import lorenz63

U0 = lorenz63.initial_condition(0)


def _nilss(**kw):
    return NILSS(lorenz63.rhs, lorenz63.J, kw.pop('params', ('rho',)), dt=0.005, T_seg=0.5, **kw)


def test_run_equals_run_segments_and_result_fields():
    nil = _nilss()
    res = nil.run(lorenz63.DEFAULTS, U0, T=10.0, T_spinup=2.0, seed=3)
    Javg, dJdp, info = nil.run_segments(lorenz63.DEFAULTS, U0, 20, 4, seed=3)
    assert isinstance(res, NILSSResult) and res.finite
    assert res.J == pytest.approx(Javg, rel=1e-12) and res.dJdp['rho'] == pytest.approx(dJdp[0], rel=1e-12)
    assert res.nseg == 20 and res.segments.shape == (20, 1) and res.vnorm.shape == (20, 1) and res.J_segments.shape == (20,)
    assert res.segments.sum() == pytest.approx(res.dJdp['rho'], rel=1e-12)
    assert res.T == pytest.approx(10.0) and res.params == ('rho',)
    assert res.J_segments.mean() == pytest.approx(res.J, rel=1e-12)
    assert 'NILSS run' in res.summary() and 'rho' in res.summary()


def test_result_save_load_roundtrip(tmp_path):
    res = _nilss(params=('rho', 'beta'), nus=1).run(lorenz63.DEFAULTS, U0, T=5.0, T_spinup=1.0)
    res.save(tmp_path / 'r.npz')
    back = NILSSResult.load(tmp_path / 'r.npz')
    assert back.params == res.params and back.dJdp == pytest.approx(res.dJdp) and back.finite
    np.testing.assert_array_equal(back.segments, res.segments)
    np.testing.assert_array_equal(back.vnorm, res.vnorm)


def test_short_run_warns_that_no_exponent_is_resolved():
    res = _nilss().run(lorenz63.DEFAULTS, U0, T=0.5)
    assert res.lyapunov_time_product < 3 and 'lambda_1 * T < 3' in res.summary()


def test_non_finite_run_is_reported_not_raised():
    res = _nilss().run({**lorenz63.DEFAULTS, 'sigma': 1e4}, U0, T=5.0)       # RK4 blows up
    assert not res.finite and np.isnan(res.J) and 'non-finite' in res.summary()
    assert res.vnorm.shape == (10, 1) and np.all(np.isnan(res.segments))


def test_input_validation():
    with pytest.raises(ValueError, match='non-empty'):
        NILSS(lorenz63.rhs, lorenz63.J, (), 0.01, 0.5)
    with pytest.raises(ValueError, match='distinct'):
        NILSS(lorenz63.rhs, lorenz63.J, ('rho', 'rho'), 0.01, 0.5)
    with pytest.raises(ValueError, match='dt'):
        NILSS(lorenz63.rhs, lorenz63.J, ('rho',), 0.01, 0.001)
    with pytest.raises(ValueError, match='nus'):
        NILSS(lorenz63.rhs, lorenz63.J, ('rho',), 0.01, 0.5, nus=0)
    with pytest.warns(UserWarning, match='multiple of dt'):
        NILSS(lorenz63.rhs, lorenz63.J, ('rho',), 0.01, 0.503)
    nil = _nilss()
    with pytest.raises(KeyError, match='rho'):
        nil.run({'sigma': 10.0, 'beta': 2.0}, U0, T=1.0)
    with pytest.raises(ValueError, match='scalar'):
        nil.run({**lorenz63.DEFAULTS, 'rho': np.array([1.0, 2.0])}, U0, T=1.0)
    with pytest.raises(ValueError, match='finite'):
        nil.run(lorenz63.DEFAULTS, np.array([np.nan, 0, 0]), T=1.0)
    with pytest.raises(ValueError, match='shorter than one segment'):
        nil.run(lorenz63.DEFAULTS, U0, T=0.1)
    with pytest.raises(ValueError, match='nus'):
        NILSS(lorenz63.rhs, lorenz63.J, ('rho',), 0.01, 0.5, nus=3).run(lorenz63.DEFAULTS, U0, T=1.0)
    bad = NILSS(lambda u, p: u[:2], lorenz63.J, ('rho',), 0.01, 0.5)
    with pytest.raises(ValueError, match='shape'):
        bad.run(lorenz63.DEFAULTS, U0, T=1.0)


def _lorenz96(u, p):
    return (jnp.roll(u, -1) - jnp.roll(u, 2)) * jnp.roll(u, 1) - u + p['F']


def _kinetic(u):
    return jnp.mean(u ** 2)


def test_eight_dimensional_system_matches_the_dense_reference():
    # Lorenz 96 with 8 variables, F = 8, two homogeneous tangents: the Jacobian-vector products of the core give the same J and
    # dJ/dF as the dense reference implementation fed with full Jacobians (jax.jacfwd)
    from nilss_jax.reference import nilss as reference
    dt, T_seg, nseg = 0.01, 0.5, 6
    u0 = 8.0 + 0.5 * np.random.RandomState(1).randn(8)
    w0 = np.random.RandomState(2).rand(2, 8)
    jac = jax.jit(jax.jacfwd(_lorenz96))

    @jax.jit
    def ddt(u, w, v, F):
        pp = {'F': F}
        Df = jac(u, pp)
        dfdF = jax.jacfwd(lambda x: _lorenz96(u, {'F': x}))(F)
        return _lorenz96(u, pp), (Df @ w.T).T, Df @ v + dfdF

    def integrator(u, w, vstar, par, s):
        y = tuple(jnp.asarray(a) for a in (u, w, vstar))
        k0 = ddt(*y, s)
        k1 = ddt(*(a + 0.5 * dt * k for a, k in zip(y, k0)), s)
        k2 = ddt(*(a + 0.5 * dt * k for a, k in zip(y, k1)), s)
        k3 = ddt(*(a + dt * k for a, k in zip(y, k2)), s)
        return tuple(np.asarray(a + dt / 6.0 * (c0 + 2 * c1 + 2 * c2 + c3)) for a, c0, c1, c2, c3 in zip(y, k0, k1, k2, k3))
    integrator.dt = dt

    def fJJu(u, par, s):
        u = jnp.asarray(u)
        return np.asarray(_lorenz96(u, {'F': s})), float(_kinetic(u)), np.asarray(jax.grad(_kinetic)(u))

    J_ref, dJ_ref = reference(dt, nseg, T_seg, 0, u0, 2, 'F', 8.0, integrator, fJJu, w0=w0)
    Javg, dJdp, info = NILSS(_lorenz96, _kinetic, ('F',), dt, T_seg, nus=2).run_segments({'F': 8.0}, u0, nseg, 0, w0=w0)
    assert Javg == pytest.approx(J_ref, rel=1e-10)
    assert dJdp[0] == pytest.approx(dJ_ref, rel=1e-7)
