"""The official vacuum Boozer guiding-center system (gc_official.py) and its relation to the model of app_gc.py."""
import numpy as np
import pytest
import jax
import jax.numpy as jnp

import app_gc
from nilss_jax.systems import guiding_center as go

# all symmetry-breaking terms on, N = 4, passing particle
P_FULL = {**go.default_params, 'eps_h': 0.1, 'eps_t': 0.05, 'eps_m': 0.03, 'eps_2': 0.04, 'N': 4.0, 'N2': 3.0,
          'iota': 1.2, 'lam': 0.4, 'kappa': 0.5}
U3 = jnp.array([0.1, 1.0, 2.0])


def _u4(u3, p):
    return jnp.concatenate([u3, go.pitch_vpar(u3, p)[None]])


def test_param_table_has_one_entry_per_parameter():
    assert set(go.default_params) == {'eps_h', 'N', 'eps_t', 'eps_m', 'eps_2', 'N2', 'iota', 'G', 'lam', 'kappa'}
    with pytest.raises(KeyError):
        go.ddt((U3, jnp.zeros((1, 3)), jnp.zeros(3)), go.default_params, 'bogus')


def test_reduces_to_the_model_of_app_gc():
    rng = np.random.RandomState(0)
    p = go.teacher_params(**{k: app_gc.default_params[k] for k in ('a0', 'a1', 'iota', 'G', 'lam')})
    assert p['kappa'] == pytest.approx(2.0 / app_gc.default_params['lam'])
    for _ in range(10):
        u = jnp.array([0.01 + 0.3 * rng.rand(), 2 * np.pi * rng.rand(), 2 * np.pi * rng.rand()])
        a, b = np.asarray(go.rhs3(u, p)), np.asarray(app_gc.f_ode_wrapper(u, app_gc.default_params))
        assert np.max(np.abs(a - b)) < 1e-12 * np.max(np.abs(b))


def test_B_derivatives_match_autodiff():
    s, th, ze = 0.07, 1.3, 2.9
    B, B_s, B_th, B_ze = go.B_derivs(s, th, ze, P_FULL)
    ref = jax.grad(lambda x: go.B_derivs(x[0], x[1], x[2], P_FULL)[0])(jnp.array([s, th, ze]))
    assert np.allclose([B_s, B_th, B_ze], np.asarray(ref), rtol=1e-12)


def test_energy_is_conserved_and_3d_flow_is_the_reduction_of_the_4d_flow():
    u4 = _u4(U3, P_FULL)
    n = 10000                                              # T = 50
    tr4 = np.asarray(go.trajectory(go.rhs4, u4, P_FULL, 0.005, n, 100))
    tr3 = np.asarray(go.trajectory(go.rhs3, U3, P_FULL, 0.005, n, 100))
    E = np.array([float(go.energy(jnp.array(x), P_FULL)) for x in tr4])
    assert np.max(np.abs(E - E[0])) < 1e-10
    assert np.max(np.abs(tr4[:, :3] - tr3)) < 1e-8
    v_energy = np.array([float(go.pitch_vpar(jnp.array(x[:3]), P_FULL)) for x in tr4])
    assert np.max(np.abs(tr4[:, 3] - v_energy)) < 1e-8


def test_quasi_helical_invariant():
    # B depends on theta - N zeta only: G v_par / B - (iota - N) s / kappa is conserved ...
    p = {**P_FULL, 'eps_t': 0.0, 'eps_m': 0.0, 'eps_2': 0.0}
    tr = np.asarray(go.trajectory(go.rhs4, _u4(U3, p), p, 0.005, 10000, 100))
    ph = np.array([float(go.phi_qh(x[0], x[1], x[2], x[3], p)) for x in tr])
    assert np.max(np.abs(ph - ph[0])) < 1e-10
    # ... and it is not once a second helicity is present (the check has teeth)
    p = {**P_FULL, 'eps_m': 0.0, 'eps_2': 0.0}
    tr = np.asarray(go.trajectory(go.rhs4, _u4(U3, p), p, 0.005, 10000, 100))
    ph = np.array([float(go.phi_qh(x[0], x[1], x[2], x[3], p)) for x in tr])
    assert np.max(np.abs(ph - ph[0])) > 1e-3


def test_invariant_of_the_model_of_app_gc():
    # the model of app_gc.py: G V / B - (iota - a1) lam x / 2 is conserved, so its flow is integrable
    p = app_gc.default_params
    B = lambda u: app_gc.B_func(u[0], u[1], u[2], p['a0'], p['a1'])
    phi = lambda u: p['G'] * app_gc.V_func(p['lam'], B(u)) / B(u) - (p['iota'] - p['a1']) * p['lam'] * u[0] / 2
    f = lambda u, _: app_gc.f_ode_wrapper(u, p)
    tr = np.asarray(go.trajectory(f, jnp.array([0.05, 1.0, 2.0]), None, 0.002, 5000, 50))
    ph = np.array([float(phi(jnp.array(x))) for x in tr])
    assert np.max(np.abs(ph - ph[0])) < 1e-10
    assert tr[:, 0].max() - tr[:, 0].min() > 1e-3          # x does move


@pytest.mark.parametrize('params', [P_FULL, go.teacher_params()])
def test_liouville_density_of_the_3d_flow(params):
    # the flow is Hamiltonian: 1 / (B v_par) is an invariant density on the energy shell, div(rho f) = 0
    # (so the flow is conservative, its long-time averages are integrals against rho, and it has no attractor)
    def rho_f(u):
        B = go.B_derivs(u[0], u[1], u[2], params)[0]
        return go.rhs3(u, params) / (B * jnp.sqrt(1.0 - params['lam'] * B))
    rng = np.random.RandomState(1)
    for _ in range(5):
        u = jnp.array([0.02 + 0.3 * rng.rand(), 2 * np.pi * rng.rand(), 2 * np.pi * rng.rand()])
        div = float(jnp.trace(jax.jacfwd(rho_f)(u)))
        assert abs(div) < 1e-10 * float(jnp.max(jnp.abs(jax.jacfwd(rho_f)(u))))


def test_trapped_particle_bounces_in_the_4d_system():
    # a mirror field, lam between 1 / B_max and 1 / B_min: v_par changes sign, which rhs3 cannot represent
    p = {**go.default_params, 'eps_h': 0.0, 'eps_m': 0.1, 'N': 4.0, 'iota': 0.5, 'lam': 1 / 1.05, 'kappa': 0.01}
    u0 = jnp.array([0.1, 0.0, np.pi / 4, np.sqrt(1 - 0.9 / 1.05)])
    tr = np.asarray(go.trajectory(go.rhs4, u0, p, 0.005, 20000, 100))
    assert np.any(np.diff(np.sign(tr[:, 3])) != 0)
    E = np.array([float(go.energy(jnp.array(x), p)) for x in tr])
    assert np.max(np.abs(E - E[0])) < 1e-10


@pytest.mark.parametrize('par', sorted(go.default_params))
def test_tangent_equations_match_finite_differences(par):
    p = {**P_FULL, 'N2': 3.0}
    u, T, dt, eps = U3, 0.2, 1e-3, 1e-6
    n = int(round(T / dt))

    def run(u0, s):
        uwv = (jnp.array(u0), jnp.zeros((1, 3)), jnp.zeros(3))
        for _ in range(n):
            uwv = go.RK4(uwv, {**p, par: s}, par, dt)
        return [np.asarray(a) for a in uwv]
    vstar = run(u, p[par])[2]
    fd = (run(u, p[par] + eps)[0] - run(u, p[par] - eps)[0]) / (2 * eps)
    assert np.max(np.abs(vstar - fd)) < 1e-6 * max(np.max(np.abs(fd)), 1e-3)


def test_nilss_runs_on_the_official_system():
    from nilss_jax.reference import nilss
    integrator, fJJu = go.make_problem(1e-3, P_FULL)
    J, dJds = nilss(1e-3, 3, 0.05, 2, np.array([0.1, 1.0, 2.0]), 1, 'eps_t', 0.05, integrator, fJJu)
    assert np.isfinite(J) and np.isfinite(dJds)
    with pytest.raises(KeyError):
        integrator(np.zeros(3), np.zeros((1, 3)), np.zeros(3), 'bogus', 0.1)
