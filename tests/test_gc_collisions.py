"""experiments/scripts/gc_collisions.py: the official guiding-center flow with pitch-angle scattering."""
import numpy as np
import pytest
import jax
import jax.numpy as jnp

from nilss_jax.systems import guiding_center as go
import gc_collisions as gcc          # experiments/scripts/gc_collisions.py (on the pytest path)

P = {**go.default_params, 'eps_h': 0.12, 'N': 2.0, 'eps_t': 0.03, 'eps_m': 0.02, 'eps_2': 0.01, 'N2': 4.0,
     'iota': 0.78, 'kappa': 0.16, 'lam': 1.0}
UNIFORM = {**go.default_params, 'eps_h': 0.0, 'eps_t': 0.0, 'eps_m': 0.0, 'eps_2': 0.0, 'iota': 0.78, 'kappa': 0.16}


def _final_xi(p, xi0, nu, dt, nstep, n, seed=0, coupling='rotation'):
    u0 = gcc.initial_state(0.1, 1.0, 2.0, xi0)
    keys = gcc.path_keys(jax.random.PRNGKey(seed), np.arange(n))
    fn = jax.jit(jax.vmap(lambda k: gcc.simulate(u0, k, p, nu, dt, nstep, coupling=coupling)))
    return np.asarray(fn(keys))


def test_equations_are_gc_official_rhs4_on_the_energy_shell():
    rng = np.random.RandomState(0)
    for _ in range(20):
        u3 = jnp.array([0.01 + 0.3 * rng.rand(), 2 * np.pi * rng.rand(), 2 * np.pi * rng.rand()])
        xi = float(rng.uniform(-1, 1))
        lam = (1 - xi ** 2) / float(go.B_derivs(*u3, P)[0])
        a = np.asarray(gcc.rhs(gcc.initial_state(*u3, xi), P))
        b = np.asarray(go.rhs4(jnp.array([*u3, xi]), {**P, 'lam': lam}))
        assert np.allclose(a[:3], b[:3], rtol=1e-13, atol=1e-15)
        assert a[5] == pytest.approx(b[3], rel=1e-13, abs=1e-15)
        assert np.all(np.isfinite(a))
        assert abs(float(jnp.sum(gcc.initial_state(*u3, xi)[3:] * a[3:]))) < 1e-14      # d|n|/dt = 0


def test_without_collisions_the_orbit_is_the_orbit_of_gc_official():
    s0, th0, ze0, xi0 = 0.1, 1.0, 2.0, 0.3
    lam = (1 - xi0 ** 2) / float(go.B_derivs(s0, th0, ze0, P)[0])
    dt, nstep = 2e-3, 5000
    ref = np.asarray(go.trajectory(go.rhs4, jnp.array([s0, th0, ze0, xi0]), {**P, 'lam': lam}, dt, nstep, nstep))[-1]
    u = np.asarray(gcc.simulate(gcc.initial_state(s0, th0, ze0, xi0), None, P, 0.0, dt, nstep, noise=False))
    assert np.max(np.abs(u[[0, 1, 2, 5]] - ref)) < 1e-8


def test_mu_is_conserved_without_collisions_and_changed_by_them():
    u0 = gcc.initial_state(0.1, 1.0, 2.0, 0.3)
    u = gcc.simulate(u0, None, P, 0.0, 0.02, 5000, noise=False)
    assert abs(float(gcc.mu(u, P) / gcc.mu(u0, P) - 1.0)) < 1e-8
    u = gcc.simulate(u0, jax.random.PRNGKey(1), P, 0.05, 0.05, 500)
    assert abs(float(gcc.mu(u, P) / gcc.mu(u0, P) - 1.0)) > 1e-3


@pytest.mark.slow
@pytest.mark.parametrize('coupling', ['rotation', 'gradient'])
def test_lorentz_moments_in_a_uniform_field(coupling):
    # d xi = -nu xi dt + sqrt(nu (1 - xi^2)) dW:  E xi = xi0 exp(-nu t),  E xi^2 = 1/3 + (xi0^2 - 1/3) exp(-3 nu t)
    xi0, nu, dt, nstep, n = 0.6, 0.05, 0.05, 500, 12000
    t = dt * nstep
    xi = _final_xi(UNIFORM, xi0, nu, dt, nstep, n, coupling=coupling)[:, 5]
    for got, want, sem in ((xi.mean(), xi0 * np.exp(-nu * t), xi.std() / np.sqrt(n)),
                           ((xi ** 2).mean(), 1 / 3 + (xi0 ** 2 - 1 / 3) * np.exp(-3 * nu * t), (xi ** 2).std() / np.sqrt(n))):
        assert abs(got - want) < 4 * sem + 2e-4, (got, want, sem)
    # long times: the uniform distribution on [-1, 1]: E xi^2 = 1/3, E xi^4 = 1/5
    xi = _final_xi(UNIFORM, xi0, 0.4, dt, 500, n, seed=1, coupling=coupling)[:, 5]
    assert abs((xi ** 2).mean() - 1 / 3) < 4 * (xi ** 2).std() / np.sqrt(n)
    assert abs((xi ** 4).mean() - 1 / 5) < 4 * (xi ** 4).std() / np.sqrt(n)


def test_pitch_stays_in_range_and_n_stays_a_unit_vector():
    u = _final_xi(P, 0.9999, 1.0, 0.05, 500, 200)
    assert np.all(np.isfinite(u))
    assert np.max(np.abs(u[:, 5])) <= 1.0 + 1e-12
    assert np.max(np.abs(np.linalg.norm(u[:, 3:], axis=1) - 1.0)) < 1e-12


def test_collision_step_is_an_isometry_of_the_sphere():
    rng = np.random.RandomState(2)
    u = np.concatenate([[0.1, 1.0, 2.0], np.array([0.3, -0.4, 0.0])])
    u[3:] = rng.randn(3)
    u[3:] /= np.linalg.norm(u[3:])
    u = jnp.array(u)
    a = jnp.array(0.2 * rng.randn(3))
    J = np.asarray(jax.jacfwd(lambda x: gcc.collide(x, a))(u))[3:, 3:]
    n = np.asarray(u[3:])
    for _ in range(5):
        w = rng.randn(3)
        w -= (w @ n) * n
        assert np.linalg.norm(J @ w) == pytest.approx(np.linalg.norm(w), rel=1e-12)


def test_make_ics():
    p0 = {**P, 'lam': 1.03}
    u = gcc.make_ics(500, p0)
    B = np.array([float(go.B_derivs(*r[:3], p0)[0]) for r in u])
    assert u.shape == (500, 6) and np.all((u[:, 0] >= 0.02) & (u[:, 0] <= 0.2))
    assert np.all(1.0 - p0['lam'] * B >= 0.02 - 1e-12)
    assert np.allclose(u[:, 5] ** 2, 1.0 - p0['lam'] * B, atol=1e-12)
    assert np.allclose(np.linalg.norm(u[:, 3:], axis=1), 1.0, atol=1e-12)
    assert 0.3 < np.mean(u[:, 5] > 0) < 0.7
    assert np.array_equal(u[:100], gcc.make_ics(100, p0))              # a prefix of a longer set


@pytest.mark.parametrize('nu, coupling', [(0.0, 'rotation'), (0.02, 'rotation'), (0.02, 'gradient')])
def test_crn_pathwise_derivative_matches_finite_differences(nu, coupling):
    ics = gcc.make_ics(12, {**P, 'lam': 1.03})
    keys = gcc.path_keys(jax.random.PRNGKey(3), np.arange(12))
    out = gcc.run_paths(ics, keys, P, 0.03, nu, 0.05, [50.0, 100.0], tangent=True, coupling=coupling)
    h = 1e-7                                                        # (the derivative of a chaotic path has a large curvature)
    up = gcc.run_paths(ics, keys, P, 0.03 + h, nu, 0.05, [50.0, 100.0], coupling=coupling)
    um = gcc.run_paths(ics, keys, P, 0.03 - h, nu, 0.05, [50.0, 100.0], coupling=coupling)
    for name, dname in (('s', 'ds'), ('xi', 'dxi')):
        fd = (up[name] - um[name]) / (2 * h)
        assert np.max(np.abs(out[dname] - fd)) < 1e-6 * max(np.max(np.abs(fd)), 1e-3)
        assert np.max(np.abs(out[dname])) > 1e-4                    # the check has teeth
    plain = gcc.run_paths(ics, keys, P, 0.03, nu, 0.05, [50.0, 100.0], coupling=coupling)
    assert np.allclose(out['s'], plain['s'], atol=1e-10)             # same primal path with and without the tangent
    assert np.all(out['s_min'] <= out['s'].min(axis=0) + 1e-12)


def test_noise_is_a_function_of_the_key_only():
    ics = gcc.make_ics(6, {**P, 'lam': 1.03})
    k1 = gcc.path_keys(jax.random.PRNGKey(5), np.arange(6))
    k2 = gcc.path_keys(jax.random.PRNGKey(6), np.arange(6))
    a = gcc.run_paths(ics, k1, P, 0.03, 0.02, 0.05, [100.0])
    b = gcc.run_paths(ics, k1, P, 0.03, 0.02, 0.05, [100.0])
    c = gcc.run_paths(ics, k2, P, 0.03, 0.02, 0.05, [100.0])
    assert np.array_equal(a['s'], b['s'])
    assert np.max(np.abs(a['xi'] - c['xi'])) > 1e-3


def test_the_two_couplings_have_the_same_law_but_different_pairs_of_paths():
    # the collision step of the gradient coupling is not an isometry (the rotation coupling is)
    rng = np.random.RandomState(7)
    u = np.concatenate([[0.1, 1.0, 2.0], rng.randn(3)])
    u[3:] /= np.linalg.norm(u[3:])
    u = jnp.array(u)
    z = jnp.array(rng.randn(3))
    sq = 0.3
    J = np.asarray(jax.jacfwd(lambda x: gcc.normalise(jnp.concatenate(
        [x[:3], x[3:] + sq * (z - (z @ x[3:]) * x[3:])])))(u))[3:, 3:]
    n = np.asarray(u[3:])
    w = rng.randn(3)
    w -= (w @ n) * n
    assert abs(np.linalg.norm(J @ w) / np.linalg.norm(w) - 1.0) > 1e-3
