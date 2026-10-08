"""nilss_stream.py (streaming, multi-parameter NILSS) against nilss.py, which is validated on Lorenz 63."""
import numpy as np
import pytest
import jax
import jax.numpy as jnp

from nilss_jax.systems import lorenz63
from nilss_jax.systems import guiding_center as go
from nilss_jax.reference import nilss, solve_shadowing_coeffs
from nilss_jax.core import NILSSStreamer, solve_shadowing_coeffs_multi


def lorenz_rhs(u, p):
    return jnp.array([p['sigma'] * (u[1] - u[0]), u[0] * (p['rho'] - u[2]) - u[1], u[0] * u[1] - p['beta'] * u[2]])


def lorenz_J(u):
    return u[2]


U0 = [1.0, 1.0, 20.0]
W0 = np.random.RandomState(0).rand(2, 3)


@pytest.mark.parametrize('par', ['rho', 'sigma', 'beta'])
def test_matches_nilss_on_lorenz(par):
    dt, T_seg, nseg = 0.005, 0.5, 30
    integrator, fJJu = lorenz63.make_problem(dt)
    J_old, dJ_old, info_old = nilss(dt, nseg, T_seg, 0, U0, 2, par, lorenz63.DEFAULTS[par], integrator, fJJu, w0=W0, return_info=True)
    streamer = NILSSStreamer(lorenz_rhs, lorenz_J, (par,), dt, T_seg, 2)
    J_new, dJ_new, info_new = streamer.run_segments(lorenz63.DEFAULTS, U0, nseg, 0, w0=W0)
    assert J_new == pytest.approx(J_old, rel=1e-10)
    assert dJ_new[0] == pytest.approx(dJ_old, rel=1e-7)
    assert np.allclose(info_new['lyap'], info_old['lyap'], rtol=1e-7)
    assert np.allclose(info_new['dJdp_segments'][:, 0], info_old['dJds_segments'], rtol=1e-6, atol=1e-10)


def test_parameters_share_the_homogeneous_tangents():
    dt, T_seg, nseg = 0.005, 0.5, 30
    together = NILSSStreamer(lorenz_rhs, lorenz_J, ('rho', 'sigma', 'beta'), dt, T_seg, 2).run_segments(lorenz63.DEFAULTS, U0, nseg, 3, seed=5)[1]
    for k, par in enumerate(('rho', 'sigma', 'beta')):
        alone = NILSSStreamer(lorenz_rhs, lorenz_J, (par,), dt, T_seg, 2).run_segments(lorenz63.DEFAULTS, U0, nseg, 3, seed=5)[1]
        assert together[k] == pytest.approx(alone[0], rel=1e-9)


def test_matches_nilss_on_the_guiding_center_flow():
    p = {**go.default_params, 'eps_t': 0.03, 'eps_m': 0.02, 'kappa': 0.4, 'lam': 0.3, 'N': 4.0, 'iota': 1.1}
    dt, T_seg, nseg = 0.005, 0.5, 12
    u0 = np.array([0.1, 1.0, 2.0])
    w0 = np.random.RandomState(1).rand(1, 3)
    integrator, fJJu = go.make_problem(dt, p)
    J_old, dJ_old = nilss(dt, nseg, T_seg, 0, u0, 1, 'eps_t', p['eps_t'], integrator, fJJu, w0=w0)
    J_new, dJ_new, _ = NILSSStreamer(go.rhs3, lambda u: u[0], ('eps_t',), dt, T_seg, 1).run_segments(p, u0, nseg, 0, w0=w0)
    assert J_new == pytest.approx(J_old, rel=1e-10)
    assert dJ_new[0] == pytest.approx(dJ_old, rel=1e-6)


def test_sparse_schur_solve_matches_the_dense_one():
    rng = np.random.RandomState(3)
    nseg, nus, npar = 7, 2, 3
    Cs = np.array([(lambda A: A @ A.T + np.eye(nus))(rng.randn(nus, nus)) for _ in range(nseg)])
    Ds, Rs, Bs = rng.randn(nseg, nus, npar), rng.randn(nseg - 1, nus, nus), rng.randn(nseg - 1, nus, npar)
    a = solve_shadowing_coeffs_multi(Cs, Ds, Rs, Bs)
    for k in range(npar):
        ref = solve_shadowing_coeffs(list(Cs), list(Ds[:, :, k]), list(Rs), list(Bs[:, :, k]))
        assert np.allclose(a[:, :, k], ref, rtol=1e-9, atol=1e-11)
        # the constraints a_{i+1} = R_i a_i + b_i hold
        assert np.allclose(a[1:, :, k], np.einsum('ijk,ik->ij', Rs, a[:-1, :, k]) + Bs[:, :, k], atol=1e-9)


def test_invariant_surface_is_kept_by_the_tangent_equations():
    p = {**go.default_params, 'eps_h': 0.1, 'eps_t': 0.04, 'N': 4.0, 'iota': 1.1, 'lam': 0.9, 'kappa': 0.3}
    u0 = np.array([0.1, 1.0, 2.0, 0.3])
    u0[3] = np.sqrt(1.0 - p['lam'] * float(go.B_derivs(*u0[:3], p)[0]))
    streamer = NILSSStreamer(go.rhs4, lambda u: u[0], ('eps_t', 'kappa'), 0.005, 5.0, 1, invariant=go.energy)
    W, V = streamer._start(jnp.array(u0), jnp.array(np.random.RandomState(2).rand(1, 4)), p)
    u = jnp.array(u0)

    def constraint_residuals(u, W, V):
        g = jax.grad(go.energy, 0)(u, p)
        dEdp = np.array([float(jax.grad(lambda x, n=n: go.energy(u, {**p, n: x}))(p[n])) for n in ('eps_t', 'kappa')])
        return float(jnp.abs(W @ g).max()), float(np.abs(np.asarray(V @ g) + dEdp).max())
    assert max(constraint_residuals(u, W, V)) < 1e-14
    assert abs(jax.grad(lambda x: go.energy(u, {**p, 'eps_t': x}))(p['eps_t'])) > 1e-3        # the constraint is not trivial
    for _ in range(3):
        u, W, V = streamer._segment(u, W, V, p)[:3]
    assert max(constraint_residuals(u, W, V)) < 1e-8


def _orbit(rhs, u0, p, dt, nstep, every):
    """RK4 orbit sampled every `every` steps (the scheme of nilss_stream.py)."""
    @jax.jit
    def chunk(u):
        def step(u, _):
            k1 = rhs(u, p)
            k2 = rhs(u + 0.5 * dt * k1, p)
            k3 = rhs(u + 0.5 * dt * k2, p)
            k4 = rhs(u + dt * k3, p)
            return u + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4), None
        return jax.lax.scan(step, u, None, length=every)[0]
    us = [jnp.asarray(u0, dtype=float)]
    for _ in range(nstep // every):
        us.append(chunk(us[-1]))
    return np.array(us)


def test_shadowing_direction_is_the_displacement_of_the_shadowing_orbit():
    # to first order in delta, the orbit of the system with rho + delta that starts from u0 + delta v(0) is displaced from the
    # orbit of the system with rho by delta v(t) (up to a shift along the flow): the norm of v perpendicular to the flow
    # that info['vnorm'] reports must be that displacement
    dt, T_seg, nseg, delta = 0.005, 0.5, 20, 1e-7
    streamer = NILSSStreamer(lorenz_rhs, lorenz_J, ('rho',), dt, T_seg, 2)
    _, _, info = streamer.run_segments(lorenz63.DEFAULTS, U0, nseg, 0, w0=W0)
    W, _ = streamer._start(jnp.array(U0), jnp.array(W0), lorenz63.DEFAULTS)
    v0 = np.asarray(info['a'][0, :, 0] @ np.asarray(W))
    every = int(round(T_seg / dt))
    base = _orbit(lorenz_rhs, np.array(U0), lorenz63.DEFAULTS, dt, nseg * every, every)
    pert = _orbit(lorenz_rhs, np.array(U0) + delta * v0, {**lorenz63.DEFAULTS, 'rho': lorenz63.DEFAULTS['rho'] + delta},
                  dt, nseg * every, every)
    # (the representation v* + sum a w and the two orbits agree up to differences of the discretisations, which the unstable
    # direction amplifies by e^(0.9 t): 1e-5 after 10 segments, 1e-2 after 20 -- hence the segments checked)
    for k in (5, 10, 15):
        d = pert[k] - base[k]
        f = np.asarray(lorenz_rhs(jnp.array(base[k]), lorenz63.DEFAULTS))
        d_perp = d - (d @ f) / (f @ f) * f
        assert np.linalg.norm(d_perp) / delta == pytest.approx(info['vnorm'][k - 1, 0], rel=2e-3)
    assert np.all(np.isfinite(info['vnorm'])) and info['vnorm'].shape == (nseg, 1)


def test_shadowing_direction_on_the_energy_surface_of_the_guiding_center_flow():
    # same check for the 4D flow with the energy constraint: the orbit of the system with eps_t + delta that starts from
    # u0 + delta v(0) is displaced by delta v(t), and it stays on its own energy surface (the constraint of the initial v)
    p = {**go.default_params, 'eps_h': 0.1, 'eps_t': 0.04, 'eps_m': 0.02, 'N': 4.0, 'iota': 1.1, 'lam': 0.9, 'kappa': 0.3}
    u0 = np.array([0.1, 1.0, 2.0, 0.3])
    u0[3] = np.sqrt(1.0 - p['lam'] * float(go.B_derivs(*u0[:3], p)[0]))
    dt, T_seg, nseg, delta = 0.01, 5.0, 12, 1e-7
    streamer = NILSSStreamer(go.rhs4, lambda u: u[0], ('eps_t',), dt, T_seg, 1, invariant=go.energy)
    w0 = np.random.RandomState(4).rand(1, 4)
    _, _, info = streamer.run_segments(p, u0, nseg, 0, w0=w0)
    W, V = streamer._start(jnp.array(u0), jnp.array(w0), p)
    v0 = np.asarray(V[0] + info['a'][0, :, 0] @ W)
    every = int(round(T_seg / dt))
    p_pert = {**p, 'eps_t': p['eps_t'] + delta}
    base = _orbit(go.rhs4, u0, p, dt, nseg * every, every)
    pert = _orbit(go.rhs4, u0 + delta * v0, p_pert, dt, nseg * every, every)
    assert abs(float(go.energy(jnp.array(u0 + delta * v0), p_pert)) - float(go.energy(jnp.array(u0), p))) < 1e-12
    for k in (4, 8, 12):
        d = pert[k] - base[k]
        f = np.asarray(go.rhs4(jnp.array(base[k]), p))
        d_perp = d - (d @ f) / (f @ f) * f
        assert np.linalg.norm(d_perp) / delta == pytest.approx(info['vnorm'][k - 1, 0], rel=1e-4)
        assert abs(float(go.energy(jnp.array(pert[k]), p_pert)) - float(go.energy(jnp.array(base[k]), p))) < 1e-9
