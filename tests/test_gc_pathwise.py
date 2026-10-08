"""experiments/scripts/gc_pathwise.py and summarize_gc_pathwise.py: forward tangents of the official guiding-center flow with an
initial map on the energy shell, read out at several horizons from one integration; and the statistics that are computed
from them."""
import os
import sys

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from nilss_jax.systems import guiding_center as go
import gc_pathwise as gp                # experiments/scripts (on the pytest path)
import summarize_gc_pathwise as sm

PFIX = {**{k: v for k, v in go.default_params.items()}, 'eps_h': 0.1, 'N': 2.0, 'eps_2': 0.01, 'N2': 3.0, 'lam': 1.0, 'G': 1.0}
Q0 = np.array([0.03, 0.02, 0.15, 0.8])                   # eps_t, eps_m, kappa, iota
PF = {k: v for k, v in PFIX.items() if k not in gp.PARS}
BOUNCING = np.array([[0.177, 2.63, 6.098, -1.0]])        # a trapped particle of PF, Q0: about nine sign changes of v_par in T = 200


def _positions(n=4):
    return gp.sample_positions(5, n, PF, Q0)


def test_initial_map_stays_on_the_energy_shell_and_its_derivative_is_the_initial_tangent():
    pos = _positions()
    z0, _ = gp.initial_pathwise_state(pos, PF, Q0)
    for i in range(len(pos)):
        q = jnp.asarray(Q0)
        u = z0[i, :4]
        p = gp.with_q(PF, q)
        assert abs(float(go.energy(u, p)) - 0.5) < 1e-14
        assert abs(float(u[3])) >= np.sqrt(gp.VPAR2_MIN) - 1e-12
        V0 = np.asarray(z0[i, 4:4 + 4 * gp.NPAR]).reshape(gp.NPAR, 4)
        g = np.asarray(jax.grad(go.energy, 0)(u, p))
        dEdq = np.array([float(jax.grad(lambda x, n=n: go.energy(u, gp.with_q(PF, jnp.where(jnp.arange(4) == n, x, q))))(q[n]))
                         for n in range(gp.NPAR)])
        assert np.max(np.abs(V0 @ g + dEdq)) < 1e-13            # the shell moves with the parameters and the tangent follows it
        assert np.max(np.abs(dEdq)) > 1e-3                      # (the check has teeth: eps_t, eps_m do move the shell)
        assert np.all(V0[2:, :] == 0.0)                         # kappa and iota do not enter the initial map
    # a different parameter vector: the shell is that of the new parameters
    q2 = Q0 * np.array([1.1, 0.8, 1.0, 1.0])
    u2 = jax.vmap(lambda x: gp.ic_map(x, jnp.asarray(q2), PF))(jnp.asarray(pos))
    assert max(abs(float(go.energy(u, gp.with_q(PF, q2))) - 0.5) for u in u2) < 1e-14


@pytest.mark.parametrize('n', range(gp.NPAR))
def test_pathwise_tangent_matches_finite_differences(n):
    pos = _positions(3)
    times = (10.0, 20.0)
    res = gp.run_pathwise(pos, PF, Q0, times)
    h = 1e-6 * abs(Q0[n])
    qp, qm = Q0.copy(), Q0.copy()
    qp[n] += h
    qm[n] -= h
    up, um = gp.run_primal(pos, PF, qp, times), gp.run_primal(pos, PF, qm, times)
    fd_u, fd_A = (up['u'] - um['u']) / (2 * h), (up['A'] - um['A']) / (2 * h)
    assert np.max(np.abs(res['V'][:, :, n, :] - fd_u)) < 1e-6 * np.max(np.abs(fd_u))
    assert np.max(np.abs(res['dA'][:, :, n] - fd_A)) < 1e-6 * np.max(np.abs(fd_A))
    assert np.max(np.abs(fd_u)) > 1e-2                           # the derivative is not trivially zero


def test_checkpoints_do_not_depend_on_where_the_orbit_is_read_out():
    pos = _positions(3)
    a = gp.run_pathwise(pos, PF, Q0, (10.0, 30.0))
    b = gp.run_pathwise(pos, PF, Q0, (10.0, 20.0, 30.0))
    for key in ('u', 'V', 'A', 'dA', 'logw', 'nflip', 'smax', 'smin'):
        assert np.array_equal(a[key][:, 0], b[key][:, 0]) and np.array_equal(a[key][:, 1], b[key][:, 2]), key
    # the orbit and A of the primal integrator are those of the tangent integrator
    c = gp.run_primal(pos, PF, Q0, (10.0, 30.0))
    assert np.max(np.abs(c['u'] - a['u'])) < 1e-12 and np.max(np.abs(c['A'] - a['A'])) < 1e-12


def test_counters_follow_the_orbit():
    # a trapped particle (v_par changes sign): sign changes, max and min of s, sampled once per time unit
    pos, pf = BOUNCING, PF
    z0, _ = gp.initial_pathwise_state(pos, pf, Q0)
    u0 = np.asarray(z0[0, :4])
    assert np.all(np.isfinite(u0))
    res = gp.run_pathwise(pos, pf, Q0, (100.0, 200.0))
    traj = np.concatenate([[u0], np.asarray(go.trajectory(go.rhs4, jnp.asarray(u0), gp.with_q(pf, Q0), gp.DT, 20000, 100))])
    for k, T in enumerate((100, 200)):
        tr = traj[:T + 1]
        flips = int(np.sum(np.sign(tr[1:, 3]) * np.sign(tr[:-1, 3]) < 0))
        assert res['nflip'][0, k] == flips
        assert res['smax'][0, k] == pytest.approx(tr[:, 0].max(), abs=1e-12)
        assert res['smin'][0, k] == pytest.approx(tr[:, 0].min(), abs=1e-12)
    assert res['nflip'][0, 1] >= 4                               # the orbit does bounce (the check has teeth)


def test_homogeneous_tangent_growth_is_accumulated_over_the_renormalisations():
    # log|W(T)| of the tangent that is renormalised every time unit equals the log norm of the unrenormalised one
    pos, pf = BOUNCING, PF
    z0, _ = gp.initial_pathwise_state(pos, pf, Q0, w_seed=3)
    assert np.all(np.isfinite(z0))
    p = gp.with_q(pf, Q0)
    Df = jax.jacfwd(go.rhs4, 0)

    @jax.jit
    def run(u, W):
        def deriv(u, W):
            return go.rhs4(u, p), Df(u, p) @ W

        def step(c, _):
            u, W = c
            k1u, k1W = deriv(u, W)
            k2u, k2W = deriv(u + 0.5 * gp.DT * k1u, W + 0.5 * gp.DT * k1W)
            k3u, k3W = deriv(u + 0.5 * gp.DT * k2u, W + 0.5 * gp.DT * k2W)
            k4u, k4W = deriv(u + gp.DT * k3u, W + gp.DT * k3W)
            return (u + gp.DT / 6 * (k1u + 2 * k2u + 2 * k3u + k4u), W + gp.DT / 6 * (k1W + 2 * k2W + 2 * k3W + k4W)), None
        return jax.lax.scan(step, (u, W), None, length=20000)[0]
    iw = 4 + 4 * gp.NPAR
    u, W = run(z0[0, :4], z0[0, iw:iw + 4])
    res = gp.run_pathwise(pos, pf, Q0, (200.0,), w_seed=3)
    assert res['logw'][0, 0] == pytest.approx(float(jnp.log(jnp.linalg.norm(W))), abs=1e-9)
    assert res['logw'][0, 0] > 0.5                               # (the check has teeth: the tangent does grow)


def test_hill_estimator_recovers_pareto_indices():
    rng = np.random.RandomState(0)
    for alpha in (1.0, 2.5):
        x = rng.rand(200000) ** (-1.0 / alpha)
        assert sm.hill(x, 2000) == pytest.approx(alpha, rel=0.08)
    assert sm.hill(np.abs(rng.randn(200000)), 500) > 4.0         # a light tail has a large index


def test_batch_dispersion_scales_like_inverse_square_root_for_finite_variance():
    rng = np.random.RandomState(1)
    x = rng.randn(4000)
    disp = sm.batch_dispersion(x, (10, 40, 160, 640), nboot=4000, rng=np.random.RandomState(2))
    assert disp[10] / disp[640] == pytest.approx(8.0, rel=0.25)
    c = rng.standard_cauchy(4000)
    disp_c = sm.batch_dispersion(c, (10, 40, 160, 640), nboot=4000, rng=np.random.RandomState(3))
    assert disp_c[10] / disp_c[640] < 2.5                        # Cauchy: the spread of the mean does not shrink with the batch size


def test_usable_horizon_rule():
    # relative SEM < 30 % and tail index > 2 at the largest horizon for which both hold
    table = [{'T': 100, 'rel_sem': 0.05, 'index': 5.0}, {'T': 300, 'rel_sem': 0.2, 'index': 3.0},
             {'T': 1000, 'rel_sem': 0.5, 'index': 3.0}, {'T': 3000, 'rel_sem': 0.1, 'index': 1.0}]
    assert sm.usable_horizon(table) == 300
    assert sm.usable_horizon([{'T': 100, 'rel_sem': 0.9, 'index': 1.0}]) is None
    assert sm.index_horizon(table) == 1000                     # the index alone: above 2 up to T = 1000
    assert sm.index_horizon([{'T': 100, 'rel_sem': 0.9, 'index': 1.0}]) is None


def test_command_line_modes_and_the_summary_read_each_other(tmp_path, monkeypatch, capsys):
    out = str(tmp_path)
    for argv in (['0', '--mode', 'pathwise', '--ens', 'E1', '--n', '8', '--per-task', '4', '--times', '10', '20'],
                 ['1', '--mode', 'pathwise', '--ens', 'E1', '--n', '8', '--per-task', '4', '--times', '10', '20'],
                 ['0', '--mode', 'fd', '--ens', 'E1', '--n', '8', '--per-task', '8', '--times', '10', '20', '--h-rel', '1e-6'],
                 ['0', '--mode', 'pop', '--par', 'eps_t', '--per-task', '6', '--times', '10', '20', '--h-rel', '0.1'],
                 ['1', '--mode', 'pop', '--par', 'eps_t', '--per-task', '6', '--times', '10', '20', '--h-rel', '0.1']):
        monkeypatch.setattr(sys, 'argv', ['gc_pathwise.py'] + argv + ['--outdir', out])
        gp.main()
    data = sm.load_pathwise(out, 'E1')
    assert data['u'].shape == (8, 2, 4) and data['V'].shape == (8, 2, gp.NPAR, 4) and list(data['idx']) == list(range(8))
    summary = {}
    sm.pathwise_report(data, 'E1', summary)
    sm.fd_report(out, 'E1', data, summary)
    sm.population_report(out, data, summary)
    text = capsys.readouterr().out
    assert 'finite differences of the sample mean' in text and 'population finite differences' in text
    # with a relative step of 1e-6 the central differences of the sample mean ARE the pathwise derivative (T = 10)
    for key, row in summary['fd'].items():
        if '|10|' in key:
            assert row['fd_mean'] == pytest.approx(row['pw_mean'], rel=1e-5, abs=1e-8)


def test_orbit_can_be_continued_from_a_saved_state():
    pos = _positions(3)
    full, Z, AUX = gp.run_pathwise(pos, PF, Q0, (10.0, 20.0, 30.0), raw=True)
    _, Z1, AUX1 = gp.run_pathwise(pos, PF, Q0, (10.0,), raw=True)
    rest = gp.run_pathwise(pos, PF, Q0, (20.0, 30.0), state=(Z1[:, -1], AUX1[:, -1]), t_start=10.0)
    for key in ('u', 'V', 'A', 'dA', 'logw', 'nflip', 'smax', 'smin'):
        assert np.array_equal(full[key][:, 1:], rest[key]), key


def test_a_task_that_was_cut_off_resumes_from_its_last_horizon(tmp_path, monkeypatch):
    out = str(tmp_path)
    argv = ['gc_pathwise.py', '0', '--mode', 'pathwise', '--ens', 'E1', '--n', '4', '--per-task', '4', '--times', '10', '20', '30', '--outdir', out]
    real = gp._pathwise_advance
    calls = {'n': 0}

    def flaky(*a, **k):
        advance = real(*a, **k)

        def wrapped(z, aux, q, nchunk):
            calls['n'] += 1
            if calls['n'] == 3:                                   # the horizon 30 is never reached
                raise RuntimeError('time limit')
            return advance(z, aux, q, nchunk)
        return wrapped
    monkeypatch.setattr(sys, 'argv', argv)
    monkeypatch.setattr(gp, '_pathwise_advance', flaky)
    with pytest.raises(RuntimeError):
        gp.main()
    assert os.path.exists(os.path.join(out, 'pw_E1_0000.part.npz')) and not os.path.exists(os.path.join(out, 'pw_E1_0000.npz'))
    monkeypatch.setattr(gp, '_pathwise_advance', real)
    gp.main()
    assert not os.path.exists(os.path.join(out, 'pw_E1_0000.part.npz'))
    got = np.load(os.path.join(out, 'pw_E1_0000.npz'))
    pf, q0 = gp.base_parameters()
    pf = {k: v for k, v in pf.items() if k not in gp.PARS}
    ref = gp.run_pathwise(got['pos'], pf, q0, (10.0, 20.0, 30.0), w_seed=1000)
    for key in ('u', 'V', 'A', 'dA', 'logw', 'nflip'):
        assert np.array_equal(got[key], ref[key]), key
