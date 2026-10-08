"""NILSS core, validated on Lorenz 63 (independent of the guiding-center model).

Reference values: Ni & Wang arXiv:1611.00880 sec. 6.1 (sigma=10, beta=8/3, J=z,
dJ/drho ~ 1, 50 segments of 2 time units, dt=0.01, one homogeneous tangent)
and the Lorenz 63 leading Lyapunov exponent 0.9056.
"""
import numpy as np
import pytest

from nilss_jax.systems import lorenz63
from nilss_jax.reference import nilss, solve_shadowing_coeffs

DT = 0.01


def _u0(seed):
    rng = np.random.RandomState(seed)
    return np.array([12.0, 6.8, 36.5]) + rng.rand(3)


def test_schur_complement_matches_full_kkt_system():
    rng = np.random.RandomState(0)
    for nseg, nus in [(5, 2), (4, 1), (1, 3)]:
        Cs, ds = [], []
        for _ in range(nseg):
            A = rng.randn(nus, nus)
            Cs.append(A @ A.T + nus * np.eye(nus))
            ds.append(rng.randn(nus))
        Rs = [rng.randn(nus, nus) for _ in range(nseg - 1)]
        bs = [rng.randn(nus) for _ in range(nseg - 1)]
        a = solve_shadowing_coeffs(Cs, ds, Rs, bs)

        # constraint a_{i+1} = R_{i+1} a_i + b_{i+1}
        for i in range(nseg - 1):
            np.testing.assert_allclose(a[i + 1], Rs[i] @ a[i] + bs[i], atol=1e-10)
        # optimality: solve the KKT system directly, [C B^T; B 0] [a; lambda] = [-d; b]
        n, m = nseg * nus, (nseg - 1) * nus
        K = np.zeros((n + m, n + m))
        rhs = np.zeros(n + m)
        for i in range(nseg):
            K[i * nus:(i + 1) * nus, i * nus:(i + 1) * nus] = Cs[i]
            rhs[i * nus:(i + 1) * nus] = -ds[i]
        for i in range(nseg - 1):
            rows = slice(n + i * nus, n + (i + 1) * nus)
            K[rows, (i + 1) * nus:(i + 2) * nus] = np.eye(nus)
            K[rows, i * nus:(i + 1) * nus] = -Rs[i]
            K[(i + 1) * nus:(i + 2) * nus, rows] = np.eye(nus)
            K[i * nus:(i + 1) * nus, rows] = -Rs[i].T
            rhs[rows] = bs[i]
        np.testing.assert_allclose(a.ravel(), np.linalg.solve(K, rhs)[:n], atol=1e-8)


def test_integrator_dt_must_match():
    integ, fjj = lorenz63.make_problem(DT)
    with pytest.raises(ValueError, match='dt'):
        nilss(2 * DT, 3, 0.5, 0, _u0(0), 1, 'rho', 28.0, integ, fjj)


@pytest.mark.parametrize('nus', [1, 2])
def test_single_segment_equals_multi_segment(nus):
    """The multi-segment problem (rescaling + continuity) is the single-segment
    problem in other coordinates, so both must give the same sensitivity."""
    integ, fjj = lorenz63.make_problem(DT)
    u0 = _u0(1)
    w0 = np.random.RandomState(2).rand(nus, 3)
    single = nilss(DT, 1, 2.0, 0, u0, nus, 'rho', 28.0, integ, fjj, w0=w0)
    multi = nilss(DT, 4, 0.5, 0, u0, nus, 'rho', 28.0, integ, fjj, w0=w0)
    np.testing.assert_allclose(single[0], multi[0], rtol=1e-12)
    # equal up to the quadrature of d(J)/dt over the segments
    np.testing.assert_allclose(single[1], multi[1], rtol=2e-3)


@pytest.mark.parametrize('par', ['sigma', 'rho', 'beta'])
def test_every_parameter_runs(par):
    integ, fjj = lorenz63.make_problem(DT)
    J, dJ = nilss(DT, 5, 1.0, 5, _u0(3), 1, par, lorenz63.DEFAULTS[par], integ, fjj)
    assert np.isfinite(J) and np.isfinite(dJ)


def test_lorenz_sensitivity_and_lyapunov_exponent_match_literature():
    integ, fjj = lorenz63.make_problem(DT)
    dJ, lyap = [], []
    for seed in range(6):
        np.random.seed(seed)
        _, g, info = nilss(DT, 50, 2.0, 20, _u0(seed), 1, 'rho', 28.0, integ, fjj, return_info=True)
        dJ.append(g)
        lyap.append(info['lyap'][0])
    assert abs(np.mean(dJ) - 1.01) < 0.1, dJ
    assert abs(np.mean(lyap) - 0.9056) < 0.06, lyap
