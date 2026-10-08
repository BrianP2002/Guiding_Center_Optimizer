"""nilss_jax.diagnostics: tail index, error scaling and the reliability verdicts."""
import numpy as np
import pytest

from nilss_jax import NILSSResult, hill_index, reliability_report, tail_index
from nilss_jax.diagnostics import OK, CAUTION, UNRELIABLE, convergence_exponent

DATA = 'tests/data/gc581_runs.npz'


def _fake_run(rng, nseg=400, npar=1, vnorm=None, segments=None, J=1.0, lyap=0.9, T=200.0, params=('p',)):
    vnorm = np.abs(rng.randn(nseg, npar)) + 1.0 if vnorm is None else vnorm
    segments = rng.randn(nseg, npar) * 0.01 if segments is None else segments
    return NILSSResult(J=J + 0.01 * rng.randn(), dJdp={n: float(segments[:, i].sum()) for i, n in enumerate(params)},
                       lyapunov=np.array([lyap]), T=T, params=tuple(params), nus=1, T_seg=0.5, dt=0.005, segments=segments,
                       vnorm=vnorm, J_segments=J + 0.1 * rng.randn(nseg), coeffs=np.zeros((nseg, 1, npar)))


def test_hill_index_recovers_pareto_and_flags_light_tails():
    rng = np.random.RandomState(0)
    for alpha in (1.0, 2.0):
        x = rng.pareto(alpha, 200000) + 1.0                        # P(X > t) = t^-alpha
        assert hill_index(x, 2000) == pytest.approx(alpha, rel=0.1)
    assert hill_index(np.abs(rng.randn(50000)), 500) > 4.0
    assert np.isnan(hill_index(np.arange(1.0, 6.0), 10))
    assert tail_index(rng.pareto(1.0, 20000) + 1.0) == pytest.approx(1.0, rel=0.25)


def test_convergence_exponent():
    rng = np.random.RandomState(1)
    assert convergence_exponent(rng.randn(20, 512)) == pytest.approx(0.5, abs=0.1)         # finite variance
    assert convergence_exponent(rng.standard_cauchy((20, 512))) < 0.2                        # no convergence
    assert np.isnan(convergence_exponent(rng.randn(1, 10)))                                  # too short


def test_well_behaved_runs_pass_all_checks():
    rng = np.random.RandomState(2)
    runs = [_fake_run(rng) for _ in range(12)]
    rep = reliability_report(runs)
    assert rep.verdict == OK, str(rep)
    assert rep.n_used == 12 and 'verdict: OK' in str(rep)


def test_heavy_tails_are_unreliable():
    rng = np.random.RandomState(3)
    runs = [_fake_run(rng, vnorm=(rng.pareto(1.0, (400, 1)) + 1.0), segments=rng.standard_cauchy((400, 1)) * 0.01) for _ in range(12)]
    rep = reliability_report(runs)
    assert rep.verdict == UNRELIABLE
    assert any(c.name.startswith('tail of |v|') and c.status == UNRELIABLE for c in rep.checks)
    assert 'infinite variance' in str(rep)


def test_unresolved_exponent_and_few_runs():
    rng = np.random.RandomState(4)
    rep = reliability_report([_fake_run(rng, lyap=1e-4) for _ in range(10)])
    assert rep.verdict == UNRELIABLE and rep.n_used == 0                                     # all dropped: lambda_1 T = 0.02
    rep = reliability_report([_fake_run(rng) for _ in range(3)])
    assert rep.verdict == CAUTION and any(c.name == 'number of runs' for c in rep.checks)


def test_ergodicity_and_exponent_spread_checks():
    rng = np.random.RandomState(5)
    runs = [_fake_run(rng, J=1.0 + 2.0 * (i % 2)) for i in range(12)]                       # two groups with different <J>
    rep = reliability_report(runs)
    assert any(c.name == 'ergodicity' and c.status == CAUTION for c in rep.checks)
    runs = [_fake_run(rng, lyap=0.9 if i else 0.2) for i in range(12)]                      # one orbit with a much smaller exponent
    assert any(c.name == 'exponent spread' and c.status == CAUTION for c in reliability_report(runs).checks)


def test_nus_consistency():
    rng = np.random.RandomState(6)
    a = [_fake_run(rng) for _ in range(12)]
    b = [_fake_run(rng, segments=r.segments.copy()) for r in a]                              # same sensitivities
    assert any(c.name == 'nus consistency' and c.status == OK for c in reliability_report(a, nus_check=b).checks)
    b = [_fake_run(rng, segments=r.segments + (5.0 if i < 5 else 0.0)) for i, r in enumerate(a)]
    assert any(c.name.startswith('nus consistency') and c.status == CAUTION for c in reliability_report(a, nus_check=b).checks)


def test_real_data_of_the_chaotic_sea_is_flagged_unreliable():
    # 10 NILSS runs (T = 2e5) on the chaotic sea of the guiding-center flow (scan draw 581): the shadowing direction has a
    # power-law tail with index 1, and the diagnostics must say so
    z = np.load(DATA)
    pars = tuple(str(s) for s in z['pars'])
    runs = [NILSSResult(J=float(z['J'][i]), dJdp=dict(zip(pars, z['dJdp'][i])), lyapunov=np.array([z['lyap'][i]]), T=float(z['T']),
                        params=pars, nus=1, T_seg=float(z['T_seg']), dt=float(z['dt']), segments=z['segments'][i].astype(float),
                        vnorm=z['vnorm'][i].astype(float), J_segments=np.full(z['segments'].shape[1], float(z['J'][i])),
                        coeffs=np.zeros((z['segments'].shape[1], 1, len(pars)))) for i in range(10)]
    rep = reliability_report(runs)
    assert rep.verdict == UNRELIABLE
    tails = [c for c in rep.checks if c.name.startswith('tail of |v|')]
    assert len(tails) == 4 and all(c.status == UNRELIABLE for c in tails)
    alpha = [float(c.value.split()[-1]) for c in tails]
    assert all(0.7 < a < 1.4 for a in alpha), alpha
