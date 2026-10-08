"""nilss_jax.ensemble: statistics over runs, saving and collecting, worker processes."""
import numpy as np
import pytest

from nilss_jax import NILSS, load_ensemble, run_ensemble
from nilss_jax.systems import lorenz63


def _nilss(params=('rho',)):
    return NILSS(lorenz63.rhs, lorenz63.J, params, dt=0.01, T_seg=0.5, nus=1)


def _u0s(n):
    return np.array([lorenz63.initial_condition(i) for i in range(n)])


def test_serial_ensemble_statistics_and_summary(tmp_path):
    ens = run_ensemble(_nilss(('rho', 'beta')), lorenz63.DEFAULTS, _u0s(4), T=30.0, T_spinup=5.0, save_dir=str(tmp_path))
    assert len(ens) == 4 and ens.kept.all() and ens.values().shape == (4, 2)
    for stat in (ens.mean(), ens.sem(), ens.median(), ens.trimmed_mean(0.25)):
        assert set(stat) == {'rho', 'beta'}
    lo, hi = ens.median_interval()['rho']
    assert lo <= ens.median()['rho'] <= hi
    text = ens.summary()
    assert 'rho' in text and 'beta' in text and 'reliability' in text
    back = load_ensemble(str(tmp_path))
    assert len(back) == 4 and back.mean() == pytest.approx(ens.mean())


def test_runs_without_resolved_exponent_are_dropped():
    ens = run_ensemble(_nilss(), lorenz63.DEFAULTS, _u0s(3), T=2.0, min_lyap_time=1e6)
    assert not ens.kept.any()
    assert 'none usable' in ens.summary()


def test_seeds_and_input_checks():
    with pytest.raises(ValueError, match='seed'):
        run_ensemble(_nilss(), lorenz63.DEFAULTS, _u0s(3), T=2.0, seeds=[0, 1])
    a = run_ensemble(_nilss(), lorenz63.DEFAULTS, _u0s(1), T=4.0, seeds=[7])
    b = run_ensemble(_nilss(), lorenz63.DEFAULTS, _u0s(1), T=4.0, seeds=[7])
    assert a.results[0].dJdp == b.results[0].dJdp


def test_worker_processes_give_the_same_numbers():
    serial = run_ensemble(_nilss(), lorenz63.DEFAULTS, _u0s(2), T=30.0, T_spinup=2.0)
    par = run_ensemble(_nilss(), lorenz63.DEFAULTS, _u0s(2), T=30.0, T_spinup=2.0, workers=2)
    np.testing.assert_allclose(par.values(), serial.values(), rtol=1e-10)


def test_workers_need_importable_functions():
    nil = NILSS(lambda u, p: lorenz63.rhs(u, p), lorenz63.J, ('rho',), 0.01, 0.5)
    with pytest.raises(ValueError, match='top-level'):
        run_ensemble(nil, lorenz63.DEFAULTS, _u0s(2), T=2.0, workers=2)
