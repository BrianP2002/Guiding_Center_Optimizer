"""NILSS applied to the guiding-center model: end-to-end smoke tests of the entry points,
and the single/multi segment consistency on the real model."""
import sys

import numpy as np
import pytest

import app_gc
import app_gc_opt
from nilss_jax.reference import nilss

U0 = np.array([0.05, 1.0, 2.0])


@pytest.mark.parametrize('par', ['a0', 'a1', 'iota', 'lam', 'lambda', 'G'])
def test_nilss_runs_for_every_parameter(par):
    integ, fjj = app_gc.make_problem(1e-3)
    s = app_gc.default_params[app_gc.canon_par(par)]
    J, dJ = nilss(1e-3, 4, 0.01, 4, U0, 1, par, s, integ, fjj)
    assert np.isfinite(J) and np.isfinite(dJ)


def test_nilss_rejects_wrong_dt():
    # app_gc_opt used to pass dt=1e-4 to nilss while the integrator stepped with 1e-3
    integ, fjj = app_gc.make_problem(1e-3)
    with pytest.raises(ValueError, match='dt'):
        nilss(1e-4, 4, 0.01, 0, U0, 1, 'a0', 0.1, integ, fjj)


def test_single_segment_equals_multi_segment_on_guiding_center():
    integ, fjj = app_gc.make_problem(1e-3)
    w0 = np.random.RandomState(0).rand(1, 3)
    single = nilss(1e-3, 1, 0.2, 0, U0, 1, 'a0', 0.1, integ, fjj, w0=w0)
    multi = nilss(1e-3, 4, 0.05, 0, U0, 1, 'a0', 0.1, integ, fjj, w0=w0)
    np.testing.assert_allclose(single[0], multi[0], rtol=1e-12)
    np.testing.assert_allclose(single[1], multi[1], rtol=2e-3)


def test_app_gc_cli_runs_and_warns_about_unresolved_lyapunov(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['app_gc.py', '--par', 'lambda', '--nseg', '4', '--nseg-ps', '4',
                                      '--step-size', '0.1', '--outdir', str(tmp_path)])
    with pytest.warns(UserWarning, match='lambda_1'):
        app_gc.main()
    data = np.load(tmp_path / 'guiding_center_lam.npz')
    assert data['par_arr'].shape == data['dJdpar_arr'].shape == (3,)
    assert np.all(np.isfinite(data['dJdpar_arr']))
    assert (tmp_path / 'guiding_center_lam.png').exists()


def test_app_gc_opt_cli_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['app_gc_opt.py', '--par', 'lambda', '--nseg', '4', '--nseg-ps', '4',
                                      '--maxiter', '2', '--outdir', str(tmp_path)])
    app_gc_opt.main()
    text = (tmp_path / 'optimization_guiding_center_results_lam.txt').read_text()
    assert 'Optimal lam' in text and 'Evaluations' in text
