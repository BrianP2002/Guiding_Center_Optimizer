"""The command line and the bundled guiding-center helpers."""
import numpy as np
import pytest

import nilss_jax
from nilss_jax import cli
from nilss_jax.systems import guiding_center as gc


def test_version(capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(['--version'])
    assert e.value.code == 0 and nilss_jax.__version__ in capsys.readouterr().out


def test_lorenz_command(capsys):
    assert cli.main(['lorenz', '--runs', '2', '--T', '30']) == 0
    out = capsys.readouterr().out
    assert 'NILSS ensemble' in out and 'reliability report' in out


def test_selftest_passes(capsys):
    assert cli.main(['selftest']) == 0
    assert 'selftest passed' in capsys.readouterr().out


@pytest.mark.parametrize('seed', [581, 835])
def test_guiding_center_draws(seed):
    d = gc.load_draw(seed)
    assert set(d) >= {'params', 'sea_ics', 'reference_fd', 'reference_nilss'} and len(d['sea_ics']) >= 5
    for u in d['sea_ics']:
        u = gc.project_to_energy_shell(u, d['params'])
        assert abs(float(gc.energy(u, d['params'])) - 0.5) < 1e-12                      # on the shell E = 1/2
    with pytest.raises(KeyError):
        gc.load_draw(1)
    with pytest.raises(ValueError):
        gc.project_to_energy_shell([0.01, 0.0, 0.0, 0.1], {**d['params'], 'lam': 5.0})


def test_guiding_center_command(capsys):
    assert cli.main(['gc', '--draw', '835', '--runs', '2', '--T', '400', '--T-spinup', '200', '--par', 'eps_2']) == 0
    out = capsys.readouterr().out
    assert 'eps_2' in out and 'finite differences' in out
