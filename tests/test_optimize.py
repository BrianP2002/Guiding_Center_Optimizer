"""nilss_jax.optimize: validation and a short optimisation on Lorenz 63."""
import numpy as np
import pytest

from nilss_jax import NILSS, optimize
from nilss_jax.systems import lorenz63


def _setup(params=('rho',)):
    return NILSS(lorenz63.rhs, lorenz63.J, params, dt=0.01, T_seg=0.5, nus=1), np.array([lorenz63.initial_condition(i) for i in range(4)])


def test_argument_checks():
    nil, u0s = _setup()
    with pytest.raises(ValueError, match='not differentiated'):
        optimize.minimize(nil, lorenz63.DEFAULTS, ['sigma'], u0s, T=10.0)
    with pytest.raises(ValueError, match='fd_h'):
        optimize.minimize(nil, lorenz63.DEFAULTS, ['rho'], u0s, T=10.0, gradient='fd')
    with pytest.raises(ValueError, match='gradient'):
        optimize.minimize(nil, lorenz63.DEFAULTS, ['rho'], u0s, T=10.0, gradient='adjoint')
    with pytest.raises(ValueError, match='on_unreliable'):
        optimize.minimize(nil, lorenz63.DEFAULTS, ['rho'], u0s, T=10.0, on_unreliable='x')


def test_lorenz_target_with_nilss_gradients():
    # <z>(rho) is close to rho - 4.5 on 25 < rho < 40 with slope about 1: ask for <z> = 30.5, expect rho near 35
    nil, u0s = _setup()
    res = optimize.minimize(nil, lorenz63.DEFAULTS, ['rho'], u0s, T=60.0, T_spinup=10.0, bounds={'rho': (26.0, 42.0)}, target=30.5,
                            maxiter=8, on_unreliable='ignore')
    assert abs(res.J - 30.5) < 1.0 and 31.0 < res.x['rho'] < 40.0
    assert len(res.history) >= 3 and all(h['verdict'] in ('ok', 'caution', 'unreliable') for h in res.history)


def test_lorenz_maximise_with_finite_difference_gradients():
    nil, u0s = _setup()
    res = optimize.minimize(nil, lorenz63.DEFAULTS, ['rho'], u0s, T=60.0, T_spinup=10.0, bounds={'rho': (26.0, 34.0)}, maximize=True,
                            gradient='fd', fd_h=1.0, maxiter=6)
    assert res.x['rho'] > 32.0                      # <z> grows with rho: the optimiser runs to the upper bound
