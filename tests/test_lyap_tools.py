"""nilss_jax.lyapunov: the QR Lyapunov spectrum used to decide whether NILSS can apply."""
import numpy as np
import jax.numpy as jnp

from nilss_jax import lyapunov as lyap_tools

SIG, RHO, BETA = 10.0, 28.0, 8.0 / 3.0


def lorenz(u):
    return jnp.array([SIG * (u[1] - u[0]), u[0] * (RHO - u[2]) - u[1], u[0] * u[1] - BETA * u[2]])


def test_lorenz_spectrum_matches_the_literature():
    r = lyap_tools.lyapunov_spectrum(lorenz, [1.0, 1.0, 20.0], 2e-3, 20.0, 600.0)
    assert abs(r['lyap'][0] - 0.9056) < 0.06
    assert abs(r['lyap'][1]) < 0.02
    assert abs(r['lyap'][2] + 14.5723) < 0.1
    assert abs(r['lyap'].sum() + (1 + SIG + BETA)) < 1e-6      # the divergence is constant, so the sum is exact


def test_regular_flow_has_no_positive_exponent():
    # harmonic oscillators: exponents 0
    f = lambda u: jnp.array([u[1], -u[0], 2.0 * u[3], -2.0 * u[2]])
    r = lyap_tools.lyapunov_spectrum(f, [1.0, 0.0, 0.5, 0.5], 1e-2, 0.0, 200.0)
    assert np.max(np.abs(r['lyap'])) < 0.02


def test_extra_arguments_are_traced_and_the_runner_is_reusable():
    f = lambda u, rho: jnp.array([SIG * (u[1] - u[0]), u[0] * (rho - u[2]) - u[1], u[0] * u[1] - BETA * u[2]])
    runner = lyap_tools.make_runner(f, 2e-3)
    a = lyap_tools.lyapunov_spectrum(f, [1.0, 1.0, 20.0], 2e-3, 20.0, 100.0, args=(RHO,), runner=runner)
    b = lyap_tools.lyapunov_spectrum(f, [1.0, 1.0, 20.0], 2e-3, 20.0, 100.0, args=(10.0,), runner=runner)   # rho = 10: no chaos
    ref = lyap_tools.lyapunov_spectrum(lorenz, [1.0, 1.0, 20.0], 2e-3, 20.0, 100.0)
    assert np.allclose(a['lyap'], ref['lyap'], atol=1e-10)
    assert b['lyap'][0] < 0.05


def test_observable_statistics():
    # u[0] = cos(t): the extremes over the run are +-1
    f = lambda u: jnp.array([u[1], -u[0]])
    r = lyap_tools.lyapunov_spectrum(f, [1.0, 0.0], 1e-2, 0.0, 100.0)
    assert abs(r['x_max'] - 1.0) < 1e-3 and abs(r['x_min'] + 1.0) < 1e-3
    assert abs(r['x_mean_parts']).max() < 0.1
