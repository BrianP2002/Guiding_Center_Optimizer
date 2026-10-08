"""Lorenz 63, the reference problem for validating the NILSS implementation independently of any application.

Ni & Wang (arXiv:1611.00880, sec. 6.1) use sigma = 10, beta = 8/3, J = z and report d<z>/d(rho) ~ 1 for rho in the chaotic
range; the leading Lyapunov exponent at rho = 28 is 0.9056.

Two interfaces:

* ``rhs(u, p)`` / ``J(u)`` for :class:`nilss_jax.NILSS` (JAX, differentiated automatically);
* ``make_problem(dt)`` for :mod:`nilss_jax.reference` (NumPy, hand-written Jacobian; used to cross-check the JAX code).
"""
import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

DEFAULTS = {'sigma': 10.0, 'rho': 28.0, 'beta': 8.0 / 3.0}


def rhs(u, p):
    """du/dt of the Lorenz 63 flow; u = (x, y, z), p a dict with sigma, rho, beta."""
    x, y, z = u
    return jnp.array([p['sigma'] * (y - x), x * (p['rho'] - z) - y, x * y - p['beta'] * z])


def J(u):
    """The objective of the NILSS paper: the height z."""
    return u[2]


def initial_condition(seed=0):
    """A point near the attractor (random in a small box), reproducible from ``seed``."""
    return np.array([12.0, 6.8, 36.5]) + np.random.RandomState(seed).rand(3)


# ---- the NumPy interface of nilss_jax.reference -----------------------------------------------------------------

def _f(u, p):
    x, y, z = u
    return np.array([p['sigma'] * (y - x), x * (p['rho'] - z) - y, x * y - p['beta'] * z])


def _Df(u, p):
    x, y, z = u
    return np.array([[-p['sigma'], p['sigma'], 0.0],
                     [p['rho'] - z, -1.0, -x],
                     [y, x, -p['beta']]])


def _dfdpar(u, p, par):
    x, y, z = u
    if par == 'sigma':
        return np.array([y - x, 0.0, 0.0])
    if par == 'rho':
        return np.array([0.0, x, 0.0])
    if par == 'beta':
        return np.array([0.0, 0.0, -z])
    raise KeyError(f'unknown Lorenz63 parameter {par!r}')


def make_problem(dt, base=None):
    """Return (integrator, fJJu) in the :mod:`nilss_jax.reference` interface; J = z."""
    base = dict(DEFAULTS, **(base or {}))

    def ddt(u, w, vstar, p, par):
        Df = _Df(u, p)
        return _f(u, p), (Df @ w.T).T, Df @ vstar + _dfdpar(u, p, par)

    def integrator(u, w, vstar, par, s):
        p = dict(base)
        p[par] = s
        y = (u, w, vstar)
        k0 = ddt(*y, p, par)
        k1 = ddt(*(a + 0.5 * dt * k for a, k in zip(y, k0)), p, par)
        k2 = ddt(*(a + 0.5 * dt * k for a, k in zip(y, k1)), p, par)
        k3 = ddt(*(a + dt * k for a, k in zip(y, k2)), p, par)
        return tuple(a + dt / 6.0 * (c0 + 2 * c1 + 2 * c2 + c3)
                     for a, c0, c1, c2, c3 in zip(y, k0, k1, k2, k3))

    def fJJu(u, par, s):
        p = dict(base)
        p[par] = s
        return _f(u, p), u[2], np.array([0.0, 0.0, 1.0])

    integrator.dt = dt
    return integrator, fJJu
