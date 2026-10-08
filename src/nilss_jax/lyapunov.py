"""Lyapunov spectrum of an autonomous flow du/dt = f(u, *args) by the QR (Benettin) method.

Independent of the NILSS code on purpose: it is used to check, before a sensitivity is trusted, whether a system has the
positive Lyapunov exponent that the NILSS assumptions need. The whole integration runs inside ``jax.lax.scan``, so
10^6 steps take seconds on one CPU core.
"""
from functools import partial

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np

__all__ = ['make_runner', 'lyapunov_spectrum']


def make_runner(f, dt, steps_per_chunk=100, obs=lambda u: u[:1]):
    """
    Returns run(u, P, nchunk, *args) -> (u, P, per-chunk mean of obs, max of obs, min of obs, log|diag R|).
    f(u, *args) is the flow; the extra arguments are traced, so one compiled runner serves any parameter values.
    obs(u) is a vector of observables, by default u[0].
    """
    Df = jax.jacfwd(f)

    @partial(jax.jit, static_argnames=['nchunk'])
    def run(u, P, nchunk, *args):
        def rhs(u, P):
            return f(u, *args), Df(u, *args) @ P

        def rk4(carry, _):
            u, P = carry
            k1u, k1P = rhs(u, P)
            k2u, k2P = rhs(u + 0.5 * dt * k1u, P + 0.5 * dt * k1P)
            k3u, k3P = rhs(u + 0.5 * dt * k2u, P + 0.5 * dt * k2P)
            k4u, k4P = rhs(u + dt * k3u, P + dt * k3P)
            u = u + dt / 6.0 * (k1u + 2 * k2u + 2 * k3u + k4u)
            P = P + dt / 6.0 * (k1P + 2 * k2P + 2 * k3P + k4P)
            return (u, P), obs(u)

        def chunk(carry, _):
            carry, os_ = jax.lax.scan(rk4, carry, None, length=steps_per_chunk)
            u, P = carry
            Q, R = jnp.linalg.qr(P)
            return (u, Q), (os_.mean(axis=0), os_.max(axis=0), os_.min(axis=0), jnp.log(jnp.abs(jnp.diag(R))))

        (u, P), out = jax.lax.scan(chunk, (u, P), None, length=nchunk)
        return (u, P) + out

    return run


def lyapunov_spectrum(f, u0, dt, T_spinup, T, steps_per_chunk=100, checkpoints=4, args=(), runner=None):
    """
    Spectrum (descending) of the flow f from u0 after a spin-up of T_spinup time units, estimated over T.
    Also returns the trajectory averages of u[0] over `checkpoints` equal parts and the
    finite-time estimates of the exponents at the end of each part.
    A runner from make_runner(f, dt, steps_per_chunk) can be given to avoid recompiling for every call.
    """
    dtc = dt * steps_per_chunk
    n_spin, n_run = int(round(T_spinup / dtc)), int(round(T / dtc)) // checkpoints * checkpoints
    run = runner if runner is not None else make_runner(f, dt, steps_per_chunk)
    u = jnp.asarray(u0, dtype=float)
    P = jnp.eye(len(u0))
    if n_spin > 0:
        u, P = run(u, P, n_spin, *args)[:2]
    u, P, omean, omax, omin, logs = run(u, P, n_run, *args)
    omean, omax, omin, logs = map(np.asarray, (omean, omax, omin, logs))
    parts = n_run // checkpoints
    cum = np.cumsum(logs, axis=0)
    finite_time = np.array([cum[(k + 1) * parts - 1] / ((k + 1) * parts * dtc) for k in range(checkpoints)])
    return {
        'lyap': np.sort(finite_time[-1])[::-1],
        'lyap_finite_time': np.sort(finite_time, axis=1)[:, ::-1],
        'x_mean_parts': omean[:, 0].reshape(checkpoints, parts).mean(axis=1),
        'x_max': float(omax[:, 0].max()),
        'x_min': float(omin[:, 0].min()),
        'obs_max': omax.max(axis=0),
        'obs_min': omin.min(axis=0),
        'finite': bool(np.all(np.isfinite(logs)) and np.all(np.isfinite(omean))),
        'u_final': np.asarray(u),
        'T': n_run * dtc,
    }
