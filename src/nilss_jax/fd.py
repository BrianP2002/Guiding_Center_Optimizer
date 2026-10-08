"""Finite-difference reference for the sensitivity of a long-time average.

The long-time average <J>(p) of an ergodic system is a function of the parameters that can be differentiated numerically with
no tangent equations at all: average J over many independent orbits at several parameter values and fit a line. It costs
``npoint * ncopy * (T + T_spinup) / dt`` Runge-Kutta steps, which is why NILSS exists, but it is the check that tells whether
a NILSS result can be believed in a system that is not uniformly hyperbolic (see ``docs/reliability.md``).

    fd = finite_difference(rhs, J, p, 'rho', h=0.5, u0s=u0s, T=2000., dt=0.005, T_spinup=50.)
    print(fd.summary())
    print(compare(ens, {'rho': fd}))        # NILSS ensemble against the finite-difference slope

The same initial conditions (displaced once by ``jitter``) are used at every parameter value, so that the points of the line
differ by the parameter and not by the sample. ``project`` re-projects them at every value (the energy shell of a Hamiltonian
flow depends on the parameters); ``select`` keeps only some copies at each value (e.g. those on the chaotic sea).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import Callable, Mapping, Optional

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

__all__ = ['FDResult', 'time_average', 'finite_difference', 'fit_slope', 'compare']


@partial(jax.jit, static_argnames=('rhs', 'J', 'nblock', 'steps'))
def _block_means(rhs, J, u0s, p, dt, nblock, steps):
    """Mean of J over each of ``nblock`` blocks of ``steps`` RK4 steps, for every initial condition (vmap)."""
    def rk4(u):
        k1 = rhs(u, p)
        k2 = rhs(u + 0.5 * dt * k1, p)
        k3 = rhs(u + 0.5 * dt * k2, p)
        k4 = rhs(u + dt * k3, p)
        return u + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

    def one(u0):
        def inner(u, _):
            u = rk4(u)
            return u, J(u)

        def block(u, _):
            u, js = jax.lax.scan(inner, u, None, length=steps)
            return u, js.mean()
        return jax.lax.scan(block, u0, None, length=nblock)[1]
    return jax.vmap(one)(u0s)


def time_average(rhs: Callable, J: Callable, p: Mapping, u0s, T: float, dt: float, *, T_spinup: float = 0.0, parts: int = 20,
                 block_steps: int = 100) -> np.ndarray:
    """Long-time average of J along an orbit from each row of ``u0s``, as ``parts`` equal sub-averages: [ncopy, parts].

    The orbit is integrated for ``T_spinup`` (not counted) and then ``T`` (rounded down to a multiple of ``parts`` blocks of
    ``block_steps`` steps). ``.mean(axis=1)`` is the long-time average of each copy; the scatter of the parts is its sampling error.
    """
    u0s = jnp.atleast_2d(jnp.asarray(u0s, dtype=float))
    block = dt * block_steps
    nspin, nrun = int(round(T_spinup / block)), int(round(T / block)) // parts * parts
    if nrun < parts:
        raise ValueError(f'T = {T} is too short for {parts} blocks of {block:g}')
    p = {k: (float(v) if np.ndim(v) == 0 else v) for k, v in p.items()}
    means = np.asarray(_block_means(rhs, J, u0s, p, dt, nspin + nrun, block_steps))[:, nspin:]
    return means.reshape(len(u0s), parts, -1).mean(axis=2)


def fit_slope(values, means, sems):
    """Weighted straight-line fit: slope, its error, chi2 and degrees of freedom."""
    v, m, se = (np.asarray(a, dtype=float) for a in (values, means, sems))
    w = 1.0 / se ** 2
    X = np.vstack([np.ones_like(v), v - v.mean()]).T
    cov = np.linalg.inv(X.T @ (X * w[:, None]))
    b = cov @ (X.T @ (w * m))
    return float(b[1]), float(np.sqrt(cov[1, 1])), float(np.sum(w * (m - X @ b) ** 2)), len(v) - 2


@dataclass
class FDResult:
    """Finite-difference sensitivity of <J> with respect to one parameter."""
    par: str
    values: np.ndarray            # parameter values
    means: np.ndarray             # ensemble mean of <J> at each value
    sems: np.ndarray              # standard error of that mean (spread of the copies)
    n_copies: np.ndarray          # copies kept at each value
    slope: float
    slope_err: float
    chi2: float
    dof: int
    copy_means: list = field(default_factory=list)      # per value, the long-time average of each kept copy

    def summary(self) -> str:
        lines = [f'finite differences in {self.par}: slope d<J>/d{self.par} = {self.slope:+.5g} +- {self.slope_err:.2g} '
                 f'(straight line through {len(self.values)} points, chi2/dof = {self.chi2:.1f}/{self.dof})']
        lines += [f'  {self.par} = {v:.6g}: <J> = {m:.6g} +- {s:.2g}  ({n} copies)'
                  for v, m, s, n in zip(self.values, self.means, self.sems, self.n_copies)]
        if self.dof > 0 and self.chi2 / self.dof > 3:
            lines.append('  chi2/dof is large: the response is not linear over this window or the copies are not one ergodic set; use a smaller h or more copies')
        return '\n'.join(lines)


def finite_difference(rhs: Callable, J: Callable, p: Mapping, par: str, h: float, u0s, T: float, dt: float, *, npoint: int = 5,
                      T_spinup: float = 0.0, project: Optional[Callable] = None, jitter: float = 1e-6, seed: int = 0,
                      select: Optional[Callable] = None, parts: int = 20, block_steps: int = 100) -> FDResult:
    """Slope of the ensemble-averaged long-time <J> with respect to ``par`` at ``p[par]``, from ``npoint`` values spaced by ``h``.

    Parameters
    ----------
    u0s : [ncopy, n] initial conditions of the copies (on the attractor, or brought there by ``T_spinup``).
    project : ``project(u, p) -> u`` applied to every copy at every parameter value (e.g. back onto an energy shell).
    select : ``select(parts) -> bool mask [ncopy]`` given the sub-averages [ncopy, parts] of the copies at one value; the copies
        it rejects are left out (e.g. copies stuck on regular tori). The default keeps all.
    jitter : the copies are displaced once by this much (Gaussian) so that they are independent orbits even from identical inputs.
    """
    if par not in p:
        raise KeyError(f'{par!r} is not in p (keys: {sorted(p)})')
    u0s = np.atleast_2d(np.asarray(u0s, dtype=float))
    u0s = u0s + jitter * np.random.RandomState(seed).randn(*u0s.shape)
    values = p[par] + h * (np.arange(npoint) - npoint // 2)
    means, sems, ns, per_copy = [], [], [], []
    for v in values:
        pv = {**p, par: float(v)}
        u = np.array([project(u0, pv) for u0 in u0s]) if project is not None else u0s
        sub = time_average(rhs, J, pv, u, T, dt, T_spinup=T_spinup, parts=parts, block_steps=block_steps)
        keep = np.ones(len(sub), dtype=bool) if select is None else np.asarray(select(sub), dtype=bool)
        cm = sub.mean(axis=1)[keep]
        if len(cm) < 2:
            raise ValueError(f'fewer than 2 copies kept at {par} = {v:g}')
        means.append(cm.mean())
        sems.append(cm.std(ddof=1) / np.sqrt(len(cm)))
        ns.append(len(cm))
        per_copy.append(cm)
    slope, err, chi2, dof = fit_slope(values, means, sems)
    return FDResult(par, values, np.array(means), np.array(sems), np.array(ns), slope, err, chi2, dof, per_copy)


def compare(ensemble, fd: Mapping) -> str:
    """Table of the NILSS ensemble (mean +- SEM and median) against finite-difference results ``{parameter: FDResult}``."""
    mean, sem, med = ensemble.mean(), ensemble.sem(), ensemble.median()
    lines = ['parameter     NILSS mean +- SEM        NILSS median     finite differences        (NILSS mean - FD) / combined error']
    for n, r in fd.items():
        z = (mean[n] - r.slope) / np.hypot(sem[n], r.slope_err)
        lines.append(f'{n:<12} {mean[n]:+10.4g} +- {sem[n]:<9.3g} {med[n]:+10.4g}     {r.slope:+10.4g} +- {r.slope_err:<9.3g}  {z:+7.1f}')
    return '\n'.join(lines)
